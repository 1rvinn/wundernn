"""
Single Fold Training Script - LSTM with Attention + Input Projection + Post-Attention Norm.

- Intended for parallel training: Set 'fold_to_train' in Args to 0, 1, 2, 3, or 4.
- Uses K-Fold Cross Validation (5 Folds).
- Uses the LSTMAttentionModel with:
  - Input Projection (Improvement C)
  - **Post-Attention LayerNorm (New)**
- Trains on RAW, unscaled data.
"""

import argparse
import math
from pathlib import Path
import time
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

# --- DATASET (Unchanged) ---
class SequenceDataset(Dataset):
    """
    Dataset that yields (raw_input_window, raw_target_state) pairs.
    Data is NOT scaled, and the target is the absolute next state.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.examples = []

        grouped = df.groupby("seq_ix")
        # print(f"Creating {mode} examples...") 
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            # states is now the RAW data
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                target = states[t + 1]
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1]
                
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# --- NEW MODEL: LSTM with Attention AND Input Projection AND Post-Attention Norm ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- IMPROVEMENT C: Input Projection Layer ---
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU() # GELU is also a good option here
        )
        
        # 1. Encoder (The LSTM)
        # Input size is now 'hidden_size' due to projection
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # --- IMPROVEMENT: Post-Attention Normalization ---
        # Stabilizes the context vector before the final projection.
        self.context_norm = nn.LayerNorm(hidden_size * 2)
        
        # 3. Decoder
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # 1. Apply Input Projection
        x_proj = self.input_projection(x)
        
        # 2. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 4. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # --- Apply Post-Attention Norm ---
        context_vector = self.context_norm(context_vector)
        
        # 5. Decode
        return self.fc(context_vector)


# --- EVALUATION (Unchanged) ---
def evaluate(model, dataloader, device):
    model.eval()
    ys_true_abs = []
    ys_pred_abs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            ys_true_abs.append(y.cpu().numpy())
            ys_pred_abs.append(pred.cpu().numpy())

    y_true = np.vstack(ys_true_abs)
    y_pred = np.vstack(ys_pred_abs)
    
    r2s = []
    for i in range(y_true.shape[1]):
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        except Exception:
            r2 = float('nan')
        r2s.append(r2)
    mean_r2 = float(np.nanmean(r2s))
    return mean_r2, r2s


# --- TRAIN FOLD FUNCTION ---
def train_fold(fold_idx, train_ids, val_ids, df_all, args, device):
    print(f"\n--- Starting Fold {fold_idx} / {args.n_folds - 1} ---")
    print(f"Train seqs: {len(train_ids)}, Val seqs: {len(val_ids)}")

    # 1. Prepare Data for this Fold
    df_train = df_all[df_all['seq_ix'].isin(train_ids)]
    df_val = df_all[df_all['seq_ix'].isin(val_ids)]

    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    input_size = len(train_ds.feature_cols)

    # 2. Initialize FRESH Model
    model = LSTMAttentionModel(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    # 3. Initialize FRESH Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    # --- replace optimizer + scheduler setup in train_fold ---
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # # OneCycleLR: choose max_lr (you can use args.lr or a slightly higher value)
    # max_lr = args.lr  # or 3*args.lr to try
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=max_lr,
    #     steps_per_epoch=len(train_loader),
    #     epochs=args.epochs,
    #     pct_start=0.1,       # short warmup
    #     anneal_strategy='cos',
    #     div_factor=10.0,     # initial lr = max_lr / div_factor
    #     final_div_factor=1e4
    # )
    
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_r2 = -math.inf
    checkpoint_name = f"{args.checkpoint_prefix}_fold_{fold_idx}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        val_mean_r2, _ = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step() 

        # Logging
        if val_mean_r2 > best_val_r2:
            best_val_r2 = val_mean_r2
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  [Fold {fold_idx} Epoch {epoch}] New Best R2: {val_mean_r2:.4f} (Saved)")
        else:
            if epoch % 1 == 0: # Print every epoch
                print(f"  [Fold {fold_idx} Epoch {epoch}] Loss: {avg_loss:.4f} | R2: {val_mean_r2:.4f}")

    print(f"--- Finished Fold {fold_idx}. Best R2: {best_val_r2:.4f} ---")
    return best_val_r2


# --- MAIN ---
def main(args):
    p = Path(args.data)
    print(f"Loading data from {p}...")
    df = pd.read_parquet(p)

    seq_ids = df['seq_ix'].unique()
    
    # Initialize K-Fold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(seq_ids)):
        
        # --- SELECTION: ONLY TRAIN THE REQUESTED FOLD ---
        if fold_idx != args.fold_to_train:
            continue
        # ------------------------------------------------
        
        print(f"Selected Fold: {fold_idx} (Target: {args.fold_to_train})")

        train_seq_ids = seq_ids[train_idx]
        val_seq_ids = seq_ids[val_idx]
        
        # Train this fold
        score = train_fold(fold_idx, train_seq_ids, val_seq_ids, df, args, device)
        
        print("\n===============================")
        print(f" FINISHED FOLD {fold_idx} RESULTS")
        print(f" R2 = {score:.6f}")
        print("===============================")
        
        # Break after training the single requested fold
        break


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 20
            
            # K-Fold settings
            self.n_folds = 5
            self.checkpoint_prefix = 'lstm_attention_proj' 
            
            # --- IMPORTANT: CHANGE THIS FOR EACH NOTEBOOK (0, 1, 2, 3, or 4) ---
            self.fold_to_train = 1
            
    args = Args()
    main(args)
