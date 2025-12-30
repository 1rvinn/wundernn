"""
2-Fold Ensemble LSTM with Attention (Raw Data).

- Trains 2 separate models based on deterministic data splits:
  1. Model A: Validates on the BOTTOM 5% of seq_ids (Trains on top 95%).
  2. Model B: Validates on the TOP 5% of seq_ids (Trains on bottom 95%).
- Uses the LSTMAttentionModel.
- Trains on RAW, unscaled data.
"""

import argparse
import math
from pathlib import Path
import time
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
# train_test_split removed in favor of deterministic slicing

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
        print(f"Creating {mode} examples from {len(grouped)} sequences...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            # states is now the RAW data
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                # --- Target is now the RAW ABSOLUTE state ---
                target = states[t + 1]
                # ---

                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1] # This data is RAW
                
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


# --- MODEL (Unchanged) ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Encoder (The LSTM)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder (The final classifier)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
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


def train_one_model(model_name, train_ids, val_ids, df, args, device):
    """
    Helper function to run a full training session for one specific data split.
    """
    print(f"\n=== Starting Training: {model_name} ===")
    print(f"Train Sequences: {len(train_ids)} | Val Sequences: {len(val_ids)}")

    # 1. Subset Data
    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    # 2. Create Datasets
    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    input_size = len(train_ds.feature_cols)

    # 3. Instantiate FRESH Model
    model = LSTMAttentionModel(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # 4. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val = -math.inf
    save_path = f"{model_name}_{args.checkpoint}" # e.g. "top_val_lstm_checkpoint.pt"

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
        
        scheduler.step() 

        print(f"[{model_name}] Epoch {epoch}: Train Loss={avg_loss:.6f} Val R2={val_mean_r2:.6f}")

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), save_path)
            print(f"  --> New Best Saved: {best_val:.6f}")

    print(f"Finished {model_name}. Best R2: {best_val:.6f}")
    return best_val


def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    # 1. Get Unique Sequence IDs and SORT them to ensure deterministic Top/Bottom splitting
    seq_ids = np.sort(df['seq_ix'].unique())
    n_seqs = len(seq_ids)
    
    # Calculate cutoff count based on val_frac (e.g., 0.05)
    n_val = int(n_seqs * args.val_frac)
    
    print(f"Total Sequences: {n_seqs}. Validation Split Size: {n_val} (Top/Bottom 5%)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- MODEL 1: BOTTOM Validation (Last 5% is Val) ---
    # Train on [0 ... N-n_val], Val on [N-n_val ... N]
    val_ids_bottom = seq_ids[-n_val:]
    train_ids_bottom = seq_ids[:-n_val]
    
    train_one_model(
        model_name="model_bottom_val", 
        train_ids=train_ids_bottom, 
        val_ids=val_ids_bottom, 
        df=df, 
        args=args, 
        device=device
    )

    # --- MODEL 2: TOP Validation (First 5% is Val) ---
    # Val on [0 ... n_val], Train on [n_val ... N]
    val_ids_top = seq_ids[:n_val]
    train_ids_top = seq_ids[n_val:]

    train_one_model(
        model_name="model_top_val", 
        train_ids=train_ids_top, 
        val_ids=val_ids_top, 
        df=df, 
        args=args, 
        device=device
    )

    print("\nAll ensemble models trained.")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/content/drive/MyDrive/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 7
            self.val_frac = 0.05 # This determines the Top/Bottom percentage
            self.checkpoint = 'lstm_attention_raw_checkpoint.pt'
            
    args = Args()
    main(args)
