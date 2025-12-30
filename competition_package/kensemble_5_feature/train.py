"""
K-Fold Ensemble Training for LSTMAttentionModel (Raw Data + Rich FE).

- Trains 5 separate models on 5 different 80/20 splits (folds).
- Uses the stateless "sliding window" dataset.
- Uses the LSTMAttentionModel.
- **ADDS FEATURE ENGINEERING (ROC, Volatility, Moving Avg)** to the raw input.
- Trains to predict the absolute next state (X[t+1]).
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """
    Dataset that yields (raw_input_window, raw_target_state) pairs.
    The input 'df' is assumed to already have the engineered features.
    """

    def __init__(self, df: pd.DataFrame, original_feature_cols, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        
        # --- FEATURE ENGINEERING ---
        # The input df already has the new features.
        # Get ALL feature columns (original + new)
        self.all_feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        # Get just the original N features (for the target)
        self.original_feature_cols = original_feature_cols
        self.original_feature_indices = [self.all_feature_cols.index(c) for c in self.original_feature_cols]
        # ---
        
        self.examples = []

        grouped = df.groupby("seq_ix")
        print(f"Creating {mode} examples...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            
            # states now includes all 4*N features
            states = g[self.all_feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                # Target is *only* the raw absolute state (the first N features)
                target = states[t + 1, self.original_feature_indices]
                
                # Input window is the full 4*N features
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1, :]
                
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    # Pad with zeros
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Encoder (The LSTM)
        # Input size is now 4*N
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder (The final classifier)
        # Output size is N (just the original features)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context_vector)


def evaluate(model, dataloader, device):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            ys.append(y.cpu().numpy())
            yps.append(pred.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
    r2s = []
    for i in range(y_true.shape[1]): # y_true is now just N features
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        except Exception:
            r2 = float('nan')
        r2s.append(r2)
    mean_r2 = float(np.nanmean(r2s))
    return mean_r2, r2s


def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    # --- FEATURE ENGINEERING (FIXED) ---
    print("Starting feature engineering...")
    original_feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    
    g = df.groupby('seq_ix')[original_feature_cols]
    
    # 1. Add Momentum (Rate of Change) - 10 steps
    print("Calculating ROC(10)...")
    roc_cols = [f'{c}_roc_10' for c in original_feature_cols]
    # Add .values to strip the index
    df[roc_cols] = g.diff(periods=10).values 
    
    # 2. Add Volatility (Rolling Std Dev) - 20 steps
    print("Calculating VOL(20)...")
    vol_cols = [f'{c}_vol_20' for c in original_feature_cols]
    # Add .values to strip the MultiIndex
    df[vol_cols] = g.rolling(window=20).std().values
    
    # 3. Add Trend (Moving Average) - 20 steps
    print("Calculating MA(20)...")
    ma_cols = [f'{c}_ma_20' for c in original_feature_cols]
    # Add .values to strip the MultiIndex
    df[ma_cols] = g.rolling(window=20).mean().values
    
    print("Filling NaNs...")
    df = df.fillna(0.0) # Fill with 0
    
    all_feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    input_size = len(all_feature_cols) 
    output_size = len(original_feature_cols)
    
    print(f"Feature engineering complete. Input size: {input_size}, Output size: {output_size}")
    # --- END FIX ---

    seq_ids = df['seq_ix'].unique()
    
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_best_val_r2s = []
    base_checkpoint_name = args.checkpoint.replace('.pt', '')

    for fold, (train_index, val_index) in enumerate(kf.split(seq_ids)):
        print("-" * 30)
        print(f"--- FOLD {fold + 1}/{N_SPLITS} ---")
        print("-" * 30)
        
        train_ids = seq_ids[train_index]
        val_ids = seq_ids[val_index]

        df_train = df[df['seq_ix'].isin(train_ids)]
        df_val = df[df['seq_ix'].isin(val_ids)]

        print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")
        
        # Pass the original_feature_cols to the dataset
        train_ds = SequenceDataset(df_train, original_feature_cols, seq_len=args.seq_len, mode='train')
        val_ds = SequenceDataset(df_val, original_feature_cols, seq_len=args.seq_len, mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = LSTMAttentionModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=args.hidden_size, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5)
        criterion = nn.MSELoss()

        best_val = -math.inf
        fold_checkpoint_path = f'{base_checkpoint_name}_fold_{fold+1}.pt'

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
            # Only print every epoch
            print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")
            
            scheduler.step(val_mean_r2)

            if val_mean_r2 > best_val:
                best_val = val_mean_r2
                torch.save(model.state_dict(), fold_checkpoint_path)

        print(f"Fold {fold+1} complete. Best Val Mean R2: {best_val:.6f}")
        all_best_val_r2s.append(best_val)
    
    print("-" * 30)
    print("K-Fold Training Finished.")
    print(f"All Best Val R2 scores: {all_best_val_r2s}")
    print(f"Average Best Val R2: {np.mean(all_best_val_r2s):.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 150
            self.batch_size = 32
            self.hidden_size = 128 
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 7
            self.val_frac = 0.1
            self.checkpoint = 'lstm_attention_rich_fe_checkpoint.pt'
            
    args = Args()
    main(args)