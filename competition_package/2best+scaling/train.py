"""
Hybrid LSTM training script (Stateless Window + Scaling + Delta-Prediction).

- Uses the "sliding window" dataset (Code 1), which works best.
- Adds StandardScaler (fit on train data)
- Adds Delta Prediction (model predicts X[t+1] - X[t])
- Evaluates R^2 on the un-scaled, absolute predictions.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Added
import joblib # Added

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    Dataset that yields (input_window, target_delta) pairs.
    Based on your high-performing "Code 1".
    """

    def __init__(self, df: pd.DataFrame, scaler, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.scaler = scaler

        self.examples = []  # list of (input_window, target_delta)
        
        # --- Pre-scale the entire dataframe. This is much faster. ---
        # Note: The DF passed in should be the *original* un-scaled data.
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        # ---

        grouped = df_scaled.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)
            need_pred = g["need_prediction"].to_numpy(dtype=bool)

            for t in range(len(g)):
                if not need_pred[t]:
                    continue
                if t + 1 >= len(g):
                    continue
                
                # --- MODIFIED: Target is now the DELTA ---
                # target = absolute_state[t + 1]
                # We want delta = state[t+1] - state[t]
                target_delta = states[t + 1] - states[t]
                # ---
                
                # input window: up to t (inclusive), last seq_len steps
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1] # This is already scaled
                
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    # Pad with zeros (or scaled zeros, which is non-trivial)
                    # For simplicity, we'll pad with zeros.
                    # A better pad would be (0 - mean) / scale
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target_delta))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y_delta = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y_delta)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last) # This is a predicted_delta


def evaluate(model, dataloader, device, scaler):
    model.eval()
    ys_true_abs = []
    ys_pred_abs = []
    
    # Store scaler stats on device for speed
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)

    with torch.no_grad():
        for x, y_delta in dataloader:
            x = x.to(device) # x is the scaled window
            y_delta = y_delta.to(device) # y_delta is the scaled delta
            
            # Get the last state of the input window (which is scaled)
            # x shape is (batch, seq_len, features)
            x_last_scaled = x[:, -1, :]
            
            pred_delta = model(x) # This is the predicted scaled delta
            
            # --- MODIFIED: Convert back to absolute for R2 ---
            # 1. Get scaled absolute values
            y_true_scaled = x_last_scaled + y_delta
            y_pred_scaled = x_last_scaled + pred_delta
            
            # 2. Un-scale
            # unscaled = (scaled * scale) + mean
            y_true = (y_true_scaled * scaler_scale) + scaler_mean
            y_pred = (y_pred_scaled * scaler_scale) + scaler_mean
            
            ys_true_abs.append(y_true.cpu().numpy())
            ys_pred_abs.append(y_pred.cpu().numpy())
            
    y_true = np.vstack(ys_true_abs)
    y_pred = np.vstack(ys_pred_abs)
    
    # compute per-feature R2
    r2s = []
    for i in range(y_true.shape[1]):
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

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]
    
    # --- ADDED: SCALING ---
    feature_cols = [c for c in df_train.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols]) 
    
    scaler_path = 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    # ---

    # Pass the *un-scaled* dataframes and the *fitted scaler*
    train_ds = SequenceDataset(df_train, scaler, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, scaler, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)
    model = LSTMModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    # Use AdamW and a scheduler for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader: # y is y_delta
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x) # out is pred_delta
            
            # Loss is on the deltas
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # Pass scaler to evaluate
        val_mean_r2, _ = evaluate(model, val_loader, device, scaler)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")

        scheduler.step(val_mean_r2) # Step the scheduler

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 256
            self.num_layers = 2
            self.dropout = 0.1 # Re-enabled dropout, 0.0 is often not enough
            self.lr = 1e-3
            self.epochs = 150
            self.val_frac = 0.2
            self.checkpoint = 'lstm_window_delta_checkpoint.pt'
            
            # --- MODIFIED: Model selection and dynamic checkpoint path ---
            # Change this to 'better_lstm' or 'lstm' to switch models
            self.model_type = 'better_lstm' 
            self.checkpoint = f'{self.model_type}_checkpoint.pt'
            # ---
            
    args = Args()
    main(args)