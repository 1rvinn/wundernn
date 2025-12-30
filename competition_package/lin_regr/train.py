"""
Training script for a Linear Autoregressive Model.

- Predicts next state as a linear combination of the past 100 states.
- Uses the "sliding window" dataset to create (window, target) pairs.
- Uses StandardScaler for robust training.
- Trains a PyTorch model with a single nn.Linear layer.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SlidingWindowDataset(Dataset):
    """
    Dataset that yields (input_window, target_state) pairs.
    Uses a 100-step window to predict the next absolute state.
    """
    def __init__(self, df: pd.DataFrame, scaler, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.scaler = scaler

        self.examples = []  # list of (window, target)
        
        # Pre-scale the entire dataframe for efficiency
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        grouped = df_scaled.groupby("seq_ix")
        print(f"[{mode}] Creating sliding window examples...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)
            need_pred = g["need_prediction"].to_numpy(dtype=bool)

            for t in range(len(g)):
                if not need_pred[t]:
                    continue
                if t + 1 >= len(g):
                    continue
                
                # Target is the absolute scaled state at t+1
                target = states[t + 1]
                
                # Input window: up to t (inclusive), last seq_len steps
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1] # This is already scaled
                
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


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        
        # Flatten the (seq_len, input_size) window
        self.flatten = nn.Flatten()
        
        # A single linear layer
        # Input: (batch, seq_len * input_size)
        # Output: (batch, input_size)
        self.linear = nn.Linear(seq_len * input_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_flat = self.flatten(x)
        return self.linear(x_flat)


def evaluate(model, dataloader, device, scaler):
    model.eval()
    ys_true_scaled = []
    ys_pred_scaled = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device) # This is the scaled target
            
            pred = model(x) # This is the scaled prediction
            
            ys_true_scaled.append(y.cpu().numpy())
            ys_pred_scaled.append(pred.cpu().numpy())
            
    # Un-scale *after* stacking for efficiency
    y_true_scaled = np.vstack(ys_true_scaled)
    y_pred_scaled = np.vstack(ys_pred_scaled)
    
    # Un-scale to get true values
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # compute per-feature R2
    r2s = []
    for i in range(y_true.shape[1]):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
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
    
    # --- SCALING ---
    feature_cols = [c for c in df_train.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols]) 
    
    scaler_path = 'linreg_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    # ---

    # Pass the *un-scaled* dataframes and the *fitted scaler*
    train_ds = SlidingWindowDataset(df_train, scaler, seq_len=args.seq_len, mode='train')
    val_ds = SlidingWindowDataset(df_val, scaler, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)
    print(f"Using device: {device}; Detected {input_size} features.")
    
    model = LinearRegressionModel(input_size=input_size, seq_len=args.seq_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val = -math.inf
    print("Starting training for LinearRegressionModel...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        val_mean_r2, _ = evaluate(model, val_loader, device, scaler)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")

        scheduler.step(val_mean_r2)

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  -> New best validation R2! Saving model to {args.checkpoint}")

    print(f"Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../datasets/train.parquet')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--checkpoint', type=str, default='linreg_checkpoint.pt')
    
    try:
        args = parser.parse_args()
    except:
        print("Running in notebook, using default args")
        args = parser.parse_args([])
        
    main(args)