"""
Simple LSTM training script for sequence prediction.

- Reads `datasets/train.parquet` (same format as competition data)
- Splits sequences by `seq_ix` into train/val (80/20)
- Trains a small PyTorch LSTM to predict next-step features
- Evaluates using R^2 per-feature and mean R^2

**UPDATED with Gradient Clipping and LR Scheduler**
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# --- ADDED ---
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SequenceDataset(Dataset):
    """Dataset that yields (input_seq, target) pairs for next-step prediction.

    For each sequence, we produce many training examples: for each time t,
    input is previous states up to t (truncated/padded
    to `seq_len`), target is state at t+1. We return float32 arrays.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        # df contains multiple sequences; we'll group by seq_ix
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]

        # Store per-sequence arrays
        self.examples = []  # list of (input_window, target_vector)

        grouped = df.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)
            need_pred = g["need_prediction"].to_numpy(dtype=bool)

            # For each timestep create an example
            for t in range(len(g)):
                # target is at t+1 (if exists)
                if t + 1 >= len(g):
                    continue
                target = states[t + 1]

                # input window: up to t (inclusive), last seq_len steps
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1]
                # pad at the front if needed
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        # Return as tensors
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last)


def evaluate(model, dataloader, device):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            ys.append(y.cpu().numpy())
            yps.append(pred.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
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

    # split sequences by seq_ix
    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    # dataset and loaders
    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)
    model = LSTMModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    # --- Use AdamW for better weight decay ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # --- ADDED: LR Scheduler ---
    # Monitors 'val_mean_r2' and reduces LR if it stops improving.
    # Note: 'verbose=True' will print a message when LR is reduced.
    # You may see a UserWarning about 'verbose' being deprecated, which is normal.
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)


    best_val = -math.inf
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
            
            # --- ADDED: Gradient Clipping ---
            # Prevents exploding gradients, common in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        val_mean_r2, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")
        
        # --- ADDED: Scheduler Step ---
        # Tell the scheduler to check the validation R2
        scheduler.step(val_mean_r2)

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
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 10 # You may want to increase this, as the scheduler needs patience
            self.val_frac = 0.1
            self.checkpoint = 'lstm_checkpoint1.pt'
            
    args = Args()
    main(args)