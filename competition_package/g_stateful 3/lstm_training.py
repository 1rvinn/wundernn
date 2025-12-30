"""
Stateful LSTM training script for sequence prediction.

- Feeds FULL sequences to a PyTorch LSTM.
- Uses a mask to calculate loss ONLY on scored timesteps (100-998).
- **Trains on RAW, unscaled data.**
- **Predicts the RAW, absolute next state.**
- **ADDED: AdamW Optimizer, LR Scheduler, and Gradient Clipping.**
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
from torch.optim.lr_scheduler import ReduceLROnPlateau  # <-- IMPORTED

class SequenceDataset(Dataset):
    """
    Dataset that yields (raw_input_seq, raw_target_seq, mask) for full sequences.
    """
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.num_features = len(self.feature_cols)

        self.sequences = []
        grouped = df.groupby("seq_ix")
        print(f"Loading {len(grouped)} sequences for {mode} set...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)
            need_pred = g["need_prediction"].to_numpy(dtype=bool)
            self.sequences.append((states, need_pred))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        states, need_pred = self.sequences[idx]
        x = states[:-1]
        y = states[1:]
        mask = need_pred[:-1]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x) 
        predictions = self.fc(out)
        return predictions


def evaluate(model, dataloader, device, criterion):
    """
    Evaluate the model on raw, absolute states.
    """
    model.eval()
    ys_true = []
    ys_pred = []
    
    with torch.no_grad():
        for x, y, mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            pred = model(x)
            
            # Select only the masked (scored) predictions
            ys_true.append(y[mask].cpu().numpy())
            ys_pred.append(pred[mask].cpu().numpy())
            
    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    
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
    print(f"Reading data from {p}...")
    df = pd.read_parquet(p)

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")
    
    train_ds = SequenceDataset(df_train, mode='train')
    val_ds = SequenceDataset(df_val, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = len(train_ds.feature_cols)
    print(f"Detected {input_size} features.")
    
    model = LSTMModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    # --- 1. UPGRADED OPTIMIZER & SCHEDULER ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # Use AdamW
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True) # Patience of 10 epochs
    criterion = nn.MSELoss()
    # ---

    best_val = -math.inf
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_predictions = 0
        
        start_time = time.time()
        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            
            # --- 2. ADDED GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # ---
            
            optimizer.step()
            
            num_predictions = torch.sum(mask).item()
            running_loss += loss.item() * num_predictions
            total_predictions += num_predictions
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_predictions if total_predictions > 0 else 0.0

        val_mean_r2, _ = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.6f} | Val Mean R2: {val_mean_r2:.6f}")

        # --- 3. STEP THE SCHEDULER ---
        scheduler.step(val_mean_r2)
        # ---

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            print(f"  -> New best validation R2! Saving model to {args.checkpoint}")
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Training complete. Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/content/drive/MyDrive/datasets/train.parquet'
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.2
            self.lr = 1e-3
            self.epochs = 200
            self.val_frac = 0.1
            # Updated checkpoint name to reflect new training method
            self.checkpoint = 'lstm_stateful_raw_stable_checkpoint.pt'
            
    args = Args()
    main(args)