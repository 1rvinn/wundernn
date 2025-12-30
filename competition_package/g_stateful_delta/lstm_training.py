"""
Stateful LSTM training script with scaling and delta prediction.

- Reads `datasets/train.parquet`
- **Scales data** using StandardScaler fit on training set
- Splits sequences by `seq_ix` into train/val (80/20)
- Feeds FULL sequences to a PyTorch LSTM
- **Model predicts the *delta* (change) from t to t+1**
- Uses a mask to calculate loss ONLY on scored timesteps
- **Uses Gradient Clipping** for stability
- **Uses ReduceLROnPlateau scheduler** for learning rate
- Evaluates using R^2 per-feature and mean R^2
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
import joblib  # To save the scaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Added for LR scheduling


class SequenceDataset(Dataset):
    """
    Dataset that yields (input_seq, target_delta_seq, mask) for full sequences.
    
    Target `y` is now the *change* from t to t+1 (X[t+1] - X[t]).
    """

    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.num_features = len(self.feature_cols)

        # Store per-sequence arrays
        self.sequences = []  # List of tuples (states, need_pred)
        
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

        # Input X is steps 0...998 (total 999 steps)
        x = states[:-1]
        
        # --- MODIFIED: Target Y is now the DELTA ---
        y_abs = states[1:]        # Absolute state at t+1
        y_delta = y_abs - x       # Target is the change (delta)
        # ---
        
        mask = need_pred[:-1]

        # Return x (original state), y (delta), and mask
        return torch.from_numpy(x), torch.from_numpy(y_delta), torch.from_numpy(mask)


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
        predictions = self.fc(out)  # These are predicted *deltas*
        return predictions


def evaluate(model, dataloader, device, criterion):
    """
    Evaluate the model. Since model predicts deltas, we convert
    back to absolute values for R^2 calculation.
    """
    model.eval()
    ys_true = []
    ys_pred = []
    
    with torch.no_grad():
        # y_delta is the target (true change)
        for x, y_delta, mask in dataloader:
            x = x.to(device)
            y_delta = y_delta.to(device)
            mask = mask.to(device)
            
            # Model predicts the delta
            pred_delta = model(x) 
            
            # --- MODIFIED: Convert back to absolute for R2 ---
            # y_true_abs = x (state at t) + y_delta (true change)
            # y_pred_abs = x (state at t) + pred_delta (predicted change)
            y_true_abs = x + y_delta
            y_pred_abs = x + pred_delta
            
            # Now, mask and append the *absolute* values
            ys_true.append(y_true_abs[mask].cpu().numpy())
            ys_pred.append(y_pred_abs[mask].cpu().numpy())
            # ---
            
    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    
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
    print(f"Reading data from {p}...")
    df = pd.read_parquet(p)

    # split sequences by seq_ix
    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    # --- ADDED: SCALING ---
    feature_cols = [c for c in df_train.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    
    # 1. Fit scaler ONLY on training data
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols]) 
    
    # 2. Save the scaler for inference
    scaler_path = 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # 3. Apply scaling to both dataframes (use .loc to avoid SettingWithCopyWarning)
    print("Applying scaling...")
    df_train.loc[:, feature_cols] = scaler.transform(df_train[feature_cols])
    df_val.loc[:, feature_cols] = scaler.transform(df_val[feature_cols])
    # --- END SCALING ---

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # --- ADDED: LR SCHEDULER ---
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)
    # ---

    best_val = -math.inf
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_predictions = 0
        
        start_time = time.time()
        # y is now y_delta
        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)  # This is the target delta
            mask = mask.to(device)

            optimizer.zero_grad()
            
            out = model(x)  # This is the predicted delta
            
            # Loss is calculated on the deltas
            loss = criterion(out[mask], y[mask])
            
            loss.backward()
            
            # --- ADDED: GRADIENT CLIPPING ---
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

        # --- ADDED: Scheduler Step ---
        scheduler.step(val_mean_r2)
        # ---

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            print(f"  -> New best validation R2! Saving model to {args.checkpoint}")
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Training complete. Best Val Mean R2: {best_val:.4f}")


if __name__ == '__main__':
    # Using your Args class for consistency
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 150
            self.val_frac = 0.2
            self.checkpoint = 'lstm_stateful_checkpoint.pt'
    
    args = Args()
    
    main(args)