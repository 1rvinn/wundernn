"""
Stateful RNN training script with multiple model options.
Includes a CNN-GRU hybrid model for local feature extraction.

- Reads `datasets/train.parquet`
- Scales data using StandardScaler fit on training set
- Feeds FULL sequences to a PyTorch RNN
- Model predicts the *delta* (change) from t to t+1
- Uses Gradient Clipping and ReduceLROnPlateau scheduler
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
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F  # <-- ADDED for CNN padding and activations


class SequenceDataset(Dataset):
    """
    Dataset that yields (input_seq, target_delta_seq, mask) for full sequences.
    Target `y` is now the *change* from t to t+1 (X[t+1] - X[t]).
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
        y_abs = states[1:]
        y_delta = y_abs - x  # Target is the change (delta)
        mask = need_pred[:-1]
        return torch.from_numpy(x), torch.from_numpy(y_delta), torch.from_numpy(mask)

# --- MODEL DEFINITIONS ---

class LSTMModel(nn.Module):
    """Baseline LSTM Model"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out)

class GRUModel(nn.Module):
    """GRU Model (Alternative to LSTM)"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        out, _ = self.gru(x); return self.fc(out)

class BetterLSTMModel(nn.Module):
    """LSTM Model with Residual Connections and a more complex head"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size; self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, input_size)
        )
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    def forward(self, x):
        residual = self.residual_proj(x)
        lstm_out, _ = self.lstm(x)
        out = F.relu(lstm_out + residual)
        return self.head(out)

class CnnGruModel(nn.Module):
    """
    NEW: Hybrid CNN-GRU Model.
    The 1D-CNN acts as a sliding-window feature extractor.
    The GRU models the sequence of these extracted features.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.1, cnn_kernel_size=10):
        super().__init__()
        
        self.cnn_kernel_size = cnn_kernel_size
        # This layer finds local patterns.
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_size, # Let CNN output 'hidden_size' features
            kernel_size=cnn_kernel_size, 
            padding=0 # We will add "causal" padding manually
        )
        self.relu = nn.ReLU()
        
        # The GRU now looks at the *sequence of patterns* from the CNN
        self.gru = nn.GRU(
            input_size=hidden_size, # Input is from the CNN
            hidden_size=hidden_size, 
            num_layers=num_layers, # 1 layer is often enough after a CNN
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # 1. CNN Part
        # We need (batch, features, seq_len) for Conv1d
        x_cnn_in = x.permute(0, 2, 1)
        
        # Add causal padding manually to the left
        x_cnn_in = F.pad(x_cnn_in, (self.cnn_kernel_size - 1, 0))
        
        x_cnn_out = self.conv1(x_cnn_in)
        x_cnn_out = self.relu(x_cnn_out)
        
        # 2. GRU Part
        # Change back to (batch, seq_len, features) for GRU
        x_gru_in = x_cnn_out.permute(0, 2, 1)
        
        out, _ = self.gru(x_gru_in) 
        predictions = self.fc(out)
        return predictions

# --- END MODEL DEFINITIONS ---


def evaluate(model, dataloader, device, criterion):
    model.eval()
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for x, y_delta, mask in dataloader:
            x, y_delta, mask = x.to(device), y_delta.to(device), mask.to(device)
            pred_delta = model(x) 
            y_true_abs = x + y_delta
            y_pred_abs = x + pred_delta
            ys_true.append(y_true_abs[mask].cpu().numpy())
            ys_pred.append(y_pred_abs[mask].cpu().numpy())
            
    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return float(np.nanmean(r2s)), r2s


def main(args):
    p = Path(args.data); df = pd.read_parquet(p)

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)
    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    feature_cols = [c for c in df_train.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    
    print("Fitting scaler..."); scaler = StandardScaler()
    scaler.fit(df_train[feature_cols]) 
    scaler_path = 'scaler.joblib'
    joblib.dump(scaler, scaler_path); print(f"Scaler saved to {scaler_path}")

    print("Applying scaling...")
    df_train.loc[:, feature_cols] = scaler.transform(df_train[feature_cols])
    df_val.loc[:, feature_cols] = scaler.transform(df_val[feature_cols])

    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")
    train_ds = SequenceDataset(df_train, mode='train')
    val_ds = SequenceDataset(df_val, mode='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)
    print(f"Using device: {device}; Detected {input_size} features.")
    
    # --- MODIFIED: Model Selection ---
    print(f"Using model_type: {args.model_type}")
    if args.model_type == 'gru':
        model = GRUModel(input_size, args.hidden_size, args.num_layers, args.dropout)
    elif args.model_type == 'better_lstm':
        model = BetterLSTMModel(input_size, args.hidden_size, args.num_layers, args.dropout)
    elif args.model_type == 'cnngru':
        model = CnnGruModel(
            input_size=input_size, hidden_size=args.hidden_size,
            num_layers=args.num_layers, dropout=args.dropout,
            cnn_kernel_size=args.cnn_kernel_size
        )
    else:
        model = LSTMModel(input_size, args.hidden_size, args.num_layers, args.dropout)
    model.to(device)
    # ---

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)

    best_val = -math.inf
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0; total_predictions = 0
        start_time = time.time()
        
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            num_predictions = torch.sum(mask).item()
            running_loss += loss.item() * num_predictions
            total_predictions += num_predictions
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_predictions if total_predictions > 0 else 0.0
        val_mean_r2, _ = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.6f} | Val Mean R2: {val_mean_r2:.6f}")

        scheduler.step(val_mean_r2)
        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            print(f"  -> New best validation R2! Saving model to {args.checkpoint}")
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Training complete. Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 1 # 1 layer is often best for GRU/LSTM *after* a CNN
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 100
            self.val_frac = 0.2
            
            # --- NEW: Model selection ---
            self.model_type = 'cnngru'
            self.cnn_kernel_size = 10 # Window size for the CNN
            self.checkpoint = f'{self.model_type}_k{self.cnn_kernel_size}_checkpoint.pt'
            
    args = Args()
    main(args)