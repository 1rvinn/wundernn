"""
LSTM with Attention training script.

- Uses the stateless "sliding window" dataset (your best model).
- **Uses a new LSTMAttentionModel** (Encoder-Decoder with Attention).
- Applies StandardScaler for all features.
- Trains the model to predict the *delta* (change) from t to t+1.
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
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """
    Dataset that yields (input_window, target_delta) pairs.
    Data is scaled, and the target is the delta.
    """

    def __init__(self, df: pd.DataFrame, scaler, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.scaler = scaler
        self.examples = []

        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        grouped = df_scaled.groupby("seq_ix")
        print(f"Creating {mode} examples...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                target_delta = states[t + 1] - states[t]

                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1]
                
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target_delta))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y_delta = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y_delta)


# --- NEW MODEL: LSTM with Attention ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Encoder (The LSTM)
        # We set bidirectional=True to make it even more powerful.
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True) # <-- Reads sequence forwards and backwards
        
        # 2. Attention Mechanism
        # The LSTM output is (hidden_size * 2) because it's bidirectional
        # This layer learns to compute an "importance score" for each time step
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder (The final classifier)
        # It takes the "attended context vector" as input
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # 1. Pass through Encoder (LSTM)
        # lstm_out: (batch, seq_len, hidden_size * 2) -> All hidden states
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 2. Calculate Attention Scores
        # Pass all 100 hidden states through the attention layer
        # attention_logits: (batch, seq_len, 1)
        attention_logits = self.attention_fc(lstm_out)
        
        # Convert logits (raw scores) to probabilities (sum to 1)
        # These are the "importance scores" for each time step
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 3. Create Context Vector
        # Multiply each hidden state by its importance score:
        # (batch, seq_len, hidden_size * 2) * (batch, seq_len, 1)
        # Then, sum them up to get a single, weighted-average vector
        # context_vector: (batch, hidden_size * 2)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 4. Decode (Make Prediction)
        # Pass the smart, "attended" context vector to the final layer
        return self.fc(context_vector)
# --- END NEW MODEL ---


def evaluate(model, dataloader, device, scaler):
    model.eval()
    ys_true_abs = []
    ys_pred_abs = []

    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)

    with torch.no_grad():
        for x, y_delta in dataloader:
            x, y_delta = x.to(device), y_delta.to(device)
            x_last_scaled = x[:, -1, :]
            pred_delta = model(x)

            y_true_scaled = x_last_scaled + y_delta
            y_pred_scaled = x_last_scaled + pred_delta
            y_true = (y_true_scaled * scaler_scale) + scaler_mean
            y_pred = (y_pred_scaled * scaler_scale) + scaler_mean

            ys_true_abs.append(y_true.cpu().numpy())
            ys_pred_abs.append(y_pred.cpu().numpy())

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
    
    scaler_path = 'lstm_attention_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    # ---

    train_ds = SequenceDataset(df_train, scaler, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, scaler, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    # --- Instantiate NEW LSTMAttentionModel ---
    model = LSTMAttentionModel(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    # ---

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=False)
    criterion = nn.MSELoss()

    best_val = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader: # y is y_delta
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) # out is pred_delta
            loss = criterion(out, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

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
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 20
            self.val_frac = 0.1
            self.checkpoint = 'lstm_attention_checkpoint.pt'
            
    args = Args()
    main(args)