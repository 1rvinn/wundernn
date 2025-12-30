"""
LSTM with Attention training script (Rolling Z-Score Scaled Version).

- Uses the stateless "sliding window" dataset.
- Uses the LSTMAttentionModel (Encoder-Decoder with Attention).
- **Trains on SCALED data (Rolling Z-Score).**
- **Trains the model to predict the *scaled next state*** (X[t+1]).
- Evaluates R^2 on the SCALED predictions.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """
    Dataset that yields (scaled_input_window, scaled_target_state) pairs.
    Data is scaled using Rolling Z-Score: (x - rolling_mean) / rolling_std.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.examples = []
        
        # Define window size for rolling calculation
        self.rolling_window = 60 

        grouped = df.groupby("seq_ix")
        print(f"Creating {mode} examples (with rolling z-score scaling)...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            
            # --- SCALING LOGIC START ---
            # Calculate Rolling Statistics
            # min_periods=1 ensures we get values even at the start of the sequence
            roll = g[self.feature_cols].rolling(window=self.rolling_window, min_periods=1)
            roll_mean = roll.mean()
            roll_std = roll.std()

            # Handle division by zero (constant values have std=0) and NaNs (start of seq)
            roll_std = roll_std.fillna(1.0)
            roll_std[roll_std == 0] = 1.0
            
            # Apply Z-Score Formula: (X - Mean) / Std
            g[self.feature_cols] = (g[self.feature_cols] - roll_mean) / roll_std
            
            # Fill any remaining NaNs (e.g., the very first row where std might be undefined) with 0
            g[self.feature_cols] = g[self.feature_cols].fillna(0.0)
            # --- SCALING LOGIC END ---

            # states is now the SCALED data
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                # --- Target is now the SCALED state ---
                target = states[t + 1]
                # ---

                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1] # This data is SCALED
                
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
        # x: (batch, seq_len, input_size)
        
        # 1. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 2. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 3. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 4. Decode (Make Prediction)
        return self.fc(context_vector)


def evaluate(model, dataloader, device):
    model.eval()
    ys_true = []
    ys_pred = []

    with torch.no_grad():
        for x, y in dataloader: 
            x, y = x.to(device), y.to(device)

            # Model predicts the SCALED state
            pred = model(x)

            ys_true.append(y.cpu().numpy())
            ys_pred.append(pred.cpu().numpy())

    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    
    # Calculate R2 on the SCALED data
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

    # Pass the dataframe to Dataset (Scaling happens inside)
    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    # --- Instantiate LSTMAttentionModel ---
    model = LSTMAttentionModel(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    # ---

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    
    criterion = nn.MSELoss()

    best_val = -math.inf
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
        print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")
        
        scheduler.step() 

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
            self.checkpoint = 'lstm_attention_scaled_checkpoint.pt'
            
    args = Args()
    main(args)
