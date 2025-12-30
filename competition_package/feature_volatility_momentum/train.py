"""
LSTM with Attention training script (Feature Engineered Version).

- **MODIFIED: Adds rolling volatility AND 1-step momentum features.**
- Uses the stateless "sliding window" dataset.
- Uses the LSTMAttentionModel (Encoder-Decoder with Attention).
- Trains on RAW, unscaled data.
- Trains to predict the absolute next state (X[t+1]).
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

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# --- Helper function for volatility feature engineering ---
def add_volatility_features(df: pd.DataFrame, feature_cols: list, window_size: int) -> (pd.DataFrame, list):
    """
    Calculates rolling volatility (std dev) for each feature, grouped by sequence.
    Returns the dataframe and the list of NEW column names.
    """
    df_out = df.copy()
    grouped = df_out.groupby('seq_ix')
    
    new_cols = []
    for col in feature_cols:
        vol_col_name = f"{col}_vol{window_size}"
        df_out[vol_col_name] = grouped[col].transform(lambda x: x.rolling(window=window_size).std()).fillna(0)
        new_cols.append(vol_col_name)
        
    return df_out, new_cols
# ---

# --- Helper function for momentum feature engineering ---
def add_momentum_features(df: pd.DataFrame, feature_cols: list) -> (pd.DataFrame, list):
    """
    Calculates 1-step momentum for each feature, grouped by sequence.
    Returns the dataframe and the list of NEW column names.
    """
    df_out = df.copy()
    grouped = df_out.groupby('seq_ix')
    
    new_cols = []
    for col in feature_cols:
        mom_col_name = f"{col}_mom1"
        df_out[mom_col_name] = grouped[col].transform(lambda x: x.diff(periods=1)).fillna(0)
        new_cols.append(mom_col_name)
        
    return df_out, new_cols
# ---

class SequenceDataset(Dataset):
    """
    Dataset that yields (raw_input_window, raw_target_state) pairs.
    
    - `all_feature_cols`: List of cols to use for the INPUT window (e.g., raw + vol + mom)
    - `target_cols`: List of cols to use for the TARGET (e.g., raw only)
    """

    def __init__(self, df: pd.DataFrame, all_feature_cols: list, target_cols: list, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.all_feature_cols = all_feature_cols
        self.target_cols = target_cols
        self.examples = []

        grouped = df.groupby("seq_ix")
        print(f"Creating {mode} examples...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            
            # Input features (raw + vol + mom)
            input_states = g[self.all_feature_cols].to_numpy(dtype=np.float32)
            # Target features (raw only)
            target_states = g[self.target_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                
                target = target_states[t + 1]

                start = max(0, t - self.seq_len + 1)
                window = input_states[start : t + 1]
                
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


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context_vector)


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


def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    # --- MODIFIED: Feature Engineering Step ---
    sample_df = df.iloc[:1]
    original_feature_cols = [c for c in sample_df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    target_cols = original_feature_cols
    
    print(f"Original features: {len(original_feature_cols)}")
    
    # 1. Add Volatility features
    print(f"Adding volatility features with window={args.vol_window_size}...")
    df, vol_cols = add_volatility_features(df, original_feature_cols, args.vol_window_size)
    
    # 2. Add Momentum features
    print("Adding 1-step momentum features...")
    # We add momentum on the *original* features, not the vol features
    df, mom_cols = add_momentum_features(df, original_feature_cols) 
    
    # 3. Combine all features
    all_input_cols = original_feature_cols + vol_cols + mom_cols
    
    input_size = len(all_input_cols)
    output_size = len(target_cols)
    print(f"New input size: {input_size} (Orig: {len(original_feature_cols)}, Vol: {len(vol_cols)}, Mom: {len(mom_cols)})")
    print(f"Target output size: {output_size}")
    # ---

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    train_ds = SequenceDataset(
        df_train, 
        all_feature_cols=all_input_cols, 
        target_cols=target_cols, 
        seq_len=args.seq_len, 
        mode='train'
    )
    val_ds = SequenceDataset(
        df_val, 
        all_feature_cols=all_input_cols, 
        target_cols=target_cols, 
        seq_len=args.seq_len, 
        mode='val'
    )
    
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
        
        scheduler.step(val_mean_r2)

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            # --- IMPORTANT: Update this path ---
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 150
            self.batch_size = 32
            self.hidden_size = 256
            self.num_layers = 3
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 20
            self.val_frac = 0.1
            
            self.vol_window_size = 10
            
            self.checkpoint = 'lstm_attention_raw_vol_mom_checkpoint.pt' # Renamed
            
    args = Args()
    
    # Check for data file before running
    if not Path(args.data).exists():
        print(f"Error: Data file not found at {args.data}")
        print("Please update the 'self.data' path in the Args class.")
    else:
        main(args)
