"""
Stateful *Transformer* training script for sequence prediction.

- Feeds FULL sequences to a PyTorch TransformerEncoder.
- **Uses a Causal Attention Mask** to prevent the model from "cheating"
  by looking at the future. This is critical for stateful forecasting.
- Uses a mask to calculate loss ONLY on scored timesteps (100-998).
- Trains on RAW, unscaled data.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau # Added
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """
    Dataset that yields (input_seq, target_seq, mask) for full sequences.
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


# --- NEW: Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- NEW MODEL: Causal Transformer Encoder ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Projection (like an embedding layer)
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1000) # 999 steps
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Final Decoder Head
        self.fc = nn.Linear(d_model, input_size)
        
        # This is a static mask, so we can register it
        self.register_buffer("causal_mask", None)

    def _generate_causal_mask(self, sz, device):
        # Creates a square matrix where the future is masked
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_size) e.g., (32, 999, N)
        seq_len = x.size(1)
        
        # Generate the causal mask if it doesn't exist or is the wrong size
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = self._generate_causal_mask(seq_len, x.device)
            
        # 1. Embed input
        x = self.input_proj(x) # (batch, 999, d_model)
        
        # 2. Add positional info
        x = self.pos_encoder(x) # (batch, 999, d_model)
        
        # 3. Pass through Transformer with the CAUSAL MASK
        # This mask prevents a position from attending to future positions.
        out = self.transformer_encoder(x, mask=self.causal_mask) # (batch, 999, d_model)
        
        # 4. Make prediction at every step
        predictions = self.fc(out) # (batch, 999, input_size)
        return predictions
# --- END NEW MODEL ---


def evaluate(model, dataloader, device, criterion):
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            
            ys_true.append(y[mask].cpu().numpy())
            ys_pred.append(pred[mask].cpu().numpy())
            
    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    
    r2s = []
    for i in range(y_true.shape[1]):
        try: r2 = r2_score(y_true[:, i], y_pred[:, i])
        except Exception: r2 = float('nan')
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
    
    # --- Instantiate NEW TransformerModel ---
    model = TransformerModel(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    # ---

    # --- Use stability improvements ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10)
    criterion = nn.MSELoss()
    # ---

    best_val = -math.inf
    print("Starting training (Stateful Transformer)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_predictions = 0
        
        start_time = time.time()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
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
            self.data = '/content/drive/MyDrive/datasets/train.parquet'
            self.batch_size = 16 # Transformers use a lot of memory
            self.dropout = 0.2
            self.lr = 1e-4 # Transformers need a smaller LR
            self.epochs = 50 # Transformers take longer to converge
            self.val_frac = 0.1
            
            # --- Transformer Hyperparameters ---
            self.d_model = 128        # Internal dimension (must be divisible by nhead)
            self.nhead = 8            # Number of "attention heads"
            self.num_layers = 4       # Number of transformer blocks
            
            self.checkpoint = 'transformer_stateful_checkpoint.pt'
            
    args = Args()
    main(args)