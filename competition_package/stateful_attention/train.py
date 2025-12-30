"""
True Stateful Attention LSTM training script.

WARNING: This model is (intentionally) not parallelized and will
be *significantly* slower to train than previous models,
as it runs a loop over the 999-step sequence.

- Uses the "full sequence" dataset.
- The model loops 999 times, applying attention at each step.
- Trains on RAW, unscaled data.
- Predicts the absolute next state.
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

class SequenceDataset(Dataset):
    """
    Dataset that yields (input_seq, target_seq, mask) for full sequences.
    This is the "Attempt 2" dataset.
    """
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.sequences = []
        
        grouped = df.groupby("seq_ix")
        print(f"Loading {len(grouped)} full sequences for {mode} set...")
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


# --- NEW MODEL: Stateful Looping Attention LSTM ---
class StatefulAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # We must use LSTMCell to manually loop one step at a time
        # We'll just model one layer for simplicity. Multi-layer is much more complex.
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        
        # Attention layers (for Dot-Product Attention)
        # We need to project the "query" (current state) and "keys" (past states)
        # to the same dimension if they aren't already. Here, they are.
        
        # Decoder Head: takes [current_thought, smart_summary]
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # h_t + context_t
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state and cell state
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)

        past_hidden_states = [] # To store all h_0, h_1, ...
        outputs = []            # To store all predictions

        # This loop is SLOW and cannot be parallelized
        for t in range(seq_len):
            input_t = x[:, t, :]
            
            # Run one step of the LSTM
            (h, c) = self.lstm_cell(input_t, (h, c))
            h = self.dropout(h)
            
            # Store this "thought"
            past_hidden_states.append(h)
            
            # --- Attention Mechanism ---
            # Stack all past thoughts: (batch, t+1, hidden_size)
            past_states = torch.stack(past_hidden_states, dim=1)
            
            # Current thought "h" is the query: (batch, hidden_size, 1)
            query = h.unsqueeze(2)
            
            # 1. Calculate scores
            # (batch, t+1, hidden_size) @ (batch, hidden_size, 1) -> (batch, t+1, 1)
            scores = torch.bmm(past_states, query)
            
            # 2. Convert to probabilities
            weights = F.softmax(scores, dim=1)
            
            # 3. Create context vector
            # (batch, t+1, hidden) * (batch, t+1, 1) -> sum -> (batch, hidden)
            context = torch.sum(weights * past_states, dim=1)
            
            # 4. Combine current thought + smart summary
            combined_out = torch.cat((h, context), dim=1) # (batch, hidden*2)
            
            # 5. Make prediction
            pred = self.fc_head(combined_out)
            outputs.append(pred)

        # Stack all predictions into (batch, seq_len, features)
        return torch.stack(outputs, dim=1)

# --- END NEW MODEL ---


def evaluate(model, dataloader, device):
    model.eval()
    ys_true_abs = []
    ys_pred_abs = []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            pred = model(x) # (batch, 999, features)
            
            # We must apply the mask *before* stacking
            ys_true_abs.append(y[mask].cpu().numpy())
            ys_pred_abs.append(pred[mask].cpu().numpy())

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
    
    train_ds = SequenceDataset(df_train, mode='train')
    val_ds = SequenceDataset(df_val, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    # --- Instantiate StatefulAttentionLSTM ---
    model = StatefulAttentionLSTM(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=1, # Note: Model is hardcoded to 1 layer
        dropout=args.dropout
    ).to(device)
    # ---

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2)
    criterion = nn.MSELoss()

    best_val = -math.inf
    print(f"Starting training (Stateful Looping Attention)... This will be SLOW.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_predictions = 0
        
        start_time = time.time()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            
            out = model(x) # (batch, 999, features)
            
            # Apply mask to get loss
            loss = criterion(out[mask], y[mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            num_preds = torch.sum(mask).item()
            running_loss += loss.item() * num_preds
            total_predictions += num_preds
            
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_predictions

        val_mean_r2, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Time: {epoch_time:.2f}s | Train Loss={avg_loss:.6f} | Val Mean R2={val_mean_r2:.6f}")
        
        scheduler.step(val_mean_r2)

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/content/drive/MyDrive/datasets/train.parquet'
            self.batch_size = 32 # Smaller batch due to high memory/time
            self.hidden_size = 128
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 20
            self.val_frac = 0.1
            self.checkpoint = 'stateful_attention_raw_checkpoint.pt'
            
    args = Args()
    main(args)