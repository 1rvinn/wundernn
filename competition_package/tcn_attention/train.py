"""
K-Fold Ensemble Training for TCNAttentionModel (Raw Data).

- Trains 5 separate models on 5 different 80/20 splits (folds).
- Uses the stateless "sliding window" dataset.
- *** MODIFIED: Uses a TCN Encoder + Attention Decoder. ***
- Trains on RAW, unscaled data.
- Trains to predict the absolute next state (X[t+1]).
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """
    Dataset that yields (raw_input_window, raw_target_state) pairs.
    Data is NOT scaled, and the target is the absolute next state.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.examples = []

        grouped = df.groupby("seq_ix")
        print(f"Creating {mode} examples...")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                target = states[t + 1]
                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1]
                
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


# --- NEW TCN + ATTENTION MODEL ---
class TCNAttentionModel(nn.Module):
    """
    A TCN Encoder followed by an Attention mechanism.
    """
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.1):
        super().__init__()
        
        # --- 1. TCN Encoder ---
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            dilation = 2**i  # Exponentially growing dilation
            out_channels = hidden_size
            
            # Use 'same' padding to keep seq_len constant
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    padding=padding, 
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels  # Input for next layer
        
        self.tcn_stack = nn.Sequential(*layers)
        
        # --- 2. Attention Mechanism ---
        # Takes the TCN output (hidden_size) and maps to a score (1)
        self.attention_fc = nn.Linear(hidden_size, 1)
        
        # --- 3. Final Decoder ---
        # Takes the context vector (hidden_size) and maps to prediction (input_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_size]
        
        # --- TCN Encoder ---
        # Conv1d expects: [batch_size, channels, seq_len]
        x_tcn = x.permute(0, 2, 1)
        
        # tcn_out shape: [batch_size, hidden_size, seq_len]
        tcn_out = self.tcn_stack(x_tcn)
        
        # --- Attention Decoder ---
        # Attention layer expects: [batch_size, seq_len, features]
        # So, permute back:
        tcn_out = tcn_out.permute(0, 2, 1)
        # tcn_out shape: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        # attention_logits shape: [batch_size, seq_len, 1]
        attention_logits = self.attention_fc(tcn_out)
        
        # Normalize scores to weights
        # attention_weights shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Create context vector (weighted sum)
        # context_vector shape: [batch_size, hidden_size]
        context_vector = torch.sum(attention_weights * tcn_out, dim=1)
        
        # Final prediction
        return self.fc(context_vector)


def evaluate(model, dataloader, device):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            ys.append(y.cpu().numpy())
            yps.append(pred.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
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
    
    # --- K-FOLD SPLITTING ---
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_best_val_r2s = []
    
    sample_df = df.iloc[:1]
    feature_cols = [c for c in sample_df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    input_size = len(feature_cols)
    
    base_checkpoint_name = args.checkpoint.replace('.pt', '')

    for fold, (train_index, val_index) in enumerate(kf.split(seq_ids)):
        print("-" * 30)
        print(f"--- FOLD {fold + 1}/{N_SPLITS} ---")
        print("-" * 30)
        
        train_ids = seq_ids[train_index]
        val_ids = seq_ids[val_index]

        df_train = df[df['seq_ix'].isin(train_ids)]
        df_val = df[df['seq_ix'].isin(val_ids)]

        print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")
        
        train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
        val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Instantiate NEW TCN + Attention Model for each fold ---
        model = TCNAttentionModel(
            input_size=input_size, 
            hidden_size=args.hidden_size, 
            num_layers=args.num_layers, 
            kernel_size=args.kernel_size,
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5)
        criterion = nn.MSELoss()

        best_val = -math.inf
        fold_checkpoint_path = f'{base_checkpoint_name}_fold_{fold+1}.pt'

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
                torch.save(model.state_dict(), fold_checkpoint_path)

        print(f"Fold {fold+1} complete. Best Val Mean R2: {best_val:.6f}")
        all_best_val_r2s.append(best_val)
    
    print("-" * 30)
    print("K-Fold Training Finished.")
    print(f"All Best Val R2 scores: {all_best_val_r2s}")
    print(f"Average Best Val R2: {np.mean(all_best_val_r2s):.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/content/drive/MyDrive/train.parquet'
            self.seq_len = 150
            self.batch_size = 32
            self.hidden_size = 128    # TCN conv channels
            
            # --- TCN Hyperparameters ---
            self.num_layers = 7       # Needs to be larger for TCN to get receptive field
            self.kernel_size = 3      # Kernel size for TCN
            # ---
            
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 5 
            self.val_frac = 0.1 
            self.checkpoint = 'tcn_attention_raw_checkpoint.pt' # Renamed checkpoint
            
    args = Args()
    main(args)