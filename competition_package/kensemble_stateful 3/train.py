"""
K-Fold Ensemble Training for Stateful LSTM.

- This script trains 5 separate models on 5 different (80/20) "folds" of the data.
- This is a powerful technique to combat overfitting and improve generalization.
- Feeds FULL sequences to a PyTorch LSTM.
- Uses a mask to calculate loss ONLY on scored timesteps (100-998).
- Uses AdamW, LR Scheduler, and Gradient Clipping for stability.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold  # <-- IMPORTED
from sklearn.model_selection import train_test_split # Still used for a quick input_size check

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # <-- IMPORTED

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
        
        # This will print 5 times (once per fold)
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

    def forward(self, x, h=None): # <-- Modified to accept hidden state
        """
        In training, 'h' is None, and the LSTM gets the full 999-step sequence.
        In inference (solution.py), 'x' is 1 step and 'h' is the memory.
        """
        out, h_out = self.lstm(x, h) 
        predictions = self.fc(out)
        return predictions, h_out # <-- Return hidden state for inference


def evaluate(model, dataloader, device, criterion):
    model.eval()
    ys_true = []
    ys_pred = []
    
    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            # We don't pass hidden state 'h' during eval,
            # as we are evaluating on full sequences, not step-by-step
            pred, _ = model(x) 
            
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

    # --- K-FOLD SPLITTING ---
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Get input_size before the loop
    sample_df = df.iloc[:1]
    feature_cols = [c for c in sample_df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    input_size = len(feature_cols)
    print(f"Detected {input_size} features.")
    
    all_best_val_r2s = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(seq_ids)):
        print("-" * 30)
        print(f"--- FOLD {fold + 1}/{N_SPLITS} ---")
        print("-" * 30)
        
        train_ids = seq_ids[train_index]
        val_ids = seq_ids[val_index]

        df_train = df[df['seq_ix'].isin(train_ids)]
        df_val = df[df['seq_ix'].isin(val_ids)]

        print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")
        
        train_ds = SequenceDataset(df_train, mode='train')
        val_ds = SequenceDataset(df_val, mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # --- Instantiate a NEW model for each fold ---
        model = LSTMModel(
            input_size=input_size, 
            hidden_size=args.hidden_size, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        ).to(device)

        # --- Use stable optimizer and scheduler ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)
        criterion = nn.MSELoss()

        best_val = -math.inf
        fold_checkpoint_path = f"{args.checkpoint.replace('.pt', '')}_fold_{fold+1}.pt"
        print(f"Saving checkpoints to: {fold_checkpoint_path}")

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            total_predictions = 0
            
            start_time = time.time()
            for x, y, mask in train_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)

                optimizer.zero_grad()
                
                # Model now returns predictions and hidden state,
                # but we only need predictions for training.
                out, _ = model(x) 
                
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
                print(f"  -> New best validation R2! Saving model to {fold_checkpoint_path}")
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
            self.data = '/content/drive/MyDrive/datasets/train.parquet'
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.4
            self.lr = 1e-3
            self.epochs = 120 # This will run 200 epochs *for each fold*
            self.val_frac = 0.1 # This is ignored, KFold(n_splits=5) is used
            self.checkpoint = 'lstm_stateful_checkpoint.pt' # This is now a base name
            
    args = Args()
    main(args)