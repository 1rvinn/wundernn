"""
Stateful LSTM training script for sequence prediction.

- Reads `datasets/train.parquet` (same format as competition data)
- Splits sequences by `seq_ix` into train/val (80/20)
- Feeds FULL sequences to a PyTorch LSTM.
- Uses a mask to calculate loss ONLY on scored timesteps (100-998).
- Evaluates using R^2 per-feature and mean R^2.
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


class SequenceDataset(Dataset):
    """
    Dataset that yields (input_seq, target_seq, mask) for full sequences.
    
    Instead of a sliding window, this dataset returns the *entire* sequence.
    __len__ is the number of sequences.
    __getitem__ returns one full sequence.
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
        # The length of the dataset is the number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        states, need_pred = self.sequences[idx]

        # Input X is steps 0...998 (total 999 steps)
        x = states[:-1]
        # Target Y is steps 1...999 (total 999 steps)
        y = states[1:]
        
        # The mask tells us which predictions to score/train on.
        # We need to predict y[t] (state t+1) based on x[t] (state t).
        # We score this if need_prediction[t] is True.
        # need_pred[t] corresponds to the prediction for t+1.
        # So, the mask for our (x, y) pair is need_pred from step 0 to 998.
        mask = need_pred[:-1]

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,  # This is crucial: (batch, seq_len, features)
            dropout=dropout
        )
        # The Linear layer is applied to EVERY time-step's output
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> e.g., (64, 999, 120)
        
        # out: (batch, seq_len, hidden_size)
        # lstm returns (output_seq, (hidden_state, cell_state))
        # We don't need the final hidden state, just the full output sequence.
        out, _ = self.lstm(x) 
        
        # predictions: (batch, seq_len, input_size)
        # Apply the FC layer to all time-step outputs
        predictions = self.fc(out)
        return predictions


def evaluate(model, dataloader, device, criterion):
    """
    Evaluate the model, using the mask to select the scored predictions.
    """
    model.eval()
    ys_true = []
    ys_pred = []
    
    with torch.no_grad():
        for x, y, mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device) # mask is (batch, 999)
            
            # pred is (batch, 999, N_features)
            pred = model(x)
            
            # CRITICAL: Only select the predictions and targets where the mask is True
            # y[mask] flattens and selects all masked elements
            # -> (total_masked_elements, N_features)
            ys_true.append(y[mask].cpu().numpy())
            ys_pred.append(pred[mask].cpu().numpy())
            
    # vstack will now work on lists of 2D arrays, resulting in a single 2D array
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

    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

    # dataset and loaders
    # Note: We can't use seq_len anymore, the dataset handles the full length
    train_ds = SequenceDataset(df_train, mode='train')
    val_ds = SequenceDataset(df_val, mode='val')
    
    # We batch sequences, not windows.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = len(train_ds.feature_cols)
    print(f"Detected {input_size} features.")
    
    model = LSTMModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

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
            mask = mask.to(device) # (batch, 999)

            optimizer.zero_grad()
            
            # out will be (batch_size, 999, N_features)
            out = model(x)
            
            # CRITICAL: Use the mask to calculate loss
            # We select only the predictions and targets where the mask is True
            # out[mask] and y[mask] will be flattened tensors of shape 
            # (N_true_predictions_in_batch, N_features)
            loss = criterion(out[mask], y[mask])
            
            loss.backward()
            optimizer.step()
            
            num_predictions = torch.sum(mask).item()
            running_loss += loss.item() * num_predictions
            total_predictions += num_predictions
        
        epoch_time = time.time() - start_time
        # Calculate loss per scored prediction
        avg_loss = running_loss / total_predictions if total_predictions > 0 else 0.0

        val_mean_r2, _ = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.6f} | Val Mean R2: {val_mean_r2:.6f}")

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            print(f"  -> New best validation R2! Saving model to {args.checkpoint}")
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Training complete. Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='/kaggle/input/trainds/train.parquet')
    # # seq_len is no longer needed, we always process the full sequence
    # # parser.add_argument('--seq_len', type=int, default=100) 
    # parser.add_argument('--batch_size', type=int, default=32) # Batch of full sequences
    # parser.add_argument('--hidden_size', type=int, default=128)
    # parser.add_argument('--num_layers', type=int, default=2)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--val_frac', type=float, default=0.2)
    # parser.add_argument('--checkpoint', type=str, default='lstm_stateful_checkpoint.pt')
    
    # # Handle environment where args might not be parsable (e.g., notebook)
    # try:
    #     args = parser.parse_args()
    # except Exception:
    #     print("Running with default args")
    #     args = parser.parse_args([]) 

    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 100
            self.val_frac = 0.2
            self.checkpoint = 'lstm_stateful_checkpoint.pt'
            # self.predict_delta = True # Add this attribute
    args = Args()
    
    main(args)