"""
Modified version of your LSTM+Attention script.

CHANGES (only these):
- Adds per-feature loss weighting:
    * mode "var" (default): weights ∝ feature variance (features with larger variance get higher weight)
    * mode "r2": weights ∝ 1 / (r2 + eps) where r2 is provided in a numpy file (shape [D])
  After computing raw weights we normalize them so mean(weight)=1.0.

- Optional "separate_heads" mode: splits features into "easy" and "hard" groups
  (based on weight median) and creates two separate linear heads. Each head
  outputs only its group's features; final output is assembled in original order.

All other training/eval behavior is preserved.
"""

import argparse
import math
from pathlib import Path
import time
import json

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
    Dataset that yields (raw_input_window, raw_target_state) pairs.
    Data is NOT scaled; target is the absolute next state (kept same as your original).
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

                target = states[t + 1]  # absolute next state (unchanged)
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

# --- MODEL: LSTM with Attention AND Input Projection ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1,
                 separate_heads=False, easy_idx=None, hard_idx=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.separate_heads = separate_heads

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.attention_fc = nn.Linear(hidden_size * 2, 1)

        if not separate_heads:
            self.fc = nn.Linear(hidden_size * 2, input_size)
        else:
            # easy_idx and hard_idx are lists of feature indices (ints)
            self.easy_idx = easy_idx
            self.hard_idx = hard_idx
            # create heads that output only required dims
            easy_out = len(easy_idx)
            hard_out = len(hard_idx)
            self.fc_easy = nn.Linear(hidden_size * 2, easy_out) if easy_out > 0 else None
            self.fc_hard = nn.Linear(hidden_size * 2, hard_out) if hard_out > 0 else None

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_proj = self.input_projection(x)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)

        if not self.separate_heads:
            return self.fc(context_vector)
        else:
            parts = []
            if self.fc_easy is not None:
                parts.append(self.fc_easy(context_vector))
            if self.fc_hard is not None:
                parts.append(self.fc_hard(context_vector))
            # Now assemble into original feature order
            # parts[0] corresponds to easy features (in order of easy_idx), parts[1] to hard_idx
            # We need to create an output tensor of shape (batch, input_size)
            batch = context_vector.size(0)
            device = context_vector.device
            out = torch.zeros((batch, self.input_size), device=device)
            p_i = 0
            if self.fc_easy is not None:
                out[:, self.easy_idx] = parts[0]
            if self.fc_hard is not None:
                out[:, self.hard_idx] = parts[-1 if len(parts)>1 else 0]
            return out

def compute_feature_weights_from_variance(df, feature_cols, mode='var', r2_array=None, eps=1e-6):
    # mode 'var' returns weights proportional to variance (so bigger-variance features weighed more)
    # mode 'r2' returns weights inversely proportional to r2: w = 1 / (r2 + eps)
    if mode == 'var':
        variances = df[feature_cols].var(axis=0).to_numpy(dtype=np.float32)
        # If variance is zero (constant feature), give small positive
        variances = np.maximum(variances, eps)
        weights = variances.copy()
    elif mode == 'r2':
        if r2_array is None:
            raise ValueError("r2_array must be provided for mode='r2'")
        r2_arr = np.array(r2_array, dtype=np.float32)
        r2_arr = np.clip(r2_arr, -1.0 + eps, None)  # avoid -1 or worse
        weights = 1.0 / (r2_arr + 1e-6)  # inverse to r2 (higher weight for lower r2)
        # If any r2 <= 0 produce larger weight (that's intended) but clip
        weights = np.maximum(weights, eps)
    else:
        raise ValueError("Unknown mode for computing weights")

    # Normalize so mean weight = 1.0 (keeps scale of loss comparable)
    weights = weights.astype(np.float32)
    weights = weights / (weights.mean() + 1e-12)
    return weights  # numpy array shape (D,)

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

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    # create dataset instances (unchanged)
    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')

    # compute feature weights
    feature_cols = train_ds.feature_cols
    D = len(feature_cols)
    if args.feature_weight_mode == 'r2':
        # load r2s from provided file (must be length D)
        r2_arr = np.load(args.feature_r2_path)
        if len(r2_arr) != D:
            raise ValueError(f"Loaded r2 length {len(r2_arr)} doesn't match feature dim {D}")
        weights_np = compute_feature_weights_from_variance(None, feature_cols, mode='r2', r2_array=r2_arr)
    else:
        # default: variance mode; we will compute variance from df_train
        weights_np = compute_feature_weights_from_variance(df_train, feature_cols, mode='var')

    # optional: save weights for inspection
    np.save(args.checkpoint + ".feature_weights.npy", weights_np)

    # determine easy/hard split if separate_heads True
    easy_idx = hard_idx = None
    if args.separate_heads:
        median = np.median(weights_np)
        hard_mask = weights_np > median
        hard_idx = [i for i, m in enumerate(hard_mask) if m]
        easy_idx = [i for i, m in enumerate(hard_mask) if not m]
        print(f"Separate heads ON. Easy features: {len(easy_idx)}, Hard features: {len(hard_idx)}")
    else:
        easy_idx = hard_idx = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        separate_heads=args.separate_heads,
        easy_idx=easy_idx,
        hard_idx=hard_idx
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    criterion = nn.MSELoss(reduction='none')  # we'll apply per-feature weighting manually

    # convert weights to torch tensor on device
    feature_weights = torch.from_numpy(weights_np).to(device)  # shape (D,)
    # make sure the feature_weights broadcast properly: we'll multiply per-feature MSE by this

    best_val = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # y shape: (batch, D)
            optimizer.zero_grad()
            out = model(x)  # (batch, D)
            # compute per-element squared error
            se = (out - y) ** 2  # (batch, D)
            # multiply by per-feature weights
            weighted_se = se * feature_weights.unsqueeze(0)  # broadcast to (batch, D)
            # reduce to scalar: mean over batch and features
            loss = weighted_se.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        val_mean_r2, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Weighted Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")

        scheduler.step()

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'feature_weights': weights_np.tolist(),
                'args': vars(args)
            }, args.checkpoint)

    print(f"Best Val Mean R2: {best_val:.6f}")

if __name__ == '__main__':
    class Args:
        def __init__(self):
            # data and basic training
            self.data = '/content/drive/MyDrive/datasets/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 20
            self.val_frac = 0.1
            self.checkpoint = 'lstm_attention_proj_checkpoint_weighted.pt'

            # NEW options:
            # feature_weight_mode: 'var' (default) or 'r2'
            # - 'var' uses per-feature variance computed from the training set to weight loss.
            # - 'r2' requires feature R2 array to be provided: set feature_r2_path to a numpy .npy file with shape (D,)
            self.feature_weight_mode = 'var'
            self.feature_r2_path = ''  # e.g., 'per_feature_val_r2.npy' if using 'r2' mode

            # separate_heads: if True, create separate easy/hard heads (split by median weight)
            self.separate_heads = False

    args = Args()
    main(args)
