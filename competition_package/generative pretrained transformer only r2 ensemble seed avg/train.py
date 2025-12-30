#!/usr/bin/env python3
"""
train_5fold_ensemble.py

5-Fold ensemble training for LSTMAttentionModel (with per-feature weighting and optional separate heads).

Usage (example):
    python train_5fold_ensemble.py --data /kaggle/input/trainds/train.parquet --n_splits 5 --epochs 20

Outputs (in working dir):
    - lstm_attention_proj_checkpoint_weighted.pt.fold{0..4}.pt
    - oof_preds_fold{0..4}.npy
    - oof_trues_fold{0..4}.npy
    - per_feature_val_r2.npy
    - (optional) test_preds_fold{0..4}.npy and test_preds_ensemble.npy
"""

import argparse
import math
from pathlib import Path
import time
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import random

# ---------------------------
# Dataset & Model (same as your modified code)
# ---------------------------
class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.examples = []

        grouped = df.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                target = states[t + 1]  # absolute next state
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
            self.easy_idx = easy_idx or []
            self.hard_idx = hard_idx or []
            easy_out = len(self.easy_idx)
            hard_out = len(self.hard_idx)
            self.fc_easy = nn.Linear(hidden_size * 2, easy_out) if easy_out > 0 else None
            self.fc_hard = nn.Linear(hidden_size * 2, hard_out) if hard_out > 0 else None

    def forward(self, x):
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        if not self.separate_heads:
            return self.fc(context_vector)
        else:
            batch = context_vector.size(0)
            device = context_vector.device
            out = torch.zeros((batch, self.input_size), device=device)
            if self.fc_easy is not None:
                out[:, self.easy_idx] = self.fc_easy(context_vector)
            if self.fc_hard is not None:
                out[:, self.hard_idx] = self.fc_hard(context_vector)
            return out


# ---------------------------
# Utilities for weights, eval
# ---------------------------
def compute_feature_weights_from_variance(df, feature_cols, mode='var', r2_array=None, eps=1e-6):
    if mode == 'var':
        variances = df[feature_cols].var(axis=0).to_numpy(dtype=np.float32)
        variances = np.maximum(variances, eps)
        weights = variances.copy()
    elif mode == 'r2':
        if r2_array is None:
            raise ValueError("r2_array must be provided for mode='r2'")
        r2_arr = np.array(r2_array, dtype=np.float32)
        r2_arr = np.clip(r2_arr, -1.0 + eps, None)
        weights = 1.0 / (r2_arr + 1e-6)
        weights = np.maximum(weights, eps)
    else:
        raise ValueError("Unknown mode for computing weights")
    weights = weights.astype(np.float32)
    weights = weights / (weights.mean() + 1e-12)
    return weights

def evaluate(model, dataloader, device):
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            ys_true.append(y.cpu().numpy())
            ys_pred.append(pred.cpu().numpy())
    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    r2s = []
    for i in range(y_true.shape[1]):
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        except:
            r2 = float('nan')
        r2s.append(r2)
    mean_r2 = float(np.nanmean(r2s))
    return mean_r2, r2s, y_true, y_pred


# ---------------------------
# Training per-fold function
# ---------------------------
def train_one_fold(fold, train_df, val_df, args, easy_idx=None, hard_idx=None, seed=0, test_df=None):
    print(f"\n=== Fold {fold} — seed={seed} ===")
    # fix seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = SequenceDataset(train_df, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(val_df, seq_len=args.seq_len, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    # Determine easy/hard indices if separate_heads requested and not already given
    if args.separate_heads and (easy_idx is None or hard_idx is None):
        # compute weights from training df (var mode or r2 mode)
        if args.feature_weight_mode == 'r2':
            r2_arr = np.load(args.feature_r2_path)
            weights_np = compute_feature_weights_from_variance(None, train_ds.feature_cols, mode='r2', r2_array=r2_arr)
        else:
            weights_np = compute_feature_weights_from_variance(train_df, train_ds.feature_cols, mode='var')
        median = np.median(weights_np)
        hard_mask = weights_np > median
        hard_idx = [i for i, m in enumerate(hard_mask) if m]
        easy_idx = [i for i, m in enumerate(hard_mask) if not m]
        print(f"Fold {fold}: easy={len(easy_idx)} hard={len(hard_idx)}")
    elif not args.separate_heads:
        easy_idx = hard_idx = None

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
    criterion = nn.MSELoss(reduction='none')

    # compute feature weights for loss
    if args.feature_weight_mode == 'r2':
        r2_arr = np.load(args.feature_r2_path)
        if len(r2_arr) != input_size:
            raise ValueError("r2 array length mismatch")
        weights_np = compute_feature_weights_from_variance(None, train_ds.feature_cols, mode='r2', r2_array=r2_arr)
    else:
        weights_np = compute_feature_weights_from_variance(train_df, train_ds.feature_cols, mode='var')
    feature_weights = torch.from_numpy(weights_np).to(device)

    best_val = -math.inf
    best_state = None
    accumulated_steps = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            se = (out - y) ** 2
            weighted_se = se * feature_weights.unsqueeze(0)
            loss = weighted_se.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            accumulated_steps += 1
        avg_loss = running_loss / len(train_loader.dataset)

        val_mean_r2, val_r2s, _, _ = evaluate(model, val_loader, device)
        print(f"Fold {fold} Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f} (time {time.time()-t0:.1f}s)")
        scheduler.step()

        # save best
        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            best_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'feature_weights': weights_np.tolist(),
                'args': vars(args)
            }
            ckpt_path = f"{args.checkpoint}.fold{fold}.pt"
            torch.save(best_state, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # After training, evaluate best model on val and test (if provided)
    print(f"Fold {fold}: loading best checkpoint for evaluation.")
    ckpt = best_state
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_mean_r2, val_r2s, y_true_val, y_pred_val = evaluate(model, val_loader, device)
    print(f"Fold {fold} final Val Mean R2={val_mean_r2:.6f}")

    # Save OOF for this fold
    np.save(f"oof_trues_fold{fold}.npy", y_true_val.astype(np.float32))
    np.save(f"oof_preds_fold{fold}.npy", y_pred_val.astype(np.float32))
    print(f"Saved OOF arrays for fold {fold} (shape {y_pred_val.shape})")

    # Test predictions if test_df provided
    test_preds = None
    if test_df is not None:
        # Build dataset for test - we still need windows; targets are dummy
        test_ds = SequenceDataset(test_df, seq_len=args.seq_len, mode='test')
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                p = model(x).cpu().numpy()
                preds.append(p)
        preds = np.vstack(preds)
        np.save(f"test_preds_fold{fold}.npy", preds.astype(np.float32))
        print(f"Saved test preds for fold {fold} shape={preds.shape}")
        test_preds = preds

    return {
        'fold': fold,
        'val_mean_r2': val_mean_r2,
        'val_r2s': val_r2s,
        'oof_true': y_true_val,
        'oof_pred': y_pred_val,
        'test_pred': test_preds
    }


# ---------------------------
# Main: K-Fold orchestration
# ---------------------------
def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    seq_ids = np.array(sorted(df['seq_ix'].unique()))
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_results = []
    test_preds_all = []
    oof_trues_list = []
    oof_preds_list = []

    # optional test set
    test_df = None
    if args.test_data is not None:
        test_df = pd.read_parquet(Path(args.test_data))
        print(f"Loaded test data with {test_df['seq_ix'].nunique()} sequences.")

    for fold, (train_idx, val_idx) in enumerate(kf.split(seq_ids)):
        train_ids = seq_ids[train_idx]
        val_ids = seq_ids[val_idx]

        df_train = df[df['seq_ix'].isin(train_ids)].reset_index(drop=True)
        df_val = df[df['seq_ix'].isin(val_ids)].reset_index(drop=True)

        res = train_one_fold(
            fold=fold,
            train_df=df_train,
            val_df=df_val,
            args=args,
            seed=args.seed + fold,
            test_df=test_df
        )
        fold_results.append(res)

        # collect OOF pieces
        oof_trues_list.append(res['oof_true'])
        oof_preds_list.append(res['oof_pred'])
        if res['test_pred'] is not None:
            test_preds_all.append(res['test_pred'])

    # Concatenate OOF arrays in the order of folds (they are per-fold val sets)
    all_oof_trues = np.vstack(oof_trues_list)
    all_oof_preds = np.vstack(oof_preds_list)
    print(f"Combined OOF shape: {all_oof_preds.shape}")

    # compute per-feature R2 and save
    per_feature_r2 = []
    for i in range(all_oof_trues.shape[1]):
        try:
            r2 = r2_score(all_oof_trues[:, i], all_oof_preds[:, i])
        except:
            r2 = float('nan')
        per_feature_r2.append(r2)
    mean_r2_overall = np.nanmean(per_feature_r2)
    print(f"Overall OOF Mean R2 (all folds): {mean_r2_overall:.6f}")
    np.save("per_feature_val_r2.npy", np.array(per_feature_r2, dtype=np.float32))
    print("Saved per_feature_val_r2.npy")

    # If test preds exist, average them across folds (simple mean)
    if len(test_preds_all) > 0:
        # ensure same shape
        test_preds_all = np.stack(test_preds_all, axis=0)  # (n_folds, N_test, D)
        ensemble_test = test_preds_all.mean(axis=0)
        np.save("test_preds_ensemble.npy", ensemble_test.astype(np.float32))
        print(f"Saved ensemble test preds shape {ensemble_test.shape}")

    # Save summary JSON
    summary = {
        'n_folds': args.n_splits,
        'mean_oof_r2': float(mean_r2_overall),
        'per_feature_r2': [float(x) for x in per_feature_r2]
    }
    with open("ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote ensemble_summary.json")

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True, help='train parquet path')
    # parser.add_argument('--test_data', type=str, default=None, help='optional test parquet to predict')
    # parser.add_argument('--n_splits', type=int, default=5)
    # parser.add_argument('--seq_len', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--hidden_size', type=int, default=128)
    # parser.add_argument('--num_layers', type=int, default=2)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--val_frac', type=float, default=0.1)
    # parser.add_argument('--checkpoint', type=str, default='lstm_attention_proj_checkpoint_weighted.pt',
                        # help='base checkpoint filename; .fold{fold}.pt will be appended')
    # parser.add_argument('--feature_weight_mode', type=str, default='r2', choices=['r2', 'var'])
    # parser.add_argument('--feature_r2_path', type=str, default='per_feature_val_r2.npy',
                        # help='if feature_weight_mode==r2, path to numpy array of per-feature r2 (length D)')
    # parser.add_argument('--separate_heads', action='store_true')
    # parser.add_argument('--seed', type=int, default=42)

    # args = parser.parse_args()
    # main(args)

if __name__ == '__main__':
    class Args:
        def __init__(self):
            # data and basic training
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 5
            self.val_frac = 0.1
            self.checkpoint = 'lstm_attention_proj_checkpoint_weighted_'
            self.n_splits = 5
            self.test_data = None

            # NEW options:
            # feature_weight_mode: 'var' (default) or 'r2'
            # - 'var' uses per-feature variance computed from the training set to weight loss.
            # - 'r2' requires feature R2 array to be provided: set feature_r2_path to a numpy .npy file with shape (D,)
            self.feature_weight_mode = 'r2'
            self.feature_r2_path = '/kaggle/input/r2vals/per_feature_val_r2.npy'  # e.g., 'per_feature_val_r2.npy' if using 'r2' mode

            # separate_heads: if True, create separate easy/hard heads (split by median weight)
            self.separate_heads = False
            self.seed = 42

    args = Args()
    main(args)
