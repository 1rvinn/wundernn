# -------------------------------
# Full K-Fold OOF Stacking Script
# (drop into your existing file / replace main)
# -------------------------------
import argparse
import math
from pathlib import Path
import time
import os
import heapq

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

# --- Your existing SequenceDataset (unchanged) ---
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
        # print(f"Creating {mode} examples...") 
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            # states is now the RAW data
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


# --- LSTMAttentionModel (unchanged) ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- IMPROVEMENT C: Input Projection Layer ---
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # 1. Encoder (The LSTM)
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # 1. Apply Input Projection
        x_proj = self.input_projection(x)
        
        # 2. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 4. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 5. Decode
        return self.fc(context_vector)


# --- Evaluate utility (unchanged) ---
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


# --- Train single fold (returns trained model path) ---
def train_fold_save_checkpoint(fold_idx, train_ids, val_ids, df_all, args, device):
    print(f"\n--- Starting Fold {fold_idx} / {args.n_folds - 1} ---")
    print(f"Train seqs: {len(train_ids)}, Val seqs: {len(val_ids)}")

    df_train = df_all[df_all['seq_ix'].isin(train_ids)]
    df_val = df_all[df_all['seq_ix'].isin(val_ids)]

    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    input_size = len(train_ds.feature_cols)

    model = LSTMAttentionModel(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0, T_mult=1, eta_min=args.cosine_eta_min)
    criterion = nn.MSELoss()

    best_val_r2 = -math.inf
    checkpoint_name = f"{args.checkpoint_prefix}_fold_{fold_idx}.pt"

    # (optional) top-K save if you want multiple ckpts (not used directly here but saved)
    top_k = args.top_k_checkpoints
    saved = []  # min-heap of (val_r2, epoch, filepath)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        val_mean_r2, _ = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step() 

        # Save per-epoch checkpoint (optional; useful for top-K or snapshot)
        ep_ckpt = f"{args.checkpoint_prefix}_fold{fold_idx}_ep{epoch}.pt"
        torch.save(model.state_dict(), ep_ckpt)

        # maintain top-k
        if top_k > 0:
            if len(saved) < top_k:
                heapq.heappush(saved, (val_mean_r2, epoch, ep_ckpt))
            else:
                if val_mean_r2 > saved[0][0]:
                    heapq.heapreplace(saved, (val_mean_r2, epoch, ep_ckpt))

        # Save best
        if val_mean_r2 > best_val_r2:
            best_val_r2 = val_mean_r2
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  [Fold {fold_idx} Epoch {epoch}] New Best R2: {val_mean_r2:.4f} (Saved best)")
        else:
            print(f"  [Fold {fold_idx} Epoch {epoch}] Loss: {avg_loss:.6f} | R2: {val_mean_r2:.6f}")

    # Optionally save top-k list for later ensembling
    topk_list = sorted(saved, key=lambda x: -x[0])  # desc by r2
    np.save(f"{args.checkpoint_prefix}_fold{fold_idx}_topk.npy", np.array([(r,e,p) for (r,e,p) in topk_list], dtype=object))

    print(f"--- Finished Fold {fold_idx}. Best R2: {best_val_r2:.4f} ---")
    return checkpoint_name, val_ds  # return path to best ckpt and validation dataset (for producing OOF preds)


# --- Helpers for inference/prediction with a saved model ---
def predict_with_checkpoint(checkpoint_path, dataset, device, args):
    """Load checkpoint -> predict in same ordering as dataset (no shuffle)."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = LSTMAttentionModel(input_size=len(dataset.feature_cols),
                               hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               dropout=args.dropout).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    truths = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            truths.append(y.numpy())
    preds = np.vstack(preds)
    truths = np.vstack(truths)
    return preds, truths


# --- Stack training orchestration (train all folds, collect OOF preds, fit Ridge) ---
def train_full_oof_stack(df, args):
    seq_ids = df['seq_ix'].unique()
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # storage for OOF
    oof_preds_list = []
    oof_y_list = []
    fold_ckpts = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(seq_ids)):
        print(f"\n=== Running Fold {fold_idx} ===")
        train_seq_ids = seq_ids[train_idx]
        val_seq_ids = seq_ids[val_idx]

        ckpt_path, val_ds = train_fold_save_checkpoint(fold_idx, train_seq_ids, val_seq_ids, df, args, device)
        fold_ckpts.append(ckpt_path)

        # produce predictions on the held-out validation examples
        preds, truths = predict_with_checkpoint(ckpt_path, val_ds, device, args)
        # preds shape: (N_val_examples, output_dim)
        print(f"  Fold {fold_idx} produced {preds.shape[0]} val examples.")

        oof_preds_list.append(preds)
        oof_y_list.append(truths)

    # concatenate OOFs (order doesn't need to match seq order; meta is trained on these pairs)
    X_oof = np.vstack(oof_preds_list)  # shape: (N_total_examples, output_dim)
    Y_oof = np.vstack(oof_y_list)      # same shape: (N_total_examples, output_dim)

    # To build meta features that include all base-model predictions, we need each base's prediction for each example.
    # The simplest approach: retrain/predict each saved fold model on the FULL training set (all examples) THEN take columns per model.
    # but that is expensive. Instead, we will:
    # - Train meta using concatenation of per-fold preds for the OOF examples only (one model per fold).
    #   For that, we'll stack per-fold predictions horizontally for OOF examples in the correct example order.
    #
    # Build meta features for OOF examples: for OOF examples, we have exactly one base-model prediction (the fold model that didn't see them).
    # We will therefore create a meta feature matrix by concatenating each base model's predictions for the OOF subset it predicted.
    # To do stacking correctly (so meta sees all base model outputs for each example), it's typical to:
    #  - Train K base models (one per fold)
    #  - Create a matrix where for each training example you have predictions from all K base models (via cross-predicting)
    # Simpler pragmatic approach:
    #  - Do a second pass: for each saved fold model, predict on ALL training examples -> results in (N_total_examples, output_dim) per model.
    #  - Then construct X_meta by concatenating per-model predictions horizontally: (N_total_examples, output_dim * n_models).
    #
    # We'll do that below (requires predicting with each ckpt on the entire training dataset). This is the standard approach.

    print("\n--- Building full meta feature matrix: predicting each base model on the full training set ---")
    # full training dataset (all examples) to obtain consistent ordering
    full_train_ds = SequenceDataset(df, seq_len=args.seq_len, mode='train')
    full_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=False)

    # For memory reasons, we will accumulate per-model preds as columns
    per_model_preds = []
    for i, ckpt in enumerate(fold_ckpts):
        print(f" Predicting full train set with model {i} ({ckpt}) ...")
        preds_i, _ = predict_with_checkpoint(ckpt, full_train_ds, device, args)
        per_model_preds.append(preds_i)  # shape (N_examples, output_dim)

    # Stack horizontally: (N_examples, output_dim * n_models)
    X_meta = np.hstack(per_model_preds)
    Y_meta = np.vstack([y for y in oof_y_list])  # Y_meta corresponds to full_train_ds ordering, but we must ensure alignment.
    # Important: Y_meta above is only OOF truths concatenated; we need full ground truth for full_train_ds
    _, full_truths = predict_with_checkpoint(fold_ckpts[0], full_train_ds, device, args)  # reuse loader to get truths (loader gives y)
    Y_full = full_truths  # (N_examples, output_dim)

    # Fit Ridge meta on full dataset (X_meta -> Y_full)
    print("Meta training: X_meta.shape =", X_meta.shape, "Y_full.shape =", Y_full.shape)
    meta = Ridge(alpha=args.meta_alpha)
    meta.fit(X_meta, Y_full)  # Ridge supports multi-output

    # Save meta model and ckpt list
    import joblib
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, f"{args.checkpoint_prefix}_meta_ridge.pkl")
    joblib.dump(meta, meta_path)
    np.save(os.path.join(args.output_dir, "stacking_base_ckpts.npy"), np.array(fold_ckpts, dtype=object))

    print("Saved meta to:", meta_path)
    return fold_ckpts, meta_path, full_train_ds  # return artifacts needed for stacked prediction


# --- Stacked prediction function (for new data) ---
def stacked_predict(df_new, fold_ckpts, meta_path, args, device):
    """
    df_new: DataFrame with the same schema as training. We'll build SequenceDataset and get predictions per base.
    fold_ckpts: list of base model checkpoint paths (one per fold)
    meta_path: path to trained ridge meta
    Returns: final_preds (N_examples, output_dim)
    """
    # Load meta
    import joblib
    meta = joblib.load(meta_path)

    ds_new = SequenceDataset(df_new, seq_len=args.seq_len, mode='test')
    # For each base model, predict on df_new and collect predictions
    per_model_preds = []
    for ckpt in fold_ckpts:
        preds_i, _ = predict_with_checkpoint(ckpt, ds_new, device, args)
        per_model_preds.append(preds_i)

    X_meta_new = np.hstack(per_model_preds)  # (N_examples, output_dim * n_models)
    final_preds = meta.predict(X_meta_new)   # (N_examples, output_dim) - Ridge multi-output
    return final_preds


# --- MAIN wrapper to run everything ---
def main_full_stack(args):
    p = Path(args.data)
    print(f"Loading data from {p}...")
    df = pd.read_parquet(p)

    # Train all folds, collect ckpts and meta
    fold_ckpts, meta_path, full_train_ds = train_full_oof_stack(df, args)

    # Example: evaluate stacked predictions on training set (sanity check)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_on_train = stacked_predict(df, fold_ckpts, meta_path, args, device)
    _, train_truths = predict_with_checkpoint(fold_ckpts[0], full_train_ds, device, args)
    # compute R2 per-feature
    r2s = []
    for i in range(train_truths.shape[1]):
        try:
            r2s.append(r2_score(train_truths[:, i], final_on_train[:, i]))
        except Exception:
            r2s.append(float('nan'))
    print("Stacked model R2 on train examples (sanity): mean =", np.nanmean(r2s), "per-feature:", r2s)

    print("Done. Artifacts:")
    print(" Base ckpts:", fold_ckpts)
    print(" Meta:", meta_path)


# ---------------------------
# Arg defaults & run
# ---------------------------
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.0
            self.lr = 1e-3
            self.epochs = 5

            # K-Fold settings
            self.n_folds = 5
            self.checkpoint_prefix = 'lstm_attention_raw_checkpoint_fold'
            self.output_dir = './stacking_artifacts'
            self.seed = 42

            # training niceties
            self.weight_decay = 1e-5
            self.cosine_T0 = 5
            self.cosine_eta_min = 1e-6
            self.clip_grad_norm = 1.0
            self.top_k_checkpoints = 0  # set >0 to keep top-K per-epoch checkpoints
            self.meta_alpha = 1.0

    args = Args()
    main_full_stack(args)
