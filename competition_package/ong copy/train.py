###############################################################
#        TRAINING FILE — KAGGLE FRIENDLY (ONE-FOLD MODE)
###############################################################

import os
import math
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ============================================================
#                     CONFIGURATION
# ============================================================

class Args:
    # ---------------- Required ----------------
    data = "/kaggle/input/trainds/train.parquet"

    # ---------------- Model & Training ----------------
    n_splits = 5
    seq_len = 150
    batch_size = 32
    hidden_size = 64
    num_layers = 2
    dropout = 0.0
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 5
    checkpoint = "lstm_weighted_ordered"

    # ---------------- Feature weighting ----------------
    feature_weight_mode = "var"     # "var" or "r2"
    feature_r2_path = "/kaggle/input/r2/transformers/default/1/per_feature_val_r2.npy"

    # ---------------- Ordered fold ----------------
    seed = 42
    # If None -> run all folds (old behavior)
    # If integer 0..n_splits-1 -> run only that fold and exit (useful for 1-notebook-per-fold)
    fold_to_train = 2

    # ---------------- Scheduler ----------------
    T_0 = 5
    eta_min = 1e-6

    # ---------------- Gradient ----------------
    max_grad_norm = 1.0

    # ---------------- Model Improvements ----------------
    use_mha = True
    attn_heads = 4
    use_pos_emb = True
    use_gate = True
    proj_dropout = 0.0
    attn_dropout = 0.0
    max_seq_len = 200

    # ---------------- Separate heads ----------------
    separate_heads = True


args = Args()  # instantiate configuration



# ============================================================
#                         DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [
            c for c in df.columns 
            if c not in ("seq_ix", "step_in_seq", "need_prediction")
        ]
        self.examples = []

        grouped = df.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g):
                    continue
                target = states[t + 1]
                start = max(0, t - self.seq_len + 1)
                window = states[start:t+1]

                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
#                  IMPROVED MODEL (FULL)
# ============================================================

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1,
                 use_mha=True, attn_heads=4, use_pos_emb=True, use_gate=True,
                 proj_dropout=0.1, attn_dropout=0.1, max_seq_len=128,
                 separate_heads=False, easy_idx=None, hard_idx=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_mha = use_mha
        self.attn_heads = attn_heads
        self.use_pos_emb = use_pos_emb
        self.use_gate = use_gate
        self.max_seq_len = max_seq_len
        self.separate_heads = separate_heads
        self.easy_idx = easy_idx or []
        self.hard_idx = hard_idx or []

        # Input projection + residual
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.input_dropout = nn.Dropout(proj_dropout)
        self.input_skip = nn.Linear(input_size, hidden_size, bias=False)

        # Positional embeddings
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, hidden_size))
            nn.init.normal_(self.pos_emb, mean=0.0, std=hidden_size ** -0.5)
        else:
            self.pos_emb = None

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True
        )

        # Multi-head attention or linear attention
        if use_mha:
            self.mha = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=attn_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            self.query_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        else:
            self.attention_fc = nn.Linear(hidden_size * 2, 1)

        self.post_attn_ln = nn.LayerNorm(hidden_size * 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, input_size)
        )

        # Output gating
        if use_gate:
            self.gate_fc = nn.Linear(hidden_size * 2, input_size)

        # Output scale
        self.out_scale = nn.Parameter(torch.ones(input_size))

        # Separate heads
        if separate_heads:
            self.fc_easy = nn.Linear(hidden_size * 2, len(self.easy_idx)) if len(self.easy_idx)>0 else None
            self.fc_hard = nn.Linear(hidden_size * 2, len(self.hard_idx)) if len(self.hard_idx)>0 else None

        self._init_weights()

    def _init_weights(self):
        # Linear Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        b, seq_len, _ = x.size()

        # Projection + residual
        x_proj = self.input_projection(x)
        x_proj = x_proj + self.input_skip(x)
        x_proj = self.input_dropout(x_proj)

        # Positional embeddings
        if self.pos_emb is not None:
            # robust pos emb handling in case seq_len > max_seq_len:
            pe_len = self.pos_emb.shape[0]
            if seq_len <= pe_len:
                pos = self.pos_emb[:seq_len].unsqueeze(0).to(x_proj.device)
            else:
                # tile to cover seq_len
                repeats = (seq_len + pe_len - 1) // pe_len
                pos_long = self.pos_emb.unsqueeze(0).repeat(repeats, 1, 1).view(-1, self.pos_emb.size(1))
                pos = pos_long[:seq_len].unsqueeze(0).to(x_proj.device)
            x_proj = x_proj + pos

        # LSTM encoder
        lstm_out, _ = self.lstm(x_proj)

        # Attention pooling
        if self.use_mha:
            last = lstm_out[:, -1:, :]
            query = self.query_proj(last)
            attn_out, _ = self.mha(query, lstm_out, lstm_out, need_weights=False)
            context = attn_out.squeeze(1)
        else:
            scores = self.attention_fc(lstm_out)
            weights = F.softmax(scores, dim=1)
            context = torch.sum(weights * lstm_out, dim=1)

        # Residual + LN
        context = self.post_attn_ln(context + lstm_out[:, -1, :])

        # Decode
        if not self.separate_heads:
            model_out = self.decoder(context)
        else:
            out = torch.zeros((b, self.input_size), device=x.device)
            if self.fc_easy is not None:
                out[:, self.easy_idx] = self.fc_easy(context)
            if self.fc_hard is not None:
                out[:, self.hard_idx] = self.fc_hard(context)
            model_out = out

        # Output gating
        if self.use_gate:
            gate = torch.sigmoid(self.gate_fc(context))
            last_obs = x[:, -1, :]
            return gate * (model_out * self.out_scale) + (1 - gate) * last_obs
        else:
            return model_out * self.out_scale



# ============================================================
#         FEATURE WEIGHTING + EVALUATION HELPERS
# ============================================================

def compute_feature_weights(df, feature_cols, mode="var", r2_array=None, eps=1e-6):
    if mode == "var":
        vals = df[feature_cols].var().values.astype(np.float32)
        vals = np.maximum(vals, eps)
        w = vals
    else:
        r2 = np.array(r2_array, dtype=np.float32)
        w = 1.0 / (r2 + eps)
    w = w / (w.mean() + 1e-12)
    return w

def evaluate(model, loader, device):
    model.eval()
    ys_t, ys_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            ys_t.append(y.cpu().numpy())
            ys_p.append(p.cpu().numpy())
    y_t = np.vstack(ys_t) if len(ys_t)>0 else np.zeros((0, model.input_size))
    y_p = np.vstack(ys_p) if len(ys_p)>0 else np.zeros((0, model.input_size))

    per_r2 = []
    for i in range(y_t.shape[1]) if y_t.shape[0]>0 else range(model.input_size):
        try:
            per_r2.append(r2_score(y_t[:, i], y_p[:, i]))
        except:
            per_r2.append(float('nan'))
    return float(np.nanmean(per_r2)) if len(per_r2)>0 else float('nan'), per_r2, y_t, y_p



# ============================================================
#             TRAIN ONE FOLD (ORDERED)
# ============================================================

def train_one_fold(fold, train_df, val_df, args, fold_seed):

    print(f"\n===== Fold {fold} | Seed {fold_seed} =====")

    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(fold_seed)

    train_ds = SequenceDataset(train_df, args.seq_len, "train")
    val_ds   = SequenceDataset(val_df,   args.seq_len, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(train_ds.feature_cols)

    # feature weights
    if args.feature_weight_mode == "r2":
        if os.path.exists(args.feature_r2_path):
            r2_arr = np.load(args.feature_r2_path)
        else:
            raise FileNotFoundError(f"Requested feature_weight_mode='r2' but file not found: {args.feature_r2_path}")
        weights_np = compute_feature_weights(None, train_ds.feature_cols, mode="r2", r2_array=r2_arr)
    else:
        weights_np = compute_feature_weights(train_df, train_ds.feature_cols, mode='var')

    easy_idx = hard_idx = None
    if args.separate_heads:
        m = np.median(weights_np)
        hard_idx = [i for i,w in enumerate(weights_np) if w > m]
        easy_idx = [i for i,w in enumerate(weights_np) if w <= m]

    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_mha=args.use_mha,
        attn_heads=args.attn_heads,
        use_pos_emb=args.use_pos_emb,
        use_gate=args.use_gate,
        proj_dropout=args.proj_dropout,
        attn_dropout=args.attn_dropout,
        max_seq_len=args.max_seq_len,
        separate_heads=args.separate_heads,
        easy_idx=easy_idx,
        hard_idx=hard_idx
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, eta_min=args.eta_min
    )

    criterion = nn.MSELoss(reduction="none")
    feature_w = torch.from_numpy(weights_np).to(device)

    best_r2 = -999
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            p = model(x)
            se = (p - y)**2
            loss = (se * feature_w.unsqueeze(0)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        val_mean, val_perr2, y_true, y_pred = evaluate(model, val_loader, device)
        print(f"Epoch {ep:02d} | TrainLoss={avg_loss:.5f} | ValR2={val_mean:.5f} | {(time.time()-t0):.1f}s")

        scheduler.step()

        if val_mean > best_r2:
            best_r2 = val_mean
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_weights": weights_np.tolist(),
                "args": args.__dict__
            }
            save_path = f"{args.checkpoint}.fold{fold}.pt"
            # Compact save: move to CPU and store as FP16 to save space
            model_sd = best_state["model_state_dict"]
            model_sd_half = {k: v.detach().cpu().half() for k, v in model_sd.items()}
            compact_ckpt = {
                'model_state_dict': model_sd_half,
                'feature_weights': best_state["feature_weights"],
                'args': best_state["args"]
            }
            torch.save(compact_ckpt, save_path)
            print(f"Saved: {save_path} (compact, fp16)")

    # After final epoch, ensure best_state exists
    if best_state is None:
        raise RuntimeError("Training completed but no checkpoint was saved (maybe validation failed)")

    # Save OOF arrays for this fold
    # reload best model and compute val predictions
    state = best_state["model_state_dict"]
    state_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in state.items()}
    model.load_state_dict(state_fp32)
    model.eval()
    _, _, y_t, y_p = evaluate(model, val_loader, device)
    np.save(f"oof_trues_fold{fold}.npy", y_t.astype(np.float32))
    np.save(f"oof_preds_fold{fold}.npy", y_p.astype(np.float32))
    print(f"Saved OOF arrays for fold {fold}")

    return {
        'fold': fold,
        'val_mean_r2': best_r2,
        'feature_weights': weights_np,
        'oof_true': y_t,
        'oof_pred': y_p,
        'ckpt_path': save_path
    }


# ============================================================
#                        TRAIN LOOP
# ============================================================

def main(args):

    df = pd.read_parquet(args.data)

    seq_ids = sorted(df["seq_ix"].unique())
    N = len(seq_ids)
    fold_size = N // args.n_splits

    # Build ordered folds
    folds = []
    for f in range(args.n_splits):
        start = f * fold_size
        end   = (f+1) * fold_size if f < args.n_splits-1 else N
        val_ids = seq_ids[start:end]
        train_ids = seq_ids[:start] + seq_ids[end:]
        folds.append((train_ids, val_ids))

    print("Ordered Folds:")
    for i, (_, val_ids) in enumerate(folds):
        print(f"Fold {i}: val={len(val_ids)}")

    # If user asked to train only one fold, do it and exit
    if args.fold_to_train is not None:
        fold = int(args.fold_to_train)
        if not (0 <= fold < args.n_splits):
            raise ValueError(f"fold_to_train must be in [0, {args.n_splits-1}]")

        train_ids, val_ids = folds[fold]
        df_train = df[df["seq_ix"].isin(train_ids)].reset_index(drop=True)
        df_val   = df[df["seq_ix"].isin(val_ids)].reset_index(drop=True)

        print(f"Running ONLY fold {fold} on this notebook.")
        res = train_one_fold(fold, df_train, df_val, args, fold_seed=args.seed + fold)

        # Save per-feature weights & fold summary for later combining
        np.save(f"{args.checkpoint}.fold{fold}.feature_weights.npy", res['feature_weights'].astype(np.float32))
        with open(f"{args.checkpoint}.fold{fold}.summary.json", "w") as f:
            json.dump({'fold': fold, 'val_mean_r2': float(res['val_mean_r2']), 'ckpt_path': res['ckpt_path']}, f, indent=2)

        print(f"Fold {fold} done. Checkpoint: {res['ckpt_path']}")
        return

    # Otherwise run full all-fold training (legacy behavior)
    all_oof_t, all_oof_p = [], []
    fold_results = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        df_train = df[df["seq_ix"].isin(train_ids)].reset_index(drop=True)
        df_val   = df[df["seq_ix"].isin(val_ids)].reset_index(drop=True)

        res = train_one_fold(fold, df_train, df_val, args, fold_seed=args.seed + fold)
        fold_results.append(res)

        # load saved compact checkpoint and compute val preds to accumulate OOF
        ckpt = torch.load(res['ckpt_path'], map_location='cpu')
        state = ckpt['model_state_dict']
        feat_w = ckpt['feature_weights']
        margs  = ckpt['args']

        D = len(feat_w)
        if margs.get("separate_heads", False):
            m = np.array(feat_w)
            med = np.median(m)
            hard_idx = [i for i,w in enumerate(m) if w > med]
            easy_idx = [i for i,w in enumerate(m) if w <= med]
        else:
            easy_idx = hard_idx = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMAttentionModel(
            input_size=D,
            hidden_size=margs.get('hidden_size', args.hidden_size),
            num_layers=margs.get('num_layers', args.num_layers),
            dropout=margs.get('dropout', args.dropout),
            use_mha=margs.get('use_mha', args.use_mha),
            attn_heads=margs.get('attn_heads', args.attn_heads),
            use_pos_emb=margs.get('use_pos_emb', args.use_pos_emb),
            use_gate=margs.get('use_gate', args.use_gate),
            proj_dropout=margs.get('proj_dropout', args.proj_dropout),
            attn_dropout=margs.get('attn_dropout', args.attn_dropout),
            max_seq_len=margs.get('max_seq_len', args.max_seq_len),
            separate_heads=margs.get('separate_heads', args.separate_heads),
            easy_idx=easy_idx,
            hard_idx=hard_idx
        ).to(device)

        # convert half->float if compact fp16 saved
        state_fp32 = {k: (v.float() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state.items()}
        model.load_state_dict(state_fp32)
        model.eval()

        val_ds   = SequenceDataset(df_val, args.seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        _, _, y_t, y_p = evaluate(model, val_loader, device)
        all_oof_t.append(y_t)
        all_oof_p.append(y_p)

    # Combine OOF if ran all folds
    all_oof_t = np.vstack(all_oof_t)
    all_oof_p = np.vstack(all_oof_p)

    per_r2 = [r2_score(all_oof_t[:, i], all_oof_p[:, i]) for i in range(all_oof_t.shape[1])]
    OOF_mean_r2 = float(np.nanmean(per_r2))

    print("\n====================")
    print(f"OOF Mean R² = {OOF_mean_r2:.6f}")
    print("====================\n")

    np.save("per_feature_val_r2.npy", np.array(per_r2, np.float32))
    print("Saved per_feature_val_r2.npy")

    with open("ensemble_summary.json", "w") as f:
        json.dump({"oof_mean_r2": OOF_mean_r2, "per_feature_r2": per_r2}, f, indent=2)

    print("Done.")


# ============================================================
#                    RUN TRAINING
# ============================================================

main(args)
