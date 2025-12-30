"""
solution.py

Inference wrapper for the improved LSTMAttentionModel training pipeline.

Behavior:
 - Prefer single averaged_checkpoint.pt if present.
 - Otherwise, load fold checkpoints produced by your training script:
     lstm_weighted_ordered.fold0.pt ... lstm_weighted_ordered.fold4.pt
 - Handles compact FP16 checkpoints (converts weights back to float).
 - Reconstructs 'separate_heads' easy/hard indices from saved feature_weights.
"""

import os
from pathlib import Path
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Competition requirement: DataPoint type from utils
try:
    from utils import DataPoint
except Exception:
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray


# ---------------------------
# Model definition (must match training)
# ---------------------------
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1,
                 use_mha=True, attn_heads=4, use_pos_emb=True, use_gate=True,
                 proj_dropout=0.1, attn_dropout=0.1, max_seq_len=200,
                 separate_heads=True, easy_idx=None, hard_idx=None):
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

        # Input projection + skip
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
        else:
            self.pos_emb = None

        # LSTM encoder (bidirectional)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True
        )

        # Attention
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

        if use_gate:
            self.gate_fc = nn.Linear(hidden_size * 2, input_size)

        self.out_scale = nn.Parameter(torch.ones(input_size))

        if separate_heads:
            easy_out = len(self.easy_idx)
            hard_out = len(self.hard_idx)
            self.fc_easy = nn.Linear(hidden_size * 2, easy_out) if easy_out > 0 else None
            self.fc_hard = nn.Linear(hidden_size * 2, hard_out) if hard_out > 0 else None

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        b, seq_len, _ = x.size()

        x_proj = self.input_projection(x)
        x_proj = x_proj + self.input_skip(x)
        x_proj = self.input_dropout(x_proj)

        # robust pos embedding: tile if seq_len > pos_emb_len
        if self.pos_emb is not None:
            pe_len = self.pos_emb.shape[0]
            if seq_len <= pe_len:
                pos = self.pos_emb[:seq_len].unsqueeze(0).to(x_proj.device)
            else:
                repeats = (seq_len + pe_len - 1) // pe_len
                pos_long = self.pos_emb.unsqueeze(0).repeat(repeats, 1, 1).view(-1, self.pos_emb.size(1))
                pos = pos_long[:seq_len].unsqueeze(0).to(x_proj.device)
            x_proj = x_proj + pos

        lstm_out, _ = self.lstm(x_proj)  # (b, seq, hidden*2)

        if self.use_mha:
            last = lstm_out[:, -1:, :]
            query = self.query_proj(last)
            attn_out, _ = self.mha(query, lstm_out, lstm_out, need_weights=False)
            context = attn_out.squeeze(1)
        else:
            scores = self.attention_fc(lstm_out)
            weights = F.softmax(scores, dim=1)
            context = torch.sum(weights * lstm_out, dim=1)

        context = self.post_attn_ln(context + lstm_out[:, -1, :])

        if not self.separate_heads:
            model_out = self.decoder(context)
        else:
            out = torch.zeros((b, self.input_size), device=x.device)
            if getattr(self, 'fc_easy', None) is not None:
                out[:, self.easy_idx] = self.fc_easy(context)
            if getattr(self, 'fc_hard', None) is not None:
                out[:, self.hard_idx] = self.fc_hard(context)
            model_out = out

        if self.use_gate:
            gate = torch.sigmoid(self.gate_fc(context))
            last_obs = x[:, -1, :]
            final = gate * (model_out * self.out_scale) + (1.0 - gate) * last_obs
        else:
            final = model_out * self.out_scale

        return final


# ---------------------------
# Prediction wrapper (ensemble-aware)
# ---------------------------
class PredictionModel:
    def __init__(self):
        print("Initializing PredictionModel (ensemble-aware)...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default checkpoint base used by training script
        self.fold_base = "lstm_weighted_ordered"
        self.n_folds = 5
        self.seq_len = 150  # default; will override if saved args include seq_len

        # Prefer single averaged model if present
        averaged_path = os.path.join(script_dir, "averaged_checkpoint.pt")
        self.models = []  # list of torch.nn models
        self.model_args = None
        self.feature_weights = None

        if os.path.exists(averaged_path):
            print("Found averaged_checkpoint.pt — loading single averaged model.")
            ckpt = torch.load(averaged_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            feat_w = ckpt.get('feature_weights', None)
            saved_args = ckpt.get('args', {}) or {}
            self._load_single_state(state, feat_w, saved_args)
        else:
            # try loading fold files
            loaded = 0
            for i in range(self.n_folds):
                path = os.path.join(script_dir, f"{self.fold_base}.fold{i}.pt")
                if not os.path.exists(path):
                    # try alternative naming
                    alt = os.path.join(script_dir, f"{self.fold_base}.fold{i}.pt")
                    if not os.path.exists(alt):
                        continue
                    path = alt
                ckpt = torch.load(path, map_location='cpu')
                state = ckpt.get('model_state_dict', ckpt)
                feat_w = ckpt.get('feature_weights', None)
                saved_args = ckpt.get('args', {}) or {}
                m = self._build_model_from_state(state, feat_w, saved_args)
                if m is not None:
                    self.models.append(m)
                    loaded += 1
            if loaded == 0:
                raise RuntimeError("No checkpoints found. Place 'averaged_checkpoint.pt' or fold checkpoints next to solution.py")
            print(f"Loaded {loaded} fold models for ensemble.")

        # inference-state
        self.current_seq_ix = -1
        self.state_buffer = []

    def _load_single_state(self, state, feat_w, saved_args):
        # Build model from saved_args and load the averaged state
        # infer input_size: check input_projection weight or decoder final layer
        input_size = None
        if isinstance(state, dict):
            # state[k] can be half tensors
            if 'input_projection.0.weight' in state:
                w = state['input_projection.0.weight']
                input_size = int(w.shape[1])
            else:
                # try decoder final weight 'decoder.4.weight' (depends on model structure)
                for k, v in state.items():
                    if k.endswith('.weight') and v.ndim == 2:
                        # heuristic: if out_features==in_features for decoder final we can't know; just take latest found
                        input_size = int(v.shape[1])
                        break
        if input_size is None:
            raise RuntimeError("Cannot infer input_size from checkpoint.")

        # reconstruct separate_heads easy/hard if needed
        separate_heads = True
        easy_idx = hard_idx = None
        if separate_heads and feat_w is not None:
            fw = np.array(feat_w, dtype=np.float32)
            med = np.median(fw)
            hard_idx = [i for i,w in enumerate(fw) if w > med]
            easy_idx = [i for i,w in enumerate(fw) if w <= med]

        # read other args (with sensible defaults)
        hidden = int(saved_args.get('hidden_size', saved_args.get('hidden', 64)))
        num_layers = int(saved_args.get('num_layers', 2))
        dropout = float(saved_args.get('dropout', 0.0))
        use_mha = bool(saved_args.get('use_mha', True))
        attn_heads = int(saved_args.get('attn_heads', 4))
        use_pos_emb = bool(saved_args.get('use_pos_emb', True))
        use_gate = bool(saved_args.get('use_gate', True))
        proj_dropout = float(saved_args.get('proj_dropout', 0.1))
        attn_dropout = float(saved_args.get('attn_dropout', 0.1))
        max_seq_len = int(saved_args.get('max_seq_len', 200))
        self.seq_len = int(saved_args.get('seq_len', self.seq_len))

        model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            use_mha=use_mha,
            attn_heads=attn_heads,
            use_pos_emb=use_pos_emb,
            use_gate=use_gate,
            proj_dropout=proj_dropout,
            attn_dropout=attn_dropout,
            max_seq_len=max_seq_len,
            separate_heads=separate_heads,
            easy_idx=easy_idx,
            hard_idx=hard_idx
        )

        # convert half->float if needed and tensors->torch.tensor
        state_fp32 = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state_fp32[k] = v.float()
            else:
                # numpy array
                state_fp32[k] = torch.tensor(v, dtype=torch.float32)
        model.load_state_dict(state_fp32)
        model.to(self.device)
        model.eval()
        self.models = [model]
        self.model_args = saved_args
        self.feature_weights = feat_w

    def _build_model_from_state(self, state, feat_w, saved_args):
        # similar to above but returns model instance
        # infer input size
        input_size = None
        if 'input_projection.0.weight' in state:
            w = state['input_projection.0.weight']
            input_size = int(w.shape[1])
        else:
            for k, v in state.items():
                if k.endswith('.weight') and v.ndim == 2:
                    input_size = int(v.shape[1])
                    break
        if input_size is None:
            return None

        separate_heads = True
        easy_idx = hard_idx = None
        if separate_heads and feat_w is not None:
            fw = np.array(feat_w, dtype=np.float32)
            med = np.median(fw)
            hard_idx = [i for i,w in enumerate(fw) if w > med]
            easy_idx = [i for i,w in enumerate(fw) if w <= med]

        hidden = int(saved_args.get('hidden_size', saved_args.get('hidden', 64)))
        num_layers = int(saved_args.get('num_layers', 2))
        dropout = float(saved_args.get('dropout', 0.0))
        use_mha = bool(saved_args.get('use_mha', True))
        attn_heads = int(saved_args.get('attn_heads', 4))
        use_pos_emb = bool(saved_args.get('use_pos_emb', True))
        use_gate = bool(saved_args.get('use_gate', True))
        proj_dropout = float(saved_args.get('proj_dropout', 0.1))
        attn_dropout = float(saved_args.get('attn_dropout', 0.1))
        max_seq_len = int(saved_args.get('max_seq_len', 200))
        self.seq_len = int(saved_args.get('seq_len', self.seq_len))

        model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            use_mha=use_mha,
            attn_heads=attn_heads,
            use_pos_emb=use_pos_emb,
            use_gate=use_gate,
            proj_dropout=proj_dropout,
            attn_dropout=attn_dropout,
            max_seq_len=max_seq_len,
            separate_heads=separate_heads,
            easy_idx=easy_idx,
            hard_idx=hard_idx
        )

        # convert half->float if needed
        state_fp32 = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state_fp32[k] = v.float()
            else:
                state_fp32[k] = torch.tensor(v, dtype=torch.float32)

        model.load_state_dict(state_fp32)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, data_point: DataPoint):
        # sequence reset when seq_ix changes
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        self.state_buffer.append(data_point.state.astype(np.float32))
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # prepare window with front-padding zeros
        window = np.array(self.state_buffer, dtype=np.float32)
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])

        x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)  # (1, seq, D)

        preds = []
        with torch.no_grad():
            for model in self.models:
                out = model(x)  # (1, D)
                preds.append(out.cpu().numpy()[0])
        preds = np.stack(preds, axis=0)  # (n_models, D)
        avg = np.mean(preds, axis=0)
        return avg


# Local sanity-check (optional)
if __name__ == "__main__":
    print("Local check: Attempt to initialize PredictionModel (will require checkpoints present).")
    try:
        pm = PredictionModel()
        print("PredictionModel initialized. Loaded models:", len(pm.models))
    except Exception as e:
        print("Initialization failed:", e)
