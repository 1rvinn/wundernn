"""
solution.py
Stacked 5-Fold Ensemble for LSTMAttentionModel.

Behavior:
 - Loads up to 5 base model checkpoints (fold_0 ... fold_4).
 - Attempts to load a stacking meta-learner:
    * tries joblib pickle at ./stacking_artifacts/*meta*.pkl
    * or tries numpy weights (meta_weights.npy & meta_bias.npy)
 - If no meta is found, falls back to averaging base model predictions.
 - Prediction API: PredictionModel.predict(data_point: DataPoint) -> np.ndarray | None
"""

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# --- DataPoint compatibility with hidden harness ---
try:
    from utils import DataPoint
except Exception:
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

# ------------------------
# Model definition (matches training script)
# ------------------------
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projection (Improv C)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Attention: single linear layer producing score per timestep
        self.attention_fc = nn.Linear(hidden_size * 2, 1)

        # Decoder to original input size
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_proj = self.input_projection(x)  # -> (batch, seq_len, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)  # lstm_out: (batch, seq_len, hidden*2)

        # attention logits and weights
        logits = self.attention_fc(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        context = torch.sum(weights * lstm_out, dim=1)  # (batch, hidden*2)

        out = self.fc(context)  # (batch, input_size)
        return out

# ------------------------
# Prediction wrapper used by harness
# ------------------------
class PredictionModel:
    def __init__(self):
        # Config (tweak if your artifact names / fold count differ)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_folds = 5
        self.seq_len = 100          # must match training seq_len used when creating dataset
        self.hidden_size = 128     # must match training hidden_size
        self.num_layers = 2
        self.dropout = 0.0

        # Prefix used by training script for saved checkpoints:
        # training used f"{args.checkpoint_prefix}_fold_{fold_idx}.pt"
        # default checkpoint_prefix used earlier: 'lstm_attention_raw_checkpoint_fold'
        self.checkpoint_prefix = 'lstm_attention_raw_checkpoint_fold'
        # location where stacking meta may be saved by training pipeline
        self.meta_paths_to_try = [
            './stacking_artifacts/{}_meta_ridge.pkl'.format(self.checkpoint_prefix),
            './stacking_artifacts/stacking_meta_ridge.pkl',
            './stacking_artifacts/meta_ridge.pkl',
            './stacking_artifacts/meta_weights.npy'  # we'll check weights/bias pair
        ]

        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # loaded models
        self.models = []
        self.input_size = None

        # attempt to load models and meta
        self._load_ensemble_and_meta()

        # sliding window state
        self.current_seq_ix = None
        self.state_buffer = []

    def _infer_input_size_from_state_dict(self, state_dict):
        # input_projection.0.weight shape: (hidden_size, input_size)
        key = 'input_projection.0.weight'
        if key in state_dict:
            w = state_dict[key]
            if isinstance(w, np.ndarray):
                return w.shape[1]
            else:
                # tensor
                return list(w.shape)[1]
        # fallback: try fc.bias shape (input_size)
        key2 = 'fc.bias'
        if key2 in state_dict:
            b = state_dict[key2]
            if isinstance(b, np.ndarray):
                return b.shape[0]
            else:
                return list(b.shape)[0]
        raise RuntimeError("Couldn't infer input size from state dict.")

    def _load_ensemble_and_meta(self):
        # load base model checkpoints
        loaded_any = False
        for i in range(self.n_folds):
            ckpt_name = f"{self.checkpoint_prefix}_fold_{i}.pt"
            ckpt_path = os.path.join(self.script_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                # try alternative naming (sometimes saved as prefix_fold{idx}.pt)
                alt = os.path.join(self.script_dir, f"{self.checkpoint_prefix}_{i}.pt")
                if os.path.exists(alt):
                    ckpt_path = alt
                else:
                    # skip missing
                    print(f"[PredictionModel] checkpoint not found: {ckpt_name} (skipping)")
                    continue

            try:
                state = torch.load(ckpt_path, map_location='cpu')
            except Exception as e:
                print(f"[PredictionModel] failed to load {ckpt_path}: {e}")
                continue

            if self.input_size is None:
                try:
                    self.input_size = self._infer_input_size_from_state_dict(state)
                    print(f"[PredictionModel] inferred input_size = {self.input_size}")
                except Exception as e:
                    print("[PredictionModel] could not infer input size:", e)
                    # continue but must set input_size later from DataPoint

            # Build model and load weights
            model = LSTMAttentionModel(input_size=self.input_size or 1,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.num_layers,
                                       dropout=self.dropout)
            try:
                model.load_state_dict(state)
            except Exception as e:
                # try to load partial (if saved with prefix)
                try:
                    model.load_state_dict(state, strict=False)
                except Exception:
                    print(f"[PredictionModel] Warning: state_dict mismatch loading {ckpt_path}: {e}")
            model.to(self.device).eval()
            self.models.append(model)
            loaded_any = True

        if not loaded_any:
            raise RuntimeError("[PredictionModel] No base model checkpoints were loaded. Place checkpoints next to solution.py or adjust paths.")

        print(f"[PredictionModel] Loaded {len(self.models)} base models.")

        # Try to load meta learner (joblib or numpy)
        self.meta = None
        # meta can be either a joblib sklearn model, or numpy weights + bias for linear meta
        try:
            import joblib
        except Exception:
            joblib = None

        for p in self.meta_paths_to_try:
            p_full = os.path.join(self.script_dir, p) if not os.path.isabs(p) else p
            if os.path.exists(p_full):
                # if it's a .pkl and joblib available, load
                if p_full.endswith('.pkl') and joblib is not None:
                    try:
                        self.meta = joblib.load(p_full)
                        print(f"[PredictionModel] Loaded meta (joblib) from {p_full}")
                        break
                    except Exception as e:
                        print(f"[PredictionModel] Failed to load joblib meta from {p_full}: {e}")
                # if it's weights npy, load weights & bias pair
                if p_full.endswith('meta_weights.npy') or p_full.endswith('meta_weights.npy'):
                    try:
                        w = np.load(p_full)
                        b_path = os.path.join(os.path.dirname(p_full), 'meta_bias.npy')
                        if os.path.exists(b_path):
                            b = np.load(b_path)
                        else:
                            b = np.zeros(w.shape[1], dtype=np.float32)
                        self.meta = ('linear_numpy', w.astype(np.float32), b.astype(np.float32))
                        print(f"[PredictionModel] Loaded numpy meta weights from {p_full}")
                        break
                    except Exception as e:
                        print(f"[PredictionModel] Failed to load numpy meta weights from {p_full}: {e}")
                # also accept a generic meta_weights.npy path
                if p_full.endswith('meta_weights.npy') and os.path.exists(p_full):
                    try:
                        w = np.load(p_full)
                        b_path = p_full.replace('weights', 'bias')
                        if os.path.exists(b_path):
                            b = np.load(b_path)
                        else:
                            b = np.zeros(w.shape[1], dtype=np.float32)
                        self.meta = ('linear_numpy', w.astype(np.float32), b.astype(np.float32))
                        print(f"[PredictionModel] Loaded numpy meta weights & bias.")
                        break
                    except Exception as e:
                        print(f"[PredictionModel] Failed to load numpy meta weights from {p_full}: {e}")
        if self.meta is None:
            print("[PredictionModel] No meta learner found — will use simple averaging of base model predictions.")

    def predict(self, data_point: DataPoint):
        """
        data_point: DataPoint(seq_ix, step_in_seq, need_prediction, state)
        Returns: np.ndarray prediction (shape (input_size,)) or None if not need_prediction
        """
        # manage sequence buffer
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        state = np.asarray(data_point.state, dtype=np.float32)
        # infer input_size if still unknown
        if self.input_size is None:
            self.input_size = state.shape[0]

        self.state_buffer.append(state)
        # enforce max length
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # prepare window: pad at front with zeros to seq_len
        window = np.array(self.state_buffer, dtype=np.float32)
        if window.shape[0] < self.seq_len:
            pad_len = self.seq_len - window.shape[0]
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)

        preds_per_model = []
        with torch.no_grad():
            for model in self.models:
                try:
                    out = model(x)  # (1, input_size)
                    preds_per_model.append(out.cpu().numpy())
                except Exception as e:
                    # fallback: try casting to float
                    out = model(x.float())
                    preds_per_model.append(out.cpu().numpy())

        preds_per_model = np.concatenate(preds_per_model, axis=0)  # (n_models, input_size)

        # Build meta-feature vector: horizontally stack per-model outputs -> shape (1, input_size * n_models)
        X_meta = preds_per_model.reshape(1, -1).astype(np.float32)

        # If meta exists and is a sklearn-like estimator
        if self.meta is not None:
            if isinstance(self.meta, tuple) and self.meta[0] == 'linear_numpy':
                # ('linear_numpy', weights, bias) where weights shape = (input_dim, output_dim)
                _, W, b = self.meta
                # W expects shape (meta_dim, output_dim)
                try:
                    out = X_meta.dot(W) + b
                    return out.ravel().astype(np.float32)
                except Exception as e:
                    # fallback to averaging
                    avg = preds_per_model.mean(axis=0)
                    return avg.ravel().astype(np.float32)
            else:
                # assume sklearn-like .predict available
                try:
                    out = self.meta.predict(X_meta)  # shape (1, input_size)
                    return out.ravel().astype(np.float32)
                except Exception as e:
                    # fallback to averaging
                    avg = preds_per_model.mean(axis=0)
                    return avg.ravel().astype(np.float32)
        else:
            # fallback: simple mean across models
            avg = preds_per_model.mean(axis=0)
            return avg.ravel().astype(np.float32)


# When running locally for quick sanity check (if harness exists)
if __name__ == "__main__":
    print("Running local check of solution.py")
    # locate a local dataset if available (same pattern as example)
    # The harness may provide utils.ScorerStepByStep — try to use it when present
    try:
        from utils import ScorerStepByStep
        # attempt to find local dataset relative to repo
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'train.parquet')
        candidate = os.path.abspath(candidate)
        if not os.path.exists(candidate):
            print(f"Local dataset not found at {candidate}. Please pass dataset or run within competition harness.")
        else:
            print("Initializing PredictionModel and running harness scorer (this may take long).")
            model = PredictionModel()
            scorer = ScorerStepByStep(candidate)
            results = scorer.score(model)
            print("Results:", results)
    except Exception as e:
        print("Local harness not available or failed to run:", e)
        print("You can still import PredictionModel and call predict() from the platform harness.")
