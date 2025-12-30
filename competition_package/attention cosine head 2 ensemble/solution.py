"""
solution.py

Modified PredictionModel to perform inference using TWO models:
 - lstm_attention_model_b.pt  (Model B)
 - lstm_attention_model_a.pt  (Model A)

Behavior:
 - A fixed list of feature indices (`b_indices`) will be taken from Model B.
 - For those indices, the final prediction is a WEIGHTED BLEND between Model B and Model A.
   * A scalar `alpha_b` controls how much of Model B's prediction is used (0.0..1.0).
   * Optionally, a per-feature `b_weights` numpy array (shape = n_features) may be provided to
     control blending per-output-feature; where provided it overrides scalar alpha for
     that index.
 - For all other indices (complement of b_indices), the final prediction uses Model A's output.
 - `b_indices` is constant (set at initialization).
 - Everything preserves the original buffering / windowing / padding semantics.

Notes:
 - This file assumes both checkpoints produce outputs of identical length (same number of features).
 - If there's a mismatch in the inferred input/ output sizes between saved checkpoints, an exception
   will be raised to avoid silent incorrect behavior.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

# This import is required by the competition environment.
try:
    from utils import DataPoint
except ImportError:
    print("Running in local mode, defining dummy DataPoint.")
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

# --- Model Definition (must match training class exactly) ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input Projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        # Attention scoring
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # Decoder -> produces same size as input (predict next absolute state)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_proj = self.input_projection(x)                         # (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)                 # (batch, seq_len, hidden_size*2)
        attention_logits = self.attention_fc(lstm_out)           # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_logits, dim=1)   # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        return self.fc(context_vector)                           # (batch, input_size)
# --- END MODEL ---


# --- Prediction Class using TWO models + blending ---
class PredictionModel:
    def __init__(
        self,
        model_a_path: str = "lstm_attention_model_a.pt",
        model_b_path: str = "lstm_attention_model_b.pt",
        seq_len: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        # Provide the fixed feature indices (integers) that will be blended using Model B.
        b_indices: list | np.ndarray | None = None,
        # Scalar blending weight for Model B (0.0 -> only A, 1.0 -> only B).
        alpha_b: float = 1,
        # Optional per-feature blend weights for Model B. If provided, must be same length as output.
        # Values should be in [0,1]. Where provided, these override alpha_b for those features.
        b_weights: np.ndarray | None = None
    ):
        """
        Initialize, load two models (A and B), set blending indices and weights.

        - model_a_path: filename for Model A weights (used as default for non-b_indices)
        - model_b_path: filename for Model B weights (used for b_indices blending)
        - b_indices: constant list/array of feature indices to take from/blend with Model B.
        - alpha_b: scalar interpolation factor (B contribution). Final value for an index i is:
                   final[i] = (1 - w_i) * pred_a[i] + w_i * pred_b[i]
                   where w_i = b_weights[i] if b_weights is provided else alpha_b (for i in b_indices)
        - b_weights: optional per-feature weights (np.ndarray of shape [n_features]); only used where not None.
        """
        print("Initializing PredictionModel (LSTM+Proj) with dual-model blending...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Paths
        self.model_a_path = os.path.join(script_dir, model_a_path)
        self.model_b_path = os.path.join(script_dir, model_b_path)

        # Hyperparams
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load both checkpoints -> infer input/output sizes and construct models
        state_dict_a = self._load_checkpoint(self.model_a_path)
        state_dict_b = self._load_checkpoint(self.model_b_path)

        # Helper to extract input/output sizes:
        input_size_a = self._infer_input_size_from_state_dict(state_dict_a)
        input_size_b = self._infer_input_size_from_state_dict(state_dict_b)
        output_size_a = self._infer_output_size_from_state_dict(state_dict_a)
        output_size_b = self._infer_output_size_from_state_dict(state_dict_b)

        # Basic sanity checks: shapes must match
        if input_size_a != input_size_b:
            raise ValueError(f"Input size mismatch between A ({input_size_a}) and B ({input_size_b}).")
        if output_size_a != output_size_b:
            raise ValueError(f"Output size mismatch between A ({output_size_a}) and B ({output_size_b}).")

        self.input_size = int(input_size_a)
        self.output_size = int(output_size_a)

        print(f"Inferred input/output size (N_features): {self.input_size}")

        # Build model instances
        self.model_a = LSTMAttentionModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.model_b = LSTMAttentionModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # Load state_dicts into models (support state wrapping keys)
        self._load_state_dict_into_model(self.model_a, state_dict_a, "A")
        self._load_state_dict_into_model(self.model_b, state_dict_b, "B")

        # Move to device and eval mode
        self.model_a.to(self.device).eval()
        self.model_b.to(self.device).eval()

        # Buffering state
        self.current_seq_ix = -1
        self.state_buffer = []

        # Blending indices (constant)
        if b_indices is None:
            # Default: no indices from B (i.e., everything from A)
            self.b_indices = np.array([], dtype=np.int64)
        else:
            self.b_indices = np.array(b_indices, dtype=np.int64)

        # Validate indices
        if np.any(self.b_indices < 0) or np.any(self.b_indices >= self.output_size):
            raise ValueError(f"b_indices contains out-of-range indices for output size {self.output_size}.")

        # Precompute A indices (complement)
        all_idx = np.arange(self.output_size, dtype=np.int64)
        self.a_indices = np.setdiff1d(all_idx, self.b_indices, assume_unique=False)

        # Blending weights
        if b_weights is not None:
            b_weights = np.asarray(b_weights, dtype=np.float32)
            if b_weights.shape[0] != self.output_size:
                raise ValueError("b_weights must have length equal to number of output features.")
            # Clip to [0,1]
            b_weights = np.clip(b_weights, 0.0, 1.0)
            self.b_weights = b_weights
            print("Using per-feature B weights (b_weights).")
        else:
            # Scalar alpha_b applied to all b_indices
            if not (0.0 <= alpha_b <= 1.0):
                raise ValueError("alpha_b must be in [0.0, 1.0].")
            self.alpha_b = float(alpha_b)
            self.b_weights = None
            print(f"Using scalar alpha_b = {self.alpha_b} for blending on b_indices.")

        print(f"Model A path: {self.model_a_path}")
        print(f"Model B path: {self.model_b_path}")
        print(f"Number of features: {self.output_size}, b_indices count: {len(self.b_indices)}")


    # ----------------- Internal helper functions -----------------
    def _load_checkpoint(self, path: str) -> dict:
        """Load a torch checkpoint and return a state_dict (supporting several common layouts)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{path}': {e}")

        # If it's a dict that already is a state_dict (contains parameter tensors), return as-is.
        # If it's wrapped like {'state_dict': {...}} or {'model_state': {...}} return the inner dict.
        if isinstance(ckpt, dict):
            # Heuristic keys
            for key in ("state_dict", "model_state", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            # Otherwise assume ckpt *is* the state_dict
            # (might also contain extra metadata keys, but param tensors are present)
            return ckpt
        else:
            raise RuntimeError(f"Unexpected checkpoint type for '{path}': {type(ckpt)}")

    def _infer_input_size_from_state_dict(self, state_dict: dict) -> int:
        """
        Infer the model input_size by checking the shape of the input_projection weight
        (expected key: 'input_projection.0.weight') or fallback to fc.bias size.
        """
        if 'input_projection.0.weight' in state_dict:
            w = state_dict['input_projection.0.weight']
            return int(w.shape[1])  # [hidden_size, input_size]
        # fallback: try fc.bias (output size) -> assume input_size == output_size if projection missing
        if 'fc.bias' in state_dict:
            return int(state_dict['fc.bias'].shape[0])
        raise RuntimeError("Unable to infer input size from state_dict (missing expected keys).")

    def _infer_output_size_from_state_dict(self, state_dict: dict) -> int:
        """Infer model output size from fc.bias shape (preferred)."""
        if 'fc.bias' in state_dict:
            return int(state_dict['fc.bias'].shape[0])
        # fallback: try last Linear weight 'fc.weight'
        if 'fc.weight' in state_dict:
            return int(state_dict['fc.weight'].shape[0])
        raise RuntimeError("Unable to infer output size from state_dict (missing 'fc.bias'/'fc.weight').")

    def _load_state_dict_into_model(self, model: nn.Module, state_dict: dict, label: str):
        """
        Load state_dict into model; tries strict load, else attempts to adapt common prefixes.
        """
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded state_dict into model {label} (strict).")
        except RuntimeError as e:
            # Try non-strict load (allow missing / unexpected keys) and inform user
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded state_dict into model {label} (non-strict). Some keys may be missing/extra.")
            except Exception as e2:
                raise RuntimeError(f"Failed to load state_dict into model {label}: {e} | {e2}")


    # ----------------- Public predict API -----------------
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next absolute state vector.

        - Maintains a per-sequence buffer like the original code.
        - Uses both models to produce predictions and then merges them according to
          b_indices and blending weights.
        """
        # Reset buffer for a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # Append current raw state
        self.state_buffer.append(data_point.state.astype(np.float32))

        # Keep buffer length <= seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # If not time to predict, return None (same behavior)
        if not data_point.need_prediction:
            return None

        # Build window array and pad if necessary
        window = np.array(self.state_buffer, dtype=np.float32)
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])  # shape = (seq_len, input_size)

        # Convert to tensor and move to device
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # shape (1, seq_len, input_size)

        # Run inference on both models
        with torch.no_grad():
            out_a = self.model_a(x)  # (1, input_size)
            out_b = self.model_b(x)  # (1, input_size)

        # Move to CPU numpy
        pred_a = out_a.cpu().numpy().squeeze(0).astype(np.float32)  # shape (input_size,)
        pred_b = out_b.cpu().numpy().squeeze(0).astype(np.float32)

        # Sanity shapes
        if pred_a.shape[0] != self.output_size or pred_b.shape[0] != self.output_size:
            raise RuntimeError("Model outputs do not match expected output size after forward pass.")

        # Compose final prediction
        final_pred = np.empty_like(pred_a)

        # For indices NOT in b_indices -> use Model A
        if self.a_indices.size > 0:
            final_pred[self.a_indices] = pred_a[self.a_indices]

        # For b_indices -> blend
        if self.b_indices.size > 0:
            if self.b_weights is not None:
                # Use per-feature weights vector (0..1)
                w = self.b_weights[self.b_indices]  # array of weights for those indices
            else:
                # Use scalar alpha_b for all b_indices
                w = np.full(self.b_indices.shape, fill_value=self.alpha_b, dtype=np.float32)

            # Weighted combination: final = (1 - w)*A + w*B
            a_vals = pred_a[self.b_indices]
            b_vals = pred_b[self.b_indices]
            final_pred[self.b_indices] = (1.0 - w) * a_vals + w * b_vals

        return final_pred


# If this file is run directly, demonstrate initialization (useful for local debug).
if __name__ == "__main__":
    # Example initialization (adjust paths / b_indices as needed)
    try:
        pm = PredictionModel(
            model_a_path="lstm_attention_model_a.pt",
            model_b_path="lstm_attention_model_b.pt",
            b_indices=[0,1,4,5,12,17,19,21],   # example indices to blend from B
            alpha_b=1            # example scalar weight (60% from B on b_indices)
        )
        print("PredictionModel initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize PredictionModel: {e}")
