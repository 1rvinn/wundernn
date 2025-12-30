"""
Solution file for the Weighted-Loss LSTM with Attention + Input Projection.

This version:
1. Loads the 'LSTMAttentionModel' EXACTLY matching the training architecture,
   including support for separate heads (but inference does not use loss weighting).
2. Loads the checkpoint created by the NEW training script:
       torch.save({
           'model_state_dict': ...,
           'optimizer_state_dict': ...,
           'feature_weights': ...,
           'args': ...
       })
3. Predicts the absolute next state.
4. Maintains a rolling 100-step buffer as input window.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

# Competition requirement
try:
    from utils import DataPoint
except ImportError:
    print("Running in local mode; defining dummy DataPoint.")
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray


# --------------------------------------------------------------------------
# MODEL - MUST MATCH TRAINING SCRIPT EXACTLY
# --------------------------------------------------------------------------

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1,
                 separate_heads=False, easy_idx=None, hard_idx=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.separate_heads = separate_heads

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # LSTM Encoder (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention
        self.attention_fc = nn.Linear(hidden_size * 2, 1)

        # Normal single-head mode
        if not separate_heads:
            self.fc = nn.Linear(hidden_size * 2, input_size)
        else:
            # Two-head architecture (hard and easy features)
            self.easy_idx = easy_idx
            self.hard_idx = hard_idx
            self.fc_easy = nn.Linear(hidden_size * 2, len(easy_idx)) if len(easy_idx) > 0 else None
            self.fc_hard = nn.Linear(hidden_size * 2, len(hard_idx)) if len(hard_idx) > 0 else None

    def forward(self, x):
        # Projection
        x_proj = self.input_projection(x)

        # LSTM
        lstm_out, _ = self.lstm(x_proj)

        # Attention weights
        att_logits = self.attention_fc(lstm_out)
        att_weights = F.softmax(att_logits, dim=1)

        # Context
        context = torch.sum(att_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Output
        if not self.separate_heads:
            return self.fc(context)
        else:
            batch = context.size(0)
            device = context.device
            out = torch.zeros((batch, self.input_size), device=device)

            if self.fc_easy is not None:
                out[:, self.easy_idx] = self.fc_easy(context)
            if self.fc_hard is not None:
                out[:, self.hard_idx] = self.fc_hard(context)

            return out


# --------------------------------------------------------------------------
# PREDICTION MODEL WRAPPER
# --------------------------------------------------------------------------

class PredictionModel:
    def __init__(self):
        print("Initializing PredictionModel (Weighted-Loss LSTM+Attention)...")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Match training hyperparameters
        self.seq_len = 100
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1

        # Checkpoint name from training
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_proj_checkpoint_weighted.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ------------------------------
        # LOAD CHECKPOINT PROPERLY
        # ------------------------------
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

            # Load args saved during training
            train_args = ckpt.get("args", {})

            # Extract "separate_heads"
            separate_heads = train_args.get("separate_heads", False)

            # Load feature weights to reconstruct easy/hard split
            feature_weights = np.array(ckpt.get("feature_weights", []), dtype=np.float32)
            if len(feature_weights) > 0 and separate_heads:
                median = np.median(feature_weights)
                hard_idx = [i for i, w in enumerate(feature_weights) if w > median]
                easy_idx = [i for i, w in enumerate(feature_weights) if w <= median]
            else:
                easy_idx, hard_idx = None, None

            # Infer input_size from projection layer
            state_dict = ckpt["model_state_dict"]
            self.input_size = state_dict["input_projection.0.weight"].shape[1]

            # Build model
            self.model = LSTMAttentionModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                separate_heads=separate_heads,
                easy_idx=easy_idx,
                hard_idx=hard_idx
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully.")

        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model: {e}")
            raise e

        # Internal seq buffer
        self.current_seq_ix = -1
        self.state_buffer = []

    # ----------------------------------------------------------------------
    # PREDICTION LOGIC
    # ----------------------------------------------------------------------
    def predict(self, data_point: DataPoint) -> np.ndarray | None:

        # Reset buffer on new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # Append raw state
        self.state_buffer.append(data_point.state.astype(np.float32))

        # Limit buffer to seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # Only predict when needed
        if not data_point.need_prediction:
            return None

        # Build window (pad with zeros if too short)
        window = np.array(self.state_buffer, dtype=np.float32)
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            window = np.vstack([
                np.zeros((pad_len, self.input_size), dtype=np.float32),
                window
            ])

        # Run model
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x).cpu().numpy().squeeze(0)

        return pred
