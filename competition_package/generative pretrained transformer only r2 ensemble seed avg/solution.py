"""
solution.py
Single-model inference using averaged_checkpoint.pt
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

# Provided by competition:
try:
    from utils import DataPoint
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray


# ========== MODEL DEFINITION ==========
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 separate_heads=False, easy_idx=None, hard_idx=None):
        super().__init__()
        self.input_size = input_size
        self.separate_heads = separate_heads
        self.easy_idx = easy_idx
        self.hard_idx = hard_idx

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True
        )

        self.attention_fc = nn.Linear(hidden_size * 2, 1)

        if not separate_heads:
            self.fc = nn.Linear(hidden_size * 2, input_size)
        else:
            self.fc_easy = nn.Linear(hidden_size * 2, len(easy_idx)) if easy_idx else None
            self.fc_hard = nn.Linear(hidden_size * 2, len(hard_idx)) if hard_idx else None

    def forward(self, x):
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)

        att_logits = self.attention_fc(lstm_out)
        att_w = F.softmax(att_logits, dim=1)

        ctx = torch.sum(att_w * lstm_out, dim=1)

        if not self.separate_heads:
            return self.fc(ctx)

        out = torch.zeros((ctx.size(0), self.input_size), device=ctx.device)
        if self.fc_easy is not None:
            out[:, self.easy_idx] = self.fc_easy(ctx)
        if self.fc_hard is not None:
            out[:, self.hard_idx] = self.fc_hard(ctx)
        return out


# ========== PREDICTION MODEL ==========
class PredictionModel:
    def __init__(self):
        print("Loading averaged model...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(script_dir, "averaged_checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        state = ckpt["model_state_dict"]
        feature_weights = ckpt["feature_weights"]
        args = ckpt["args"]

        # infer input_size
        in_size = state["input_projection.0.weight"].shape[1]

        separate_heads = args.get("separate_heads", False)
        if separate_heads:
            fw = np.array(feature_weights, np.float32)
            med = np.median(fw)
            hard_idx = [i for i,w in enumerate(fw) if w > med]
            easy_idx = [i for i,w in enumerate(fw) if w <= med]
        else:
            easy_idx = hard_idx = None

        hidden = args.get("hidden_size", 128)
        layers = args.get("num_layers", 2)
        drop = args.get("dropout", 0.1)

        self.seq_len = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMAttentionModel(
            input_size=in_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=drop,
            separate_heads=separate_heads,
            easy_idx=easy_idx,
            hard_idx=hard_idx
        ).to(self.device)

        # convert to FP32 tensors if needed
        state = {k: v.float() for k, v in state.items()}

        self.model.load_state_dict(state)
        self.model.eval()

        self.current_seq_ix = -1
        self.state_buffer = []

        print("Averaged model loaded.")

    def predict(self, dp: DataPoint):
        if dp.seq_ix != self.current_seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.state_buffer = []

        self.state_buffer.append(dp.state.astype(np.float32))
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not dp.need_prediction:
            return None

        # build window
        window = np.array(self.state_buffer, dtype=np.float32)
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            window = np.vstack([np.zeros((pad_len, window.shape[1]), np.float32), window])

        x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred = self.model(x).cpu().numpy().squeeze(0)

        return pred
