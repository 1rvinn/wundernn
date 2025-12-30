"""
Solution file for the Stateful TransformerModel (Raw Data Version).

This version:
1. Loads the 'TransformerModel' and 'PositionalEncoding' classes.
2. Loads the 'transformer_stateful_checkpoint.pt' weights.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Manages a "growing buffer" of all past states for the current sequence.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
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

# --- Model Definitions ---
# These MUST be identical to your training script.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1000)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, input_size)
        
        self.register_buffer("causal_mask", None)

    def _generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        seq_len = x.size(1)
        
        # This logic is for training. In inference, the mask size changes
        # so we must generate it dynamically.
        causal_mask = self._generate_causal_mask(seq_len, x.device)
            
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, mask=causal_mask)
        
        # Take the output *only at the last time step* for prediction
        # This is different from training, but more efficient for inference.
        # However, to be 100% consistent with training, we'll
        # just let the fc layer project all outputs.
        predictions = self.fc(out) 
        return predictions
# --- END MODEL DEFINITIONS ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (Stateful Transformer, Raw Data)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dropout = 0.2
        
        self.checkpoint_path = os.path.join(script_dir, 'transformer_stateful_checkpoint.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.input_size = state_dict['input_proj.weight'].shape[1]
            print(f"Inferred input_size (N_features): {self.input_size}")

            self.model = TransformerModel(
                input_size=self.input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model '{self.checkpoint_path}': {e}")
            raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.state_buffer = [] # This will be a list of *raw* np.ndarrays

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Add Raw State to Buffer
        self.state_buffer.append(data_point.state.astype(np.float32))
        
        # 3. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 4. Prepare Input Sequence (NO padding needed)
        # This will grow from (1, 1, N) to (1, 999, N)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        # 5. Model Inference
        # Add batch dimension: (seq_len, features) -> (1, seq_len, features)
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts for all steps: (1, seq_len, features)
            prediction_seq = self.model(x)
            
            # We only care about the *last* prediction
            prediction_tensor = prediction_seq[:, -1, :] # Shape: (1, features)

        # 6. Un-scaling Logic (NOT NEEDED)
        # Convert to 1D numpy array and return
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        
        return prediction