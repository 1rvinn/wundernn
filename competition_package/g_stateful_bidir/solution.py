"""
Solution file for a STATEFUL, BIDIRECTIONAL LSTM.
"""

import numpy as np
import torch
from torch import nn
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

# --- Model Definition ---
# This MUST be identical to your training script.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # <-- Must match
        )
        # --- THE FIX ---
        # Must be hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, input_size) 
        # --- END FIX ---

    def forward(self, x, h=None):
        # We need to accept the hidden state for step-by-step inference
        out, h_out = self.lstm(x, h) 
        predictions = self.fc(out)
        return predictions, h_out
# --- END MODEL DEFINITION ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        print("Initializing PredictionModel (Stateful, Bidirectional)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.4 # From your Args
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_stateful_checkpoint.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            # Infer input_size
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            print(f"Inferred input_size (N_features): {self.input_size}")

            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}"); raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.hidden_state = None # Will store (h_n, c_n)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            # Reset state for the new sequence
            self.hidden_state = None 

        # 1. Prepare Input
        # (N,) -> (1, 1, N) for (batch, seq_len, features)
        x = torch.from_numpy(data_point.state.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. Model Inference
        with torch.no_grad():
            # Pass the *current* input and the *previous* hidden state
            prediction_tensor, self.hidden_state = self.model(x, self.hidden_state)
        
        if not data_point.need_prediction:
            return None

        # 3. Return
        # prediction_tensor is (1, 1, N), squeeze to (N,)
        prediction = prediction_tensor.cpu().numpy().squeeze()
        
        return prediction