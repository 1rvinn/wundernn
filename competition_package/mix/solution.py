"""
Solution file for stateful RNN inference (with Scaling and Delta Prediction).

This file defines the `PredictionModel` class. It loads:
1. The pre-trained stateful model (e.g., `gru_checkpoint.pt`)
2. The fitted StandardScaler (`scaler.joblib`)

It correctly scales inputs, interprets the model's output as a "delta",
and un-scales the final prediction.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F # Need this for BetterLSTMModel
import joblib
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

# --- MODEL DEFINITIONS ---
# You MUST include the definition for the model you want to use.
# ---
class LSTMModel(nn.Module):
    """Baseline LSTM Model"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Note: This is the training-time forward.
        # In inference, we call the layers separately.
        out, _ = self.lstm(x) 
        predictions = self.fc(out)
        return predictions

class GRUModel(nn.Module):
    """GRU Model (Alternative to LSTM)"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Note: This is the training-time forward.
        # In inference, we call the layers separately.
        out, _ = self.gru(x) 
        predictions = self.fc(out)
        return predictions

class BetterLSTMModel(nn.Module):
    """LSTM Model with Residual Connections and a more complex head"""
    def __init__(self, input_size, hidden_size=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size)
        )
        if input_size != hidden_size:
            self.residual_proj = nn.Linear(input_size, hidden_size)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # Note: This is the training-time forward.
        # In inference, we call the layers separately.
        residual = self.residual_proj(x)
        lstm_out, _ = self.lstm(x)
        out = F.relu(lstm_out + residual)
        predictions = self.head(out)
        return predictions
# --- END MODEL DEFINITIONS ---


class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel...")
        
        # --- Get the directory where solution.py is located ---
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Hyperparameters (must match training) ---
        self.hidden_size = 512
        self.num_layers = 4
        self.dropout = 0.1 # Make sure this matches your Args class
        
        # --- 1. CHANGE THIS to the checkpoint file you want to use ---
        self.checkpoint_path = os.path.join(script_dir, 'better_lstm_checkpoint.pt') 
        # e.g., 'better_lstm_checkpoint.pt' or 'lstm_stateful_checkpoint.pt'
        
        self.scaler_path = os.path.join(script_dir, 'scaler.joblib')

        # --- Device Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # **Dynamically determine input_size from the checkpoint**
            # This works because all models end with 'fc.weight' or 'head.X.weight'
            # We find the first linear layer's output weights to get input_size
            if 'fc.weight' in state_dict:
                self.input_size = state_dict['fc.weight'].shape[0] # For GRU/LSTM
            else:
                self.input_size = state_dict['head.3.weight'].shape[0] # For BetterLSTM
                
            print(f"Inferred input_size (N_features) from checkpoint: {self.input_size}")

            # --- 2. CHANGE THIS to the model class you want to use ---
            self.model = BetterLSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            # e.g., BetterLSTMModel(...) or LSTMModel(...)
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # --- Load Scaler ---
        try:
            self.scaler = joblib.load(self.scaler_path)
            self.scaler_mean = torch.from_numpy(self.scaler.mean_).float().to(self.device)
            self.scaler_scale = torch.from_numpy(self.scaler.scale_).float().to(self.device)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.hidden_state = None


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        # 1. Manage State: Reset if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_state = None

        # 2. Scaling and Tensor Conversion
        raw_state_tensor = torch.from_numpy(data_point.state).float().to(self.device)
        scaled_state_tensor = (raw_state_tensor - self.scaler_mean) / self.scaler_scale
        x = scaled_state_tensor.view(1, 1, self.input_size)

        # 3. Model Inference (Stateful)
        # This part is tricky. We must call the layers *manually*
        # to correctly pass the hidden state.
        with torch.no_grad():
            if isinstance(self.model, GRUModel):
                # GRU: rnn_out, self.hidden_state = self.model.gru(x, self.hidden_state)
                rnn_out, self.hidden_state = self.model.gru(x, self.hidden_state)
                scaled_delta_pred = self.model.fc(rnn_out.squeeze(1))
                
            elif isinstance(self.model, BetterLSTMModel):
                # BetterLSTM:
                residual = self.model.residual_proj(x)
                lstm_out, self.hidden_state = self.model.lstm(x, self.hidden_state)
                out = F.relu(lstm_out + residual)
                scaled_delta_pred = self.model.head(out.squeeze(1))
                
            else: # Default to baseline LSTMModel
                rnn_out, self.hidden_state = self.model.lstm(x, self.hidden_state)
                scaled_delta_pred = self.model.fc(rnn_out.squeeze(1))


        # 4. Return Prediction (or None)
        if not data_point.need_prediction:
            return None

        # 5. Un-scaling Logic
        scaled_abs_pred = scaled_state_tensor + scaled_delta_pred.squeeze(0)
        unscaled_abs_pred = (scaled_abs_pred * self.scaler_scale) + self.scaler_mean
        prediction = unscaled_abs_pred.cpu().numpy()
        
        return prediction