"""
Solution file for stateful LSTM inference (with Scaling and Delta Prediction).

This file defines the `PredictionModel` class. It loads:
1. The pre-trained stateful LSTM model (`lstm_stateful_checkpoint.pt`)
2. The fitted StandardScaler (`scaler.joblib`)

It correctly scales inputs, interprets the model's output as a "delta",
and un-scales the final prediction.
"""

import numpy as np
import torch
from torch import nn
import joblib  # Added for loading the scaler

# This import is required by the competition environment.
# We'll include a placeholder definition for clarity.
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
# This class MUST be identical to the one used in training.
class LSTMModel(nn.Module):
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
        out, _ = self.lstm(x) 
        predictions = self.fc(out)
        return predictions


# --- Prediction Class ---

class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel...")
        
        # --- Hyperparameters (must match training) ---
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.0  # From your args
        self.checkpoint_path = 'lstm_stateful_checkpoint.pt'
        self.scaler_path = 'scaler.joblib'

        # --- Device Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            # Load the checkpoint
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Dynamically determine input_size from the checkpoint
            self.input_size = state_dict['fc.weight'].shape[0]
            print(f"Inferred input_size (N_features) from checkpoint: {self.input_size}")

            # Instantiate the model
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.input_size = 10 # Placeholder
            
        # --- NEW: Load Scaler ---
        try:
            self.scaler = joblib.load(self.scaler_path)
            # Store mean and scale as tensors on the device for fast math
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
            self.hidden_state = None  # LSTM will auto-initialize

        # --- NEW: Scaling and Tensor Conversion ---
        # Convert raw numpy state to a torch tensor
        raw_state_tensor = torch.from_numpy(data_point.state).float().to(self.device)
        
        # Scale the state: (X_t - mean) / scale
        scaled_state_tensor = (raw_state_tensor - self.scaler_mean) / self.scaler_scale
        
        # Reshape for LSTM: (1, 1, N_features)
        x = scaled_state_tensor.view(1, 1, self.input_size)

        # 3. Model Inference (Stateful)
        with torch.no_grad():
            lstm_out, self.hidden_state = self.model.lstm(x, self.hidden_state)
            
            # Model now predicts a *scaled delta*
            scaled_delta_pred = self.model.fc(lstm_out.squeeze(1))

        # 4. Return Prediction (or None)
        if not data_point.need_prediction:
            return None

        # --- NEW: Un-scaling Logic ---
        # 1. Calculate the *scaled* absolute prediction
        #    X_t+1_scaled = X_t_scaled + Delta_t_scaled
        scaled_abs_pred = scaled_state_tensor + scaled_delta_pred.squeeze(0)
        
        # 2. Un-scale the absolute prediction
        #    X_t+1_unscaled = (X_t+1_scaled * scale) + mean
        unscaled_abs_pred = (scaled_abs_pred * self.scaler_scale) + self.scaler_mean
        
        # 3. Convert back to numpy
        prediction = unscaled_abs_pred.cpu().numpy()
        
        return prediction