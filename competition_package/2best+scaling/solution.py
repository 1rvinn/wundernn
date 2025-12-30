"""
Solution file for the Stateless Sliding-Window LSTM.
This version implements:
1. StandardScaler for inputs/outputs.
2. Delta Prediction (model predicts X[t+1] - X[t]).
3. A state buffer to manually create the 100-step sliding window.
"""

import numpy as np
import torch
from torch import nn
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

# --- Model Definition ---
# This MUST be the model class from your *first* script
# (the one that takes the last timestep).
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
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last)


# --- Prediction Class ---

class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (Sliding Window w/ Scaling)...")
        
        # --- Get the directory where solution.py is located ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training) ---
        self.seq_len = 100  # From your first script's args
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_window_delta_checkpoint.pt')
        self.scaler_path = os.path.join(script_dir, 'scaler.joblib')

        # --- Device Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Scaler ---
        try:
            self.scaler = joblib.load(self.scaler_path)
            # Store mean/scale as numpy and as device tensors for speed
            self.scaler_mean_np = self.scaler.mean_.astype(np.float32)
            self.scaler_scale_np = self.scaler.scale_.astype(np.float32)
            self.scaler_mean = torch.from_numpy(self.scaler_mean_np).float().to(self.device)
            self.scaler_scale = torch.from_numpy(self.scaler_scale_np).float().to(self.device)
            self.input_size = len(self.scaler.mean_)
            print(f"Scaler loaded. Input size: {self.input_size}")
        except Exception as e:
            print(f"CRITICAL: Failed to load scaler: {e}"); raise e

        # --- Load Model ---
        try:
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}"); raise e
            
        # --- Internal State Management ---
        # This model is "stateless" but needs a buffer for the window
        self.current_seq_ix = -1 
        self.state_buffer = [] # This will be a list of scaled np.ndarrays


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Scale Input and Update Buffer
        # (X_t - mean) / scale
        scaled_state = (data_point.state - self.scaler_mean_np) / self.scaler_scale_np
        self.state_buffer.append(scaled_state)
        
        # Keep the buffer at the correct seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0) # Remove the oldest state

        # 3. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 4. Prepare Input Window (with padding)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            # Use zeros for padding, as in the training script
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 5. Model Inference
        # Convert window to (1, seq_len, features) tensor
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the *scaled delta*
            pred_scaled_delta = self.model(x) # Shape: (1, features)

        # 6. Un-scaling Logic
        # Convert scaled_state (current) to a tensor
        scaled_state_tensor = torch.from_numpy(scaled_state).float().to(self.device)
        
        # X_t+1_scaled = X_t_scaled + Delta_t_scaled
        pred_scaled_abs = scaled_state_tensor + pred_scaled_delta.squeeze(0)
        
        # X_t+1_unscaled = (X_t+1_scaled * scale) + mean
        unscaled_abs_pred = (pred_scaled_abs * self.scaler_scale) + self.scaler_mean
        
        # Convert back to numpy
        prediction = unscaled_abs_pred.cpu().numpy()
        
        return prediction