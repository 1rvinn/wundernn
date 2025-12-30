"""
Solution file for the stateless Linear Regression Model.

- Loads the trained PyTorch LinearRegressionModel.
- Loads the StandardScaler (`linreg_scaler.joblib`).
- Manages a 100-step "state buffer" to build the input window.
- Predicts the absolute next state.
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
# This MUST be the model class from your training script.
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(seq_len * input_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_flat = self.flatten(x)
        return self.linear(x_flat)


# --- Prediction Class ---

class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (Linear Regression)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training) ---
        self.seq_len = 100
        
        self.checkpoint_path = os.path.join(script_dir, 'linreg_checkpoint.pt')
        self.scaler_path = os.path.join(script_dir, 'linreg_scaler.joblib')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Scaler ---
        try:
            self.scaler = joblib.load(self.scaler_path)
            self.scaler_mean_np = self.scaler.mean_.astype(np.float32)
            self.scaler_scale_np = self.scaler.scale_.astype(np.float32)
            self.input_size = len(self.scaler.mean_)
            print(f"Scaler loaded. Input size: {self.input_size}")
        except Exception as e:
            print(f"CRITICAL: Failed to load scaler: {e}"); raise e

        # --- Load Model ---
        try:
            self.model = LinearRegressionModel(
                input_size=self.input_size,
                seq_len=self.seq_len
            )
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}"); raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.state_buffer = [] # This will be a list of *scaled* np.ndarrays


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Scale Input and Update Buffer
        scaled_state = (data_point.state - self.scaler_mean_np) / self.scaler_scale_np
        self.state_buffer.append(scaled_state)
        
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0) # Remove the oldest state

        # 3. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 4. Prepare Input Window (with padding)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 5. Model Inference
        # Convert window to (1, seq_len, features) tensor
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the *scaled absolute state*
            pred_scaled = self.model(x) # Shape: (1, features)

        # 6. Un-scaling Logic
        # Un-scale to get the final absolute prediction
        pred_unscaled = self.scaler.inverse_transform(pred_scaled.cpu().numpy())
        
        # Convert back to a 1D numpy array
        prediction = pred_unscaled.squeeze(0)
        
        return prediction