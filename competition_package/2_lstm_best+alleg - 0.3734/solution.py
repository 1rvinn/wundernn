"""
Solution file for the Stateless "Global" CNN Model.

This version:
1. Loads the 'CNNModel' (with large kernel).
2. Loads the 'cnn_global_kernel_checkpoint.pt' weights.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Manages a 150-step "state buffer" to build the input window.
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

# --- Model Definition ---
# This MUST be the model class from your training script.
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, kernel_size=100, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_size, 
            kernel_size=kernel_size, 
            padding=0
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.squeeze(2) 
        x = self.dropout(x)
        return self.fc(x)

# --- Prediction Class ---

class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (Global CNN)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 150
        self.kernel_size = 100
        self.hidden_size = 128
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'cnn_global_kernel_checkpoint.pt')
        # No scaler path is needed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.input_size = state_dict['conv1.weight'].shape[1]
            print(f"Inferred input_size (N_features): {self.input_size}")

            self.model = CNNModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                kernel_size=self.kernel_size,
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
        
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        self.state_buffer.append(data_point.state.astype(np.float32))
        
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # Prepare Input Window
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # Model Inference
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction_tensor = self.model(x)

        # No un-scaling needed
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        
        return prediction