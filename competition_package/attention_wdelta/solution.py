"""
Solution file for the Stateless LSTMAttentionModel.

This version implements:
1. StandardScaler (from 'lstm_attention_scaler.joblib').
2. Delta Prediction (model predicts X[t+1] - X[t]).
3. A state buffer to manually create the 100-step sliding window.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context_vector)
# --- END NEW MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (LSTM w/ Attention)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training) ---
        self.seq_len = 100
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_checkpoint.pt')
        self.scaler_path = os.path.join(script_dir, 'lstm_attention_scaler.joblib')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Scaler ---
        try:
            self.scaler = joblib.load(self.scaler_path)
            self.scaler_mean_np = self.scaler.mean_.astype(np.float32)
            self.scaler_scale_np = self.scaler.scale_.astype(np.float32)
            self.scaler_mean = torch.from_numpy(self.scaler_mean_np).float().to(self.device)
            self.scaler_scale = torch.from_numpy(self.scaler_scale_np).float().to(self.device)
            self.input_size = len(self.scaler.mean_)
            print(f"Scaler loaded. Input size: {self.input_size}")
        except Exception as e:
            print(f"CRITICAL: Failed to load scaler '{self.scaler_path}': {e}")
            raise e

        # --- Load Model ---
        try:
            self.model = LSTMAttentionModel(
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
            print(f"CRITICAL: Failed to load model '{self.checkpoint_path}': {e}")
            raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.state_buffer = [] # This will be a list of *scaled* np.ndarrays


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Scale Input and Update Buffer
        scaled_state = (data_point.state - self.scaler_mean_np) / self.scaler_scale_np
        self.state_buffer.append(scaled_state)
        
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # 4. Prepare Input Window (with padding)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 5. Model Inference
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled_delta = self.model(x) # Shape: (1, features)

        # 6. Un-scaling Logic
        scaled_state_tensor = torch.from_numpy(scaled_state).float().to(self.device)
        pred_scaled_abs = scaled_state_tensor + pred_scaled_delta.squeeze(0)
        unscaled_abs_pred = (pred_scaled_abs * self.scaler_scale) + self.scaler_mean
        prediction = unscaled_abs_pred.cpu().numpy()
        
        return prediction
    