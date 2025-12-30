"""
Solution file for stateful inference.
Supports: LSTMModel, GRUModel, BetterLSTMModel, and CnnGruModel.

It loads:
1. The pre-trained stateful model (e.g., `cnngru_checkpoint.pt`)
2. The fitted StandardScaler (`scaler.joblib`)
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import joblib
import os

try:
    from utils import DataPoint
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int; step_in_seq: int
        need_prediction: bool; state: np.ndarray

# --- MODEL DEFINITIONS ---
# All 4 model classes must be defined here so they can be loaded.

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x, h=None): # Modified to accept hidden state
        out, h = self.lstm(x, h); return self.fc(out), h

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x, h=None): # Modified to accept hidden state
        out, h = self.gru(x, h); return self.fc(out), h

class BetterLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size; self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, input_size)
        )
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    def forward(self, x, h=None): # Modified to accept hidden state
        residual = self.residual_proj(x)
        lstm_out, h = self.lstm(x, h)
        out = F.relu(lstm_out + residual)
        return self.head(out), h

class CnnGruModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.1, cnn_kernel_size=10):
        super().__init__()
        self.cnn_kernel_size = cnn_kernel_size
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=cnn_kernel_size, padding=0)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
    
    # Inference is different: we process one step at a time.
    # The CNN needs a "buffer" of past inputs.
    def forward(self, x_buffer, h_gru=None):
        # x_buffer: (batch, features, kernel_size)
        x_cnn_out = self.conv1(x_buffer) # (batch, hidden, 1)
        x_cnn_out = self.relu(x_cnn_out)
        
        # (batch, hidden, 1) -> (batch, 1, hidden) for GRU
        x_gru_in = x_cnn_out.permute(0, 2, 1) 
        
        out, h_gru = self.gru(x_gru_in, h_gru) 
        return self.fc(out), h_gru

# --- END MODEL DEFINITIONS ---


class PredictionModel:
    def __init__(self):
        print("Initializing PredictionModel...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Model Hyperparameters (Must match training) ---
        self.hidden_size = 128
        self.num_layers = 1 # For CnnGruModel
        self.dropout = 0.1
        self.cnn_kernel_size = 10
        self.model_type = 'cnngru' # <-- Set this to your chosen model
        
        self.checkpoint_path = os.path.join(script_dir, f'{self.model_type}_k{self.cnn_kernel_size}_checkpoint.pt')
        self.scaler_path = os.path.join(script_dir, 'scaler.joblib')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load Scaler
        try:
            self.scaler = joblib.load(self.scaler_path)
            self.scaler_mean = torch.from_numpy(self.scaler.mean_).float().to(self.device)
            self.scaler_scale = torch.from_numpy(self.scaler.scale_).float().to(self.device)
            self.input_size = len(self.scaler.mean_)
            print(f"Scaler loaded. Input size: {self.input_size}")
        except Exception as e:
            print(f"CRITICAL: Failed to load scaler: {e}"); raise e

        # Load Model
        try:
            if self.model_type == 'gru':
                self.model = GRUModel(self.input_size, self.hidden_size, 2, self.dropout) # Note: num_layers=2
            elif self.model_type == 'better_lstm':
                self.model = BetterLSTMModel(self.input_size, self.hidden_size, 2, self.dropout) # Note: num_layers=2
            elif self.model_type == 'cnngru':
                self.model = CnnGruModel(self.input_size, self.hidden_size, self.num_layers, self.dropout, self.cnn_kernel_size)
            else:
                self.model = LSTMModel(self.input_size, self.hidden_size, 2, self.dropout) # Note: num_layers=2

            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}"); raise e
            
        # Internal State Management
        self.current_seq_ix = -1 
        self.hidden_state = None
        
        # NEW: State buffer for CNN
        # This holds the last 'cnn_kernel_size' inputs
        self.cnn_buffer = torch.zeros((1, self.cnn_kernel_size, self.input_size), dtype=torch.float32).to(self.device)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_state = None
            # Reset CNN buffer
            self.cnn_buffer.zero_()

        # --- 1. Scaling and State Update ---
        raw_state_tensor = torch.from_numpy(data_point.state).float().to(self.device)
        scaled_state_tensor = (raw_state_tensor - self.scaler_mean) / self.scaler_scale
        
        # --- 2. Model-Specific Inference ---
        with torch.no_grad():
            if isinstance(self.model, CnnGruModel):
                # Update CNN buffer (roll and add new state)
                self.cnn_buffer = torch.roll(self.cnn_buffer, -1, dims=1)
                self.cnn_buffer[0, -1, :] = scaled_state_tensor
                
                # Input to Conv1d must be (batch, features, seq_len)
                cnn_input = self.cnn_buffer.permute(0, 2, 1)
                
                # Pass buffer and GRU hidden state
                scaled_delta_pred, self.hidden_state = self.model(cnn_input, self.hidden_state)
                scaled_delta_pred = scaled_delta_pred.squeeze(1) # (B, 1, F) -> (B, F)

            else:
                # Standard RNN inference
                x = scaled_state_tensor.view(1, 1, self.input_size)
                scaled_delta_pred, self.hidden_state = self.model(x, self.hidden_state)
                scaled_delta_pred = scaled_delta_pred.squeeze(1) # (B, 1, F) -> (B, F)

        if not data_point.need_prediction:
            return None

        # --- 3. Un-scaling Logic ---
        scaled_abs_pred = scaled_state_tensor + scaled_delta_pred.squeeze(0)
        unscaled_abs_pred = (scaled_abs_pred * self.scaler_scale) + self.scaler_mean
        prediction = unscaled_abs_pred.cpu().numpy()
        
        return prediction