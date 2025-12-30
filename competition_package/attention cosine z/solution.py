"""
Solution file for LSTMAttentionModel with Rolling Z-Score Scaling.

- Model: LSTMAttentionModel (Standard bidirectional + simple attention).
- Input: Rolling Z-Score SCALED data.
- Scaling: Calculated on-the-fly using a raw history buffer.
- Output: Scaled prediction -> Unscaled using current window stats.
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
# Matches the class in your training script EXACTLY.
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Encoder (The LSTM)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True) 
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder (The final classifier)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # 1. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 2. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 3. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 4. Decode (Make Prediction)
        return self.fc(context_vector)
# --- END MODEL DEFINITION ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model and state buffers.
        """
        print("Initializing PredictionModel (LSTM Attention + Rolling Scaling)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training script) ---
        self.seq_len = 100
        self.rolling_window = 60
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_scaled_checkpoint.pt')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            # Load the checkpoint to infer input_size
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Infer input_size: 'lstm.weight_ih_l0' is [4*hidden, input]
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            print(f"Inferred input_size (N_Features): {self.input_size}")

            self.model = LSTMAttentionModel(
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
            print(f"CRITICAL: Failed to load model '{self.checkpoint_path}': {e}")
            raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        
        # Buffer 1: Stores RAW states to calculate rolling Mean/Std
        # We need enough history for the rolling window calculation
        self.raw_buffer = [] 
        
        # Buffer 2: Stores SCALED states to feed into the model
        # We need 'seq_len' history for the LSTM window
        self.scaled_buffer = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state.
        1. Update raw buffer.
        2. Calculate rolling mean/std for current step.
        3. Scale current state -> Update scaled buffer.
        4. Feed scaled window to model.
        5. Unscale prediction using current mean/std.
        """
        
        # 1. Manage State: Reset buffers if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.raw_buffer = []
            self.scaled_buffer = []

        # 2. Update Raw Buffer
        current_raw = data_point.state.astype(np.float32)
        self.raw_buffer.append(current_raw)
        
        # We only need to keep enough raw history for the rolling window
        if len(self.raw_buffer) > self.rolling_window:
            self.raw_buffer.pop(0)

        # 3. Calculate Rolling Statistics (On-the-fly)
        # Convert buffer to numpy array: (timesteps, features)
        window_data = np.array(self.raw_buffer, dtype=np.float32)
        
        # Calculate Mean
        curr_mean = np.mean(window_data, axis=0)
        
        # Calculate Std (ddof=1 to match Pandas default)
        if len(self.raw_buffer) > 1:
            curr_std = np.std(window_data, axis=0, ddof=1)
        else:
            # Edge case: Only 1 sample, std is undefined. 
            # Training script uses .fillna(1.0), so we use 1.0
            curr_std = np.ones_like(curr_mean)
            
        # Handle division by zero (if constant values, std=0)
        # Training script: roll_std[roll_std == 0] = 1.0
        curr_std[curr_std == 0] = 1.0
        
        # 4. Scale Current State
        # Z-Score: (X - Mean) / Std
        current_scaled = (current_raw - curr_mean) / curr_std
        
        # Update Scaled Buffer
        self.scaled_buffer.append(current_scaled)
        
        # Prune scaled buffer to seq_len (input window size)
        if len(self.scaled_buffer) > self.seq_len:
            self.scaled_buffer.pop(0)

        # 5. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 6. Prepare Input Window (with padding)
        window_array = np.array(self.scaled_buffer, dtype=np.float32)
        
        if len(window_array) < self.seq_len:
            pad_len = self.seq_len - len(window_array)
            # Pad with zeros (scaled mean is 0)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window_array = np.vstack([pad, window_array])
        
        # 7. Model Inference
        x = torch.from_numpy(window_array).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the SCALED next state
            pred_scaled_tensor = self.model(x) 
            pred_scaled = pred_scaled_tensor.cpu().numpy().squeeze(0)

        # 8. Un-Scale Prediction
        # We use the CURRENT mean and std as the best proxy for the next step
        # Pred_raw = Pred_scaled * Std + Mean
        prediction = (pred_scaled * curr_std) + curr_mean
        
        return prediction
