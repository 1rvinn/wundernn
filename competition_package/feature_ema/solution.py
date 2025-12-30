"""
Solution file for the LSTMAttentionModel (Raw Data + EMA FE).

This version:
1. Loads the 'LSTMAttentionModel' (with 2N -> N signature).
2. Loads the 'lstm_attention_raw_ema_checkpoint.pt' checkpoint.
3. Does NOT use a StandardScaler.
4. **Performs EMA Feature Engineering on the fly.**
5. Manages a 150-step "state buffer" of the (state + EMA) features.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from dataclasses import dataclass # Added for the dummy DataPoint

# This import is required by the competition environment.
try:
    from utils import DataPoint
except ImportError:
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

# --- Model Definition ---
# This MUST be the model class from your training script.
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Encoder (The LSTM)
        # Input size is 2*N (raw + EMA)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # 3. Decoder (The final classifier)
        # Output size is N (raw state)
        self.fc = nn.Linear(hidden_size * 2, output_size)

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
        Initialize the model, load weights, and set up internal states.
        """
        print("Initializing PredictionModel (LSTM Attention + EMA FE)...")

        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 150
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.checkpoint_path = 'lstm_attention_raw_ema_checkpoint.pt'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Infer input_size (2*N) and output_size (N)
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            self.output_size = state_dict['fc.weight'].shape[0]
            print(f"Inferred Input size: {self.input_size}, Output size: {self.output_size}")
            
            # Sanity check
            if self.input_size != self.output_size * 2:
                 print(f"Warning: Input/Output size mismatch. Input={self.input_size}, Output={self.output_size}")

            self.model = LSTMAttentionModel(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            print("Model loaded successfully.")

        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}")
            raise e

        # --- Define Alphas (MUST MATCH TRAINING) ---
        # This list MUST be identical to the one in your 'main' function
        common_alphas = [0.5, 0.2, 0.05]
        self.alphas_list = np.array(
            [common_alphas[i % len(common_alphas)] for i in range(self.output_size)], 
            dtype=np.float32
        )
        self.one_minus_alphas = 1.0 - self.alphas_list
        # ---
        
        # --- Internal State Management ---
        self.current_seq_ix = -1
        # This buffer will store the (state + EMA) features
        self.state_buffer = [] 
        # We need to store the *last EMA value* to calculate the next one
        self.last_ema_np = None


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict by running the model on the window of engineered features.
        """
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []
            self.last_ema_np = None # Reset the EMA state

        # 2. --- FEATURE ENGINEERING (on the fly) ---
        current_state_np = data_point.state.astype(np.float32)
        
        if self.last_ema_np is None:
            # This is the first step (t=0)
            # The first EMA is just the first data point
            current_ema_np = current_state_np
        else:
            # Calculate the new EMA: EMA_t = alpha*X_t + (1-alpha)*EMA_t-1
            current_ema_np = (self.alphas_list * current_state_np) + (self.one_minus_alphas * self.last_ema_np)
            
        # Combine into the (2*N) feature vector
        combined_features = np.concatenate([current_state_np, current_ema_np])
        
        # Store the current EMA for the *next* step's calculation
        self.last_ema_np = current_ema_np
        # --- END FE ---

        # 3. Add combined features to buffer
        self.state_buffer.append(combined_features)
        
        # 4. Prune buffer to seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # 5. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 6. Prepare Input Window (with padding)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # Add batch dimension: (seq_len, features) -> (1, seq_len, features)
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        # 7. Model Inference
        with torch.no_grad():
            pred_tensor = self.model(x) # Output is (1, N)

        # 8. Return Prediction
        # Squeeze to 1D numpy array: (1, N) -> (N,)
        prediction = pred_tensor.cpu().numpy().squeeze(0)
        
        return prediction
