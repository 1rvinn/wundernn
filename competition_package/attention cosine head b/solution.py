"""
Solution file for the LSTM with Attention + Input Projection (Raw Data Version).

This version:
1. Loads the 'LSTMAttentionModel' with the Input Projection Layer.
2. Loads the 'lstm_attention_proj_checkpoint.pt' weights.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Manages a 100-step "state buffer" to build the input window.
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
# This MUST match the class from your training script EXACTLY.
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- IMPROVEMENT C: Input Projection Layer ---
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU() 
        )
        
        # 1. Encoder (The LSTM)
        # Note: input_size for LSTM is now 'hidden_size' because of the projection.
        self.lstm = nn.LSTM(input_size=hidden_size, 
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
        # x: (batch, seq_len, input_size)
        
        # 1. Apply Input Projection
        x_proj = self.input_projection(x)
        
        # 2. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 4. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 5. Decode (Make Prediction)
        return self.fc(context_vector)
# --- END MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (LSTM+Proj, Raw Data)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 100
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        # Note: Checkpoint name updated to match training script
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_proj_checkpoint.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            # Load the checkpoint
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Infer input_size. 
            # Since we have an input_projection layer, the first weight is 'input_projection.0.weight'
            # Shape is [hidden_size, input_size]
            if 'input_projection.0.weight' in state_dict:
                self.input_size = state_dict['input_projection.0.weight'].shape[1]
            else:
                # Fallback if key name differs slightly, though it shouldn't
                # Or if trying to load an old model without projection (will error later)
                self.input_size = state_dict['fc.bias'].shape[0] 
                
            print(f"Inferred input_size (N_features): {self.input_size}")

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
        self.state_buffer = [] # This will be a list of *raw* np.ndarrays

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Add Raw State to Buffer
        self.state_buffer.append(data_point.state.astype(np.float32))
        
        # 3. Prune buffer to seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # 4. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 5. Prepare Input Window (with padding)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            # Use zeros for padding
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 6. Model Inference
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the *raw absolute state*
            prediction_tensor = self.model(x) # Shape: (1, features)

        # 7. Return result
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        
        return prediction
