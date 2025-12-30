"""
Solution file for LSTMAttentionModel (Raw Data + Huber Loss Training).

- Model: Standard LSTMAttentionModel (as defined in your specific training script).
- Input: Raw, unscaled data (stateless sliding window).
- Output: Absolute next state prediction.
- Logic: Matches the training script that used HuberLoss.
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
# This matches the class in your provided training script exactly.
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
        # x: (batch, seq_len, input_size)
        
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
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (LSTM Attention, Raw Data)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match your training script) ---
        self.seq_len = 150
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_raw_checkpoint.pt')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            # Load the checkpoint
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Infer input_size dynamically
            # 'lstm.weight_ih_l0' shape is [4*hidden_size, input_size]
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
        self.state_buffer = [] # Stores raw float32 arrays

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset buffer if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # 2. Add Raw State to Buffer
        # Ensure type is float32 for the network
        self.state_buffer.append(data_point.state.astype(np.float32))
        
        # 3. Prune buffer to seq_len
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # 4. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 5. Prepare Input Window (with padding logic matching training)
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            # Pad with zeros
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 6. Model Inference
        # Add batch dimension: (seq_len, features) -> (1, seq_len, features)
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the raw absolute state
            prediction_tensor = self.model(x) # Shape: (1, features)

        # 7. Return Prediction
        # Convert to 1D numpy array
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        
        return prediction
