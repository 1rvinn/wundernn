"""
Solution file for stateful LSTM inference.

This file defines the `PredictionModel` class which is instantiated once
and used to make predictions one data point at a time. It loads the
pre-trained stateful LSTM model and manages its hidden state.
"""

import numpy as np
import torch
from torch import nn

# This import is required by the competition environment.
# We'll include a placeholder definition for clarity, but the
# official `utils` module will be used during evaluation.

from utils import DataPoint
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
            batch_first=True,  # (batch, seq_len, features)
            dropout=dropout
        )
        # The Linear layer is applied to EVERY time-step's output
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Note: This forward is for training. In inference, we will
        call the lstm and fc layers separately to manage the state.
        """
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
        # These are based on the defaults in your training script.
        # If you trained with different values, update them here.
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.0
        self.checkpoint_path = 'lstm_stateful_checkpoint.pt'

        # --- Device Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        # Load the checkpoint
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        
        # **Dynamically determine input_size from the checkpoint**
        # fc.weight has shape [input_size, hidden_size]
        self.input_size = state_dict['fc.weight'].shape[0]
        print(f"Inferred input_size (N_features) from checkpoint: {self.input_size}")

        # Instantiate the model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout  # Dropout is inactive in .eval() mode
        )
        
        # Load the weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")

            
        # --- Internal State Management ---
        # This tracks the current sequence to know when to reset.
        self.current_seq_ix = -1 
        # This stores the (h_t, c_t) tuple for the LSTM.
        self.hidden_state = None


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        # 1. Manage State: Reset if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_state = None  # LSTM will auto-initialize on next call
            # print(f"New sequence detected: {data_point.seq_ix}. Resetting state.")

        # 2. Prepare Input: Convert numpy state to a PyTorch tensor
        # The model was trained on (batch, seq_len, features).
        # For step-by-step, batch=1 and seq_len=1.
        
        # Ensure input state has the expected number of features
        if data_point.state.shape[0] != self.input_size:
            # This is a safety check.
            # In a real scenario, you might want to handle this error.
            print(f"Warning: Mismatch in feature size. Expected {self.input_size}, got {data_point.state.shape[0]}")
            # Fallback to a dummy prediction if needed
            if data_point.need_prediction:
                return np.zeros(self.input_size)
            else:
                return None
                
        # Convert state to tensor: (N,) -> (1, 1, N)
        x = torch.from_numpy(data_point.state).float().to(self.device)
        x = x.view(1, 1, self.input_size)

        # 3. Model Inference (Stateful)
        # We must call the LSTM and FC layers separately to capture the hidden state.
        with torch.no_grad():
            # (lstm_out, self.hidden_state) = self.model.lstm(x, previous_hidden_state)
            # lstm_out shape: (1, 1, hidden_size)
            # self.hidden_state: (h_n, c_n) tuple
            lstm_out, self.hidden_state = self.model.lstm(x, self.hidden_state)
            
            # Pass the LSTM output to the fully-connected layer
            # lstm_out.squeeze(1) -> (1, hidden_size)
            # prediction_tensor -> (1, input_size)
            prediction_tensor = self.model.fc(lstm_out.squeeze(1))

        # 4. Return Prediction (or None)
        # The model *always* runs to update its state, but we only
        # return a value when requested.
        
        if not data_point.need_prediction:
            return None

        # Convert the prediction tensor back to a 1D numpy array
        # prediction_tensor.squeeze(0) -> (input_size)
        prediction = prediction_tensor.squeeze(0).cpu().numpy()
        
        return prediction