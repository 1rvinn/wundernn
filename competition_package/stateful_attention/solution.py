"""
Solution file for the StatefulAttentionLSTM.

This version:
1. Loads the 'StatefulAttentionLSTM' model.
2. Is "stateful" - it maintains the LSTM state (h, c)
   and a list of all past hidden states for the attention mechanism.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
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
class StatefulAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # h_t + context_t
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        self.dropout = nn.Dropout(dropout)

    # In solution.py, we don't need the full forward loop.
    # We will call the components manually.
    def forward(self, x):
        # This is just a placeholder for loading the state_dict
        pass

# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (Stateful Attention)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.hidden_size = 128
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'stateful_attention_raw_checkpoint.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.input_size = state_dict['lstm_cell.weight_ih'].shape[1]
            print(f"Inferred input_size (N_features): {self.input_size}")

            self.model = StatefulAttentionLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
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
        self.h = None
        self.c = None
        self.past_hidden_states = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector based on the current data_point.
        """
        
        # 1. Manage State: Reset if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            # Reset all internal states
            self.h = torch.zeros(1, self.hidden_size).to(self.device)
            self.c = torch.zeros(1, self.hidden_size).to(self.device)
            self.past_hidden_states = []

        # 2. Prepare Input (Raw Data)
        input_t = torch.from_numpy(data_point.state.astype(np.float32)).unsqueeze(0).to(self.device)

        # 3. Model Inference (One Step)
        with torch.no_grad():
            # 3a. Run one step of the LSTM
            (self.h, self.c) = self.model.lstm_cell(input_t, (self.h, self.c))
            self.past_hidden_states.append(self.h)
            
            # 3b. Attention Mechanism
            if len(self.past_hidden_states) > 0:
                past_states = torch.stack(self.past_hidden_states, dim=1) # (1, t+1, hidden)
                query = self.h.unsqueeze(2) # (1, hidden, 1)
                scores = torch.bmm(past_states, query) # (1, t+1, 1)
                weights = F.softmax(scores, dim=1)
                context = torch.sum(weights * past_states, dim=1) # (1, hidden)
            else:
                context = torch.zeros_like(self.h)
            
            # 3c. Combine and Predict
            combined_out = torch.cat((self.h, context), dim=1) # (1, hidden*2)
            prediction_tensor = self.model.fc_head(combined_out) # (1, features)

        # 4. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 5. Return (No un-scaling needed)
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        return prediction