"""
Solution file for the 2-Model Regime-Switching Ensemble.

Models included:
1. 'LSTMModel': Simple Stateful LSTM.
   - Matches: train_lstm_stateful.py
2. 'LSTMAttentionModel': Attention + Input Projection.
   - Matches: train_lstm_attention.py

Strategy:
- Calculates the volatility (std dev) of the current input window.
- Switches between Model A (Stable) and Model B (Volatile) based on threshold.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

# DataPoint stub for local testing (Scorer compatibility)
try:
    from utils import DataPoint
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

# ==============================================================================
# MODEL A: Simple Stateful LSTM (Matches Training Script)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        # In the training script, fc maps hidden -> input_size
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x) 
        predictions = self.fc(out)
        return predictions

# ==============================================================================
# MODEL B: Attention + Input Projection (Matches Training Script)
# ==============================================================================
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Matches Improvement C in training script
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU() 
        )
        
        # Encoder (Input is hidden_size due to projection)
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # Attention
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # Decoder
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # 1. Apply Input Projection
        x_proj = self.input_projection(x)
        
        # 2. LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Attention
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 4. Decode
        return self.fc(context_vector)

# ==============================================================================
# PREDICTION CLASS
# ==============================================================================
class PredictionModel:
    def __init__(self):
        print("Initializing Volatility-Switching Ensemble...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Configuration ---
        self.seq_len = 100
        self.volatility_threshold = 0.5 
        self.input_size = None # Will determine from checkpoint
        
        # ======================================================================
        # LOAD MODEL A (Stateful)
        # ======================================================================
        self.model_a = None
        path_a = os.path.join(script_dir, 'lstm_stateful_raw_stable_checkpoint.pt')
        try:
            state_dict_a = torch.load(path_a, map_location=self.device)
            
            # --- CRITICAL FIX: Infer Input Size from Checkpoint Weights ---
            # Using fc.bias length ensures we match the output layer dimension
            if 'fc.bias' in state_dict_a:
                input_size_a = state_dict_a['fc.bias'].shape[0]
            else:
                # Fallback to LSTM weights if FC is missing (unlikely)
                input_size_a = state_dict_a['lstm.weight_ih_l0'].shape[1]
                
            print(f"Inferred Model A Input Size: {input_size_a}")
            
            self.model_a = LSTMModel(
                input_size=input_size_a, 
                hidden_size=128, 
                num_layers=2, 
                dropout=0.0
            )
            self.model_a.load_state_dict(state_dict_a)
            self.model_a.to(self.device).eval()
            self.input_size = input_size_a # Set global input size
            
        except Exception as e:
            print(f"CRITICAL: Failed to load Model A at {path_a}: {e}")
            raise e

        # ======================================================================
        # LOAD MODEL B (Attention)
        # ======================================================================
        self.model_b = None
        path_b = os.path.join(script_dir, 'lstm_attention_proj_checkpoint.pt')
        try:
            state_dict_b = torch.load(path_b, map_location=self.device)
            
            # --- CRITICAL FIX: Infer Input Size from Checkpoint Weights ---
            # Check fc.bias (output) or input_projection (input)
            if 'fc.bias' in state_dict_b:
                input_size_b = state_dict_b['fc.bias'].shape[0]
            else:
                input_size_b = state_dict_b['input_projection.0.weight'].shape[1]
            
            print(f"Inferred Model B Input Size: {input_size_b}")
            
            self.model_b = LSTMAttentionModel(
                input_size=input_size_b,
                hidden_size=128,
                num_layers=2,
                dropout=0.0
            )
            self.model_b.load_state_dict(state_dict_b)
            self.model_b.to(self.device).eval()
            
            if input_size_a != input_size_b:
                print(f"WARNING: Model A ({input_size_a}) and Model B ({input_size_b}) size mismatch!")
                
        except Exception as e:
            print(f"CRITICAL: Failed to load Model B at {path_b}: {e}")
            raise e

        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.state_buffer = [] 

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # 1. Manage State Buffer
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        # Convert state to float32
        state_data = data_point.state.astype(np.float32)
        
        # --- Handle Feature Mismatch ---
        # If incoming data has more features than model expects (e.g. 128 vs 32),
        # slice it. If less, we are in trouble (but usually this is a subset issue).
        if state_data.shape[0] != self.input_size:
            # Assuming the relevant features are the first N
            state_data = state_data[:self.input_size]

        self.state_buffer.append(state_data)
        
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # 2. Check Volatility
        current_data = np.array(self.state_buffer, dtype=np.float32)
        current_volatility = np.std(current_data, axis=0).mean()
        
        use_model_b = current_volatility >= self.volatility_threshold

        # 3. Prepare Input Window
        window = current_data
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        # 4. Inference
        prediction = None
        with torch.no_grad():
            if use_model_b:
                # Model B: Input (batch, seq, feat) -> Output (batch, feat)
                pred_tensor = self.model_b(x)
                prediction = pred_tensor.cpu().numpy().squeeze(0)
            else:
                # Model A: Input (batch, seq, feat) -> Output (batch, seq, feat)
                # We take the last timestep
                pred_seq = self.model_a(x)
                pred_tensor = pred_seq[:, -1, :] 
                prediction = pred_tensor.cpu().numpy().squeeze(0)
        
        return prediction

if __name__ == "__main__":
    # Local Test Stub
    test_file = f"{os.path.dirname(os.path.abspath(__file__))}/../datasets/train.parquet"
    if os.path.exists(test_file):
        try:
            print("Running local test...")
            model = PredictionModel()
            from utils import ScorerStepByStep
            scorer = ScorerStepByStep(test_file)
            results = scorer.score(model)
            print(f"Mean R2: {results['mean_r2']:.6f}")
        except Exception as e:
            print(f"Error during local test: {e}")
