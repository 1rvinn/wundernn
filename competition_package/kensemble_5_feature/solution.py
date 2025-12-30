"""
Solution file for the K-Fold Ensemble of the LSTMAttentionModel (Rich FE Version).

This version:
1. Loads the 'LSTMAttentionModel' (with new 4N -> N signature).
2. Loads all 5 '...fold_X.pt' checkpoint files.
3. Does NOT use a StandardScaler.
4. Performs Feature Engineering (ROC, Vol, MA) on the fly.
5. Manages two separate buffers for states and features.
6. Averages the predictions from all 5 models.
7. **FIXED: Casts 't' to int() to prevent TypeError.**
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from dataclasses import dataclass

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
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.1):
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
        Initialize the model, load ALL 5 models, and set up internal states.
        """
        print("Initializing PredictionModel (K-Fold LSTMAttentionModel + Rich FE)...")

        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 150
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.0
        self.n_folds = 5
        self.base_checkpoint_name = 'lstm_attention_rich_fe_checkpoint'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load All 5 Models ---
        self.models = []
        try:
            first_model_path = f'{self.base_checkpoint_name}_fold_1.pt'
            state_dict = torch.load(first_model_path, map_location=self.device)
            
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            self.output_size = state_dict['fc.weight'].shape[0]
            print(f"Inferred Input size: {self.input_size}, Output size: {self.output_size}")

            # Load model 1
            model_1 = LSTMAttentionModel(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            model_1.load_state_dict(state_dict)
            model_1.to(self.device).eval()
            self.models.append(model_1)
            
            # Load models 2 through 5
            for i in range(2, self.n_folds + 1):
                model_path = f'{self.base_checkpoint_name}_fold_{i}.pt'
                model = LSTMAttentionModel(
                    input_size=self.input_size,
                    output_size=self.output_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device).eval()
                self.models.append(model)
                
            print(f"Successfully loaded all {len(self.models)} ensemble models.")

        except Exception as e:
            print(f"CRITICAL: Failed to load one or more models: {e}")
            raise e

        # --- Internal State Management ---
        self.current_seq_ix = -1
        self.feature_buffer = [] 
        self.raw_state_buffer = []


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict by running all 5 models on the same window and averaging their outputs.
        """
        # 1. Manage State
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.feature_buffer = []
            self.raw_state_buffer = []

        # 2. --- FEATURE ENGINEERING (on the fly) ---
        current_state_np = data_point.state.astype(np.float32)
        self.raw_state_buffer.append(current_state_np)
        
        # --- FIX: Cast t to int() ---
        t = int(data_point.step_in_seq)
        # ---
        
        # Calculate ROC(10)
        current_roc_10 = np.zeros(self.output_size, dtype=np.float32)
        if t >= 10:
            current_roc_10 = current_state_np - self.raw_state_buffer[t - 10]
            
        # Calculate Vol(20) and MA(20)
        current_vol_20 = np.zeros(self.output_size, dtype=np.float32)
        current_ma_20 = np.zeros(self.output_size, dtype=np.float32)
        if t >= 19: 
            # Slice needs integers, which is now guaranteed by int(t)
            window_20 = np.array(self.raw_state_buffer[t - 19 : t + 1])
            current_vol_20 = np.std(window_20, axis=0)
            current_ma_20 = np.mean(window_20, axis=0)
            
        # Combine into the (4*N) feature vector
        combined_features = np.concatenate([
            current_state_np, 
            current_roc_10, 
            current_vol_20, 
            current_ma_20
        ])
        # --- END FE ---

        # 3. Add combined features to the model's input buffer
        self.feature_buffer.append(combined_features)
        
        # 4. Prune model's input buffer to seq_len
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)

        # 5. Check if prediction is needed
        if not data_point.need_prediction:
            return None

        # 6. Prepare Input Window (with padding)
        window = np.array(self.feature_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        # 7. Model Inference (Ensemble)
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                pred_tensor = model(x)
                all_predictions.append(pred_tensor)

        # 8. Average the predictions
        avg_pred_tensor = torch.mean(torch.stack(all_predictions), dim=0)
        prediction = avg_pred_tensor.cpu().numpy().squeeze(0)
        
        return prediction