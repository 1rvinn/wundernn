"""
Solution file for the *K-Fold Ensemble* of the *Upgraded* LSTMAttentionModel.

This version:
1. Loads the 'LSTMAttentionModel' (with dot-product attention and deeper head).
2. Loads all 5 (or 6) '...fold_X.pt' checkpoint files.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Manages a 150-step "state buffer".
6. Averages the predictions from all models.
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
# This MUST be the model class from your training script.
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
        
        # 2. Attention Mechanism (Dot-Product) - No separate layer needed
        
        # 3. Decoder (Deeper Head)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # e.g., 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size) # 128 -> N_features
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # 1. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 2. Create the "Query"
        query = h_n[-2:].permute(1, 0, 2) # (batch, 2, hidden_size)
        query = query.reshape(x.size(0), 1, -1) # (batch, 1, hidden_size * 2)
        
        # 3. Calculate Attention Scores (Dot-Product)
        attention_scores = torch.bmm(lstm_out, query.transpose(1, 2))
        
        # 4. Convert scores to probabilities
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 5. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 6. Decode
        return self.fc_head(context_vector)
# --- END NEW MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (K-Fold Ensemble, Upgraded Attention)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 150
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.n_folds = 5 # Set this to the N_SPLITS you used (5 or 6)
        
        self.base_checkpoint_name = 'lstm_attention_raw_checkpoint'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load All Models ---
        self.models = []
        try:
            # Load the first model to infer input_size
            first_model_path = os.path.join(script_dir, f'{self.base_checkpoint_name}_fold_1.pt')
            state_dict = torch.load(first_model_path, map_location=self.device)
            
            # Infer input_size from the 'lstm.weight_ih_l0' shape
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            print(f"Inferred input_size (N_features): {self.input_size}")

            # Load model 1
            model_1 = LSTMAttentionModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            model_1.load_state_dict(state_dict)
            model_1.to(self.device).eval()
            self.models.append(model_1)
            
            # Load models 2 through N
            for i in range(2, self.n_folds + 1):
                model_path = os.path.join(script_dir, f'{self.base_checkpoint_name}_fold_{i}.pt')
                model = LSTMAttentionModel(
                    input_size=self.input_size,
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
        self.state_buffer = [] # This will be a list of *raw* np.ndarrays

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state vector by averaging all model predictions.
        """
        
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state_buffer = []

        self.state_buffer.append(data_point.state.astype(np.float32))
        
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        # Prepare Input Window
        window = np.array(self.state_buffer, dtype=np.float32)
        
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        # --- Run ENSEMBLE Inference ---
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                pred_tensor = model(x)
                all_predictions.append(pred_tensor)

        # Stack predictions and average them
        # (N_models, batch, features) -> (1, features)
        stacked_preds = torch.stack(all_predictions, dim=0)
        avg_pred_tensor = torch.mean(stacked_preds, dim=0)
        # ---
        
        prediction = avg_pred_tensor.cpu().numpy().squeeze(0)
        
        return prediction