"""
Solution file for the K-Fold Ensemble of the TCNAttentionModel (Raw Data).

This version:
1. Loads the 'TCNAttentionModel' (TCN Encoder + Attention Decoder).
2. Loads all 5 '...fold_X.pt' checkpoint files.
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Manages a 150-step "state buffer" (the sliding window).
6. Averages the predictions from all 5 models.
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
class TCNAttentionModel(nn.Module):
    """
    A TCN Encoder followed by an Attention mechanism.
    """
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.1):
        super().__init__()
        
        # --- 1. TCN Encoder ---
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            dilation = 2**i  # Exponentially growing dilation
            out_channels = hidden_size
            
            # Use 'same' padding to keep seq_len constant
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    padding=padding, 
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels  # Input for next layer
        
        self.tcn_stack = nn.Sequential(*layers)
        
        # --- 2. Attention Mechanism ---
        self.attention_fc = nn.Linear(hidden_size, 1)
        
        # --- 3. Final Decoder ---
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_size]
        
        # --- TCN Encoder ---
        x_tcn = x.permute(0, 2, 1) # [batch, channels, seq_len]
        tcn_out = self.tcn_stack(x_tcn) # [batch, hidden_size, seq_len]
        
        # --- Attention Decoder ---
        tcn_out = tcn_out.permute(0, 2, 1) # [batch, seq_len, hidden_size]
        
        attention_logits = self.attention_fc(tcn_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * tcn_out, dim=1)
        
        return self.fc(context_vector)
# --- END NEW MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load ALL 5 models, and set up internal states.
        """
        print("Initializing PredictionModel (K-Fold TCNAttentionModel)...")

        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 150
        self.hidden_size = 64
        self.num_layers = 7
        self.kernel_size = 3
        self.dropout = 0.0
        self.n_folds = 5 # From your N_SPLITS
        self.base_checkpoint_name = 'tcn_attention_raw_checkpoint'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load All 5 Models ---
        self.models = []
        try:
            # Load the first model to infer input_size
            first_model_path = f'{self.base_checkpoint_name}_fold_1.pt'
            state_dict = torch.load(first_model_path, map_location=self.device)
            
            # Infer input_size from the TCN stack's first conv layer
            # Weight shape is [out_channels, in_channels, kernel_size]
            self.input_size = state_dict['tcn_stack.0.weight'].shape[1]
            print(f"Inferred input_size (N_Features): {self.input_size}")

            # Load model 1
            model_1 = TCNAttentionModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                kernel_size=self.kernel_size,
                dropout=self.dropout
            )
            model_1.load_state_dict(state_dict)
            model_1.to(self.device).eval()
            self.models.append(model_1)
            
            # Load models 2 through 5
            for i in range(2, self.n_folds + 1):
                model_path = f'{self.base_checkpoint_name}_fold_{i}.pt'
                model = TCNAttentionModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    kernel_size=self.kernel_size,
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
        # We only need ONE buffer for the stateless window
        self.state_buffer = [] 


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict by running all 5 models on the same window and averaging their outputs.
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
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # Add batch dimension: (seq_len, features) -> (1, seq_len, features)
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        # 6. Model Inference (Ensemble)
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                # All models see the *exact same* window
                pred_tensor = model(x)
                all_predictions.append(pred_tensor)

        # 7. Average the predictions
        avg_pred_tensor = torch.mean(torch.stack(all_predictions), dim=0)
        
        # Squeeze to 1D numpy array: (1, N) -> (N,)
        prediction = avg_pred_tensor.cpu().numpy().squeeze(0)
        
        return prediction