"""
Solution file for the ENSEMBLE LSTMAttentionModel (Input Projection + Post-Norm Version).

This version:
1. Loads all 5 'LSTMAttentionModel' models trained via K-Fold.
2. Uses the updated model architecture with:
   - Input Projection
   - **Post-Attention LayerNorm** (Synced with training code)
3. Does NOT use a StandardScaler.
4. Predicts the absolute next state.
5. Runs inference 5 times (once per model) and AVERAGES the predictions.
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
    # Dummy definition for local testing if utils is missing
    from dataclasses import dataclass
    @dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

# --- Model Definition ---
# This MUST match the class from 'train_kfold.py' EXACTLY.
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
        # Input size is now 'hidden_size' due to projection
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        # 2. Attention Mechanism
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        # --- IMPROVEMENT: Post-Attention Normalization ---
        # Stabilizes the context vector before the final projection.
        self.context_norm = nn.LayerNorm(hidden_size * 2)
        
        # 3. Decoder
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # 1. Apply Input Projection
        x_proj = self.input_projection(x)
        
        # 2. Pass through Encoder (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 4. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # --- Apply Post-Attention Norm ---
        context_vector = self.context_norm(context_vector)
        
        # 5. Decode
        return self.fc(context_vector)
# --- END NEW MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (ENSEMBLE, Input Projection + Post-Norm)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters ---
        self.seq_len = 100
        self.num_layers = 2
        self.dropout = 0.0 # Dropout usually disabled during inference
        self.n_folds = 5
        
        # UPDATED: Matches the prefix in 'train_kfold.py'
        base_checkpoint_name = 'lstm_attention_proj' 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load All 5 Models ---
        self.models = []
        try:
            # Load the first model to infer input_size
            # Note: Checking fold_0 first as per your training script
            first_model_path = os.path.join(script_dir, f'{base_checkpoint_name}_fold_0.pt')
            
            # Fallback check if 0 doesn't exist but 1 does (just in case)
            if not os.path.exists(first_model_path):
                 first_model_path = os.path.join(script_dir, f'{base_checkpoint_name}_fold_1.pt')
            
            if not os.path.exists(first_model_path):
                raise FileNotFoundError(f"Could not find first checkpoint: {first_model_path}")

            state_dict = torch.load(first_model_path, map_location=self.device)
            
            # --- Infer Input Size ---
            # Look at the first layer of input_projection: shape is (hidden_size, input_size)
            self.input_size = state_dict['input_projection.0.weight'].shape[1]
            self.hidden_size = state_dict['input_projection.0.weight'].shape[0]
            
            print(f"Inferred input_size (N_features): {self.input_size}")
            print(f"Inferred hidden_size: {self.hidden_size}")

            # Instantiate and load models
            for i in range(self.n_folds):
                # Assuming 0-based indexing from training script
                model_path = os.path.join(script_dir, f'{base_checkpoint_name}_fold_{i}.pt')
                
                if not os.path.exists(model_path):
                    print(f"Warning: Fold {i} checkpoint not found at {model_path}. Skipping.")
                    continue

                model = LSTMAttentionModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device).eval()
                self.models.append(model)
                
            print(f"Successfully loaded {len(self.models)} ensemble models.")
            
            if len(self.models) == 0:
                raise ValueError("No models were loaded! Check filepath.")
            
        except Exception as e:
            print(f"CRITICAL: Failed to load ensemble: {e}")
            raise e
            
        # --- Internal State Management ---
        self.current_seq_ix = -1 
        self.state_buffer = [] 

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

if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{os.path.dirname(os.path.abspath(__file__))}/../datasets/train.parquet"

    # Create and test our model
    if os.path.exists(test_file):
        try:
            model = PredictionModel()
            from utils import ScorerStepByStep
            scorer = ScorerStepByStep(test_file)
            print("Testing model...")
            results = scorer.score(model)
            print(f"Mean R2: {results['mean_r2']:.6f}")
        except Exception as e:
            print(f"Error during local test: {e}")
    else:
        print("Test file not found, skipping local test.")
