"""
Solution file for K-Fold Ensemble Stateful LSTM.

This version:
1. Loads all 10 pre-trained stateful LSTM models.
2. Manages 10 separate hidden states, one for each model.
3. Resets all hidden states for each new sequence.
4. Averages the predictions from all 10 models for the final output.
5. Operates on raw, unscaled data.
"""

import numpy as np
import torch
from torch import nn
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
# This class MUST be identical to the one used in training.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, h=None):
        """
        Modified to accept and return hidden state 'h' for inference.
        """
        out, h_out = self.lstm(x, h) 
        predictions = self.fc(out)
        return predictions, h_out # Return predictions and new state
# --- END MODEL DEFINITION ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load ALL 10 models, and set up internal states.
        """
        print("Initializing PredictionModel (K-Fold Stateful Ensemble)...")

        # --- Hyperparameters (must match training 'Args') ---
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.n_folds = 10 # From your N_SPLITS
        self.base_checkpoint_name = 'lstm_stateful_checkpoint'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load All 10 Models ---
        self.models = []
        try:
            # Load the first model to infer input_size
            first_model_path = f'{self.base_checkpoint_name}_fold_1.pt'
            state_dict = torch.load(first_model_path, map_location=self.device)
            
            # Infer input_size from the fc.weight shape: [input_size, hidden_size]
            self.input_size = state_dict['fc.weight'].shape[0]
            print(f"Inferred input_size (N_features): {self.input_size}")

            # Load model 1
            model_1 = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            model_1.load_state_dict(state_dict)
            model_1.to(self.device).eval()
            self.models.append(model_1)
            
            # Load models 2 through 10
            for i in range(2, self.n_folds + 1):
                model_path = f'{self.base_checkpoint_name}_fold_{i}.pt'
                model = LSTMModel(
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
        # We now need to store 10 hidden states, one for each model
        self.hidden_states = [None] * self.n_folds


    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict by running all 10 models and averaging their outputs.
        """
        # 1. Manage State: Reset if we are on a new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            # Reset all 10 hidden states
            self.hidden_states = [None] * self.n_folds

        # 2. Prepare Input
        # (N,) -> (1, 1, N) for (batch, seq_len, features)
        x = torch.from_numpy(data_point.state.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Model Inference (Ensemble)
        all_predictions = []
        with torch.no_grad():
            for i in range(self.n_folds):
                # Get the i-th model and its specific hidden state
                model = self.models[i]
                h_in = self.hidden_states[i]
                
                # Run the model for one step
                pred_tensor, h_out = model(x, h_in)
                
                # Save the model's new hidden state
                self.hidden_states[i] = h_out
                
                # Store this model's prediction
                all_predictions.append(pred_tensor)

        # 4. Average the predictions
        # Stack all predictions: [ (1,1,N), (1,1,N), ... ] -> (10, 1, 1, N)
        # Then average across the 0-th dimension (the models)
        avg_pred_tensor = torch.mean(torch.stack(all_predictions), dim=0)
        
        # 5. Return Prediction (or None)
        if not data_point.need_prediction:
            return None

        # Squeeze to 1D numpy array: (1, 1, N) -> (N,)
        prediction = avg_pred_tensor.cpu().numpy().squeeze()
        
        return prediction