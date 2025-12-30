"""
Solution file for the Stateless LSTMAttentionModel (Raw Data Version).

This version:
1. Loads the 'LSTMAttentionModel'.
2. Loads the 'lstm_attention_raw_checkpoint.pt' weights.
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
from utils import DataPoint, ScorerStepByStep

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
# --- END NEW MODEL ---


# --- Prediction Class ---
class PredictionModel:
    def __init__(self):
        """
        Initialize the model, load weights, and set up internal state.
        """
        print("Initializing PredictionModel (LSTM w/ Attention, Raw Data)...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Hyperparameters (must match training 'Args') ---
        self.seq_len = 200
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        self.checkpoint_path = os.path.join(script_dir, 'lstm_attention_raw_checkpoint.pt')
        # No scaler path is needed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Model ---
        try:
            # Load the checkpoint to infer input_size
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Infer input_size from the 'lstm.weight_ih_l0' shape: [4*hidden, input]
            self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
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
            # Use zeros for padding, as in the training script
            pad = np.zeros((pad_len, self.input_size), dtype=np.float32)
            window = np.vstack([pad, window])
        
        # 6. Model Inference
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model predicts the *raw absolute state*
            prediction_tensor = self.model(x) # Shape: (1, features)

        # 7. Un-scaling Logic (NOT NEEDED)
        # Convert to numpy and return
        prediction = prediction_tensor.cpu().numpy().squeeze(0)
        
        return prediction
    
if __name__ == "__main__":
    # Check existence of test file
    test_file = f"/Users/irvinsachdeva/Documents/wundernn/competition_package/datasets/train.parquet"

    # Create and test our model
    model = PredictionModel()

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)