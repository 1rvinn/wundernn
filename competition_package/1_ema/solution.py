import os
import sys
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep


# Default per-feature alphas (replace with your optimized per-feature values)
best_alphas = [0.08, 0.05, 0.11, 0.05, 0.03, 0.07, 0.03, 0.08, 0.04, 0.03,
               0.04, 0.06, 0.04, 0.08, 0.11, 0.04, 0.05, 0.05, 0.08, 0.09,
               0.11, 0.03, 0.05, 0.06, 0.05, 0.05, 0.08, 0.05, 0.07, 0.02,
               0.05, 0.09]


# Try to import torch and define a small LSTM inference model that matches
# the training script. If torch isn't available, we'll fall back to EMA.
use_torch = False
try:
    import torch
    from torch import nn
    use_torch = True
except Exception:
    use_torch = False


if use_torch:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)


class PredictionModel:
    """
    PredictionModel that uses a trained LSTM for inference when available.

    Behavior:
    - Attempts to load an LSTM checkpoint from `lstm_checkpoint.pt` in the repo root.
    - If torch or the checkpoint isn't available, falls back to per-feature EMA.
    - Maintains a sliding window buffer for the current sequence; resets on seq_ix change.
    """

    def __init__(self, alphas=None, lstm_checkpoint_path=None, seq_len=100):
        self.current_seq_ix = None
        self.ema = None
        self.seq_buffer = []  # list of recent states for LSTM input
        self.seq_len = seq_len

        if alphas is None:
            self.alphas = np.array(best_alphas)
        else:
            self.alphas = np.array(alphas)

        self.lstm_checkpoint_path = lstm_checkpoint_path or os.path.join(CURRENT_DIR, '..', 'lstm_checkpoint.pt')
        self.lstm_model = None
        self.device = None
        # try loading model lazily when we see first data point

    def _try_load_lstm(self, input_size):
        if not use_torch:
            return False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMModel(input_size=input_size)
            if os.path.exists(self.lstm_checkpoint_path):
                state = torch.load(self.lstm_checkpoint_path, map_location=self.device)
                # state could be a state_dict or a full model; try state_dict first
                if isinstance(state, dict) and not any(k.startswith('module.') for k in state.keys()):
                    model.load_state_dict(state)
                else:
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        # last resort: if state is a nested dict or contains 'model'
                        if isinstance(state, dict) and 'model' in state:
                            model.load_state_dict(state['model'])
                model.to(self.device)
                model.eval()
                self.lstm_model = model
                return True
            else:
                return False
        except Exception:
            # any error — don't crash, we'll fallback to EMA
            self.lstm_model = None
            return False

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.ema = None
            self.seq_buffer = []

        current_state = data_point.state.astype(np.float32).copy()

        # Update EMA buffer
        if self.ema is None:
            self.ema = current_state.copy()
        else:
            self.ema = self.alphas * current_state + (1 - self.alphas) * self.ema

        # Update LSTM buffer
        self.seq_buffer.append(current_state)
        if len(self.seq_buffer) > self.seq_len:
            self.seq_buffer.pop(0)

        # If no prediction required, return None
        if not data_point.need_prediction:
            return None

        # If LSTM not loaded yet, try to load now using input size
        input_size = current_state.shape[0]
        if self.lstm_model is None:
            loaded = self._try_load_lstm(input_size)
            if not loaded:
                # fallback to EMA
                return self.ema.copy()

        # Prepare LSTM input: pad front with zeros if buffer shorter than seq_len
        buf = np.array(self.seq_buffer, dtype=np.float32)
        if buf.shape[0] < self.seq_len:
            pad_len = self.seq_len - buf.shape[0]
            pad = np.zeros((pad_len, buf.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, buf])
        else:
            arr = buf[-self.seq_len:]

        try:
            import torch
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)
            with torch.no_grad():
                out = self.lstm_model(x)
            pred = out.squeeze(0).cpu().numpy()
            return pred
        except Exception:
            # On any failure, return EMA
            return self.ema.copy()


if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"

    # Create and test our model
    model = PredictionModel()

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    print("Testing model (LSTM if available, otherwise EMA)...")
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
