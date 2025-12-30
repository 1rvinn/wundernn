import os
import sys
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep


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

    def __init__(self, alphas=None, lstm_checkpoint_path=None, seq_len=100, default_alpha=0.055):
        self.current_seq_ix = None
        self.seq_buffer = []  # list of recent states for LSTM input
        self.seq_len = seq_len

        # alphas can be:
        # - None: no per-feature alpha provided (use scalar default_alpha for EMA fallback)
        # - scalar: single alpha applied to all features
        # - list/array: per-feature alphas
        if alphas is None:
            self.alphas = None
            self.scalar_alpha = float(default_alpha)
        else:
            # convert to numpy array if list-like, else store scalar
            try:
                arr = np.array(alphas)
                if arr.ndim == 0:
                    self.alphas = None
                    self.scalar_alpha = float(arr)
                else:
                    self.alphas = arr
                    self.scalar_alpha = None
            except Exception:
                # fallback to scalar
                self.alphas = None
                self.scalar_alpha = float(alphas)

        self.lstm_checkpoint_path = lstm_checkpoint_path or os.path.join(CURRENT_DIR, 'lstm_checkpoint.pt')
        self.lstm_model = None
        self.device = None
        # try loading model lazily when we see first data point

    def _try_load_lstm(self, input_size):
        if not use_torch:
            return False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not os.path.exists(self.lstm_checkpoint_path):
                return False

            state = torch.load(self.lstm_checkpoint_path, map_location=self.device)

            # Determine where the actual state_dict is stored
            model_state = None
            saved_args = None
            if isinstance(state, dict):
                # common patterns: {'model_state': state_dict, 'args': {...}}, or raw state_dict
                if 'model_state' in state and isinstance(state['model_state'], dict):
                    model_state = state['model_state']
                elif 'state_dict' in state and isinstance(state['state_dict'], dict):
                    model_state = state['state_dict']
                elif 'model' in state and isinstance(state['model'], dict):
                    model_state = state['model']
                elif all(isinstance(v, torch.Tensor) for v in state.values()):
                    # raw state_dict
                    model_state = state
                # metadata
                if 'args' in state and isinstance(state['args'], dict):
                    saved_args = state['args']
            else:
                # Unexpected type
                return False

            if model_state is None:
                return False

            # If metadata provides hidden_size / num_layers, use it. Otherwise infer.
            hidden_size = 128
            num_layers = 2
            if saved_args is not None:
                try:
                    if 'hidden_size' in saved_args:
                        hidden_size = int(saved_args['hidden_size'])
                    if 'num_layers' in saved_args:
                        num_layers = int(saved_args['num_layers'])
                    if 'seq_len' in saved_args and self.seq_len is None:
                        self.seq_len = int(saved_args['seq_len'])
                except Exception:
                    pass

            # If not provided, try to infer hidden_size and num_layers from state_dict
            try:
                # infer hidden_size from fc weight: shape (out_features=input_size, in_features=hidden_size)
                if 'fc.weight' in model_state:
                    w = model_state['fc.weight']
                elif 'fc.weight' in model_state or 'fc.weight' in model_state.keys():
                    w = model_state.get('fc.weight')
                else:
                    # try common key names
                    candidates = [k for k in model_state.keys() if k.endswith('fc.weight') or k.endswith('.fc.weight')]
                    w = model_state[candidates[0]] if candidates else None
                if w is not None:
                    # w shape: (out_features, in_features)
                    hidden_size = int(w.shape[1])
            except Exception:
                pass

            try:
                # infer num_layers from LSTM param keys like weight_ih_l{n}
                lstm_layer_keys = [k for k in model_state.keys() if 'weight_ih_l' in k]
                if lstm_layer_keys:
                    max_index = max(int(k.split('weight_ih_l')[-1].split('.')[0]) for k in lstm_layer_keys)
                    num_layers = max_index + 1
            except Exception:
                pass

            # Build model with inferred params and load
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            try:
                model.load_state_dict(model_state)
            except Exception as e:
                # try to load with strict=False to allow minor name mismatches
                model.load_state_dict(model_state, strict=False)

            model.to(self.device)
            model.eval()
            self.lstm_model = model
            return True
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

        # Only use LSTM; do not compute or update EMA

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
                raise RuntimeError(f"LSTM model not available. Ensure torch is installed and checkpoint exists at '{self.lstm_checkpoint_path}'.")

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
        except Exception as e:
            raise RuntimeError(f"LSTM inference failed: {e}") from e


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
