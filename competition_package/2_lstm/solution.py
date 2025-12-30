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
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1, bidirectional=False):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )
            fc_in = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Linear(fc_in, input_size)

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
        # scaler and predict_delta (populated when loading checkpoint)
        self.scaler_mean = None
        self.scaler_std = None
        self.predict_delta = False
    # try loading model lazily when we see first data point

    def _try_load_lstm(self, input_size):
        if not use_torch:
            return False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not os.path.exists(self.lstm_checkpoint_path):
                return False

            state = torch.load(self.lstm_checkpoint_path, map_location=self.device, weights_only=False)
            sd = None
            saved_args = None
            saved_scaler = None

            if isinstance(state, dict):
                for key in ("model_state", "state_dict", "model", "model_state_dict"):
                    if key in state:
                        sd = state[key]
                        break
                if sd is None and any("." in str(k) for k in state.keys()):
                    sd = state
                if 'args' in state and isinstance(state['args'], dict):
                    saved_args = state['args']
                if 'scaler' in state and isinstance(state['scaler'], dict):
                    saved_scaler = state['scaler']
            elif isinstance(state, nn.Module):
                state.to(self.device)
                state.eval()
                self.lstm_model = state
                return True
            else:
                sd = state

            if sd is None:
                ck_keys = list(state.keys()) if isinstance(state, dict) else type(state)
                raise RuntimeError(f"Unrecognized checkpoint format. Available keys: {ck_keys}")

            # Determine model hyperparameters before instantiation
            model_params = {'input_size': input_size}
            if saved_args:
                model_params['hidden_size'] = saved_args.get('hidden_size', 128)
                model_params['num_layers'] = saved_args.get('num_layers', 2)
                model_params['dropout'] = saved_args.get('dropout', 0.1)
                model_params['bidirectional'] = saved_args.get('bidirectional', True)  # default True for new models
                self.predict_delta = bool(saved_args.get('predict_delta', False))
                self.seq_len = int(saved_args.get('seq_len', self.seq_len))
            else:
                # Fallback: Infer from state_dict if args not available
                try:
                    # Heuristic: if reverse weights exist, it's bidirectional
                    bidirectional = any('reverse' in k for k in sd.keys())
                    model_params['bidirectional'] = bidirectional
                    model_params['hidden_size'] = sd['fc.weight'].shape[1] // (2 if bidirectional else 1)
                    num_layers = 1
                    while f'lstm.weight_hh_l{num_layers}' in sd or f'lstm.weight_hh_l{num_layers}_reverse' in sd:
                        num_layers += 1
                    model_params['num_layers'] = num_layers
                except KeyError:
                    raise RuntimeError("Could not infer model architecture from checkpoint. Please retrain and save with args.")

            model = LSTMModel(**model_params)
            model.load_state_dict(sd)
            model.to(self.device)
            model.eval()
            self.lstm_model = model

            if saved_scaler is not None:
                self.scaler_mean = np.array(saved_scaler.get('mean'))
                self.scaler_std = np.array(saved_scaler.get('std'))

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM checkpoint '{self.lstm_checkpoint_path}': {e}") from e

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

        # --- Scaling and Delta Prediction Logic ---
        # 1. Scale the input array for the model
        if self.scaler_mean is not None and self.scaler_std is not None:
            # Use a small epsilon to avoid division by zero if std is zero
            arr = (arr - self.scaler_mean) / (self.scaler_std + 1e-8)

        try:
            import torch
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)
            with torch.no_grad():
                out = self.lstm_model(x)
            pred = out.squeeze(0).cpu().numpy()

            # --- V2: Correct Scaling and Delta Prediction Logic ---
            # 1. If predicting deltas, add the predicted (scaled) delta to the last
            #    (scaled) input state to get the next (scaled) state.
            if self.predict_delta:
                # arr is the scaled input to the model
                last_scaled_state = arr[-1, :]
                pred = last_scaled_state + pred

            # 2. Un-scale the result (either the direct prediction or the state from step 1)
            #    to get the final prediction in the original data space.
            if self.scaler_mean is not None and self.scaler_std is not None:
                pred = pred * (self.scaler_std + 1e-8) + self.scaler_mean

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
