import os
import sys
import numpy as np
import math


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep


# Try to import torch and define the Transformer model.
use_torch = False
try:
    import torch
    from torch import nn
    use_torch = True
except Exception:
    use_torch = False


if use_torch:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)


    class TransformerModel(nn.Module):
        def __init__(self, input_size: int, d_model: int, nhead: int, num_encoder_layers: int, dropout: float = 0.1):
            super().__init__()
            self.model_type = 'Transformer'
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
            self.encoder = nn.Linear(input_size, d_model)
            self.d_model = d_model
            self.decoder = nn.Linear(d_model, input_size)

        def forward(self, src):
            src = self.encoder(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            output = self.decoder(output)
            return output[:, -1, :]


class PredictionModel:
    """
    PredictionModel that uses a trained Transformer for inference.
    """

    def __init__(self, lstm_checkpoint_path=None, seq_len=100):
        self.current_seq_ix = None
        self.seq_buffer = []
        self.seq_len = seq_len

        self.transformer_checkpoint_path = lstm_checkpoint_path or os.path.join(CURRENT_DIR, 'transformer_checkpoint.pt')
        self.transformer_model = None
        self.device = None
        self.scaler_mean = None
        self.scaler_std = None
        self.predict_delta = False

    def _try_load_transformer(self, input_size):
        if not use_torch:
            return False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if os.path.exists(self.transformer_checkpoint_path):
                state = torch.load(self.transformer_checkpoint_path, map_location=self.device, weights_only=False)
                
                saved_args = state['args']
                
                model = TransformerModel(
                    input_size=input_size,
                    d_model=saved_args['d_model'],
                    nhead=saved_args['nhead'],
                    num_encoder_layers=saved_args['num_encoder_layers'],
                    dropout=saved_args['dropout']
                )
                
                model.load_state_dict(state['model_state'])
                model.to(self.device)
                model.eval()
                self.transformer_model = model

                if 'scaler' in state:
                    self.scaler_mean = np.array(state['scaler'].get('mean'))
                    self.scaler_std = np.array(state['scaler'].get('std'))
                
                if 'predict_delta' in saved_args:
                    self.predict_delta = bool(saved_args.get('predict_delta', False))
                
                if 'seq_len' in saved_args:
                    self.seq_len = int(saved_args.get('seq_len'))
                
                return True
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to load Transformer checkpoint '{self.transformer_checkpoint_path}': {e}") from e

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.seq_buffer = []

        current_state = data_point.state.astype(np.float32).copy()
        self.seq_buffer.append(current_state)
        if len(self.seq_buffer) > self.seq_len:
            self.seq_buffer.pop(0)

        if not data_point.need_prediction:
            return None

        input_size = current_state.shape[0]
        if self.transformer_model is None:
            loaded = self._try_load_transformer(input_size)
            if not loaded:
                raise RuntimeError(f"Transformer model not available. Ensure torch is installed and checkpoint exists at '{self.transformer_checkpoint_path}'.")

        buf = np.array(self.seq_buffer, dtype=np.float32)
        if buf.shape[0] < self.seq_len:
            pad_len = self.seq_len - buf.shape[0]
            pad = np.zeros((pad_len, buf.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, buf])
        else:
            arr = buf[-self.seq_len:]

        if self.scaler_mean is not None and self.scaler_std is not None:
            arr = (arr - self.scaler_mean) / (self.scaler_std + 1e-8)

        try:
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.transformer_model(x)
            pred = pred.squeeze(0).cpu().numpy()

            if self.predict_delta:
                last_scaled_state = arr[-1, :]
                pred = last_scaled_state + pred

            if self.scaler_mean is not None and self.scaler_std is not None:
                pred = pred * (self.scaler_std + 1e-8) + self.scaler_mean

            return pred
        except Exception as e:
            raise RuntimeError(f"Transformer inference failed: {e}") from e


if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"
    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print("Testing Transformer model...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")
