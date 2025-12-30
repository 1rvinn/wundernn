import os
import sys
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")
from utils import DataPoint, ScorerStepByStep

class SingleFeatureLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)

class PredictionModel:
    def __init__(self):
        self.seq_len = 100
        self.num_features = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.scalers = []
        self.buffers = None
        self._load_models()

    def _load_models(self):
        self.models = []
        self.scalers = []
        for feature_idx in range(self.num_features):
            checkpoint = torch.load(f"{CURRENT_DIR}/lstm_feature_{feature_idx}.pt", map_location=self.device, weights_only=False)
            args = checkpoint['args']
            model = SingleFeatureLSTM(
                input_size=self.num_features,
                hidden_size=args['hidden_size'],
                num_layers=args['num_layers']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            self.models.append(model)
            self.scalers.append(checkpoint['scaler'])
        self.seq_len = args['seq_len']

    def reset(self):
        self.buffers = None
        self._last_seq_ix = None

    def predict(self, data_point: DataPoint):
        # Only predict if needed
        if not getattr(data_point, "need_prediction", True):
            return None

        # Robustly detect new sequence (works for all scorer implementations)
        seq_ix = getattr(data_point, "seq_ix", None)
        if not hasattr(self, "_last_seq_ix"):
            self._last_seq_ix = None
        is_new_seq = (self.buffers is None) or (seq_ix is not None and seq_ix != self._last_seq_ix)
        self._last_seq_ix = seq_ix

        x = np.array(data_point.state, dtype=np.float32)
        if is_new_seq:
            self.buffers = [np.zeros((self.seq_len, self.num_features), dtype=np.float32)]
            self.buffers[0][-1] = x
        else:
            self.buffers[0] = np.roll(self.buffers[0], -1, axis=0)
            self.buffers[0][-1] = x
        window = self.buffers[0].copy()
        preds = []
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            x_scaled = ((window - scaler['mean']) / (scaler['std'] + 1e-8)).astype(np.float32)
            x_tensor = torch.from_numpy(x_scaled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = model(x_tensor).cpu().numpy().item()
            # Unscale the prediction
            pred = pred * scaler['std'][i] + scaler['mean'][i]
            preds.append(pred)
        return np.array(preds, dtype=np.float32)

if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"
    scorer = ScorerStepByStep(test_file)
    model = PredictionModel()
    results = scorer.score(model)
    print(results)