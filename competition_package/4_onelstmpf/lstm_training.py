import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, df, feature_idx, seq_len=100, scaler=None):
        self.seq_len = seq_len
        self.feature_idx = feature_idx
        self.feature_cols = [f"{i}" for i in range(32)]
        self.states = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.examples = []
        for t in range(len(self.states) - seq_len - 1):
            x = self.states[t:t+seq_len]
            y = self.states[t+seq_len+1, feature_idx]
            self.examples.append((x, y))
        self.scaler = scaler

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        x = x.astype(np.float32)
        y = np.float32(y)
        if self.scaler:
            x = (x - self.scaler['mean']) / (self.scaler['std'] + 1e-8)
            x = x.astype(np.float32)  # <-- Ensure float32 after scaling
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
class SingleFeatureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)

def train_one_feature(df_train, df_val, feature_idx, seq_len=100, hidden_size=64, num_layers=1, epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    feature_cols = [f"{i}" for i in range(32)]
    # Fit scaler on train
    mean = df_train[feature_cols].mean().to_numpy()
    std = df_train[feature_cols].std().replace(0, 1.0).to_numpy()
    scaler = {'mean': mean, 'std': std}

    train_ds = FeatureDataset(df_train, feature_idx, seq_len, scaler)
    val_ds = FeatureDataset(df_val, feature_idx, seq_len, scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SingleFeatureLSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_r2 = -np.inf
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                y_true.append(y.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_r2 = r2_score(y_true, y_pred)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'model_state': model.state_dict(),
                'scaler': scaler,
                'args': {
                    'seq_len': seq_len,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'feature_idx': feature_idx
                }
            }, f"lstm_feature_{feature_idx}.pt")
        print(f"Feature {feature_idx} Epoch {epoch+1}: Val R2={val_r2:.4f}")
    print(f"Feature {feature_idx} Best Val R2: {best_val_r2:.4f}")

def main():
    df = pd.read_parquet("/datasets/train.parquet")
    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=0.2, random_state=42)
    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for feature_idx in range(32):
        print(f"\nTraining LSTM for feature {feature_idx}")
        train_one_feature(df_train, df_val, feature_idx, seq_len=100, hidden_size=64, num_layers=1, epochs=10, batch_size=64, lr=1e-3, device=device)

if __name__ == "__main__":
    main()