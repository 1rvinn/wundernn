"""
Transformer training script for sequence prediction.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """Dataset that yields (input_seq, target) pairs for next-step prediction."""

    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train", predict_delta: bool = False):
        self.seq_len = seq_len
        self.mode = mode
        self.predict_delta = predict_delta
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]

        self.examples = []

        grouped = df.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)
            need_pred = g["need_prediction"].to_numpy(dtype=bool)

            for t in range(len(g)):
                if not need_pred[t]:
                    continue
                if t + 1 >= len(g):
                    continue
                if self.predict_delta:
                    target = states[t + 1] - states[t]
                else:
                    target = states[t + 1]

                start = max(0, t - self.seq_len + 1)
                window = states[start : t + 1]
                if window.shape[0] < self.seq_len:
                    pad_len = self.seq_len - window.shape[0]
                    pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])

                self.examples.append((window, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
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

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[:, -1, :]


def evaluate(model, dataloader, device, predict_delta=False):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            if predict_delta:
                last_input_state = x[:, -1, :]
                final_state_pred = last_input_state + pred
                final_state_true = last_input_state + y
                ys.append(final_state_true.cpu().numpy())
                yps.append(final_state_pred.cpu().numpy())
            else:
                ys.append(y.cpu().numpy())
                yps.append(pred.cpu().numpy())

    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
    r2s = []
    for i in range(y_true.shape[1]):
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        except Exception:
            r2 = float('nan')
        r2s.append(r2)
    mean_r2 = float(np.nanmean(r2s))
    return mean_r2, r2s


def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    train_mean = df_train[feature_cols].mean()
    train_std = df_train[feature_cols].std().replace(0, 1.0)
    df_train.loc[:, feature_cols] = (df_train[feature_cols] - train_mean) / train_std
    df_val.loc[:, feature_cols] = (df_val[feature_cols] - train_mean) / train_std

    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train', predict_delta=args.predict_delta)
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val', predict_delta=args.predict_delta)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)
    model = TransformerModel(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        val_mean_r2, _ = evaluate(model, val_loader, device, predict_delta=args.predict_delta)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.6f} Val Mean R2={val_mean_r2:.6f}")

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "scaler": {"mean": train_mean.to_numpy(dtype=np.float32), "std": train_std.to_numpy(dtype=np.float32)},
                "feature_cols": train_ds.feature_cols
            }, args.checkpoint)

    print(f"Best Val Mean R2: {best_val:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../datasets/train.parquet')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--checkpoint', type=str, default='transformer_checkpoint.pt')
    parser.add_argument('--predict_delta', action='store_true', help='Predict the delta between steps')
    args = parser.parse_args()
    main(args)
