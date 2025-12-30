"""
Training Script for MODEL B (The Specialist).

- Strategy: **Exclusive Focus**.
- Loss: **MaskedMSELoss** (Weights = 1.0 for Hard features, 0.0 for Easy features).
- Result: This model will likely have TERRIBLE R2 on easy features, but optimal R2 on hard ones.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

# --- NEW: Masked Loss Function ---
class MaskedMSELoss(nn.Module):
    """
    MSE Loss that ONLY calculates error for specific indices.
    All other features are ignored (weight=0).
    """
    def __init__(self, focus_indices, num_features):
        super().__init__()
        self.focus_indices = focus_indices
        self.num_features = num_features

    def forward(self, pred, target):
        # 1. Base MSE (Batch, Features)
        loss = F.mse_loss(pred, target, reduction='none')
        
        # 2. Create Mask (Default 0.0)
        mask = torch.zeros(self.num_features, device=pred.device)
        
        # 3. Set Focus Indices to 1.0
        idx_tensor = torch.tensor(self.focus_indices, device=pred.device, dtype=torch.long)
        mask[idx_tensor] = 1.0
        
        # 4. Expand mask to batch size
        mask = mask.unsqueeze(0).expand_as(loss)
        
        # 5. Apply Mask
        loss = loss * mask
        
        # 6. Mean over ONLY the active features (avoid diluting with zeros)
        # We divide by the number of focus features, not total features
        return loss.sum() / (loss.size(0) * len(self.focus_indices))
# -----------------------------------

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, mode: str = "train"):
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
        self.examples = []

        grouped = df.groupby("seq_ix")
        for seq_ix, g in grouped:
            g = g.sort_values("step_in_seq")
            states = g[self.feature_cols].to_numpy(dtype=np.float32)

            for t in range(len(g)):
                if t + 1 >= len(g): continue
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


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- IMPROVEMENT C: Input Projection Layer ---
        # Projects raw features to hidden_size, normalizes, and activates.
        # This helps the model learn a better representation before the recurrent steps.
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU() # GELU is often preferred over ReLU for transformer/modern LSTM architectures
        )
        
        # 1. Encoder (The LSTM)
        # Note: input_size for LSTM is now 'hidden_size' because of the projection above.
        self.lstm = nn.LSTM(input_size=hidden_size, 
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
        
        # 1. Apply Input Projection
        # x_proj: (batch, seq_len, hidden_size)
        x_proj = self.input_projection(x)
        
        # 2. Pass through Encoder (LSTM)
        # lstm_out: (batch, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # 3. Calculate Attention Scores
        attention_logits = self.attention_fc(lstm_out)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 4. Create Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 5. Decode (Make Prediction)
        return self.fc(context_vector)


def evaluate(model, dataloader, device, focus_indices):
    model.eval()
    ys_true = []
    ys_pred = []

    with torch.no_grad():
        for x, y in dataloader: 
            x, y = x.to(device), y.to(device)
            pred = model(x)
            ys_true.append(y.cpu().numpy())
            ys_pred.append(pred.cpu().numpy())

    y_true = np.vstack(ys_true)
    y_pred = np.vstack(ys_pred)
    
    # Calculate R2 ONLY for focus indices
    r2_scores = []
    for i in focus_indices:
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        except:
            r2 = float('nan')
        r2_scores.append(r2)
    
    return np.mean(r2_scores)


def main(args):
    p = Path(args.data)
    df = pd.read_parquet(p)

    seq_ids = df['seq_ix'].unique()
    train_ids, val_ids = train_test_split(seq_ids, test_size=args.val_frac, random_state=42)

    df_train = df[df['seq_ix'].isin(train_ids)]
    df_val = df[df['seq_ix'].isin(val_ids)]

    train_ds = SequenceDataset(df_train, seq_len=args.seq_len, mode='train')
    val_ds = SequenceDataset(df_val, seq_len=args.seq_len, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_ds.feature_cols)

    model = LSTMAttentionModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    
    # --- CHANGED: Masked Loss ---
    print(f"Training SPECIALIST model on indices: {args.focus_indices}")
    criterion = MaskedMSELoss(args.focus_indices, num_features=input_size)
    # ----------------------------

    best_val = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader: 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) 
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate ONLY on the focus indices
        val_mean_r2 = evaluate(model, val_loader, device, args.focus_indices) 
        
        print(f"Epoch {epoch}: Loss={avg_loss:.6f} | Focus Features R2={val_mean_r2:.6f}")
        
        scheduler.step() 

        if val_mean_r2 > best_val:
            best_val = val_mean_r2
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Best Focus R2: {best_val:.6f}")


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/trainds/train.parquet'
            self.seq_len = 100
            self.batch_size = 32
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.lr = 1e-3
            self.epochs = 20
            self.val_frac = 0.1
            self.checkpoint = 'lstm_specialist_model_b.pt'
            
            # --- UPDATE THIS with your actual bad features ---
            self.focus_indices = [0,1,4,5,12,17,19,21] 
            
    args = Args()
    main(args)
