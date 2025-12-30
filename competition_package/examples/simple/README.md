# Simple Moving Average Solution

This example demonstrates a basic model implementation for the market state prediction competition.

## Algorithm Description

The `PredictionModel` uses a simple moving average strategy:

1. **Context accumulation**: For each new sequence, the model accumulates history of all previous states
2. **Reset on sequence change**: When `seq_ix` changes, the model resets the accumulated history
3. **Prediction**: The next state is predicted as the arithmetic mean of all previous states in the current sequence

## Solution Structure

```python
class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Main prediction logic
```

Key features:
- Maintains internal state `sequence_history` for each sequence
- Properly handles the `need_prediction` flag
- Resets state when `seq_ix` changes

## How to Run

```bash
cd examples/simple
python solution.py
```

## What to try
**Recurrent networks**: LSTM, GRU or Mamba for capturing long-term dependencies

This example serves as a starting point for developing more complex solutions!