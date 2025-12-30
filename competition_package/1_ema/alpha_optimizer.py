import os
import sys
import numpy as np
import pandas as pd

# Move imports and path setup outside the loop
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep


class PredictionModel:
    """
    Simple model that predicts the next value as an exponential moving average
    of all previous values in the current sequence.
    """

    def __init__(self, alpha=0.055):
        self.current_seq_ix = None
        self.ema = None
        self.alpha = alpha

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.ema = None

        current_state = data_point.state.copy()
        if self.ema is None:
            self.ema = current_state
        else:
            self.ema = self.alpha * current_state + (1 - self.alpha) * self.ema

        if not data_point.need_prediction:
            return None

        return self.ema.copy()


if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"
    
    # Load the scorer once to get feature names
    scorer = ScorerStepByStep(test_file)
    feature_names = scorer.features
    
    # Store results for each alpha
    all_results = []
    alphas = [a / 100 for a in range(0, 101)]

    print(f"Testing {len(alphas)} alpha values...")

    for alpha in alphas:
        model = PredictionModel(alpha=alpha)
        # We need a new scorer for each run to reset its state
        scorer_run = ScorerStepByStep(test_file)
        results = scorer_run.score(model)
        
        # Store the alpha and the individual feature scores
        scores = {feature: results[feature] for feature in feature_names}
        all_results.append({'alpha': alpha, 'scores': scores})
        
        print(f"Alpha: {alpha:.2f}, Mean R²: {results['mean_r2']:.6f}")

    # --- Post-processing to find the best alpha for each feature ---
    
    # Create a DataFrame for easier analysis
    # Rows are alphas, columns are features
    df_data = {res['alpha']: res['scores'] for res in all_results}
    scores_df = pd.DataFrame.from_dict(df_data, orient='index')

    # Get the alpha with the maximum score for each feature column
    best_alphas = scores_df.idxmax()
    list = best_alphas.to_list()
    print("\n" + "=" * 60)
    print("Best alpha for each feature:")
    print(best_alphas)
    print("=" * 60)
    print("Best alphas as a list:")
    print(list)

