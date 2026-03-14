import pandas as pd
import numpy as np
import os
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_sequences(feature_df: pd.DataFrame, context_length: int):
    """
    Converts a flat DataFrame into sliding window sequences.

    Input:  DataFrame of shape (num_rows, 9)
    Output: X of shape (num_samples, context_length, 8)  ← 8 input features
            y of shape (num_samples,)                    ← 1 target per sample
    """
    # Separate input features from target
    feature_cols = ["return", "log_return", "rsi", "macd",
                    "macd_signal", "bb_width", "volume_ma_ratio", "close_ma_ratio"]
    target_col   = "target"

    values = feature_df[feature_cols].values   # shape: (num_rows, 8)
    targets = feature_df[target_col].values    # shape: (num_rows,)

    X, y = [], []

    for i in range(len(values) - context_length):
        # 60 consecutive rows of features → one input window
        X.append(values[i : i + context_length])
        # The target is the value at the END of that window
        y.append(targets[i + context_length - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_all_sequences(config: dict):
    processed_dir  = config["data"]["processed_dir"]
    tickers        = config["data"]["tickers"]
    context_length = config["model"]["context_length"]

    all_X, all_y = [], []

    for ticker in tickers:
        print(f"  Building sequences for {ticker}...")
        path = os.path.join(processed_dir, f"{ticker}_features.csv")
        df   = pd.read_csv(path, index_col=0, parse_dates=True)

        X, y = make_sequences(df, context_length)
        print(f"  X: {X.shape}  y: {y.shape}")

        all_X.append(X)
        all_y.append(y)

    # Stack all tickers together into one combined dataset
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print(f"\n  Combined → X: {X_combined.shape}  y: {y_combined.shape}")

    # Save as .npy files — fast binary format for numpy arrays
    np.save(os.path.join(processed_dir, "X.npy"), X_combined)
    np.save(os.path.join(processed_dir, "y.npy"), y_combined)

    print(f"\n✓ Sequences saved → data/processed/X.npy & y.npy")
    return X_combined, y_combined


if __name__ == "__main__":
    config = load_config()
    build_all_sequences(config)

