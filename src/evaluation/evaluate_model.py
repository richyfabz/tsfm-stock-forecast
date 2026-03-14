import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.transformer_model import build_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(config: dict):
    processed_dir = config["data"]["processed_dir"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    print(f"  Test set size: {X_test.shape}")
    return X_test, y_test


def load_trained_model(config: dict) -> torch.nn.Module:
    model = build_model(config)
    model_path = os.path.join(config["training"]["model_save_path"], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"  Model loaded from {model_path}")
    return model


def run_predictions(model, X_test: np.ndarray) -> np.ndarray:
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor)
    return preds.numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Directional accuracy: did we predict the right direction (up/down)?
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    dir_acc = np.mean(correct_direction)

    return {"MAE": mae, "RMSE": rmse, "Directional Accuracy": dir_acc}


def plot_returns(y_true, y_pred, output_dir: str):
    """Plot actual vs predicted returns — Image 1 style."""
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual",    color="steelblue",  linewidth=0.8)
    plt.plot(y_pred, label="Predicted", color="darkorange", linewidth=0.8, alpha=0.8)
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.title("Actual vs Predicted Returns")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "actual_vs_predicted_returns.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved → {path}")


def plot_close_price(config: dict, model, output_dir: str):
    """
    Plot actual vs predicted close price per ticker — Image 2 style.
    Reconstructs price from predicted returns.
    """
    tickers       = config["data"]["tickers"]
    processed_dir = config["data"]["processed_dir"]
    context_length = config["model"]["context_length"]
    test_split    = config["training"]["test_split"]

    for ticker in tickers:
        # Load feature CSV to get dates and close prices
        feat_path = os.path.join(processed_dir, f"{ticker}_features.csv")
        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

        # Get the test slice (last 15% chronologically)
        n          = len(df)
        test_start = int(n * (1 - test_split))
        df_test    = df.iloc[test_start:]

        # Load raw close prices for this ticker
        raw_path = os.path.join(config["data"]["raw_dir"], f"{ticker}.csv")
        raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)

        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        # Align close prices to test dates
        close_test = raw_df["Close"].reindex(df_test.index).dropna()

        # Build sequences for just this ticker's test period
        feature_cols = ["return", "log_return", "rsi", "macd",
                        "macd_signal", "bb_width", "volume_ma_ratio", "close_ma_ratio"]

        # We need context_length rows before test start for the first window
        df_full     = df[feature_cols].values
        test_idx    = int(n * (1 - test_split))
        X_ticker, dates = [], []

        for i in range(len(df_test) - 1):
            start = test_idx + i - context_length + 1
            if start < 0:
                continue
            window = df_full[start: start + context_length]
            if len(window) == context_length:
                X_ticker.append(window)
                dates.append(df_test.index[i + 1])

        if len(X_ticker) == 0:
            continue

        X_tensor = torch.tensor(np.array(X_ticker, dtype=np.float32))
        with torch.no_grad():
            pred_returns = model(X_tensor).numpy()

        # Reconstruct predicted price from predicted returns
        actual_close = close_test.values
        pred_prices  = []

        for i, d in enumerate(dates):
            # Find the previous actual close
            loc = close_test.index.get_loc(d)
            if loc == 0:
                continue
            prev_close = close_test.iloc[loc - 1]
            pred_prices.append(prev_close * (1 + pred_returns[i]))

        # Trim to the shortest length to guarantee alignment
        min_len     = min(len(dates), len(pred_prices))
        dates       = dates[-min_len:]
        pred_prices = pred_prices[-min_len:]
        actual_plot = close_test.reindex(dates).values

        # Plot
        plt.figure(figsize=(14, 5))
        plt.plot(dates, actual_plot, label="Actual Close",
                 color="steelblue", linewidth=1.2)
        plt.plot(dates, pred_prices, label="Predicted Close",
                 color="darkorange", linewidth=1.2,
                 linestyle="--", alpha=0.85)
        plt.title(f"{ticker} — Actual vs Predicted Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        path = os.path.join(output_dir, f"{ticker}_actual_vs_predicted.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Chart saved → {path}")

def plot_chronos_forecast(ticker: str, config: dict):
    """
    Generates a 20-day Chronos forecast chart with confidence intervals.
    Shows historical context + full probabilistic forecast bands.
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from src.models.chronos_model import load_chronos, get_context, run_forecast, build_forecast_df

    print(f"  Generating Chronos 20-day forecast chart for {ticker}...")

    pipeline = load_chronos(config)
    context, last_close, last_date, history_dates = get_context(ticker, config)

    forecast = run_forecast(pipeline, context, config)
    df       = build_forecast_df(
        last_date, last_close, forecast,
        config["chronos"]["prediction_length"]
    )

    output_dir = config["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6))

    # --- Plot historical context (last 60 days) ---
    raw_dir  = config["data"]["raw_dir"]
    raw_path = os.path.join(raw_dir, f"{ticker}.csv")
    raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    history = raw_df["Close"].dropna().iloc[-config["chronos"]["context_length"]:]

    ax.plot(history.index, history.values,
            color="steelblue", linewidth=1.5,
            label="Historical Close")

    # --- Mark last known close ---
    ax.axvline(x=last_date, color="gray",
               linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate("Today", xy=(last_date, last_close),
                xytext=(10, 10), textcoords="offset points",
                fontsize=8, color="gray")

    # --- Plot 90% confidence band ---
    ax.fill_between(df["date"], df["low_90"], df["high_90"],
                    alpha=0.15, color="darkorange",
                    label="90% Confidence")

    # --- Plot 80% confidence band ---
    ax.fill_between(df["date"], df["low_80"], df["high_80"],
                    alpha=0.25, color="darkorange",
                    label="80% Confidence")

    # --- Plot median forecast ---
    ax.plot(df["date"], df["median"],
            color="darkorange", linewidth=2,
            linestyle="--", label="Median Forecast",
            marker="o", markersize=3)

    ax.set_title(f"{ticker} — Chronos T5 20-Day Forecast with Confidence Intervals",
                 fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{ticker}_chronos_20day_forecast.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved → {path}")

def evaluate(config: dict):
    output_dir = config["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading test data...")
    X_test, y_test = load_test_data(config)

    print("\nLoading model...")
    model = load_trained_model(config)

    print("\nRunning predictions...")
    y_pred = run_predictions(model, X_test)

    print("\nEvaluation Metrics")
    metrics = compute_metrics(y_test, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nGenerating charts...")
    plot_returns(y_test, y_pred, output_dir)
    plot_close_price(config, model, output_dir)

    print(f"\n✓ Evaluation complete. Charts saved → {output_dir}")
    # --- Chronos 20-day forecast charts ---
    print("\nGenerating Chronos 20-day forecast charts...")
    for ticker in config["data"]["tickers"]:
        plot_chronos_forecast(ticker, config)


if __name__ == "__main__":
    config = load_config()
    evaluate(config)