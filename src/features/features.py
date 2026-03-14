import pandas as pd
import numpy as np
import ta
import os
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=df.index)

    # --- Feature 1: Daily Return ---
    # % change from previous close — this is what the model predicts
    feature_df["return"] = df["Close"].pct_change()

    # --- Feature 2: Log Return ---
    # More statistically stable than raw return, used in quant finance
    feature_df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- Feature 3: RSI (Relative Strength Index) ---
    # Momentum indicator: 0-100. Above 70 = overbought, below 30 = oversold
    feature_df["rsi"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=14
    ).rsi()

    # --- Feature 4 & 5: MACD and MACD Signal ---
    # Trend indicator: difference between 12-day and 26-day moving averages
    macd = ta.trend.MACD(close=df["Close"])
    feature_df["macd"]        = macd.macd()
    feature_df["macd_signal"] = macd.macd_signal()

    # --- Feature 6: Bollinger Band Width ---
    # Volatility indicator: how wide the price bands are
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20)
    feature_df["bb_width"] = bb.bollinger_wband()

    # --- Feature 7: Volume Moving Average (normalised) ---
    # Relative volume vs 20-day average — spikes signal big moves
    feature_df["volume_ma_ratio"] = (
        df["Volume"] / df["Volume"].rolling(20).mean()
    )

    # --- Feature 8: Close normalised by 20-day MA ---
    # Where is price relative to its recent average?
    feature_df["close_ma_ratio"] = (
        df["Close"] / df["Close"].rolling(20).mean()
    )

    # --- Feature 9: Target — next day's return ---
    # Shift return back by 1: today's features predict tomorrow's return
    feature_df["target"] = feature_df["return"].shift(-1)

    # Drop rows with NaN (from rolling windows and indicators warming up)
    feature_df.dropna(inplace=True)

    return feature_df


def build_all_features(config: dict) -> dict:
    raw_dir       = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    tickers       = config["data"]["tickers"]

    os.makedirs(processed_dir, exist_ok=True)

    all_features = {}
    for ticker in tickers:
        print(f"  Building features for {ticker}...")
        raw_path = os.path.join(raw_dir, f"{ticker}.csv")
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        features = build_features(df)
        save_path = os.path.join(processed_dir, f"{ticker}_features.csv")
        features.to_csv(save_path)
        print(f"  {len(features)} rows, {len(features.columns)} features → {save_path}")
        all_features[ticker] = features

    print(f"\n✓ Features built for: {', '.join(tickers)}")
    return all_features


if __name__ == "__main__":
    config = load_config()
    build_all_features(config)