import yfinance as yf
import pandas as pd
import os
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_stock(ticker: str, period: str, interval: str, raw_dir: str) -> pd.DataFrame:
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    save_path = os.path.join(raw_dir, f"{ticker}.csv")
    df.to_csv(save_path)
    print(f"  Saved {len(df)} rows → {save_path}")
    return df


def fetch_all(config: dict) -> dict:
    tickers  = config["data"]["tickers"]
    period   = config["data"]["period"]
    interval = config["data"]["interval"]
    raw_dir  = config["data"]["raw_dir"]

    os.makedirs(raw_dir, exist_ok=True)

    data = {}
    for ticker in tickers:
        data[ticker] = fetch_stock(ticker, period, interval, raw_dir)

    print(f"\n✓ Fetched {len(tickers)} tickers: {', '.join(tickers)}")
    return data


if __name__ == "__main__":
    config = load_config()
    fetch_all(config)