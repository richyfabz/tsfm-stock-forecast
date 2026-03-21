# TSFM Stock Forecast

> A production-grade stock forecasting system combining a **custom PyTorch Transformer**, a **traditional LSTM baseline**, and **Amazon's Chronos T5** pretrained foundation model,delivering next-day and 20-day probabilistic price forecasts for GOOG, TSLA, and SPY through an interactive Streamlit dashboard and a Rich-formatted CLI.

## Dashboard Preview

The Streamlit dashboard lets users select any combination of ticker and model from the sidebar. Each model speaks through its own charts and metrics — no written comparison, no bias. Results speak for themselves.

```bash
streamlit run app.py
```


## Architecture — Three Models

This project is built across three model layers that share the same data pipeline but use fundamentally different architectures.


### Layer 1 — Custom Transformer (Trained From Scratch)

A `StockTransformer` built entirely with PyTorch's `nn.TransformerEncoder`. Trained on 8 engineered technical indicator features across all 3 tickers.

```
Input: (batch, 60, 8)           60-day window × 8 technical features
  ↓ nn.Linear(8 → 64)           Project features to model dimension
  ↓ PositionalEncoding(64)       Stamp time order into each step
  ↓ TransformerEncoder (2 layers, 4 heads)
  ↓ x[:, -1, :]                  Extract last time step
  ↓ nn.Linear(64 → 32 → 1)      Collapse to single predicted return
Output: (batch,)                 Next-day return prediction
```

**Model stats:** 102,657 trainable parameters | 2 encoder layers | 4 attention heads | d_model=64


### Layer 2 — LSTM Baseline (Traditional Sequential Model)

A `StockLSTM` built with PyTorch's `nn.LSTM`. Reads the sequence step-by-step using forget, input, and output gates. Trained on the same data and identical hyperparameters as the transformer for a fair architectural comparison.

```
Input: (batch, 60, 8)           60-day window × 8 technical features
  ↓ nn.LSTM(8 → 64, layers=2)   Sequential gate processing
  ↓ lstm_out[:, -1, :]           Last hidden state
  ↓ nn.Linear(64 → 32 → 1)      Collapse to single predicted return
Output: (batch,)                 Next-day return prediction
```

**Model stats:** 54,337 trainable parameters | 2 LSTM layers | hidden_size=64

### Layer 3 — Chronos T5 (Amazon Foundation Model)

A pretrained time-series transformer from Amazon, built on the T5 encoder-decoder architecture. Pretrained on millions of real-world time-series datasets. Produces full probabilistic 20-day forecasts natively.

```
Input: last 60 raw closing prices
  ↓ Chronos T5 Small (pretrained, 185MB)
  ↓ Tokenises prices into discrete bins
  ↓ T5 encoder-decoder processes sequence
  ↓ 20 stochastic samples generated
Output: 20-day forecast + 80% and 90% confidence intervals
```
## Project Structure

```
tsfm-stock-forecast/
│
├── configs/
│   └── config.yaml                  # All hyperparameters in one place
│
├── data/
│   ├── raw/                         # OHLCV CSVs from Yahoo Finance
│   └── processed/                   # Features, X.npy, y.npy, test splits
│
├── models/
│   ├── best_model.pt                # Saved transformer weights
│   └── best_lstm.pt                 # Saved LSTM weights
│
├── outputs/                         # Generated forecast charts (.png)
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py            # Pull OHLCV from Yahoo Finance
│   │   └── build_sequences.py       # Sliding window → X.npy, y.npy
│   ├── features/
│   │   └── features.py              # 8 technical indicator features
│   ├── models/
│   │   ├── transformer_model.py     # Custom transformer architecture
│   │   ├── lstm_model.py            # LSTM architecture
│   │   └── chronos_model.py         # Chronos T5 inference pipeline
│   ├── training/
│   │   ├── train.py                 # Transformer training loop
│   │   └── train_lstm.py            # LSTM training loop
│   └── evaluation/
│       └── evaluate_model.py        # Metrics + chart generation
│
├── scripts/
│   └── run_forecast.py              # CLI entry point (Rich formatted)
│
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Data Pipeline

```
Yahoo Finance API
      ↓
fetch_data.py        →  data/raw/GOOG.csv, TSLA.csv, SPY.csv
      ↓
features.py          →  8 technical indicators per row
      ↓
build_sequences.py   →  X.npy (1224, 60, 8)   y.npy (1224,)
      ↓
      ┌──────────────────┬─────────────────┬──────────────────┐
      ↓                  ↓                 ↓                  
  train.py          train_lstm.py    chronos_model.py   
  Transformer        LSTM trains      Pretrained         
  trains             same data        zero-shot          
  best_model.pt      best_lstm.pt     confidence bands   
      └──────────────────┴─────────────────┴──────────────────┘
                                ↓
                   app.py — Streamlit Dashboard
                   scripts/run_forecast.py — CLI
```

## Features Engineered

All scratch models (Transformer + LSTM) train on 8 technical indicators:

| Feature | Type | Description |
|---|---|---|
| `return` | Momentum | Daily % price change |
| `log_return` | Momentum | Log of price ratio — statistically stable |
| `rsi` | Momentum | Relative Strength Index (14-day) |
| `macd` | Trend | 12-day vs 26-day EMA difference |
| `macd_signal` | Trend | 9-day EMA of MACD |
| `bb_width` | Volatility | Bollinger Band width (20-day) |
| `volume_ma_ratio` | Volume | Volume relative to 20-day average |
| `close_ma_ratio` | Trend | Close relative to 20-day moving average |

**Target:** next day's return (`return.shift(-1)`)

> Chronos T5 uses raw closing prices directly — no feature engineering needed.


## Model Training

```
Data split (chronological — never shuffled):
├── Train:      70%   (857 samples)
├── Validation: 15%   (183 samples)
└── Test:       15%   (184 samples)

Training config (identical for Transformer and LSTM):
├── Epochs:            20 (early stopping patience: 5)
├── Batch size:        32
├── Learning rate:     1e-4 (Adam optimiser)
├── LR scheduler:      ReduceLROnPlateau (factor 0.5, patience 3)
├── Gradient clipping: max_norm=1.0
└── Loss function:     MSELoss (regression)

Training outcomes:
├── Transformer → ran all 20 epochs | best val loss: 0.000192
└── LSTM        → early stop epoch 7 | best val loss: 0.000224
```

---

## Streamlit Dashboard

The dashboard is fully user-driven. The user controls ticker and model from the sidebar.


## CLI Usage

```bash
python scripts/run_forecast.py --ticker GOOG
python scripts/run_forecast.py --ticker TSLA
python scripts/run_forecast.py --ticker SPY
```

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/richyfabz/tsfm-stock-forecast.git
cd tsfm-stock-forecast
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### 2. Run the full pipeline
```bash
python src/data/fetch_data.py
python src/features/features.py
python src/data/build_sequences.py
python src/training/train.py
python src/training/train_lstm.py
python src/evaluation/evaluate_model.py
```

### 3. Launch dashboard
```bash
streamlit run app.py
```

### 4. Use the CLI
```bash
python scripts/run_forecast.py --ticker GOOG
```


## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch** | Transformer + LSTM architecture and training |
| **Chronos T5** | Amazon pretrained time-series foundation model |
| **yfinance** | Stock data fetching (Yahoo Finance) |
| **pandas / numpy** | Data manipulation and sequence building |
| **ta** | Technical indicators (RSI, MACD, Bollinger) |
| **scikit-learn** | Evaluation metrics |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts with hover and zoom |
| **Rich** | CLI formatting and display |
| **Typer** | CLI argument parsing |
| **PyYAML** | Config file management |


## Key Concepts Applied

**Positional Encoding** — Transformers process all 60 time steps simultaneously. Sine/cosine positional encoding injects temporal order so the model distinguishes day 1 from day 60.

**LSTM Gates** — Forget gate erases from cell state, input gate writes new information to cell state, output gate exposes cell state as hidden state. Cell state = long-term memory. Hidden state = working memory.

**Sliding Window Sequences** — 60-day overlapping windows create 408 training samples per ticker × 3 tickers = 1,224 total.

**Chronological Splitting** — Never shuffle time-series. Train on the past, test on the future. Prevents data leakage.

**Probabilistic Forecasting** — Chronos generates 20 sampled futures. 80% confidence interval means 80% of samples fell within that range. Bands widen with forecast horizon — honest uncertainty.

**Target Engineering** — `return.shift(-1)` ensures today's features predict tomorrow's return, not today's (which would be data leakage).

**Regression Task** — Continuous return prediction with MSELoss. Directional Accuracy is a classification-style metric applied on top via `np.sign()`.

**Transfer Learning** — Chronos pretrained on millions of time-series. Your stocks fine-tune this broad knowledge into domain-specific predictions.


## ⚠️ Disclaimer

For **educational purposes only**. Not financial advice. Models do not account for earnings, news, macroeconomic events, or fundamental analysis.


## Author

**Fabunmi Richard**
- GitHub: [@richyfabz](https://github.com/richyfabz)
- LinkedIn: [Fabunmi Richard](https://www.linkedin.com/in/fabunmi-richard-a686ab23b/)
- Twitter/X: [@damilola356075](https://x.com/damilola356075)
- Email: dammifabz@gmail.com

---

*Built in one day — transformer architecture, LSTM baseline, Chronos T5 foundation model, Streamlit dashboard, Rich CLI. Python · PyTorch · Chronos T5 · Streamlit · Plotly.*