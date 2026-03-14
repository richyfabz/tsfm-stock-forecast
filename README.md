# TSFM Stock Forecast

> A production-grade stock forecasting system combining a **custom PyTorch Transformer** trained from scratch with **Amazon's Chronos T5** pretrained foundation model — delivering 20-day probabilistic price forecasts for GOOG, TSLA, and SPY.

## Dashboard Preview

The Streamlit dashboard shows all 3 tickers simultaneously with:
- Interactive 20-day forecast charts with confidence bands
- Full forecast tables with 80% and 90% prediction intervals
- Actual vs predicted close price comparison (test set)
- Live model performance metrics in the sidebar

Run locally with:
```bash
streamlit run app.py
```

---

## Architecture

This project is built in two layers that work alongside each other:

### Layer 1 — Custom Transformer (Trained From Scratch)
A `StockTransformer` built entirely with PyTorch's `nn.TransformerEncoder`. Trained on engineered technical indicator features across all 3 tickers.

```
Input: (batch, 60, 8)         60-day window × 8 technical features
  ↓ nn.Linear(8 → 64)         Project features to model dimension
  ↓ PositionalEncoding(64)    Stamp time order into each step
  ↓ TransformerEncoder        Self-attention across all 60 days
  ↓ x[:, -1, :]               Extract last time step representation
  ↓ nn.Linear(64 → 32 → 1)   Collapse to single predicted return
Output: (batch,)              Next-day return prediction
```

**Model stats:** 102,657 trainable parameters | 2 encoder layers | 4 attention heads

### Layer 2 — Chronos T5 (Amazon Foundation Model)
A pretrained time-series transformer from Amazon, fine-tuned on raw closing prices. Produces full probabilistic 20-day forecasts natively.

```
Input: last 60 closing prices (raw)
  ↓ Chronos T5 Small (pretrained on millions of time-series)
  ↓ 20 stochastic samples → median + quantile bands
Output: 20-day forecast with 80% and 90% confidence intervals
```

## Project Structure

```
tsfm-stock-forecast/
│
├── configs/
│   └── config.yaml              # All hyperparameters in one place
│
├── data/
│   ├── raw/                     # OHLCV CSVs from Yahoo Finance
│   └── processed/               # Features, sequences (X.npy, y.npy)
│
├── models/
│   └── best_model.pt            # Saved scratch transformer weights
│
├── outputs/                     # Forecast charts (.png)
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py        # Pull stock data via yfinance
│   │   └── build_sequences.py   # Sliding window → X.npy, y.npy
│   ├── features/
│   │   └── features.py          # Technical indicator engineering
│   ├── models/
│   │   ├── transformer_model.py # Custom transformer architecture
│   │   └── chronos_model.py     # Chronos T5 inference + forecast builder
│   ├── training/
│   │   └── train.py             # Fine-tuning loop with early stopping
│   └── evaluation/
│       └── evaluate_model.py    # Metrics + chart generation
│
├── scripts/
│   └── run_forecast.py          # CLI entry point (Rich-formatted output)
│
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Data Pipeline

```
yfinance API
     ↓
fetch_data.py        →  data/raw/GOOG.csv, TSLA.csv, SPY.csv
     ↓
features.py          →  8 technical indicators per row
     ↓
build_sequences.py   →  X.npy (1224, 60, 8)  y.npy (1224,)
     ↓
train.py             →  models/best_model.pt
     ↓
evaluate_model.py    →  MAE, RMSE, Directional Accuracy + charts
     ↓
run_forecast.py      →  CLI: live prediction per ticker
app.py               →  Streamlit: full interactive dashboard
```

---

## Features Engineered

The scratch transformer trains on 8 technical indicators:

| Feature | Type | Description |
|---|---|---|
| `return` | Momentum | Daily % price change |
| `log_return` | Momentum | Log of daily return — more statistically stable |
| `rsi` | Momentum | Relative Strength Index (14-day) — overbought/oversold |
| `macd` | Trend | MACD line — 12 vs 26-day EMA difference |
| `macd_signal` | Trend | 9-day EMA of MACD |
| `bb_width` | Volatility | Bollinger Band width (20-day) |
| `volume_ma_ratio` | Volume | Volume relative to 20-day moving average |
| `close_ma_ratio` | Trend | Close price relative to 20-day moving average |

**Target:** next day's return (`return.shift(-1)`)

---

## Model Training

```
Data split (chronological — never shuffled):
├── Train:      70%  (857 samples)
├── Validation: 15%  (183 samples)
└── Test:       15%  (184 samples)

Training config:
├── Epochs:           20 (early stopping patience: 5)
├── Batch size:       32
├── Learning rate:    1e-4 (Adam)
├── LR scheduler:     ReduceLROnPlateau (factor 0.5, patience 3)
└── Gradient clipping: max_norm=1.0
```

---

## Results

### Scratch Transformer (Test Set)
| Metric | Value |
|---|---|
| MAE | 0.0057 |
| RMSE | 0.0072 |
| Directional Accuracy | 51.1% |
| Test Samples | 184 |

### Chronos T5 — 20-Day Forecast (GOOG example)
| Horizon | Median | 80% Range |
|---|---|---|
| Day 1 | $307.44 | $300.22 – $310.00 |
| Day 10 | $312.10 | $297.19 – $328.40 |
| Day 20 | $316.76 | $299.29 – $333.30 |

> Confidence bands widen as forecast horizon extends,reflecting honest uncertainty growth over time.


## CLI Usage

```bash
# Forecast GOOG
python scripts/run_forecast.py --ticker GOOG

# Forecast TSLA
python scripts/run_forecast.py --ticker TSLA

# Forecast SPY
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
# Fetch data
python src/data/fetch_data.py

# Build features
python src/features/features.py

# Build sequences
python src/data/build_sequences.py

# Train scratch transformer
python src/training/train.py

# Evaluate both models + generate charts
python src/evaluation/evaluate_model.py
```

### 3. Launch dashboard
```bash
streamlit run app.py
```

### 4. Or use the CLI
```bash
python scripts/run_forecast.py --ticker GOOG
```

---

## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch** | Custom transformer architecture + training loop |
| **Chronos T5** | Amazon pretrained time-series foundation model |
| **yfinance** | Stock data fetching (Yahoo Finance) |
| **pandas / numpy** | Data manipulation and sequence building |
| **ta** | Technical indicator library (RSI, MACD, Bollinger) |
| **scikit-learn** | Evaluation metrics (MAE, RMSE) |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts with hover + zoom |
| **Rich** | CLI formatting and display |
| **Typer** | CLI argument parsing |

---

## Key Concepts Applied

**Positional Encoding** — Transformers process all time steps simultaneously with no inherent sense of order. Sine/cosine positional encoding injects temporal position into each token so the model distinguishes day 1 from day 60.

**Sliding Window Sequences** — The flat time-series is converted into overlapping 60-day windows, each paired with the next day's return as the target. This gives 408 training samples per ticker from ~468 rows of features.

**Chronological Data Splitting** — Time-series data must never be split randomly. Train/val/test sets are sliced chronologically to prevent future data leaking into training — a critical correctness requirement for any time-series model.

**Probabilistic Forecasting** — Chronos produces a distribution over possible futures rather than a single point prediction. The 80% confidence interval means the model expects the true price to fall within that range 80% of the time.

**Transfer Learning** — Chronos was pretrained on millions of diverse time-series datasets. Your stock data fine-tunes this broad knowledge toward GOOG, TSLA, and SPY specifically — achieving better results than training from scratch on 502 rows alone.

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. Forecasts are generated by statistical models and do not account for earnings reports, macroeconomic events, news, or any fundamental analysis. Nothing in this project constitutes financial advice.

---

## 👤 Author

**Fabunmi Richard**
- GitHub: [@richyfabz](https://github.com/richyfabz)
- LinkedIn: [Fabunmi Richard](https://www.linkedin.com/in/fabunmi-richard-a686ab23b/)
- Twitter/X: [@damilola356075](https://x.com/damilola356075)

---

*Built in one day as part of an intensive project-building streak across data engineering, AI engineering, and full-stack development.*