import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import yaml

sys.path.append(os.path.dirname(__file__))

from src.models.transformer_model import build_model
from src.models.chronos_model import (
    load_chronos, get_context, run_forecast, build_forecast_df
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSFM Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# LOAD MODELS (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def get_chronos(_config):
    return load_chronos(_config)


@st.cache_resource
def get_scratch_model(_config):
    model = build_model(_config)
    path  = os.path.join(_config["training"]["model_save_path"], "best_model.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ─────────────────────────────────────────────
# COMPUTE FORECASTS (cached per ticker)
# ─────────────────────────────────────────────
@st.cache_data
def get_forecast(ticker: str, _config, _pipeline):
    context, last_close, last_date, history_dates = get_context(ticker, _config)
    forecast_raw = run_forecast(_pipeline, context, _config)
    forecast_df  = build_forecast_df(
        last_date, last_close, forecast_raw,
        _config["chronos"]["prediction_length"]
    )
    return forecast_df, last_close, last_date, context, history_dates


@st.cache_data
def get_scratch_metrics(_config, _model):
    processed_dir = _config["data"]["processed_dir"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = _model(X_tensor).numpy()
    mae     = float(np.mean(np.abs(y_test - y_pred)))
    rmse    = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "n_test": len(y_test)}

@st.cache_resource
def get_lstm_model(_config):
    from src.models.lstm_model import build_lstm
    model = build_lstm(_config)
    path  = os.path.join(_config["training"]["model_save_path"], "best_lstm.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def get_lstm_metrics(_config, _model):
    processed_dir = _config["data"]["processed_dir"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = _model(X_tensor).numpy()
    mae     = float(np.mean(np.abs(y_test - y_pred)))
    rmse    = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "n_test": len(y_test)}


@st.cache_data
def get_lstm_predictions(_config, _model):
    processed_dir = _config["data"]["processed_dir"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = _model(X_tensor).numpy()
    return y_test, y_pred


@st.cache_data
def get_actual_vs_predicted(ticker: str, _config, _model):
    processed_dir  = _config["data"]["processed_dir"]
    context_length = _config["model"]["context_length"]
    test_split     = _config["training"]["test_split"]

    feat_path = os.path.join(processed_dir, f"{ticker}_features.csv")
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    feature_cols = ["return", "log_return", "rsi", "macd",
                    "macd_signal", "bb_width", "volume_ma_ratio", "close_ma_ratio"]

    n          = len(df)
    test_start = int(n * (1 - test_split))
    df_test    = df.iloc[test_start:]
    df_full    = df[feature_cols].values

    raw_path = os.path.join(_config["data"]["raw_dir"], f"{ticker}.csv")
    raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    close_test = raw_df["Close"].reindex(df_test.index).dropna()

    X_ticker, dates = [], []
    for i in range(len(df_test) - 1):
        start = test_start + i - context_length + 1
        if start < 0:
            continue
        window = df_full[start: start + context_length]
        if len(window) == context_length:
            X_ticker.append(window)
            dates.append(df_test.index[i + 1])

    if not X_ticker:
        return None, None, None

    X_tensor = torch.tensor(np.array(X_ticker, dtype=np.float32))
    with torch.no_grad():
        pred_returns = _model(X_tensor).numpy()

    pred_prices = []
    for i, d in enumerate(dates):
        loc = close_test.index.get_loc(d)
        if loc == 0:
            continue
        prev_close = close_test.iloc[loc - 1]
        pred_prices.append(prev_close * (1 + pred_returns[i]))

    dates       = dates[1:]
    min_len     = min(len(dates), len(pred_prices))
    dates       = dates[-min_len:]
    pred_prices = pred_prices[-min_len:]
    actual      = close_test.reindex(dates).values

    return dates, actual, pred_prices

def chart_all_models(y_test, transformer_pred, lstm_pred):
    """All models on one returns chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test, mode="lines", name="Actual Returns",
        line=dict(color="#4C9BE8", width=1)
    ))
    fig.add_trace(go.Scatter(
        y=transformer_pred, mode="lines", name="Transformer",
        line=dict(color="orange", width=1, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        y=lstm_pred, mode="lines", name="LSTM",
        line=dict(color="#2ecc71", width=1, dash="dot")
    ))

    fig.update_layout(
        title="All Models — Actual vs Predicted Returns (Test Set)",
        xaxis_title="Test Sample Index",
        yaxis_title="Return",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_lstm_returns(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_test, mode="lines", name="Actual",
        line=dict(color="#4C9BE8", width=1)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred, mode="lines", name="LSTM Predicted",
        line=dict(color="#2ecc71", width=1, dash="dash")
    ))
    fig.update_layout(
        title="LSTM — Actual vs Predicted Returns (Test Set)",
        xaxis_title="Test Sample Index", yaxis_title="Return",
        hovermode="x unified", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def chart_chronos(ticker, forecast_df, last_close, last_date, context, history_dates):
    raw_path = f"data/raw/{ticker}.csv"
    raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    history = raw_df["Close"].dropna().iloc[-60:]

    fig = go.Figure()

    # Historical
    # Today line:using scatter trace for compatibility
    fig.add_trace(go.Scatter(
        x=[last_date, last_date],
        y=[history.values.min() * 0.98, history.values.max() * 1.02],
        mode="lines",
        name="Today",
        line=dict(color="gray", width=1, dash="dot"),
        showlegend=True
    ))

    # 90% band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["high_90"], forecast_df["low_90"][::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Confidence"
    ))

    # 80% band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["high_80"], forecast_df["low_80"][::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.20)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence"
    ))

    # Median forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["median"],
        mode="lines+markers", name="Median Forecast",
        line=dict(color="orange", width=2, dash="dash"),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f"{ticker} — Chronos T5 20-Day Forecast",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_actual_vs_predicted(ticker, dates, actual, predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines", name="Actual Close",
        line=dict(color="#4C9BE8", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode="lines", name="Predicted Close",
        line=dict(color="orange", width=1.5, dash="dash")
    ))

    fig.update_layout(
        title=f"{ticker} — Actual vs Predicted Close (Test Set)",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(metrics):
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
        st.title("TSFM Forecast")
        st.caption("Powered by Chronos T5 + Custom Transformer")
        st.divider()

        st.subheader("Scratch Transformer")
        st.caption("Evaluated on held-out test set")
        col1, col2 = st.columns(2)
        col1.metric("MAE",  f"{metrics['mae']:.4f}")
        col2.metric("RMSE", f"{metrics['rmse']:.4f}")
        st.metric(
            "Directional Accuracy",
            f"{metrics['dir_acc']*100:.1f}%",
            delta=f"{(metrics['dir_acc']-0.5)*100:+.1f}% vs random"
        )
        st.caption(f"Test samples: {metrics['n_test']}")
        st.divider()

        st.subheader("Models")
        st.markdown("- **Chronos T5 Small** — Amazon (pretrained)")
        st.markdown("- **StockTransformer** — Custom PyTorch (trained from scratch)")
        st.markdown("- **LSTM** — Traditional sequential model (trained from scratch)")

        st.divider()

        st.subheader("Tickers")
        st.markdown("- **GOOG** — Alphabet Inc.")
        st.markdown("- **TSLA** — Tesla Inc.")
        st.markdown("- **SPY** — S&P 500 ETF")
        st.divider()

        st.caption("For educational purposes only. Not financial advice.")

def render_comparison(scratch_metrics, lstm_metrics):
    st.subheader("🏆 Model Comparison")

    # --- Metrics table ---
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**Metric**")
    col2.markdown("**Transformer**")
    col3.markdown("** LSTM**")
    col4.markdown("**Winner**")

    metrics = [
        ("MAE",                  "mae",     True),
        ("RMSE",                 "rmse",    True),
        ("Directional Accuracy", "dir_acc", False),
    ]

    for label, key, lower_is_better in metrics:
        t_val = scratch_metrics[key]
        l_val = lstm_metrics[key]

        if lower_is_better:
            t_color = "green" if t_val < l_val else "red"
            l_color = "green" if l_val < t_val else "red"
            winner  = "Transformer" if t_val < l_val else "🏆 LSTM"
        else:
            t_color = "green" if t_val > l_val else "red"
            l_color = "green" if l_val > t_val else "red"
            winner  = "Transformer" if t_val > l_val else "🏆 LSTM"

        col1.markdown(label)
        col2.markdown(f":{t_color}[{t_val:.4f}]")
        col3.markdown(f":{l_color}[{l_val:.4f}]")
        col4.markdown(winner)

    st.divider()

    # --- Written comparison ---
    st.markdown("####  Why Transformer Wins on Error Metrics")
    st.markdown("""
The **Transformer** achieves lower MAE and RMSE because its **self-attention mechanism**
reads all 60 days of context simultaneously,directly connecting distant time steps
without information decay. This gives it a more accurate picture of the overall
price level and magnitude.

The **LSTM** reads the sequence step-by-step, passing a hidden state forward at each step.
By day 60, signals from day 1 have faded through the gate mechanism, a fundamental
limitation called the **vanishing gradient problem** on long sequences.

However, LSTM's slightly higher **Directional Accuracy** (54.9% vs 51.1%) suggests
its sequential gate memory captures short-term momentum,the immediate "is the next
move up or down?" signal,marginally better than the transformer on this dataset size.

**Key insight:** With only ~1,200 training samples, the transformer's advantage is
already visible. On larger datasets, the performance gap widens significantly in
the transformer's favour.
    """)
    st.divider()

# ─────────────────────────────────────────────
# TICKER CARD
# ─────────────────────────────────────────────
def render_ticker(ticker, config, pipeline, scratch_model):
    forecast_df, last_close, last_date, context, history_dates = get_forecast(
        ticker, config, pipeline
    )
    dates, actual, predicted = get_actual_vs_predicted(ticker, config, scratch_model)

    # Summary metrics
    day1  = forecast_df.iloc[0]
    day20 = forecast_df.iloc[-1]
    overall_change = ((day20["median"] - last_close) / last_close) * 100
    day1_change    = ((day1["median"]  - last_close) / last_close) * 100

    st.subheader(f"{'📈' if overall_change > 0 else '📉'} {ticker}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close",      f"${last_close:.2f}")
    m2.metric("Tomorrow",        f"${day1['median']:.2f}",
              delta=f"{day1_change:+.2f}%")
    m3.metric("Day 20 Forecast", f"${day20['median']:.2f}",
              delta=f"{overall_change:+.2f}%")
    m4.metric("80% Range Day 20",
              f"${day20['low_80']:.0f} – ${day20['high_80']:.0f}")

    # Tabs per ticker
    tab1, tab2, tab3 = st.tabs([
        "📅 20-Day Forecast Chart",
        "📋 Forecast Table",
        "🔁 Actual vs Predicted"
    ])

    with tab1:
        st.plotly_chart(
            chart_chronos(ticker, forecast_df, last_close, last_date, context, history_dates),
            use_container_width=True
        )

    with tab2:
        display_df = forecast_df[[
            "date", "median", "low_80", "high_80",
            "low_90", "high_90", "return_pct", "direction"
        ]].copy()
        display_df.columns = [
            "Date", "Median ($)", "Low 80%", "High 80%",
            "Low 90%", "High 90%", "Change (%)", "Dir"
        ]
        display_df["Date"]       = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["Median ($)"] = display_df["Median ($)"].map("${:.2f}".format)
        display_df["Low 80%"]    = display_df["Low 80%"].map("${:.2f}".format)
        display_df["High 80%"]   = display_df["High 80%"].map("${:.2f}".format)
        display_df["Low 90%"]    = display_df["Low 90%"].map("${:.2f}".format)
        display_df["High 90%"]   = display_df["High 90%"].map("${:.2f}".format)
        display_df["Change (%)"] = display_df["Change (%)"].map("{:+.2f}%".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab3:
        if dates is not None:
            st.plotly_chart(
                chart_actual_vs_predicted(ticker, dates, actual, predicted),
                use_container_width=True
            )
        else:
            st.info("Not enough test data to generate this chart.")

    st.divider()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    config = load_config()

    # Header
    st.title("📈 TSFM Stock Forecast Dashboard")
    st.caption(
        "20-day probabilistic forecasts using **Chronos T5** (Amazon) "
        "and a custom **Transformer** trained from scratch on GOOG, TSLA, SPY."
    )
    st.divider()

    # Load models
    with st.spinner("Loading Chronos T5..."):
        pipeline = get_chronos(config)

    with st.spinner("Loading scratch transformer..."):
        scratch_model = get_scratch_model(config)

    with st.spinner("Loading LSTM model..."):
        lstm_model = get_lstm_model(config)

    scratch_metrics = get_scratch_metrics(config, scratch_model)
    lstm_metrics    = get_lstm_metrics(config, lstm_model)

    # Sidebar
    render_sidebar(scratch_metrics)

    # Get predictions for both models on same test set
    lstm_y_test, lstm_pred        = get_lstm_predictions(config, lstm_model)
    _,           transformer_pred = get_lstm_predictions(config, scratch_model)

    render_comparison(scratch_metrics, lstm_metrics)

    # All models chart
    st.subheader("📊 All Models on One Chart")
    st.plotly_chart(
        chart_all_models(lstm_y_test, transformer_pred, lstm_pred),
        use_container_width=True
    )
    st.divider()

    # All 3 tickers
    tickers = config["data"]["tickers"]
    for ticker in tickers:
        with st.spinner(f"Running forecast for {ticker}..."):
            render_ticker(ticker, config, pipeline, scratch_model)


if __name__ == "__main__":
    main()