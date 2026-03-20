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
from src.models.lstm_model import build_lstm
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
# CONFIG
# ─────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# MODEL LOADERS (cached — runs once each)
# ─────────────────────────────────────────────
@st.cache_resource
def get_chronos(_config):
    return load_chronos(_config)


@st.cache_resource
def get_transformer(_config):
    model = build_model(_config)
    path  = os.path.join(_config["training"]["model_save_path"], "best_model.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def get_lstm(_config):
    model = build_lstm(_config)
    path  = os.path.join(_config["training"]["model_save_path"], "best_lstm.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def fetch_chronos_forecast(ticker: str, _config, _pipeline):
    context, last_close, last_date, history_dates = get_context(ticker, _config)
    forecast_raw = run_forecast(_pipeline, context, _config)
    forecast_df  = build_forecast_df(
        last_date, last_close, forecast_raw,
        _config["chronos"]["prediction_length"]
    )
    return forecast_df, last_close, last_date, context, history_dates


@st.cache_data
def fetch_scratch_predictions(ticker: str, _config, _model):
    """Run scratch model (transformer or LSTM) on ticker test sequences."""
    processed_dir  = _config["data"]["processed_dir"]
    context_length = _config["model"]["context_length"]
    test_split     = _config["training"]["test_split"]

    feat_path = os.path.join(processed_dir, f"{ticker}_features.csv")
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    feature_cols = [
        "return", "log_return", "rsi", "macd",
        "macd_signal", "bb_width", "volume_ma_ratio", "close_ma_ratio"
    ]

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
        return None, None, None, None

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

    # Metrics
    y_true = np.array(actual)
    y_pred = np.array(pred_prices)
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Directional accuracy on returns
    X_test  = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(processed_dir, "y_test.npy"))
    Xt      = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        yp = _model(Xt).numpy()
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(yp)))

    metrics = {"mae": mae, "rmse": rmse, "dir_acc": dir_acc}
    return dates, actual, pred_prices, metrics


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def chart_chronos(ticker, forecast_df, last_close, last_date):
    raw_path = f"data/raw/{ticker}.csv"
    raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    history = raw_df["Close"].dropna().iloc[-60:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        mode="lines", name="Historical Close",
        line=dict(color="#4C9BE8", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["high_90"], forecast_df["low_90"][::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Confidence"
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["high_80"], forecast_df["low_80"][::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.22)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["median"],
        mode="lines+markers", name="Median Forecast",
        line=dict(color="orange", width=2, dash="dash"),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=[last_date, last_date],
        y=[history.values.min() * 0.98, history.values.max() * 1.02],
        mode="lines", name="Today",
        line=dict(color="gray", width=1, dash="dot")
    ))

    fig.update_layout(
        title=f"{ticker} — Chronos T5 20-Day Forecast with Confidence Intervals",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_scratch(ticker, dates, actual, predicted, model_name):
    color = "orange" if model_name == "Transformer" else "#2ecc71"
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines", name="Actual Close",
        line=dict(color="#4C9BE8", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode="lines", name=f"{model_name} Predicted",
        line=dict(color=color, width=1.5, dash="dash")
    ))

    fig.update_layout(
        title=f"{ticker} — {model_name}: Actual vs Predicted Close Price (Test Set)",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(config):
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
        st.title("TSFM Forecast")
        st.caption("Powered by Chronos T5 + PyTorch")
        st.divider()

        # Ticker selector
        st.subheader("📊 Select Ticker")
        ticker = st.selectbox(
            "Stock",
            options=config["data"]["tickers"],
            format_func=lambda x: {
                "GOOG": "GOOG — Alphabet Inc.",
                "TSLA": "TSLA — Tesla Inc.",
                "SPY":  "SPY  — S&P 500 ETF"
            }.get(x, x)
        )

        st.divider()

        # Model selector
        st.subheader("🧠 Select Model")
        model_choice = st.selectbox(
            "Forecasting Model",
            options=["Chronos T5", "Transformer", "LSTM"],
            format_func=lambda x: {
                "Chronos T5":  "Chronos T5 — Amazon (pretrained)",
                "Transformer": "Transformer — Custom PyTorch",
                "LSTM":        "LSTM — Traditional Sequential"
            }.get(x, x)
        )

        st.divider()

        # Model info card
        if model_choice == "Chronos T5":
            st.markdown("**About this model**")
            st.markdown("""
- Pretrained by Amazon on millions of time-series
- Probabilistic — returns confidence intervals
- 20-day multi-step forecast
- Input: raw closing prices
            """)
        elif model_choice == "Transformer":
            st.markdown("**About this model**")
            st.markdown("""
- Built from scratch with PyTorch
- Self-attention across all 60 days
- 102,657 trainable parameters
- Input: 8 technical indicators
            """)
        else:
            st.markdown("**About this model**")
            st.markdown("""
- Traditional sequential architecture
- Reads sequence step by step
- 54,337 trainable parameters
- Input: 8 technical indicators
            """)

        st.divider()
        st.caption("⚠️ For educational purposes only. Not financial advice.")

        return ticker, model_choice


# ─────────────────────────────────────────────
# RENDER CHRONOS
# ─────────────────────────────────────────────
def render_chronos(ticker, config, pipeline):
    forecast_df, last_close, last_date, context, _ = fetch_chronos_forecast(
        ticker, config, pipeline
    )

    day1  = forecast_df.iloc[0]
    day20 = forecast_df.iloc[-1]
    overall_change = ((day20["median"] - last_close) / last_close) * 100
    day1_change    = ((day1["median"]  - last_close) / last_close) * 100

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close",       f"${last_close:.2f}")
    m2.metric("Tomorrow",         f"${day1['median']:.2f}",
              delta=f"{day1_change:+.2f}%")
    m3.metric("Day 20 Forecast",  f"${day20['median']:.2f}",
              delta=f"{overall_change:+.2f}%")
    m4.metric("80% Range Day 20",
              f"${day20['low_80']:.0f} – ${day20['high_80']:.0f}")

    # Tabs
    tab1, tab2 = st.tabs(["📅 Forecast Chart", "📋 Forecast Table"])

    with tab1:
        st.plotly_chart(
            chart_chronos(ticker, forecast_df, last_close, last_date),
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


# ─────────────────────────────────────────────
# RENDER SCRATCH MODEL (TRANSFORMER OR LSTM)
# ─────────────────────────────────────────────
def render_scratch(ticker, config, model, model_name):
    dates, actual, predicted, metrics = fetch_scratch_predictions(
        ticker, config, model
    )

    if dates is None:
        st.warning("Not enough test data to generate forecast for this ticker.")
        return

    # Last known close
    raw_path = os.path.join(config["data"]["raw_dir"], f"{ticker}.csv")
    raw_df   = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    last_close = float(raw_df["Close"].iloc[-1])
    last_pred  = predicted[-1]
    pred_change = ((last_pred - last_close) / last_close) * 100

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close",          f"${last_close:.2f}")
    m2.metric("Latest Prediction",   f"${last_pred:.2f}",
              delta=f"{pred_change:+.2f}%")
    m3.metric("MAE",                 f"{metrics['mae']:.4f}")
    m4.metric("Directional Accuracy",f"{metrics['dir_acc']*100:.1f}%")

    # Chart
    st.plotly_chart(
        chart_scratch(ticker, dates, actual, predicted, model_name),
        use_container_width=True
    )

    # Metrics expander
    with st.expander("📐 Full Model Metrics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",                 f"{metrics['mae']:.4f}")
        c2.metric("RMSE",                f"{metrics['rmse']:.4f}")
        c3.metric("Directional Accuracy",f"{metrics['dir_acc']*100:.1f}%")
        st.caption(
            "MAE and RMSE measure prediction error on the held-out test set. "
            "Directional Accuracy measures how often the model correctly "
            "predicted whether the price would go up or down."
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    config = load_config()

    # Header
    st.title("📈 TSFM Stock Forecast")
    st.caption(
        "Select a stock and a forecasting model from the sidebar to view predictions."
    )
    st.divider()

    # Sidebar — get user selections
    ticker, model_choice = render_sidebar(config)

    # Load selected model
    if model_choice == "Chronos T5":
        with st.spinner("Loading Chronos T5..."):
            pipeline = get_chronos(config)
    elif model_choice == "Transformer":
        with st.spinner("Loading Transformer..."):
            model = get_transformer(config)
    else:
        with st.spinner("Loading LSTM..."):
            model = get_lstm(config)

    # Ticker + model header
    ticker_names = {
        "GOOG": "Alphabet Inc. (GOOG)",
        "TSLA": "Tesla Inc. (TSLA)",
        "SPY":  "S&P 500 ETF (SPY)"
    }
    st.subheader(f"{ticker_names[ticker]} — {model_choice}")
    st.divider()

    # Render selected model output
    with st.spinner(f"Running {model_choice} forecast for {ticker}..."):
        if model_choice == "Chronos T5":
            render_chronos(ticker, config, pipeline)
        elif model_choice == "Transformer":
            render_scratch(ticker, config, model, "Transformer")
        else:
            render_scratch(ticker, config, model, "LSTM")


if __name__ == "__main__":
    main()