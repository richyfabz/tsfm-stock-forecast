import torch
import numpy as np
import pandas as pd
import os
import sys
import yaml
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.transformer_model import build_model
from src.models.chronos_model import (
    load_chronos, get_context, run_forecast, build_forecast_df
)

app     = typer.Typer()
console = Console()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_scratch_model(config: dict):
    model = build_model(config)
    model_path = os.path.join(config["training"]["model_save_path"], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def get_scratch_metrics(config: dict, model) -> dict:
    processed_dir = config["data"]["processed_dir"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    mae     = float(np.mean(np.abs(y_test - y_pred)))
    rmse    = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))

    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "n_test": len(y_test)}


def display_header(ticker: str, last_close: float, last_date: str):
    console.print()
    console.print(Panel(
        f"[bold white]{ticker} — TSFM Stock Forecast[/bold white]\n"
        f"[dim]Last Close: [bold]${last_close:.2f}[/bold]   "
        f"Date: [bold]{last_date}[/bold][/dim]",
        box=box.DOUBLE,
        style="bold cyan",
        padding=(0, 2)
    ))


def display_chronos_forecast(df: pd.DataFrame, last_close: float):
    console.print(
        Panel("[bold]Chronos T5 — 20-Day Probabilistic Forecast[/bold]",
              style="cyan", box=box.SIMPLE)
    )

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1)
    )

    table.add_column("Day",       width=4,  justify="right")
    table.add_column("Date",      width=12)
    table.add_column("Median",    width=10, justify="right")
    table.add_column("Low 80%",   width=10, justify="right")
    table.add_column("High 80%",  width=10, justify="right")
    table.add_column("Change",    width=8,  justify="right")
    table.add_column("Dir",       width=4,  justify="center")

    for i, row in df.iterrows():
        day_num    = i + 1
        change_pct = row["return_pct"]
        direction  = row["direction"]
        color      = "green" if change_pct >= 0 else "red"
        sign       = "+" if change_pct >= 0 else ""

        table.add_row(
            str(day_num),
            row["date"].strftime("%Y-%m-%d"),
            f"${row['median']:.2f}",
            f"${row['low_80']:.2f}",
            f"${row['high_80']:.2f}",
            f"[{color}]{sign}{change_pct:.2f}%[/{color}]",
            f"[{color}]{direction}[/{color}]",
        )

    console.print(table)

    # Summary row
    day1    = df.iloc[0]
    day20   = df.iloc[-1]
    overall = ((day20["median"] - last_close) / last_close) * 100
    color   = "green" if overall >= 0 else "red"
    sign    = "+" if overall >= 0 else ""

    console.print(
        f"  [dim]20-day outlook:[/dim] "
        f"[{color}][bold]{sign}{overall:.2f}% "
        f"(${last_close:.2f} → ${day20['median']:.2f})[/bold][/{color}]\n"
    )


def display_scratch_metrics(metrics: dict):
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Label", style="dim",   width=24)
    table.add_column("Value", style="white", width=16)

    table.add_row("Model MAE",             f"{metrics['mae']:.4f}")
    table.add_row("Model RMSE",            f"{metrics['rmse']:.4f}")
    table.add_row("Directional Accuracy",  f"{metrics['dir_acc']*100:.1f}%")
    table.add_row("Test Samples",          str(metrics['n_test']))

    console.print(Panel(
        table,
        title="[dim]Scratch Transformer — Test Set Performance[/dim]",
        box=box.SIMPLE
    ))


@app.command()
def forecast(
    ticker: str = typer.Option(
        "GOOG",
        "--ticker", "-t",
        help="Stock ticker to forecast. Options: GOOG, TSLA, SPY"
    )
):
    ticker = ticker.upper()
    config = load_config()

    valid_tickers = config["data"]["tickers"]
    if ticker not in valid_tickers:
        console.print(f"[red]✗ '{ticker}' not supported.[/red]")
        console.print(f"  Available: {', '.join(valid_tickers)}")
        raise typer.Exit()

    # --- Chronos forecast ---
    console.print(f"\n[dim]Loading Chronos T5...[/dim]")
    pipeline = load_chronos(config)

    console.print(f"[dim]Running 20-day forecast for {ticker}...[/dim]")
    context, last_close, last_date, _ = get_context(ticker, config)
    forecast_raw = run_forecast(pipeline, context, config)
    forecast_df  = build_forecast_df(
        last_date, last_close, forecast_raw,
        config["chronos"]["prediction_length"]
    )

    # --- Scratch model metrics ---
    console.print(f"[dim]Loading scratch transformer metrics...[/dim]")
    scratch_model   = load_scratch_model(config)
    scratch_metrics = get_scratch_metrics(config, scratch_model)

    # --- Display ---
    display_header(ticker, last_close, last_date.strftime("%Y-%m-%d"))
    display_chronos_forecast(forecast_df, last_close)
    display_scratch_metrics(scratch_metrics)

    console.print(
        f"  [dim]Chart saved → "
        f"outputs/{ticker}_chronos_20day_forecast.png[/dim]\n"
    )


if __name__ == "__main__":
    app()