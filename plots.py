from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from electricity_forecasting.market.curves import CurvePair, evaluate_step_curves
from electricity_forecasting.market.clearing import ClearingResult


def _save(fig, output_path: Path | None):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_observed_curves(
    pair: CurvePair,
    clearing: ClearingResult,
    *,
    official_price: float | None = None,
    output_path: Path | None = None,
):
    grid = evaluate_step_curves(pair)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(grid["supply_mwh"], grid["price_eur_mwh"], where="post", label="Supply")
    ax.step(grid["demand_mwh"], grid["price_eur_mwh"], where="post", label="Demand")
    if clearing.estimated_price_eur_mwh is not None:
        ax.axhline(clearing.estimated_price_eur_mwh, linestyle="--", label="Approx. clearing")
    if official_price is not None:
        ax.axhline(official_price, linestyle=":", label="Official OMIE price")
    ax.set_xlabel("Energy / volume (MWh)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Observed OMIE supply and demand curves")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_curve_comparison(
    original: CurvePair,
    reconstructed: CurvePair,
    *,
    output_path: Path | None = None,
    title: str = "Original versus price-class reconstructed curves",
):
    orig = evaluate_step_curves(original)
    rec = evaluate_step_curves(reconstructed)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(orig["supply_mwh"], orig["price_eur_mwh"], where="post", label="Original supply")
    ax.step(orig["demand_mwh"], orig["price_eur_mwh"], where="post", label="Original demand")
    ax.step(rec["supply_mwh"], rec["price_eur_mwh"], where="post", linestyle="--", label="Reconstructed supply")
    ax.step(rec["demand_mwh"], rec["price_eur_mwh"], where="post", linestyle="--", label="Reconstructed demand")
    ax.set_xlabel("Energy / volume (MWh)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_prediction_timeseries(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    observed_col: str,
    prediction_cols: list[str],
    output_path: Path | None = None,
    max_points: int = 24 * 14,
):
    data = frame.sort_values(timestamp_col).tail(max_points)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data[timestamp_col], data[observed_col], label="Official OMIE price", linewidth=1.6)
    for col in prediction_cols:
        ax.plot(data[timestamp_col], data[col], label=col, linewidth=1.1)
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Delivery interval")
    ax.set_title("Out-of-sample day-ahead price forecasts")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_metric_comparison(
    metric_table: pd.DataFrame,
    *,
    output_path: Path | None = None,
):
    data = metric_table.loc[metric_table["subset"] == "all"].copy()
    if data.empty:
        raise ValueError("No all-observation metrics available")
    models = data["model"].tolist()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, data["mae"], width=width, label="MAE")
    ax.bar(x + width / 2, data["rmse"], width=width, label="RMSE")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Out-of-sample forecast performance")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)
