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
    matched: CurvePair | None = None,
    title: str = "Observed OMIE supply and demand curves",
):
    grid = evaluate_step_curves(pair)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(grid["supply_mwh"], grid["price_eur_mwh"], where="post", label="Offered supply")
    ax.step(grid["demand_mwh"], grid["price_eur_mwh"], where="post", label="Offered demand")
    if matched is not None:
        mgrid = evaluate_step_curves(matched)
        ax.step(mgrid["supply_mwh"], mgrid["price_eur_mwh"], where="post", linestyle="-.", linewidth=0.9,
                label="Matched supply (ex post)")
        ax.step(mgrid["demand_mwh"], mgrid["price_eur_mwh"], where="post", linestyle="-.", linewidth=0.9,
                label="Matched demand (ex post)")
    if clearing.estimated_price_eur_mwh is not None:
        ax.axhline(clearing.estimated_price_eur_mwh, linestyle="--", label="Approx. clearing")
    if official_price is not None:
        ax.axhline(official_price, linestyle=":", label="Official OMIE price")
    ax.set_xlabel("Energy / volume (MWh)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(title)
    ax.set_ylim(-50, max(300.0, (official_price or 0) * 2))
    ax.legend(fontsize=8)
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


def plot_clearing_diagnostics(
    frame: pd.DataFrame,
    *,
    official_col: str = "price_es_eur_mwh",
    estimate_col: str = "observed_curve_clearing_eur_mwh",
    output_path: Path | None = None,
    title: str = "Approximate clearing from observed offered curves vs official Spanish price",
):
    """Scatter, error distribution and error-by-hour for the clearing approximation."""
    data = frame[[official_col, estimate_col, "interval_start_utc"]].dropna().copy()
    if data.empty:
        raise ValueError("No finite clearing observations to plot")
    data["error"] = data[estimate_col].astype(float) - data[official_col].astype(float)
    local = pd.to_datetime(data["interval_start_utc"], utc=True).dt.tz_convert("Europe/Madrid")
    data["hour"] = local.dt.hour
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.scatter(data[official_col], data[estimate_col], s=3, alpha=0.25)
    lo = float(min(data[official_col].min(), data[estimate_col].min()))
    hi = float(max(data[official_col].max(), data[estimate_col].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, label="45° line")
    ax.set_xlabel("Official Spanish price (EUR/MWh)")
    ax.set_ylabel("Approx. clearing (EUR/MWh)")
    ax.set_title("Approximation vs official")
    ax.legend()
    ax = axes[1]
    clipped = data["error"].clip(-150, 150)
    ax.hist(clipped, bins=120)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Approx. − official (EUR/MWh, clipped ±150)")
    ax.set_ylabel("Intervals")
    ax.set_title(f"Error distribution (MAE {data['error'].abs().mean():.2f}, bias {data['error'].mean():.2f})")
    ax = axes[2]
    by_hour = data.groupby("hour")["error"].agg(mae=lambda e: e.abs().mean(), bias="mean")
    ax.bar(by_hour.index - 0.2, by_hour["mae"], width=0.4, label="MAE")
    ax.bar(by_hour.index + 0.2, by_hour["bias"], width=0.4, label="Bias")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Error by local hour")
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_reconstruction_diagnostics(
    detail: pd.DataFrame,
    *,
    output_path: Path | None = None,
):
    """Compression error by candidate class count (validation period)."""
    data = detail.dropna(subset=["original_clearing_eur_mwh", "compressed_clearing_eur_mwh"]).copy()
    if data.empty:
        raise ValueError("No finite reconstruction observations to plot")
    data["error"] = data["compressed_clearing_eur_mwh"] - data["original_clearing_eur_mwh"]
    ks = sorted(data["n_classes"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.boxplot([data.loc[data["n_classes"] == k, "error"].clip(-100, 100) for k in ks], showfliers=False)
    ax.set_xticks(range(1, len(ks) + 1), [str(k) for k in ks])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Price classes per side")
    ax.set_ylabel("Compressed − original clearing (EUR/MWh)")
    ax.set_title("Reconstruction price distortion (validation period)")
    ax = axes[1]
    summary = data.groupby("n_classes")["error"].agg(mae=lambda e: e.abs().mean(), rmse=lambda e: float(np.sqrt((e**2).mean())))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["mae"], width=0.4, label="MAE")
    ax.bar(x + 0.2, summary["rmse"], width=0.4, label="RMSE")
    ax.set_xticks(x, [str(k) for k in summary.index])
    ax.set_xlabel("Price classes per side")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Reconstruction error vs original-curve clearing")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)
