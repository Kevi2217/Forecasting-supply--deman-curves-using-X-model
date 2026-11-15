from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_forecasting.evaluation.metrics import compute_metrics


def conditional_metric_table(
    predictions: pd.DataFrame,
    *,
    observed_col: str,
    prediction_cols: list[str],
    training_prices: pd.Series,
) -> pd.DataFrame:
    train = training_prices.dropna().astype(float)
    high_threshold = float(train.quantile(0.95))
    train_change = train.diff().abs().dropna()
    large_change_threshold = float(train_change.quantile(0.95)) if len(train_change) else float("inf")

    frame = predictions.copy()
    frame["abs_price_change"] = frame[observed_col].astype(float).diff().abs()
    groups: dict[str, pd.Series] = {
        "all": pd.Series(True, index=frame.index),
        "negative_price": frame[observed_col] < 0,
        "high_price": frame[observed_col] >= high_threshold,
        "large_abs_change": frame["abs_price_change"] >= large_change_threshold,
    }
    rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        subset = frame.loc[mask]
        for pred_col in prediction_cols:
            if subset.empty:
                continue
            try:
                metrics = compute_metrics(subset[observed_col].to_numpy(), subset[pred_col].to_numpy())
            except ValueError:
                continue
            row = {"subset": group_name, "model": pred_col, **metrics.to_dict()}
            row["high_price_threshold"] = high_threshold
            row["large_change_threshold"] = large_change_threshold
            rows.append(row)

    if "hour" in frame.columns:
        for hour, subset in frame.groupby("hour"):
            for pred_col in prediction_cols:
                try:
                    metrics = compute_metrics(subset[observed_col].to_numpy(), subset[pred_col].to_numpy())
                except ValueError:
                    continue
                rows.append({"subset": f"hour_{int(hour):02d}", "model": pred_col, **metrics.to_dict()})
    return pd.DataFrame(rows)


def error_decomposition_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "price_es_eur_mwh",
        "observed_curve_clearing_eur_mwh",
        "compressed_curve_clearing_eur_mwh",
        "curve_model_prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    official = predictions["price_es_eur_mwh"].to_numpy(dtype=float)
    original = predictions["observed_curve_clearing_eur_mwh"].to_numpy(dtype=float)
    compressed = predictions["compressed_curve_clearing_eur_mwh"].to_numpy(dtype=float)
    forecasted = predictions["curve_model_prediction"].to_numpy(dtype=float)
    rows = []
    for name, truth, estimate in [
        ("A_clearing_approximation", official, original),
        ("B_curve_compression", original, compressed),
        ("C_curve_forecasting", compressed, forecasted),
        ("Total_official_to_forecast", official, forecasted),
    ]:
        metrics = compute_metrics(truth, estimate)
        rows.append({"component": name, **metrics.to_dict()})
    return pd.DataFrame(rows)
