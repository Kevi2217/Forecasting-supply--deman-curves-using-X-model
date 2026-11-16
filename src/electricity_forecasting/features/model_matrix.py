from __future__ import annotations

import pandas as pd

from electricity_forecasting.features.build_features import add_calendar_features, calendar_aware_lag


def build_model_matrix(
    interval_table: pd.DataFrame,
    *,
    class_columns: list[str],
    price_column: str = "price_es_eur_mwh",
    lag_days: tuple[int, ...] = (1, 2, 7),
) -> pd.DataFrame:
    required = {"interval_start_utc", "delivery_date", "period", price_column, *class_columns}
    missing = required - set(interval_table.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    out = interval_table.sort_values("interval_start_utc").copy()
    local = pd.to_datetime(out["interval_start_utc"], utc=True).dt.tz_convert("Europe/Madrid")
    out["interval_start_local"] = local
    out = add_calendar_features(out)
    lag_values = [*class_columns, price_column]
    for days in lag_days:
        out = calendar_aware_lag(out, value_columns=lag_values, days=days, align_col="hour")
    return out


def curve_feature_columns(
    class_columns: list[str],
    *,
    price_column: str = "price_es_eur_mwh",
    lag_days: tuple[int, ...] = (1, 2, 7),
) -> list[str]:
    cols = ["hour", "weekday", "month", "is_weekend"]
    for days in lag_days:
        cols.extend(f"{c}_lag_{days}d" for c in class_columns)
        cols.append(f"{price_column}_lag_{days}d")
    return cols


def direct_price_feature_columns(
    *,
    price_column: str = "price_es_eur_mwh",
    lag_days: tuple[int, ...] = (1, 2, 7),
) -> list[str]:
    return [
        "hour",
        "weekday",
        "month",
        "is_weekend",
        *(f"{price_column}_lag_{days}d" for days in lag_days),
    ]
