from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame, local_col: str = "interval_start_local") -> pd.DataFrame:
    if local_col not in df.columns:
        raise KeyError(f"Missing {local_col}")
    out = df.copy()
    dt = out[local_col]
    out["hour"] = dt.dt.hour.astype(int)
    out["weekday"] = dt.dt.weekday.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    return out


def calendar_aware_lag(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    days: int,
    date_col: str = "delivery_date",
    period_col: str = "period",
) -> pd.DataFrame:
    """Attach lagged values by local delivery date and OMIE period, not row offset.

    This avoids blindly assuming 24 rows per day across DST transitions.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    missing = set(value_columns + [date_col, period_col]) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    key = out[[date_col, period_col] + value_columns].copy()
    key[date_col] = pd.to_datetime(key[date_col]).dt.normalize() + pd.Timedelta(days=days)
    rename = {c: f"{c}_lag_{days}d" for c in value_columns}
    key = key.rename(columns=rename)
    merged = out.merge(key, on=[date_col, period_col], how="left", validate="many_to_one")
    return merged
