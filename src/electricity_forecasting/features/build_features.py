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
    align_col: str | None = None,
) -> pd.DataFrame:
    """Attach values from ``days`` earlier by local delivery date, not row offset.

    ``align_col`` selects the within-day alignment key. The empirical pipeline uses the
    local wall-clock ``hour`` so that, across DST changes, 03:00 is compared with 03:00
    (OMIE period numbers shift by one hour on 23/25-hour days). On a 25-hour source day
    the repeated hour keeps its first (summer-time) occurrence; a target 25-hour day maps
    both repeated hours to the same source hour; a clock hour absent from a 23-hour source
    day yields NaN. Defaults to ``period_col`` for backwards compatibility.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    key_col = align_col or period_col
    missing = set(value_columns + [date_col, key_col]) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    key = out[[date_col, key_col] + value_columns].copy()
    key[date_col] = key[date_col] + pd.Timedelta(days=days)
    key = key.drop_duplicates(subset=[date_col, key_col], keep="first")
    rename = {c: f"{c}_lag_{days}d" for c in value_columns}
    key = key.rename(columns=rename)
    merged = out.merge(key, on=[date_col, key_col], how="left", validate="many_to_one")
    return merged
