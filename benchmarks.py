from __future__ import annotations

import pandas as pd


def weekly_persistence(
    prices: pd.DataFrame,
    *,
    target_col: str = "price_es_eur_mwh",
    date_col: str = "delivery_date",
    period_col: str = "period",
) -> pd.DataFrame:
    """Calendar-aware one-week persistence benchmark."""
    required = {target_col, date_col, period_col}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    out = prices.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    lag = out[[date_col, period_col, target_col]].copy()
    lag[date_col] = pd.to_datetime(lag[date_col]).dt.normalize() + pd.Timedelta(days=7)
    lag = lag.rename(columns={target_col: "prediction"})
    return out.merge(lag, on=[date_col, period_col], how="left", validate="many_to_one")
