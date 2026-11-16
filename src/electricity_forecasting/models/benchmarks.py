from __future__ import annotations

import pandas as pd


def weekly_persistence(
    prices: pd.DataFrame,
    *,
    target_col: str = "price_es_eur_mwh",
    date_col: str = "delivery_date",
    period_col: str = "period",
    align_col: str | None = None,
) -> pd.DataFrame:
    """Calendar-aware one-week persistence: same local date minus 7 days, same key.

    ``align_col="hour"`` aligns on local wall-clock hour (DST-safe, used by the
    empirical pipeline). On a 25-hour source day the first occurrence of the repeated
    hour is used. Defaults to ``period_col`` for backwards compatibility.
    """
    key_col = align_col or period_col
    required = {target_col, date_col, key_col}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    out = prices.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    lag = out[[date_col, key_col, target_col]].copy()
    lag[date_col] = lag[date_col] + pd.Timedelta(days=7)
    lag = lag.drop_duplicates(subset=[date_col, key_col], keep="first")
    lag = lag.rename(columns={target_col: "prediction"})
    return out.merge(lag, on=[date_col, key_col], how="left", validate="many_to_one")
