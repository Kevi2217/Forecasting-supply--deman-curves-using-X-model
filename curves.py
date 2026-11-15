from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurvePair:
    interval_start_utc: pd.Timestamp
    country: str
    supply: pd.DataFrame
    demand: pd.DataFrame


def select_curve_records(
    df: pd.DataFrame,
    *,
    interval_start_utc: pd.Timestamp,
    country: str,
    curve_flag: str = "O",
) -> pd.DataFrame:
    mask = (
        (df["interval_start_utc"] == interval_start_utc)
        & (df["country"] == country)
        & (df["curve_flag"] == curve_flag)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(
            f"No curve records for interval={interval_start_utc}, country={country}, flag={curve_flag}"
        )
    return out


def _aggregate_incremental_side(records: pd.DataFrame, side: str) -> pd.DataFrame:
    side_records = records.loc[records["side"] == side, ["price_eur_mwh", "energy_mwh"]].copy()
    if side_records.empty:
        raise ValueError(f"No records for side={side}")
    if (side_records["energy_mwh"] < 0).any():
        raise ValueError("Curve increments must be non-negative")
    grouped = (
        side_records.groupby("price_eur_mwh", as_index=False, sort=True)["energy_mwh"]
        .sum()
        .rename(columns={"energy_mwh": "increment_mwh"})
    )
    return grouped


def build_supply_curve(records: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative supply S(p)=sum(sell volume with offer price <= p)."""
    grouped = _aggregate_incremental_side(records, "V").sort_values("price_eur_mwh", ascending=True)
    grouped["cumulative_mwh"] = grouped["increment_mwh"].cumsum()
    if not grouped["cumulative_mwh"].is_monotonic_increasing:
        raise AssertionError("Supply curve is not monotone increasing")
    return grouped.reset_index(drop=True)


def build_demand_curve(records: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative demand D(p)=sum(buy volume with bid price >= p)."""
    grouped = _aggregate_incremental_side(records, "C").sort_values("price_eur_mwh", ascending=False)
    grouped["cumulative_mwh"] = grouped["increment_mwh"].cumsum()
    # In descending price order the cumulative quantity increases. In ascending price order D(p) is non-increasing.
    return grouped.reset_index(drop=True)


def build_curve_pair(
    records: pd.DataFrame,
    *,
    interval_start_utc: pd.Timestamp,
    country: str,
    curve_flag: str = "O",
) -> CurvePair:
    selected = select_curve_records(
        records,
        interval_start_utc=interval_start_utc,
        country=country,
        curve_flag=curve_flag,
    )
    return CurvePair(
        interval_start_utc=interval_start_utc,
        country=country,
        supply=build_supply_curve(selected),
        demand=build_demand_curve(selected),
    )


def evaluate_step_curves(pair: CurvePair, price_grid: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate cumulative step curves on a common observed-price grid."""
    if price_grid is None:
        price_grid = np.unique(
            np.concatenate(
                [
                    pair.supply["price_eur_mwh"].to_numpy(dtype=float),
                    pair.demand["price_eur_mwh"].to_numpy(dtype=float),
                ]
            )
        )
    price_grid = np.asarray(price_grid, dtype=float)
    price_grid.sort()

    supply_prices = pair.supply["price_eur_mwh"].to_numpy(dtype=float)
    supply_increments = pair.supply["increment_mwh"].to_numpy(dtype=float)
    demand_prices = pair.demand["price_eur_mwh"].to_numpy(dtype=float)
    demand_increments = pair.demand["increment_mwh"].to_numpy(dtype=float)

    s_values = np.array([supply_increments[supply_prices <= p].sum() for p in price_grid], dtype=float)
    d_values = np.array([demand_increments[demand_prices >= p].sum() for p in price_grid], dtype=float)
    out = pd.DataFrame({"price_eur_mwh": price_grid, "supply_mwh": s_values, "demand_mwh": d_values})
    out["imbalance_mwh"] = out["supply_mwh"] - out["demand_mwh"]
    if not out["supply_mwh"].is_monotonic_increasing:
        raise AssertionError("Supply evaluation violated monotonicity")
    if not out["demand_mwh"].is_monotonic_decreasing:
        raise AssertionError("Demand evaluation violated monotonicity")
    return out
