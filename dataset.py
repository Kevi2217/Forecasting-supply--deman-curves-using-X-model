from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from electricity_forecasting.data.parse_omie import parse_curve_file, parse_price_file
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve
from electricity_forecasting.market.price_classes import PriceClassSpec, compress_to_classes, reconstruct_class_records


@dataclass(frozen=True)
class IntervalSelection:
    interval_start_utc: pd.Timestamp
    delivery_date: object
    period: int
    country: str
    records: pd.DataFrame


def iter_curve_files(raw_curve_dir: Path, start_date: str, end_date: str) -> Iterable[Path]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for day in pd.date_range(start, end, freq="D"):
        path = raw_curve_dir / f"curva_pbc_{day:%Y%m%d}.1"
        if not path.exists():
            raise FileNotFoundError(f"Missing mandatory OMIE curve file: {path}")
        yield path


def iter_price_files(raw_price_dir: Path, start_date: str, end_date: str) -> Iterable[Path]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for day in pd.date_range(start, end, freq="D"):
        path = raw_price_dir / f"marginalpdbc_{day:%Y%m%d}.1"
        if not path.exists():
            raise FileNotFoundError(f"Missing mandatory OMIE price file: {path}")
        yield path


def select_country_records_for_interval(
    interval_records: pd.DataFrame,
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
) -> tuple[str, pd.DataFrame]:
    """Select the country representation used for the Spanish target.

    During market splitting OMIE can publish Spanish/Portuguese curves separately.
    Otherwise the MIBEL-wide (MI) curve is the available aggregate. We prefer ES and
    use MI only when ES is absent. This choice is recorded for every interval.
    """
    offered = interval_records.loc[interval_records["curve_flag"] == curve_flag].copy()
    countries = set(offered["country"].dropna().astype(str))
    if preferred_country in countries:
        country = preferred_country
    elif fallback_country in countries:
        country = fallback_country
    else:
        raise ValueError(
            f"Neither preferred country {preferred_country!r} nor fallback {fallback_country!r} "
            f"is available; countries={sorted(countries)}"
        )
    chosen = offered.loc[offered["country"] == country].copy()
    if chosen.empty:
        raise AssertionError("Country selection produced no records")
    return country, chosen


def iter_selected_intervals(
    curve_frame: pd.DataFrame,
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
) -> Iterable[IntervalSelection]:
    for interval, group in curve_frame.groupby("interval_start_utc", sort=True):
        country, chosen = select_country_records_for_interval(
            group,
            preferred_country=preferred_country,
            fallback_country=fallback_country,
            curve_flag=curve_flag,
        )
        first = chosen.iloc[0]
        yield IntervalSelection(
            interval_start_utc=pd.Timestamp(interval),
            delivery_date=first["delivery_date"],
            period=int(first["period"]),
            country=country,
            records=chosen,
        )


def load_prices(
    files: Iterable[Path],
    *,
    timezone: str = "Europe/Madrid",
) -> pd.DataFrame:
    frames = [parse_price_file(path, timezone=timezone) for path in files]
    if not frames:
        raise ValueError("No price files supplied")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("interval_start_utc").reset_index(drop=True)


def aggregate_training_price_weights(
    files: Iterable[Path],
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
    timezone: str = "Europe/Madrid",
) -> dict[str, pd.DataFrame]:
    """Stream training files and aggregate energy weights by exact bid price.

    This avoids holding the complete multi-year bid table in memory while retaining
    exact weighted-quantile information for the price-class construction.
    """
    accum: dict[str, dict[float, float]] = {"C": {}, "V": {}}
    for path in files:
        frame = parse_curve_file(path, timezone=timezone)
        for selected in iter_selected_intervals(
            frame,
            preferred_country=preferred_country,
            fallback_country=fallback_country,
            curve_flag=curve_flag,
        ):
            for side in ("C", "V"):
                side_df = selected.records.loc[selected.records["side"] == side]
                grouped = side_df.groupby("price_eur_mwh")["energy_mwh"].sum()
                bucket = accum[side]
                for price, energy in grouped.items():
                    key = float(price)
                    bucket[key] = bucket.get(key, 0.0) + float(energy)
    result: dict[str, pd.DataFrame] = {}
    for side, mapping in accum.items():
        if not mapping:
            raise ValueError(f"No training curve records accumulated for side={side}")
        result[side] = pd.DataFrame(
            {
                "side": side,
                "price_eur_mwh": list(mapping.keys()),
                "energy_mwh": list(mapping.values()),
            }
        )
    return result


def build_observed_clearing_table(
    files: Iterable[Path],
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
    timezone: str = "Europe/Madrid",
) -> pd.DataFrame:
    """Build the interval-level clearing approximation from original observed curves."""
    rows: list[dict[str, object]] = []
    for path in files:
        frame = parse_curve_file(path, timezone=timezone)
        for selected in iter_selected_intervals(
            frame,
            preferred_country=preferred_country,
            fallback_country=fallback_country,
            curve_flag=curve_flag,
        ):
            pair = CurvePair(
                interval_start_utc=selected.interval_start_utc,
                country=selected.country,
                supply=build_supply_curve(selected.records),
                demand=build_demand_curve(selected.records),
            )
            result = approximate_clearing(pair)
            rows.append(
                {
                    "interval_start_utc": selected.interval_start_utc,
                    "delivery_date": selected.delivery_date,
                    "period": selected.period,
                    "curve_country": selected.country,
                    "source_file": path.name,
                    "observed_curve_clearing_eur_mwh": result.estimated_price_eur_mwh,
                    "observed_curve_clearing_status": result.status,
                    "observed_curve_crossing_low_eur_mwh": result.lower_price_eur_mwh,
                    "observed_curve_crossing_high_eur_mwh": result.upper_price_eur_mwh,
                }
            )
    if not rows:
        raise ValueError("No observed clearing rows built from curve files")
    return pd.DataFrame(rows).sort_values("interval_start_utc").reset_index(drop=True)


def build_interval_class_table(
    files: Iterable[Path],
    *,
    supply_spec: PriceClassSpec,
    demand_spec: PriceClassSpec,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
    timezone: str = "Europe/Madrid",
    include_observed_clearing: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in files:
        frame = parse_curve_file(path, timezone=timezone)
        for selected in iter_selected_intervals(
            frame,
            preferred_country=preferred_country,
            fallback_country=fallback_country,
            curve_flag=curve_flag,
        ):
            supply_vol, supply_diag = compress_to_classes(selected.records, supply_spec)
            demand_vol, demand_diag = compress_to_classes(selected.records, demand_spec)
            row: dict[str, object] = {
                "interval_start_utc": selected.interval_start_utc,
                "delivery_date": selected.delivery_date,
                "period": selected.period,
                "curve_country": selected.country,
                "source_file": path.name,
                "supply_underflow_mwh": supply_diag["underflow_mwh"],
                "supply_overflow_mwh": supply_diag["overflow_mwh"],
                "demand_underflow_mwh": demand_diag["underflow_mwh"],
                "demand_overflow_mwh": demand_diag["overflow_mwh"],
            }
            row.update({f"supply_{i:02d}": float(v) for i, v in enumerate(supply_vol)})
            row.update({f"demand_{i:02d}": float(v) for i, v in enumerate(demand_vol)})
            if include_observed_clearing:
                pair = CurvePair(
                    interval_start_utc=selected.interval_start_utc,
                    country=selected.country,
                    supply=build_supply_curve(selected.records),
                    demand=build_demand_curve(selected.records),
                )
                result = approximate_clearing(pair)
                row["observed_curve_clearing_eur_mwh"] = result.estimated_price_eur_mwh
                row["observed_curve_clearing_status"] = result.status
            rows.append(row)
    if not rows:
        raise ValueError("No interval rows built from curve files")
    return pd.DataFrame(rows).sort_values("interval_start_utc").reset_index(drop=True)


def reconstructed_clearing_from_class_row(
    row: pd.Series,
    supply_spec: PriceClassSpec,
    demand_spec: PriceClassSpec,
) -> tuple[float | None, str]:
    supply_cols = [f"supply_{i:02d}" for i in range(supply_spec.n_classes)]
    demand_cols = [f"demand_{i:02d}" for i in range(demand_spec.n_classes)]
    s_records = reconstruct_class_records(row[supply_cols].to_numpy(dtype=float), supply_spec)
    d_records = reconstruct_class_records(row[demand_cols].to_numpy(dtype=float), demand_spec)
    records = pd.concat([s_records, d_records], ignore_index=True)
    pair = CurvePair(
        interval_start_utc=pd.Timestamp(row["interval_start_utc"]),
        country=str(row.get("curve_country", "NA")),
        supply=build_supply_curve(records),
        demand=build_demand_curve(records),
    )
    result = approximate_clearing(pair)
    return result.estimated_price_eur_mwh, result.status
