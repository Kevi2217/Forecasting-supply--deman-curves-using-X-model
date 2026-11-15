from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from electricity_forecasting.data.dataset import iter_selected_intervals
from electricity_forecasting.data.parse_omie import parse_curve_file
from electricity_forecasting.evaluation.metrics import compute_metrics
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve
from electricity_forecasting.market.price_classes import (
    PriceClassSpec,
    compress_to_classes,
    reconstruct_class_records,
)


@dataclass(frozen=True)
class CandidateClassSpecs:
    n_classes: int
    supply: PriceClassSpec
    demand: PriceClassSpec


def reconstruct_pair(
    selected_records: pd.DataFrame,
    *,
    interval_start_utc: pd.Timestamp,
    country: str,
    supply_spec: PriceClassSpec,
    demand_spec: PriceClassSpec,
) -> tuple[CurvePair, dict[str, float]]:
    s_vol, s_diag = compress_to_classes(selected_records, supply_spec)
    d_vol, d_diag = compress_to_classes(selected_records, demand_spec)
    records = pd.concat(
        [reconstruct_class_records(s_vol, supply_spec), reconstruct_class_records(d_vol, demand_spec)],
        ignore_index=True,
    )
    pair = CurvePair(
        interval_start_utc=interval_start_utc,
        country=country,
        supply=build_supply_curve(records),
        demand=build_demand_curve(records),
    )
    diagnostics = {
        "supply_underflow_mwh": s_diag["underflow_mwh"],
        "supply_overflow_mwh": s_diag["overflow_mwh"],
        "demand_underflow_mwh": d_diag["underflow_mwh"],
        "demand_overflow_mwh": d_diag["overflow_mwh"],
    }
    return pair, diagnostics


def evaluate_class_candidates(
    files: list[Path],
    candidates: list[CandidateClassSpecs],
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
    curve_flag: str = "O",
    timezone: str = "Europe/Madrid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for path in files:
        frame = parse_curve_file(path, timezone=timezone)
        for selected in iter_selected_intervals(
            frame,
            preferred_country=preferred_country,
            fallback_country=fallback_country,
            curve_flag=curve_flag,
        ):
            original_pair = CurvePair(
                interval_start_utc=selected.interval_start_utc,
                country=selected.country,
                supply=build_supply_curve(selected.records),
                demand=build_demand_curve(selected.records),
            )
            original = approximate_clearing(original_pair)
            for candidate in candidates:
                reconstructed, diag = reconstruct_pair(
                    selected.records,
                    interval_start_utc=selected.interval_start_utc,
                    country=selected.country,
                    supply_spec=candidate.supply,
                    demand_spec=candidate.demand,
                )
                compressed = approximate_clearing(reconstructed)
                rows.append(
                    {
                        "interval_start_utc": selected.interval_start_utc,
                        "delivery_date": selected.delivery_date,
                        "period": selected.period,
                        "curve_country": selected.country,
                        "n_classes": candidate.n_classes,
                        "original_clearing_eur_mwh": original.estimated_price_eur_mwh,
                        "original_status": original.status,
                        "compressed_clearing_eur_mwh": compressed.estimated_price_eur_mwh,
                        "compressed_status": compressed.status,
                        **diag,
                    }
                )
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for n_classes, group in detail.groupby("n_classes"):
        original = group["original_clearing_eur_mwh"].to_numpy(dtype=float)
        compressed = group["compressed_clearing_eur_mwh"].to_numpy(dtype=float)
        try:
            m = compute_metrics(original, compressed)
            metric_dict = m.to_dict()
        except ValueError:
            metric_dict = {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "median_absolute_error": np.nan}
        summaries.append(
            {
                "n_classes": int(n_classes),
                **metric_dict,
                "invalid_original": int(group["original_clearing_eur_mwh"].isna().sum()),
                "invalid_compressed": int(group["compressed_clearing_eur_mwh"].isna().sum()),
                "mean_unrepresented_mwh": float(
                    group[[
                        "supply_underflow_mwh",
                        "supply_overflow_mwh",
                        "demand_underflow_mwh",
                        "demand_overflow_mwh",
                    ]].sum(axis=1).mean()
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("n_classes").reset_index(drop=True)
    return detail, summary


def select_class_count(summary: pd.DataFrame, *, tie_tolerance_eur_mwh: float = 0.05) -> int:
    valid = summary.dropna(subset=["mae"]).copy()
    if valid.empty:
        raise ValueError("No valid compression candidate metrics")
    best_mae = float(valid["mae"].min())
    acceptable = valid.loc[valid["mae"] <= best_mae + tie_tolerance_eur_mwh]
    return int(acceptable["n_classes"].min())
