from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    status: str
    n_rows: int
    n_intervals: int
    issues: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_curve_records(df: pd.DataFrame) -> ValidationResult:
    required = {
        "period",
        "delivery_date",
        "country",
        "side",
        "energy_mwh",
        "price_eur_mwh",
        "curve_flag",
        "interval_start_utc",
    }
    missing = sorted(required - set(df.columns))
    issues: list[str] = []
    if missing:
        return ValidationResult("FAILED", len(df), 0, (f"Missing columns: {missing}",), {})

    if df.empty:
        issues.append("No curve records")
    if not np.isfinite(df["energy_mwh"]).all() or not np.isfinite(df["price_eur_mwh"]).all():
        issues.append("Non-finite numeric values")
    if (df["energy_mwh"] < 0).any():
        issues.append("Negative energy values")
    invalid_sides = sorted(set(df["side"].dropna()) - {"C", "V"})
    if invalid_sides:
        issues.append(f"Invalid side codes: {invalid_sides}")
    invalid_flags = sorted(set(df["curve_flag"].dropna()) - {"O", "C"})
    if invalid_flags:
        issues.append(f"Invalid curve flags: {invalid_flags}")
    invalid_countries = sorted(set(df["country"].dropna()) - {"MI", "ES", "PT"})
    if invalid_countries:
        issues.append(f"Unexpected country codes: {invalid_countries}")

    duplicate_keys = ["interval_start_utc", "country", "side", "price_eur_mwh", "curve_flag", "unit"]
    duplicate_count = int(df.duplicated(subset=[c for c in duplicate_keys if c in df.columns], keep=False).sum())
    # Duplicates can be economically valid because multiple bids may share a price; report, do not fail.

    interval_counts = df[["delivery_date", "period", "interval_start_utc"]].drop_duplicates()
    by_date = interval_counts.groupby("delivery_date")["period"].nunique()
    unexpected_day_counts = by_date[~by_date.isin([23, 24, 25])].to_dict()
    if unexpected_day_counts:
        issues.append(f"Unexpected number of delivery periods for dates: {unexpected_day_counts}")

    details = {
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "curve_flags": sorted(df["curve_flag"].dropna().unique().tolist()),
        "sides": sorted(df["side"].dropna().unique().tolist()),
        "duplicate_like_rows": duplicate_count,
        "date_period_counts": {str(k): int(v) for k, v in by_date.items()},
        "price_min": float(df["price_eur_mwh"].min()) if len(df) else None,
        "price_max": float(df["price_eur_mwh"].max()) if len(df) else None,
        "energy_min": float(df["energy_mwh"].min()) if len(df) else None,
        "energy_max": float(df["energy_mwh"].max()) if len(df) else None,
    }
    return ValidationResult("VERIFIED" if not issues else "FAILED", len(df), len(interval_counts), tuple(issues), details)


def validate_price_records(df: pd.DataFrame) -> ValidationResult:
    required = {"period", "delivery_date", "price_es_eur_mwh", "price_pt_eur_mwh", "interval_start_utc"}
    missing = sorted(required - set(df.columns))
    if missing:
        return ValidationResult("FAILED", len(df), 0, (f"Missing columns: {missing}",), {})
    issues: list[str] = []
    if df.empty:
        issues.append("No price records")
    if not np.isfinite(df[["price_es_eur_mwh", "price_pt_eur_mwh"]].to_numpy()).all():
        issues.append("Non-finite marginal prices")
    if df.duplicated(subset=["interval_start_utc"]).any():
        issues.append("Duplicate physical delivery intervals")
    by_date = df.groupby("delivery_date")["period"].nunique()
    unexpected = by_date[~by_date.isin([23, 24, 25])].to_dict()
    if unexpected:
        issues.append(f"Unexpected period counts: {unexpected}")
    details = {
        "date_period_counts": {str(k): int(v) for k, v in by_date.items()},
        "spain_price_min": float(df["price_es_eur_mwh"].min()) if len(df) else None,
        "spain_price_max": float(df["price_es_eur_mwh"].max()) if len(df) else None,
    }
    return ValidationResult("VERIFIED" if not issues else "FAILED", len(df), len(df), tuple(issues), details)
