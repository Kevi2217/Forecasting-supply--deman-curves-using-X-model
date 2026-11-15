from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from electricity_forecasting.data.parse_omie import parse_curve_file, parse_price_file
from electricity_forecasting.data.validate import validate_curve_records, validate_price_records


def audit_raw_files(
    curve_files: Iterable[Path],
    price_files: Iterable[Path],
    *,
    timezone: str = "Europe/Madrid",
) -> dict[str, object]:
    """Parse and validate every supplied raw file without retaining the full raw dataset.

    This is deliberately a pre-modelling gate. A parser exception or a failed per-file
    validation is retained in the report and causes the overall status to fail.
    """
    curve_results: list[dict[str, object]] = []
    price_results: list[dict[str, object]] = []

    for path in curve_files:
        try:
            frame = parse_curve_file(path, timezone=timezone)
            result = validate_curve_records(frame).to_dict()
            result["file"] = path.name
        except Exception as exc:  # report raw-source failures rather than hiding them
            result = {"file": path.name, "status": "FAILED", "issues": [f"{type(exc).__name__}: {exc}"]}
        curve_results.append(result)

    for path in price_files:
        try:
            frame = parse_price_file(path, timezone=timezone)
            result = validate_price_records(frame).to_dict()
            result["file"] = path.name
        except Exception as exc:
            result = {"file": path.name, "status": "FAILED", "issues": [f"{type(exc).__name__}: {exc}"]}
        price_results.append(result)

    curve_failed = [r["file"] for r in curve_results if r.get("status") != "VERIFIED"]
    price_failed = [r["file"] for r in price_results if r.get("status") != "VERIFIED"]
    return {
        "status": "VERIFIED" if not curve_failed and not price_failed else "FAILED",
        "curve_files_checked": len(curve_results),
        "price_files_checked": len(price_results),
        "curve_files_failed": curve_failed,
        "price_files_failed": price_failed,
        "curve_results": curve_results,
        "price_results": price_results,
    }


def assess_curve_price_alignment(
    empirical: pd.DataFrame,
    *,
    price_tolerance_eur_mwh: float = 1e-6,
) -> dict[str, object]:
    """Assess whether the selected curve geography is compatible with a Spanish target.

    ES curves are directly zonal. MI is accepted only as a documented fallback; when an
    MI curve is used, divergence between the official Spanish and Portuguese marginal
    prices signals a zonal split for which an Iberian-wide curve cannot be assumed to
    reconstruct the Spanish zonal clearing price exactly.
    """
    required = {"curve_country", "price_es_eur_mwh", "price_pt_eur_mwh"}
    missing = required - set(empirical.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    country_counts = empirical["curve_country"].value_counts(dropna=False).to_dict()
    mi = empirical.loc[empirical["curve_country"] == "MI"].copy()
    if mi.empty:
        mi_split_count = 0
        mi_split_share = 0.0
        mi_abs_diff_mean = 0.0
        mi_abs_diff_max = 0.0
    else:
        diff = (mi["price_es_eur_mwh"].astype(float) - mi["price_pt_eur_mwh"].astype(float)).abs()
        split = diff > float(price_tolerance_eur_mwh)
        mi_split_count = int(split.sum())
        mi_split_share = float(split.mean())
        mi_abs_diff_mean = float(diff.mean())
        mi_abs_diff_max = float(diff.max())

    unsupported = empirical.loc[~empirical["curve_country"].isin(["ES", "MI"]), "curve_country"].dropna().unique()
    status = "VERIFIED" if len(unsupported) == 0 and mi_split_count == 0 else "PARTIALLY_VERIFIED"
    return {
        "status": status,
        "n_intervals": int(len(empirical)),
        "curve_country_counts": {str(k): int(v) for k, v in country_counts.items()},
        "mi_intervals": int(len(mi)),
        "mi_intervals_with_es_pt_price_split": mi_split_count,
        "mi_split_share": mi_split_share,
        "mi_mean_abs_es_pt_price_difference_eur_mwh": mi_abs_diff_mean,
        "mi_max_abs_es_pt_price_difference_eur_mwh": mi_abs_diff_max,
        "unsupported_curve_countries": [str(x) for x in unsupported],
        "interpretation": (
            "MI curves during ES/PT price divergence are not assumed to exactly reconstruct the Spanish zonal price."
        ),
    }


def clearing_validation_summary(empirical: pd.DataFrame) -> dict[str, object]:
    required = {"price_es_eur_mwh", "observed_curve_clearing_eur_mwh", "observed_curve_clearing_status"}
    missing = required - set(empirical.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    official = empirical["price_es_eur_mwh"].to_numpy(dtype=float)
    estimated = empirical["observed_curve_clearing_eur_mwh"].to_numpy(dtype=float)
    finite = np.isfinite(official) & np.isfinite(estimated)
    errors = estimated[finite] - official[finite]
    invalid = ~np.isfinite(estimated)
    if finite.any():
        summary = {
            "n_matched": int(finite.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "p95_absolute_error": float(np.quantile(np.abs(errors), 0.95)),
        }
    else:
        summary = {
            "n_matched": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "median_absolute_error": None,
            "p95_absolute_error": None,
        }
    summary.update(
        {
            "n_total": int(len(empirical)),
            "invalid_count": int(invalid.sum()),
            "invalid_share": float(invalid.mean()) if len(empirical) else None,
            "status_counts": {
                str(k): int(v) for k, v in empirical["observed_curve_clearing_status"].value_counts(dropna=False).items()
            },
        }
    )
    return summary
