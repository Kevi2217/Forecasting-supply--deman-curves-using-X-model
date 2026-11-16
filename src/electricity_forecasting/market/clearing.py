from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import CurvePair, evaluate_step_curves


@dataclass(frozen=True)
class ClearingResult:
    status: str
    estimated_price_eur_mwh: float | None
    estimated_volume_mwh: float | None
    lower_price_eur_mwh: float | None
    upper_price_eur_mwh: float | None
    ambiguous: bool
    reason: str


def approximate_clearing(pair: CurvePair) -> ClearingResult:
    """Approximate clearing using the first observed price with S(p)-D(p) >= 0.

    This intentionally respects step functions and performs no linear interpolation.
    The method is an approximation to OMIE/EUPHEMIA clearing and therefore returns
    diagnostic metadata rather than pretending to reproduce the official price.
    """
    grid = evaluate_step_curves(pair)
    if grid.empty:
        return ClearingResult("INVALID", None, None, None, None, False, "empty common price grid")

    imbalance = grid["imbalance_mwh"].to_numpy(dtype=float)
    crossing_idx = np.flatnonzero(imbalance >= 0)
    if len(crossing_idx) == 0:
        return ClearingResult("NO_CROSSING", None, None, None, None, False, "supply never reaches demand")

    idx = int(crossing_idx[0])
    upper = float(grid.iloc[idx]["price_eur_mwh"])
    lower = float(grid.iloc[idx - 1]["price_eur_mwh"]) if idx > 0 else upper
    estimated_volume = float(min(grid.iloc[idx]["supply_mwh"], grid.iloc[idx]["demand_mwh"]))

    ambiguous = False
    reason = "first observed price at which cumulative supply meets/exceeds cumulative demand"
    if np.isclose(imbalance[idx], 0.0):
        next_same = idx + 1 < len(grid) and np.isclose(imbalance[idx + 1], 0.0)
        ambiguous = bool(next_same)
        if ambiguous:
            reason += "; zero imbalance persists across more than one observed price"
    if idx == 0:
        reason += "; crossing occurs at lower domain boundary"

    return ClearingResult(
        status="OK_AMBIGUOUS" if ambiguous else "OK",
        estimated_price_eur_mwh=upper,
        estimated_volume_mwh=estimated_volume,
        lower_price_eur_mwh=lower,
        upper_price_eur_mwh=upper,
        ambiguous=ambiguous,
        reason=reason,
    )
