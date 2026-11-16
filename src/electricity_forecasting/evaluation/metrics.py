from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    n: int
    mae: float
    rmse: float
    bias: float
    median_absolute_error: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("No finite matched observations")
    err = y_pred[mask] - y_true[mask]
    abs_err = np.abs(err)
    return ForecastMetrics(
        n=int(mask.sum()),
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=float(err.mean()),
        median_absolute_error=float(np.median(abs_err)),
    )
