from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ConstraintDiagnostics:
    n_predictions: int
    n_negative_raw: int
    negative_share: float
    negative_volume_sum: float


def make_regressor(
    kind: Literal["elastic_net", "ridge", "hist_gradient_boosting"],
    *,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_seed: int = 42,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.0,
):
    if kind == "elastic_net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_seed),
                ),
            ]
        )
    if kind == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=random_seed,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def enforce_nonnegative(predictions: np.ndarray) -> tuple[np.ndarray, ConstraintDiagnostics]:
    """Apply documented non-negative post-processing and report its extent."""
    raw = np.asarray(predictions, dtype=float)
    if np.any(~np.isfinite(raw)):
        raise ValueError("Predictions contain non-finite values")
    negative = raw < 0
    constrained = raw.copy()
    constrained[negative] = 0.0
    diag = ConstraintDiagnostics(
        n_predictions=int(raw.size),
        n_negative_raw=int(negative.sum()),
        negative_share=float(negative.mean()) if raw.size else 0.0,
        negative_volume_sum=float(-raw[negative].sum()) if negative.any() else 0.0,
    )
    return constrained, diag
