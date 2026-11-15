from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from electricity_forecasting.models.curve_forecast import enforce_nonnegative, make_regressor


@dataclass(frozen=True)
class ModelSelectionResult:
    kind: str
    params: dict[str, Any]
    validation_score: float
    candidate_scores: tuple[dict[str, Any], ...]


@dataclass
class IndependentTargetModels:
    kind: str
    params: dict[str, Any]
    feature_columns: list[str]
    target_columns: list[str]
    estimators: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
        x = frame[self.feature_columns]
        predictions: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, float | int]] = {}
        for target in self.target_columns:
            raw = np.asarray(self.estimators[target].predict(x), dtype=float)
            constrained, diag = enforce_nonnegative(raw)
            predictions[target] = constrained
            diagnostics[target] = {
                "n_predictions": diag.n_predictions,
                "n_negative_raw": diag.n_negative_raw,
                "negative_share": diag.negative_share,
                "negative_volume_sum": diag.negative_volume_sum,
            }
        return pd.DataFrame(predictions, index=frame.index), diagnostics


def fit_independent_target_models(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    kind: str,
    params: dict[str, Any],
    random_seed: int = 42,
) -> IndependentTargetModels:
    x = train[feature_columns]
    estimators: dict[str, Any] = {}
    for target in target_columns:
        estimator = make_regressor(kind, random_seed=random_seed, **params)
        estimator.fit(x, train[target].to_numpy(dtype=float))
        estimators[target] = estimator
    return IndependentTargetModels(
        kind=kind,
        params=dict(params),
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        estimators=estimators,
    )


def _scaled_multi_target_mae(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    scores: list[float] = []
    for col in y_true.columns:
        scale = float(np.mean(np.abs(y_true[col].to_numpy(dtype=float))))
        scale = max(scale, 1.0)
        scores.append(float(mean_absolute_error(y_true[col], y_pred[col])) / scale)
    return float(np.mean(scores))


def select_curve_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    candidates: Iterable[tuple[str, dict[str, Any]]],
    random_seed: int = 42,
) -> ModelSelectionResult:
    results: list[dict[str, Any]] = []
    for kind, params in candidates:
        fitted = fit_independent_target_models(
            train,
            feature_columns=feature_columns,
            target_columns=target_columns,
            kind=kind,
            params=params,
            random_seed=random_seed,
        )
        pred, _ = fitted.predict(validation)
        score = _scaled_multi_target_mae(validation[target_columns], pred)
        results.append({"kind": kind, "params": dict(params), "validation_score": score})
    if not results:
        raise ValueError("No model candidates provided")
    best = min(results, key=lambda item: item["validation_score"])
    return ModelSelectionResult(
        kind=str(best["kind"]),
        params=dict(best["params"]),
        validation_score=float(best["validation_score"]),
        candidate_scores=tuple(results),
    )


def fit_direct_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    kind: str,
    params: dict[str, Any],
    random_seed: int = 42,
):
    model = make_regressor(kind, random_seed=random_seed, **params)
    model.fit(train[feature_columns], train[target_column].to_numpy(dtype=float))
    return model


def select_direct_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    candidates: Iterable[tuple[str, dict[str, Any]]],
    random_seed: int = 42,
) -> ModelSelectionResult:
    results: list[dict[str, Any]] = []
    for kind, params in candidates:
        model = fit_direct_model(
            train,
            feature_columns=feature_columns,
            target_column=target_column,
            kind=kind,
            params=params,
            random_seed=random_seed,
        )
        pred = np.asarray(model.predict(validation[feature_columns]), dtype=float)
        score = float(mean_absolute_error(validation[target_column], pred))
        results.append({"kind": kind, "params": dict(params), "validation_score": score})
    if not results:
        raise ValueError("No model candidates provided")
    best = min(results, key=lambda item: item["validation_score"])
    return ModelSelectionResult(
        kind=str(best["kind"]),
        params=dict(best["params"]),
        validation_score=float(best["validation_score"]),
        candidate_scores=tuple(results),
    )
