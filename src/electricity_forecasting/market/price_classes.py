from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceClassSpec:
    side: str
    edges: tuple[float, ...]
    representatives: tuple[float, ...]

    @property
    def n_classes(self) -> int:
        return len(self.representatives)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("total weight must be positive")
    cumulative = np.cumsum(weights) / total
    return np.interp(quantiles, cumulative, values)


def fit_price_classes(training_records: pd.DataFrame, side: str, n_classes: int) -> PriceClassSpec:
    if side not in {"C", "V"}:
        raise ValueError("side must be 'C' or 'V'")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    data = training_records.loc[training_records["side"] == side, ["price_eur_mwh", "energy_mwh"]].copy()
    if data.empty:
        raise ValueError(f"No training records for side={side}")
    prices = data["price_eur_mwh"].to_numpy(dtype=float)
    weights = data["energy_mwh"].to_numpy(dtype=float)
    qs = np.linspace(0.0, 1.0, n_classes + 1)
    edges = _weighted_quantiles(prices, weights, qs)
    edges[0] = prices.min()
    edges[-1] = prices.max()
    edges = np.unique(edges)
    if len(edges) != n_classes + 1:
        raise ValueError(
            f"Requested {n_classes} classes but weighted quantiles produced only {len(edges)-1} distinct bins; "
            "reduce n_classes or use fixed boundaries"
        )

    representatives: list[float] = []
    for i in range(n_classes):
        left, right = edges[i], edges[i + 1]
        if i == n_classes - 1:
            mask = (prices >= left) & (prices <= right)
        else:
            mask = (prices >= left) & (prices < right)
        if not mask.any():
            representatives.append(float((left + right) / 2.0))
        else:
            w = weights[mask]
            p = prices[mask]
            representatives.append(float(np.average(p, weights=w if w.sum() > 0 else None)))
    return PriceClassSpec(side=side, edges=tuple(map(float, edges)), representatives=tuple(representatives))


def compress_to_classes(records: pd.DataFrame, spec: PriceClassSpec) -> tuple[np.ndarray, dict[str, float]]:
    data = records.loc[records["side"] == spec.side, ["price_eur_mwh", "energy_mwh"]].copy()
    prices = data["price_eur_mwh"].to_numpy(dtype=float)
    energy = data["energy_mwh"].to_numpy(dtype=float)
    edges = np.asarray(spec.edges, dtype=float)
    volumes = np.zeros(spec.n_classes, dtype=float)

    under = energy[prices < edges[0]].sum()
    over = energy[prices > edges[-1]].sum()
    for i in range(spec.n_classes):
        left, right = edges[i], edges[i + 1]
        if i == spec.n_classes - 1:
            mask = (prices >= left) & (prices <= right)
        else:
            mask = (prices >= left) & (prices < right)
        volumes[i] = energy[mask].sum()
    diagnostics = {
        "underflow_mwh": float(under),
        "overflow_mwh": float(over),
        "represented_mwh": float(volumes.sum()),
        "input_mwh": float(energy.sum()),
    }
    return volumes, diagnostics


def reconstruct_class_records(volumes: np.ndarray, spec: PriceClassSpec) -> pd.DataFrame:
    volumes = np.asarray(volumes, dtype=float)
    if volumes.shape != (spec.n_classes,):
        raise ValueError(f"Expected {spec.n_classes} class volumes, got shape={volumes.shape}")
    if np.any(~np.isfinite(volumes)) or np.any(volumes < 0):
        raise ValueError("Class volumes must be finite and non-negative")
    return pd.DataFrame(
        {
            "side": spec.side,
            "price_eur_mwh": np.asarray(spec.representatives, dtype=float),
            "energy_mwh": volumes,
        }
    )
