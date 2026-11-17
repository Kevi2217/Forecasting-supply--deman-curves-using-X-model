"""Supplementary Diebold-Mariano comparisons for the curve extension (not pre-registered).

Loss = absolute error on the common evaluation mask written by run_curve_extension.py.
Hourly errors within a delivery day are strongly dependent (one auction per day), so the
loss differential is first averaged per delivery day; the DM statistic then uses a
Newey-West long-run variance with `lag` days. Two-sided p-values, normal approximation,
with the Harvey-Leybourne-Newbold small-sample correction.

Output: reports/extension/dm_tests.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

PRICE = "price_es_eur_mwh"
PAIRS = [
    ("adjusted_curve_elastic_net_prediction", "curve_elastic_net_prediction"),
    ("adjusted_curve_hgb_prediction", "curve_hgb_prediction"),
    ("adjusted_curve_elastic_net_prediction", "bias_baseline_curve_elastic_net_prediction"),
    ("adjusted_curve_hgb_prediction", "bias_baseline_curve_hgb_prediction"),
    ("adjusted_curve_elastic_net_prediction", "direct_elastic_net_prediction"),
    ("adjusted_curve_hgb_prediction", "direct_hgb_prediction"),
    ("obs_adjusted_curve_elastic_net_prediction", "direct_elastic_net_prediction"),
    ("obs_adjusted_curve_hgb_prediction", "direct_hgb_prediction"),
    ("adjusted_curve_hgb_prediction", "weekly_persistence_prediction"),
    ("direct_hgb_prediction", "direct_elastic_net_prediction"),
]


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = float(x @ x) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * float(x[k:] @ x[:-k]) / n
    return v


def dm_test(daily_diff: np.ndarray, lag: int) -> tuple[float, float]:
    n = len(daily_diff)
    var = newey_west_variance(daily_diff, lag)
    stat = daily_diff.mean() / math.sqrt(var / n)
    h = lag + 1
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", type=Path, default=Path("data/processed/extension_test_predictions.csv"))
    ap.add_argument("--lag", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("reports/extension/dm_tests.csv"))
    args = ap.parse_args()
    p = pd.read_csv(args.predictions)
    p = p.loc[p["evaluation_included"]]
    rows = []
    for period, g in p.groupby("period_label"):
        for a, b in PAIRS:
            d = (g[a] - g[PRICE]).abs() - (g[b] - g[PRICE]).abs()
            daily = d.groupby(g["delivery_date"]).mean().to_numpy(float)
            stat, pval = dm_test(daily, args.lag)
            rows.append({"period": period, "model_a": a, "model_b": b, "n_hours": int(len(g)), "n_days": int(len(daily)),
                         "mae_a_minus_mae_b": float(d.mean()), "dm_stat": stat, "p_value": pval,
                         "interpretation": "A better" if (pval < 0.05 and d.mean() < 0) else "B better" if (pval < 0.05 and d.mean() > 0) else "no significant difference (5%)"})
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
