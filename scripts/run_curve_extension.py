"""Exploratory curve-based models and the adjusted curve-based extension (X-Model-inspired).

This is a SEPARATE execution path from ``run_project.py``. The production pipeline and its
observed-curve clearing gate are unchanged and still fail; nothing here weakens that gate.

Protocol: reports/preregistration_curve_extension.md (incl. Amendment 1).
  A  unadjusted curve-based EN / HGB  - production phases 3-6, EXPLORATORY (gate failed)
  B  adjusted curve-based EN / HGB    - P_adj = P_curve + g(X); primary g trained on
                                        chronological out-of-fold (OOF) curve-forecast errors
     references: constant training-mean OOF bias; observed-curve-trained g (sensitivity)
  C  direct EN / HGB and weekly persistence - same frame, identical evaluation intervals

Periods: train 2023-01-01..2024-06-30 | validation 2024-07..09 (all selection) |
exploratory evaluation 2024-Q4 (previously examined) | confirmation 2025-01..09.
Monthly expanding walk-forward for every month 2023-07..2025-09 (2023-01..06 = burn-in).

Stages are cached so that an interrupted or repeated run does not redo expensive work:
  data/processed/extension_interval_dataset.csv, reports/extension/price_class_specs.json,
  reports/extension/model_selection.json, data/interim/extension/wf/<family>_<YYYY-MM>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_project as rp  # noqa: E402  (production helpers, imported unchanged)

from electricity_forecasting.data.coverage import audit_dir  # noqa: E402
from electricity_forecasting.data.dataset import (  # noqa: E402
    aggregate_training_price_weights,
    build_interval_class_table,
    iter_curve_files,
    iter_price_files,
    load_prices,
)
from electricity_forecasting.data.download import find_local_versions  # noqa: E402
from electricity_forecasting.data.parse_omie import parse_curve_file  # noqa: E402
from electricity_forecasting.data.quality import audit_raw_files  # noqa: E402
from electricity_forecasting.evaluation.backtest import chronological_date_split  # noqa: E402
from electricity_forecasting.evaluation.diagnostics import conditional_metric_table  # noqa: E402
from electricity_forecasting.evaluation.metrics import compute_metrics  # noqa: E402
from electricity_forecasting.evaluation.walk_forward import expanding_monthly_predictions  # noqa: E402
from electricity_forecasting.features.build_features import calendar_aware_lag  # noqa: E402
from electricity_forecasting.features.model_matrix import (  # noqa: E402
    build_model_matrix,
    curve_feature_columns,
    direct_price_feature_columns,
)
from electricity_forecasting.market.compression import CandidateClassSpecs, evaluate_class_candidates, reconstruct_pair, select_class_count  # noqa: E402
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve  # noqa: E402
from electricity_forecasting.market.price_classes import PriceClassSpec, reconstruct_class_records  # noqa: E402
from electricity_forecasting.models.benchmarks import weekly_persistence  # noqa: E402
from electricity_forecasting.models.curve_forecast import make_regressor  # noqa: E402
from electricity_forecasting.models.training import select_curve_model, select_direct_model  # noqa: E402
from electricity_forecasting.visualization.plots import (  # noqa: E402
    plot_curve_comparison,
    plot_reconstruction_diagnostics,
)

PRICE = "price_es_eur_mwh"
OUT = Path("reports/extension")
FIG = Path("reports/figures")
CACHE = Path("data/interim/extension")
FAMILIES = ("elastic_net", "hgb")
LABELS = {
    "weekly_persistence_prediction": "Weekly Persistence",
    "direct_elastic_net_prediction": "Direct Elastic Net",
    "direct_hgb_prediction": "Direct Gradient Boosting",
    "curve_elastic_net_prediction": "Unadjusted Curve-Based Elastic Net (exploratory)",
    "curve_hgb_prediction": "Unadjusted Curve-Based Gradient Boosting (exploratory)",
    "adjusted_curve_elastic_net_prediction": "Adjusted Curve-Based Elastic Net",
    "adjusted_curve_hgb_prediction": "Adjusted Curve-Based Gradient Boosting",
}
REFERENCE_LABELS = {
    "bias_baseline_curve_elastic_net_prediction": "Curve-Based EN + constant training bias (baseline)",
    "bias_baseline_curve_hgb_prediction": "Curve-Based GB + constant training bias (baseline)",
    "obs_adjusted_curve_elastic_net_prediction": "Curve-Based EN + observed-curve-trained g (sensitivity)",
    "obs_adjusted_curve_hgb_prediction": "Curve-Based GB + observed-curve-trained g (sensitivity)",
}
ALL_LABELS = {**LABELS, **REFERENCE_LABELS}
PRED_COLS = list(LABELS)
ALL_COLS = list(ALL_LABELS)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------- parallel helpers (data stage)
def _pmap(fn, items: list, workers: int, chunksize: int = 2) -> list:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))


def _quality_one(pair: tuple[str, str], tz: str) -> dict:
    kind, path = pair
    if kind == "curve":
        return audit_raw_files([Path(path)], [], timezone=tz)
    return audit_raw_files([], [Path(path)], timezone=tz)


def _weights_one(path: str, **kw) -> dict[str, pd.DataFrame]:
    return aggregate_training_price_weights([Path(path)], **kw)


def _class_table_one(path: str, **kw) -> pd.DataFrame:
    return build_interval_class_table([Path(path)], **kw)


def _compression_one(path: str, candidates, **kw) -> pd.DataFrame:
    detail, _ = evaluate_class_candidates([Path(path)], candidates, **kw)
    return detail


def _compression_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in detail.groupby("n_classes"):
        m = compute_metrics(g["original_clearing_eur_mwh"].to_numpy(float), g["compressed_clearing_eur_mwh"].to_numpy(float))
        rows.append({
            "n_classes": int(k), **m.to_dict(),
            "invalid_original": int(g["original_clearing_eur_mwh"].isna().sum()),
            "invalid_compressed": int(g["compressed_clearing_eur_mwh"].isna().sum()),
            "mean_unrepresented_mwh": float(g[["supply_underflow_mwh", "supply_overflow_mwh", "demand_underflow_mwh", "demand_overflow_mwh"]].sum(axis=1).mean()),
        })
    return pd.DataFrame(rows).sort_values("n_classes").reset_index(drop=True)


def _spec_from_dict(d: dict) -> PriceClassSpec:
    return PriceClassSpec(side=d["side"], edges=tuple(map(float, d["edges"])), representatives=tuple(map(float, d["representatives"])))


# ---------------------------------------------------------------- adjustment model g(X)
def adjustment_feature_columns(lag_days: list[int]) -> list[str]:
    cols = ["curve_price", "curve_supply_total", "curve_demand_total"]
    for d in lag_days:
        cols += [f"discrepancy_lag_{d}d", f"unmatched_supply_lag_{d}d", f"demand_gap_lag_{d}d"]
    cols += [f"hour_{h:02d}" for h in range(24)] + [f"weekday_{w}" for w in range(7)]
    return cols


def _with_curve_state(frame: pd.DataFrame, *, price, supply_total, demand_total) -> pd.DataFrame:
    out = frame.copy()
    out["curve_price"] = np.asarray(price, dtype=float)
    out["curve_supply_total"] = np.asarray(supply_total, dtype=float)
    out["curve_demand_total"] = np.asarray(demand_total, dtype=float)
    return out


def observed_curve_state(frame: pd.DataFrame, supply_cols: list[str], demand_cols: list[str]) -> pd.DataFrame:
    """Curve state of the compressed *observed* offered curves (sensitivity variant's training rows)."""
    return _with_curve_state(
        frame, price=frame["compressed_curve_clearing_eur_mwh"],
        supply_total=frame[supply_cols].sum(axis=1), demand_total=frame[demand_cols].sum(axis=1),
    )


def forecast_curve_state(frame: pd.DataFrame, implied, pred_supply: pd.DataFrame, pred_demand: pd.DataFrame) -> pd.DataFrame:
    """Curve state of the *forecast* curves (no same-day observed data)."""
    return _with_curve_state(frame, price=implied, supply_total=pred_supply.sum(axis=1), demand_total=pred_demand.sum(axis=1))


def fit_adjustment(train: pd.DataFrame, features: list[str], target: str, kind: str, params: dict, seed: int):
    data = train.dropna(subset=features + [target])
    model = make_regressor(kind, random_seed=seed, **params)
    model.fit(data[features], data[target].to_numpy(float))
    return model


def predict_adjustment(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.full(len(frame), np.nan)
    ok = frame[features].notna().all(axis=1).to_numpy()
    if ok.any():
        out[ok] = np.asarray(model.predict(frame.loc[ok, features]), dtype=float)
    return out


def trailing_hour_median(history: pd.DataFrame, target: pd.DataFrame, window_days: int, col: str = "discrepancy") -> np.ndarray:
    """Reference only: median of `col` over the previous `window_days` delivery days, same local hour."""
    h = history[["delivery_date", "hour", col]].dropna().copy()
    h["delivery_date"] = pd.to_datetime(h["delivery_date"])
    out = np.full(len(target), np.nan)
    for i, (d, hr) in enumerate(zip(pd.to_datetime(target["delivery_date"]).tolist(), target["hour"].tolist())):
        m = (h["delivery_date"] < d) & (h["delivery_date"] >= d - pd.Timedelta(days=int(window_days))) & (h["hour"] == hr)
        if m.any():
            out[i] = float(h.loc[m, col].median())
    return out


def _mae(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(p[ok] - y[ok]))), int(ok.sum())


def _adjustment_candidates(acfg: dict) -> list[tuple[str, dict]]:
    en, hg = acfg["elastic_net"], acfg["hist_gradient_boosting"]
    cands = [("elastic_net", {"alpha": float(a), "l1_ratio": float(r), "max_iter": int(en["max_iter"])})
             for a in en["alpha_grid"] for r in en["l1_ratio_grid"]]
    cands += [("hist_gradient_boosting", {"learning_rate": float(lr), "max_leaf_nodes": int(lv), "max_iter": int(hg["max_iter"]),
                                          "l2_regularization": float(hg["l2_regularization"])})
              for lr in hg["learning_rate_grid"] for lv in hg["max_leaf_nodes_grid"]]
    return cands


def select_adjustment(train: pd.DataFrame, val: pd.DataFrame, target: str, features: list[str], acfg: dict, seed: int) -> dict:
    """Validation selection: score = MAE(P_official, P_curve + g) on walk-forward validation forecasts."""
    y = val[PRICE].to_numpy(float)
    p_curve = val["curve_price"].to_numpy(float)
    rows = []
    for kind, params in _adjustment_candidates(acfg):
        g = predict_adjustment(fit_adjustment(train, features, target, kind, params, seed), val, features)
        mae, n = _mae(y, p_curve + g)
        rows.append({"kind": kind, "params": params, "validation_mae": mae, "n": n})
    best_en = min((r for r in rows if r["kind"] == "elastic_net"), key=lambda r: r["validation_mae"])
    best_hgb = min((r for r in rows if r["kind"] == "hist_gradient_boosting"), key=lambda r: r["validation_mae"])
    thr = float(acfg["hgb_min_relative_improvement_over_en"])
    chosen = best_hgb if best_hgb["validation_mae"] <= (1 - thr) * best_en["validation_mae"] else best_en
    none_mae, none_n = _mae(y, p_curve)
    return {"chosen": chosen, "best_elastic_net": best_en, "best_hist_gradient_boosting": best_hgb,
            "hgb_rule": f"HGB only if validation MAE <= (1 - {thr}) x best EN MAE",
            "no_correction_validation_mae": none_mae, "n": none_n,
            "chosen_beats_no_correction_on_validation": bool(chosen["validation_mae"] < none_mae),
            "candidates": rows}


# ---------------------------------------------------------------- parallel model selection
def _select_one(job: tuple, frames: dict) -> dict:
    which, kind, params, features, targets, seed = job
    train, val = frames["train"], frames["validation"]
    if which == "curve":
        res = select_curve_model(train, val, feature_columns=features, target_columns=targets, candidates=[(kind, params)], random_seed=seed)
    else:
        res = select_direct_model(train, val, feature_columns=features, target_column=PRICE, candidates=[(kind, params)], random_seed=seed)
    return {"kind": res.kind, "params": res.params, "validation_score": res.validation_score}


_WORKER: dict = {}


def _init_worker(payload: dict) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    _WORKER.update(payload)


def _select_worker(job: tuple) -> dict:
    return _select_one(job, _WORKER)


# ---------------------------------------------------------------- parallel walk-forward
def _wf_worker(job: tuple) -> str:
    """One (family, month) refit: all data dated before the month -> predict the month; cached CSV."""
    family, month, out_path = job
    w = _WORKER
    modelling = w["modelling"]
    dates = pd.to_datetime(modelling["delivery_date"])
    m0 = pd.Timestamp(month)
    month_rows = modelling.loc[(dates >= m0) & (dates < m0 + pd.offsets.MonthBegin(1))]
    cs, ds = w["selection"][f"curve_{family}"], w["selection"][f"direct_{family}"]
    pred, diag = expanding_monthly_predictions(
        modelling, month_rows, date_col="delivery_date", curve_feature_columns=w["c_feat"], class_target_columns=w["class_cols"],
        curve_kind=cs["kind"], curve_params=cs["params"], direct_feature_columns=w["d_feat"], direct_target_column=PRICE,
        direct_kind=ds["kind"], direct_params=ds["params"], random_seed=w["seed"],
    )
    spec = w["spec"]
    imp = rp._prediction_clearing_frame(pred, supply_spec=spec.supply, demand_spec=spec.demand,
                                        value_name="p_curve", status_name="p_curve_status", prediction_prefix="pred_")
    pred = pred.merge(imp, on="interval_start_utc", how="left", validate="one_to_one")
    neg = {d["target"]: d["n_negative_raw"] for d in diag["constraint_diagnostics"]}
    pred["n_negative_raw_class_predictions_month"] = int(sum(neg.values()))
    pred["n_class_predictions_month"] = int(len(month_rows) * len(w["class_cols"]))
    tmp = Path(str(out_path) + ".tmp")
    pred.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    return str(out_path)


# ---------------------------------------------------------------- evaluation helpers
def _metric_rows(frame: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    y = frame[PRICE].to_numpy(float)
    pers = compute_metrics(y, frame["weekly_persistence_prediction"].to_numpy(float))
    for col in ALL_COLS:
        m = compute_metrics(y, frame[col].to_numpy(float))
        rows.append({"period": period, "model": ALL_LABELS[col], "column": col, "role": "primary" if col in LABELS else "reference",
                     "n": m.n, "mae": m.mae, "rmse": m.rmse, "bias": m.bias, "median_absolute_error": m.median_absolute_error,
                     "mae_improvement_vs_persistence_pct": 100.0 * (1 - m.mae / pers.mae)})
    return rows


def _component(name: str, err: np.ndarray) -> dict:
    err = err[np.isfinite(err)]
    return {"component": name, "n": int(err.size), "signed_mean": float(err.mean()), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))), "sd": float(err.std())}


def decomposition(frame: pd.DataFrame, family: str) -> list[dict]:
    off = frame[PRICE].to_numpy(float)
    obs = frame["observed_curve_clearing_eur_mwh"].to_numpy(float)
    comp = frame["compressed_curve_clearing_eur_mwh"].to_numpy(float)
    pc = frame[f"curve_{family}_prediction"].to_numpy(float)
    d = off - comp
    rows = [
        _component("1_clearing_approximation (P_obs - P_official)", obs - off),
        _component("2_price_class_compression (P_comp - P_obs)", comp - obs),
        _component("3_curve_forecasting (P_curve - P_comp)", pc - comp),
        _component("total_unadjusted (P_curve - P_official) = 1+2+3", pc - off),
    ]
    for tag, col in (("oof_g", f"adjustment_{family}"), ("constant_bias", f"bias_constant_{family}"), ("observed_g", f"obs_adjustment_{family}")):
        g = frame[col].to_numpy(float)
        rows.append(_component(f"4_adjustment_estimation[{tag}] (g_hat - d)", g - d))
        rows.append(_component(f"total_adjusted[{tag}] (P_curve + g_hat - P_official) = 3+4", pc + g - off))
    # Exact signed identities, checked numerically.
    assert np.allclose((obs - off) + (comp - obs) + (pc - comp), pc - off, equal_nan=True)
    return rows


def _plot_comparison(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = list(dict.fromkeys(metrics["period"]))
    fig, axes = plt.subplots(1, len(periods), figsize=(7.5 * len(periods), 6), sharey=True, squeeze=False)
    colors = {"Weekly": "#8c8c8c", "Direct": "#1f77b4", "Unadjusted": "#d62728", "Adjusted": "#2ca02c", "Curve-Based": "#bcbd22"}
    for ax, period in zip(axes[0], periods):
        sub = metrics.loc[metrics["period"] == period]
        names = [m.replace(" (exploratory)", "*") for m in sub["model"]]
        cols = [colors[next(k for k in colors if m.startswith(k))] for m in sub["model"]]
        bars = ax.barh(names, sub["mae"], color=cols)
        for b, v, role in zip(bars, sub["mae"], sub["role"]):
            if role != "primary":
                b.set_alpha(0.45)
            ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
        ax.set_title(f"{period}  (N = {int(sub['n'].iloc[0])})", fontsize=10)
        ax.set_xlabel("MAE (EUR/MWh)")
        ax.invert_yaxis()
    axes[0][0].text(0, -0.12, "* exploratory: curve clearing gate failed; faded bars = reference variants", transform=axes[0][0].transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_timeseries(frame: pd.DataFrame, path: Path, title: str, days: int = 14) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame.sort_values("interval_start_utc")
    last = data["interval_start_utc"].max()
    data = data.loc[data["interval_start_utc"] > last - pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["interval_start_utc"], data[PRICE], color="k", lw=1.8, label="Official OMIE Spain")
    for col, c in (("direct_elastic_net_prediction", "#1f77b4"), ("curve_elastic_net_prediction", "#d62728"),
                   ("adjusted_curve_elastic_net_prediction", "#2ca02c"), ("adjusted_curve_hgb_prediction", "#17becf")):
        ax.plot(data["interval_start_utc"], data[col], lw=1.0, color=c, label=ALL_LABELS[col])
    ax.set_ylabel("EUR/MWh")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_conditional(cond: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, period in zip(axes, ("exploratory_2024Q4", "confirmation_2025")):
        sub = cond.loc[(cond["period"] == period) & cond["subset"].str.startswith("hour_")].copy()
        sub["h"] = sub["subset"].str[5:].astype(int)
        for col in PRED_COLS:
            s = sub.loc[sub["model"] == col].sort_values("h")
            ax.plot(s["h"], s["mae"], marker="o", ms=2.5, lw=1, label=LABELS[col].replace(" (exploratory)", "*"))
        ax.set_title(f"MAE by local hour - {period}")
        ax.set_xlabel("hour")
        ax.set_ylabel("MAE (EUR/MWh)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_forecast_curves(ax, observed: CurvePair, compressed: CurvePair, forecasts: dict[str, CurvePair], prices: dict[str, float], title: str) -> None:
    def step(curve: pd.DataFrame, **kw):
        ax.step(curve["cumulative_mwh"], curve["price_eur_mwh"], where="post", **kw)

    step(observed.supply, color="0.6", lw=1.0, label="observed offered supply/demand")
    step(observed.demand, color="0.6", lw=1.0)
    step(compressed.supply, color="k", lw=1.0, ls="--", label="observed, 40-class compressed")
    step(compressed.demand, color="k", lw=1.0, ls="--")
    for (name, pair), c in zip(forecasts.items(), ("#d62728", "#ff7f0e")):
        step(pair.supply, color=c, lw=1.4, label=f"forecast ({name})")
        step(pair.demand, color=c, lw=1.4)
    for (name, val), c in zip(prices.items(), ("k", "#d62728", "#ff7f0e", "#2ca02c", "#17becf")):
        ax.axhline(val, color=c, lw=0.8, ls=":", label=f"{name}: {val:.1f}")
    hi = max(prices.values())
    ax.set_ylim(-60, max(250.0, 1.8 * hi))
    ax.set_xlabel("cumulative MWh")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/extension.yaml"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true", help="Ignore cached stages and recompute everything")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")  # inherited by spawned workers; avoids thread oversubscription
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ext, acfg = cfg["extension"], cfg["extension"]["adjustment"]
    raw = Path(cfg["data"]["raw_dir"])
    processed = Path(cfg["data"]["processed_dir"])
    tz = cfg["project"]["timezone"]
    seed = int(cfg["project"]["random_seed"])
    splits = cfg["splits"]
    kw = dict(preferred_country=cfg["omie"]["preferred_country"], fallback_country=cfg["omie"]["fallback_country"],
              curve_flag=cfg["omie"]["offered_flag"], timezone=tz)
    global OUT, FIG, CACHE
    OUT = Path(ext.get("report_dir", OUT))
    FIG = Path(ext.get("figure_dir", FIG))
    CACHE = Path(ext.get("cache_dir", CACHE))
    for p in (OUT, FIG, CACHE / "wf", processed):
        p.mkdir(parents=True, exist_ok=True)
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    lag_days = [int(x) for x in acfg["lag_days"]]
    oof_start = pd.Timestamp(ext.get("oof_start", "2023-07-01"))

    # ---- Gate 1: coverage + per-file quality over the full extended period (fail closed; always re-run, cheap).
    coverage = {
        "curves": audit_dir(raw / "curves", cfg["omie"]["curves_prefix"], date.fromisoformat(start), date.fromisoformat(end)),
        "prices": audit_dir(raw / "prices", cfg["omie"]["prices_prefix"], date.fromisoformat(start), date.fromisoformat(end)),
    }
    _write_json(OUT / "raw_coverage.json", coverage)
    if any(v["status"] != "VERIFIED" for v in coverage.values()):
        raise SystemExit("BLOCKED: extended-period raw coverage incomplete; see reports/extension/raw_coverage.json")
    curve_files = list(iter_curve_files(raw / "curves", start, end))
    price_files = list(iter_price_files(raw / "prices", start, end))
    qual = _pmap(partial(_quality_one, tz=tz), [("curve", str(f)) for f in curve_files] + [("price", str(f)) for f in price_files], args.workers)
    failed = [f for q in qual for f in q["curve_files_failed"] + q["price_files_failed"]]
    _write_json(OUT / "data_quality.json", {"status": "VERIFIED" if not failed else "FAILED", "files_checked": len(qual), "failed": failed})
    if failed:
        raise SystemExit(f"FAILED: {len(failed)} raw files failed validation, e.g. {failed[:5]}")
    print(f"[gate] coverage + quality VERIFIED for {len(curve_files)} curve and {len(price_files)} price files", flush=True)
    prices = load_prices(price_files, timezone=tz)

    # ---- Stage 1: price classes (training only; k on validation) + interval dataset.
    dataset_path = processed / "extension_interval_dataset.csv"
    specs_path = OUT / "price_class_specs.json"
    detail_path = OUT / "compression_detail.csv"
    need_classes = args.rebuild or not specs_path.exists() or not detail_path.exists()
    if need_classes:
        train_files = list(iter_curve_files(raw / "curves", start, splits["train_end"]))
        parts = _pmap(partial(_weights_one, **kw), [str(f) for f in train_files], args.workers)
        weights = {s: pd.concat([p[s] for p in parts]).groupby(["side", "price_eur_mwh"], as_index=False)["energy_mwh"].sum() for s in ("C", "V")}
        candidates, cand_failures = [], {}
        for kk in cfg["price_classes"]["candidates"]:
            try:  # identical to production: an unfittable candidate is recorded, not silently dropped
                candidates.append(rp._fit_specs_from_weights(weights, int(kk)))
            except ValueError as exc:
                cand_failures[str(kk)] = str(exc)
        _write_json(OUT / "price_class_candidate_failures.json", cand_failures)
        if not candidates:
            raise SystemExit("FAILED: no requested price-class candidate can be fitted from training data.")
        val_start = (pd.Timestamp(splits["train_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        val_files = list(iter_curve_files(raw / "curves", val_start, splits["validation_end"]))
        detail = pd.concat(_pmap(partial(_compression_one, candidates=candidates, **kw), [str(f) for f in val_files], args.workers))
        detail.to_csv(detail_path, index=False)
        comp_summary = _compression_summary(detail)
        comp_summary.to_csv(OUT / "compression_summary.csv", index=False)
        k = select_class_count(comp_summary, tie_tolerance_eur_mwh=float(cfg["price_classes"]["tie_tolerance_eur_mwh"]))
        chosen_c = next(c for c in candidates if c.n_classes == k)
        if specs_path.exists() and not args.rebuild:
            cached = json.loads(specs_path.read_text(encoding="utf-8"))
            if cached["n_classes"] != k or list(cached["supply"]["edges"]) != list(chosen_c.supply.edges):
                raise SystemExit("FAILED: recomputed price classes differ from the cached specification; rerun with --rebuild")
        _write_json(specs_path, {"n_classes": k, "supply": rp._spec_to_dict(chosen_c.supply), "demand": rp._spec_to_dict(chosen_c.demand),
                                 "selection_rule": "smallest k within tie tolerance of best validation compression MAE; classes fit on training files only"})
    detail = pd.read_csv(detail_path)
    plot_reconstruction_diagnostics(detail, output_path=FIG / "extension_figure7_reconstruction_diagnostics.png")
    spec_json = json.loads(specs_path.read_text(encoding="utf-8"))
    k = int(spec_json["n_classes"])
    chosen = CandidateClassSpecs(n_classes=k, supply=_spec_from_dict(spec_json["supply"]), demand=_spec_from_dict(spec_json["demand"]))
    print(f"[classes] k={k} (validation compression MAE table: {OUT / 'compression_summary.csv'})", flush=True)

    supply_cols = [f"supply_{i:02d}" for i in range(k)]
    demand_cols = [f"demand_{i:02d}" for i in range(k)]
    class_cols = supply_cols + demand_cols
    if args.rebuild or not dataset_path.exists():
        audit_path = processed / "clearing_mechanism_audit.csv"
        if not audit_path.exists():
            raise SystemExit("BLOCKED: run scripts/audit_clearing_mechanism.py --config configs/extension.yaml first")
        audit = pd.read_csv(audit_path)
        audit["interval_start_utc"] = pd.to_datetime(audit["interval_start_utc"], utc=True)
        if len(audit) != len(prices) or not audit["interval_start_utc"].isin(prices["interval_start_utc"]).all():
            raise SystemExit("BLOCKED: clearing_mechanism_audit.csv does not cover the configured period exactly")
        class_table = pd.concat(_pmap(
            partial(_class_table_one, supply_spec=chosen.supply, demand_spec=chosen.demand, include_observed_clearing=False, **kw),
            [str(f) for f in curve_files], args.workers,
        )).sort_values("interval_start_utc").reset_index(drop=True)
        class_table = rp._add_compressed_clearing(class_table, chosen.supply, chosen.demand)
        audit_cols = ["interval_start_utc", "p_offered", "supply_unmatched_at_or_below_p_mwh", "demand_matched_not_in_offered_mwh"]
        empirical = class_table.merge(audit[audit_cols], on="interval_start_utc", how="inner", validate="one_to_one").rename(columns={
            "p_offered": "observed_curve_clearing_eur_mwh", "supply_unmatched_at_or_below_p_mwh": "unmatched_supply",
            "demand_matched_not_in_offered_mwh": "demand_gap"})
        empirical = rp._merge_price_and_curves(empirical, prices)
        empirical["geography_supported"] = rp._supported_geography_mask(empirical, float(cfg["market_validation"]["zonal_price_tolerance_eur_mwh"]))
        if not empirical["geography_supported"].all():
            raise SystemExit(f"FAILED: {int((~empirical['geography_supported']).sum())} intervals have unsupported curve geography")
        empirical["discrepancy"] = empirical[PRICE] - empirical["compressed_curve_clearing_eur_mwh"]
        matrix = build_model_matrix(empirical, class_columns=class_cols, price_column=PRICE)
        for d in lag_days:
            matrix = calendar_aware_lag(matrix, value_columns=["discrepancy", "unmatched_supply", "demand_gap"], days=d, align_col="hour")
        for h in range(24):
            matrix[f"hour_{h:02d}"] = (matrix["hour"] == h).astype(int)
        for w in range(7):
            matrix[f"weekday_{w}"] = (matrix["weekday"] == w).astype(int)
        matrix.to_csv(dataset_path, index=False)
    matrix = pd.read_csv(dataset_path)
    matrix["interval_start_utc"] = pd.to_datetime(matrix["interval_start_utc"], utc=True)
    matrix["delivery_date"] = pd.to_datetime(matrix["delivery_date"])
    if len(matrix) != len(prices) or matrix["interval_start_utc"].duplicated().any():
        raise SystemExit("FAILED: cached interval dataset does not match the configured period; rerun with --rebuild")
    print(f"[data] interval dataset: {len(matrix)} intervals, {matrix['delivery_date'].min():%Y-%m-%d} .. {matrix['delivery_date'].max():%Y-%m-%d}", flush=True)

    c_feat = curve_feature_columns(class_cols)
    d_feat = direct_price_feature_columns()
    g_feat = adjustment_feature_columns(lag_days)
    modelling = matrix.dropna(subset=list(dict.fromkeys(c_feat + d_feat + class_cols + [PRICE]))).reset_index(drop=True)
    split = chronological_date_split(modelling, date_col="delivery_date", train_end=splits["train_end"],
                                     validation_end=splits["validation_end"], test_start=splits["test_start"])
    periods = {name: [str(f["delivery_date"].min().date()), str(f["delivery_date"].max().date()), int(len(f))]
               for name, f in (("train", split.train), ("validation", split.validation), ("evaluation", split.test))}
    print(f"[split] {periods}", flush=True)

    # ---- Stage 2: hyperparameter selection on validation (production rules), parallel over candidates, cached.
    sel_path = OUT / "model_selection.json"
    if args.rebuild or not sel_path.exists():
        jobs = ([("curve", kd, pr, c_feat, class_cols, seed) for kd, pr in rp._elastic_candidates(cfg)]
                + [("curve", kd, pr, c_feat, class_cols, seed) for kd, pr in rp._hgb_candidates(cfg)]
                + [("direct", kd, pr, d_feat, None, seed) for kd, pr in rp._elastic_candidates(cfg)]
                + [("direct", kd, pr, d_feat, None, seed) for kd, pr in rp._hgb_candidates(cfg)])
        payload = {"train": split.train[list(dict.fromkeys(c_feat + d_feat + class_cols + [PRICE]))],
                   "validation": split.validation[list(dict.fromkeys(c_feat + d_feat + class_cols + [PRICE]))]}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(payload,)) as pool:
            results = list(pool.map(_select_worker, jobs, chunksize=1))
        selection = {}
        for name, which, kind in (("curve_elastic_net", "curve", "elastic_net"), ("curve_hgb", "curve", "hist_gradient_boosting"),
                                  ("direct_elastic_net", "direct", "elastic_net"), ("direct_hgb", "direct", "hist_gradient_boosting")):
            cands = [r for j, r in zip(jobs, results) if j[0] == which and r["kind"] == kind]
            best = min(cands, key=lambda r: r["validation_score"])  # same argmin rule as the library
            selection[name] = {**best, "candidate_scores": cands,
                               "score_definition": "scaled multi-target class MAE" if which == "curve" else "price MAE"}
        _write_json(sel_path, {"selected_on": "validation 2024-07-01..2024-09-30 only", "periods": periods, **selection})
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    print("[select] " + json.dumps({n: selection[n]["params"] for n in ("curve_elastic_net", "curve_hgb", "direct_elastic_net", "direct_hgb")}), flush=True)

    # ---- Stage 3: monthly expanding walk-forward 2023-07..2025-09 (OOF for training/validation; evaluation forecasts).
    months = pd.period_range(oof_start, pd.Timestamp(end), freq="M")
    import hashlib

    key = hashlib.sha256(json.dumps({"config": cfg, "classes": spec_json, "selection": {n: selection[n]["params"] for n in
                         ("curve_elastic_net", "curve_hgb", "direct_elastic_net", "direct_hgb")}}, sort_keys=True, default=str).encode()).hexdigest()
    key_path = CACHE / "wf" / "cache_key.txt"
    if key_path.exists() and key_path.read_text(encoding="utf-8").strip() != key:
        if not args.rebuild:
            raise SystemExit(f"FAILED: walk-forward cache in {CACHE / 'wf'} was produced with a different configuration; rerun with --rebuild")
        for old in (CACHE / "wf").glob("*.csv"):
            old.unlink()
    key_path.write_text(key, encoding="utf-8")
    jobs = []
    for fam in FAMILIES:
        for m in months:
            path = CACHE / "wf" / f"{fam}_{m}.csv"
            if args.rebuild or not path.exists():
                jobs.append((fam, str(m.start_time.date()), str(path)))
    if jobs:
        print(f"[walk-forward] {len(jobs)} family-month refits to compute ({len(FAMILIES) * len(months) - len(jobs)} cached)", flush=True)
        keep = list(dict.fromkeys(["interval_start_utc", "delivery_date", "period", PRICE] + c_feat + d_feat + class_cols))
        payload = {"modelling": modelling[keep], "selection": selection, "c_feat": c_feat, "d_feat": d_feat,
                   "class_cols": class_cols, "seed": seed, "spec": chosen}
        # Largest training windows first so the pool finishes evenly.
        jobs.sort(key=lambda j: j[1], reverse=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(payload,)) as pool:
            for i, done in enumerate(pool.map(_wf_worker, jobs, chunksize=1), 1):
                print(f"[walk-forward] {i}/{len(jobs)} {Path(done).name}", flush=True)

    fam_frames: dict[str, pd.DataFrame] = {}
    constraint = {}
    for fam in FAMILIES:
        parts = [pd.read_csv(CACHE / "wf" / f"{fam}_{m}.csv") for m in months]
        wf = pd.concat(parts, ignore_index=True)
        wf["interval_start_utc"] = pd.to_datetime(wf["interval_start_utc"], utc=True)
        pred_s = wf[[f"pred_{c}" for c in supply_cols]]
        pred_d = wf[[f"pred_{c}" for c in demand_cols]]
        base = modelling.drop(columns=[PRICE]).merge(wf[["interval_start_utc", PRICE, "p_curve", "p_curve_status", "direct_prediction"]],
                                                     on="interval_start_utc", how="inner", validate="one_to_one")
        if len(base) != len(wf):
            raise SystemExit("FAILED: walk-forward predictions do not align with the modelling frame")
        base = base.set_index("interval_start_utc").loc[wf["interval_start_utc"]].reset_index()
        fc = forecast_curve_state(base, base["p_curve"], pred_s, pred_d)
        fc["oof_residual"] = fc[PRICE] - fc["p_curve"]
        fam_frames[fam] = fc
        per_month = wf.groupby("refit_month")[["n_negative_raw_class_predictions_month", "n_class_predictions_month"]].first()
        constraint[fam] = {
            "negative_raw_class_predictions_clipped_to_zero": int(per_month.iloc[:, 0].sum()),
            "class_predictions_total": int(per_month.iloc[:, 1].sum()),
            "share": float(per_month.iloc[:, 0].sum() / per_month.iloc[:, 1].sum()),
            "implied_price_status_counts": {str(a): int(b) for a, b in wf["p_curve_status"].value_counts(dropna=False).items()},
        }
    _write_json(OUT / "curve_validity_diagnostics.json", {
        "note": "Forecast class volumes are clipped at zero (documented production post-processing); curves are then rebuilt as "
                "monotone step functions at training-derived class prices, so every forecast curve is economically valid "
                "(non-decreasing supply, non-increasing demand, non-negative volumes).", **constraint})

    # ---- Stage 4: corrections. Selection on validation; evaluation refits monthly on data before each month.
    train_end, val_end = pd.Timestamp(splits["train_end"]), pd.Timestamp(splits["validation_end"])
    obs_state = observed_curve_state(modelling, supply_cols, demand_cols)
    adj_selection, eval_frames = {}, {}
    eval_start = pd.Timestamp(splits["test_start"])
    for fam in FAMILIES:
        fc = fam_frames[fam]
        dates = pd.to_datetime(fc["delivery_date"])
        oof_train = fc.loc[(dates >= oof_start) & (dates <= train_end)]
        oof_val = fc.loc[(dates > train_end) & (dates <= val_end)]
        sel_oof = select_adjustment(oof_train, oof_val, "oof_residual", g_feat, acfg, seed)
        obs_train = obs_state.loc[pd.to_datetime(obs_state["delivery_date"]) <= train_end]
        sel_obs = select_adjustment(obs_train, oof_val, "discrepancy", g_feat, acfg, seed)
        bias_const = float(oof_train["oof_residual"].mean())
        y_val, p_val = oof_val[PRICE].to_numpy(float), oof_val["p_curve"].to_numpy(float)
        med = trailing_hour_median(fc.loc[dates <= val_end], oof_val, int(acfg["trailing_median_window_days"]), col="oof_residual")
        adj_selection[fam] = {
            "primary_oof_trained": sel_oof, "sensitivity_observed_trained": sel_obs,
            "bias_baseline": {"constant": bias_const, "estimated_on": [str(oof_start.date()), str(train_end.date())],
                              "validation_mae": _mae(y_val, p_val + bias_const)[0]},
            "reference_trailing_28d_same_hour_median_oof_residual": {"validation_mae": _mae(y_val, p_val + med)[0]},
            "oof_training_rows": int(len(oof_train)), "validation_rows": int(len(oof_val)),
        }
        print(f"[adjust:{fam}] validation MAE  none {sel_oof['no_correction_validation_mae']:.2f} | bias {adj_selection[fam]['bias_baseline']['validation_mae']:.2f} "
              f"| OOF-g {sel_oof['chosen']['kind']} {sel_oof['chosen']['validation_mae']:.2f} | obs-g {sel_obs['chosen']['kind']} {sel_obs['chosen']['validation_mae']:.2f}", flush=True)

        ev = fc.loc[dates >= eval_start].copy()
        ev_dates = pd.to_datetime(ev["delivery_date"])
        g_oof, g_obs = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
        for m in pd.period_range(eval_start, pd.Timestamp(end), freq="M"):
            m0, m1 = m.start_time, m.end_time
            mask = ((ev_dates >= m0) & (ev_dates <= m1)).to_numpy()
            hist_oof = fc.loc[(dates >= oof_start) & (dates < m0)]
            hist_obs = obs_state.loc[pd.to_datetime(obs_state["delivery_date"]) < m0]
            c1, c2 = sel_oof["chosen"], sel_obs["chosen"]
            g_oof[mask] = predict_adjustment(fit_adjustment(hist_oof, g_feat, "oof_residual", c1["kind"], c1["params"], seed), ev.loc[mask], g_feat)
            g_obs[mask] = predict_adjustment(fit_adjustment(hist_obs, g_feat, "discrepancy", c2["kind"], c2["params"], seed), ev.loc[mask], g_feat)
        ev[f"curve_{fam}_prediction"] = ev["p_curve"]
        ev[f"direct_{fam}_prediction"] = ev["direct_prediction"]
        ev[f"adjustment_{fam}"] = g_oof
        ev[f"obs_adjustment_{fam}"] = g_obs
        ev[f"bias_constant_{fam}"] = bias_const
        ev[f"adjusted_curve_{fam}_prediction"] = ev["p_curve"] + g_oof
        ev[f"obs_adjusted_curve_{fam}_prediction"] = ev["p_curve"] + g_obs
        ev[f"bias_baseline_curve_{fam}_prediction"] = ev["p_curve"] + bias_const
        ev[f"p_curve_status_{fam}"] = ev["p_curve_status"]
        eval_frames[fam] = ev
    _write_json(OUT / "adjustment_selection.json", adj_selection)

    # ---- Stage 5: assemble identical evaluation intervals, metrics, decomposition, conditional analysis.
    base_cols = ["interval_start_utc", "delivery_date", "period", "hour", PRICE, "observed_curve_clearing_eur_mwh",
                 "compressed_curve_clearing_eur_mwh", "discrepancy"]
    preds = eval_frames["elastic_net"][base_cols].copy()
    for fam in FAMILIES:
        cols = [f"curve_{fam}_prediction", f"direct_{fam}_prediction", f"adjustment_{fam}", f"obs_adjustment_{fam}", f"bias_constant_{fam}",
                f"adjusted_curve_{fam}_prediction", f"obs_adjusted_curve_{fam}_prediction", f"bias_baseline_curve_{fam}_prediction", f"p_curve_status_{fam}"]
        preds = preds.merge(eval_frames[fam][["interval_start_utc", *cols]], on="interval_start_utc", how="outer", validate="one_to_one")
    price_hist = prices[["interval_start_utc", "delivery_date", "period", PRICE]].copy()
    price_hist["hour"] = price_hist["interval_start_utc"].dt.tz_convert(tz).dt.hour
    pers = weekly_persistence(price_hist, align_col="hour")
    preds = preds.merge(pers[["interval_start_utc", "prediction"]].rename(columns={"prediction": "weekly_persistence_prediction"}),
                        on="interval_start_utc", how="left", validate="one_to_one")
    dates = pd.to_datetime(preds["delivery_date"])
    preds["period_label"] = np.where(dates <= pd.Timestamp(ext["exploratory_eval_end"]), "exploratory_2024Q4",
                                     np.where((dates >= pd.Timestamp(ext["confirmation_start"])) & (dates <= pd.Timestamp(ext["confirmation_end"])),
                                              "confirmation_2025", "other"))
    all_eval_intervals = prices.loc[prices["delivery_date"].astype(str) >= splits["test_start"], "interval_start_utc"]
    full_index = pd.DatetimeIndex(all_eval_intervals.sort_values(), name="interval_start_utc")
    preds = preds.set_index("interval_start_utc").reindex(full_index).reset_index()  # rows lost to lags stay visible as exclusions
    fallback_label = pd.Series(np.where(preds["interval_start_utc"].dt.tz_convert(tz).dt.year == 2024, "exploratory_2024Q4", "confirmation_2025"),
                               index=preds.index)
    preds["period_label"] = preds["period_label"].where(preds["period_label"].notna(), fallback_label)
    finite = np.isfinite(preds[[PRICE, *ALL_COLS]].to_numpy(float)).all(axis=1)
    preds["evaluation_included"] = finite
    preds.to_csv(processed / "extension_test_predictions.csv", index=False)

    metric_rows, decomp_rows, cond_rows, exclusions = [], [], [], {}
    for label in ("exploratory_2024Q4", "confirmation_2025"):
        sub_all = preds.loc[preds["period_label"] == label]
        sub = sub_all.loc[sub_all["evaluation_included"]]
        exclusions[label] = {"total_intervals": int(len(sub_all)), "common_finite": int(len(sub)), "excluded": int(len(sub_all) - len(sub)),
                             "non_finite_by_column": {c: int((~np.isfinite(sub_all[c].to_numpy(float))).sum()) for c in [PRICE, *ALL_COLS]}}
        if sub.empty:
            continue
        metric_rows += _metric_rows(sub, label)
        for fam in FAMILIES:
            decomp_rows += [{"period": label, "curve_family": fam, **r} for r in decomposition(sub, fam)]
        cond = conditional_metric_table(sub.sort_values("interval_start_utc"), observed_col=PRICE, prediction_cols=ALL_COLS, training_prices=split.train[PRICE])
        cond.insert(0, "period", label)
        cond_rows.append(cond)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(decomp_rows).to_csv(OUT / "error_decomposition.csv", index=False)
    conditional = pd.concat(cond_rows, ignore_index=True)
    conditional.to_csv(OUT / "conditional_metrics.csv", index=False)

    by_month = []
    inc = preds.loc[preds["evaluation_included"]]
    for (label, month), g in inc.groupby(["period_label", pd.to_datetime(inc["delivery_date"]).dt.to_period("M").astype(str)]):
        for col in ALL_COLS:
            m = compute_metrics(g[PRICE].to_numpy(float), g[col].to_numpy(float))
            by_month.append({"period": label, "month": month, "model": ALL_LABELS[col], "n": m.n, "mae": m.mae, "bias": m.bias})
    pd.DataFrame(by_month).to_csv(OUT / "monthly_metrics.csv", index=False)

    _write_json(OUT / "summary.json", {
        "status": "EXECUTED - curve-based results EXPLORATORY; production clearing gate still FAILED (unchanged)",
        "protocol": "reports/preregistration_curve_extension.md (with Amendment 1)",
        "n_classes": k, "periods": periods, "walk_forward_months": [str(months[0]), str(months[-1])],
        "exclusions": exclusions, "curve_validity": constraint,
        "headline": metrics.loc[metrics["role"] == "primary"].to_dict(orient="records"),
    })

    # ---- Stage 6: figures from actual outputs.
    _plot_comparison(metrics, FIG / "extension_model_comparison.png")
    _plot_conditional(conditional, FIG / "extension_conditional_mae_by_hour.png")
    for label, title in (("exploratory_2024Q4", "Last two weeks of 2024-Q4 (exploratory)"), ("confirmation_2025", "Last two weeks of the 2025 confirmation period")):
        sub = preds.loc[(preds["period_label"] == label) & preds["evaluation_included"]]
        if not sub.empty:
            _plot_timeseries(sub, FIG / f"extension_predictions_{label}.png", title)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed, a-priori intervals (not chosen by error): first evaluation day and first Monday of June 2025, 20:00 local.
    fixed = [pd.Timestamp(f"{splits['test_start']} 20:00").tz_localize(tz), pd.Timestamp("2025-06-02 20:00").tz_localize(tz)]
    evaluated = set(preds.loc[preds["evaluation_included"], "interval_start_utc"])
    fixed = [t for t in fixed if t.tz_convert("UTC") in evaluated]
    fig, axes = plt.subplots(1, len(fixed), figsize=(8 * len(fixed), 6))
    for ax, ts_local in zip(np.atleast_1d(axes), fixed):
        ts = ts_local.tz_convert("UTC")
        row = preds.loc[preds["interval_start_utc"] == ts].iloc[0]
        day = ts_local.date()
        rows = parse_curve_file(find_local_versions(raw / "curves", cfg["omie"]["curves_prefix"], day)[-1], timezone=tz)
        rows = rows.loc[rows["interval_start_utc"] == ts]
        country, offered = rp.select_country_records_for_interval(rows, preferred_country=kw["preferred_country"],
                                                                  fallback_country=kw["fallback_country"], curve_flag=kw["curve_flag"])
        observed = CurvePair(ts, country, build_supply_curve(offered), build_demand_curve(offered))
        compressed, _ = reconstruct_pair(offered, interval_start_utc=ts, country=country, supply_spec=chosen.supply, demand_spec=chosen.demand)
        forecasts = {}
        for fam in FAMILIES:
            wf_row = pd.read_csv(CACHE / "wf" / f"{fam}_{pd.Period(day, 'M')}.csv")
            wf_row = wf_row.loc[pd.to_datetime(wf_row["interval_start_utc"], utc=True) == ts].iloc[0]
            recs = pd.concat([reconstruct_class_records(wf_row[[f"pred_{c}" for c in supply_cols]].to_numpy(float), chosen.supply),
                              reconstruct_class_records(wf_row[[f"pred_{c}" for c in demand_cols]].to_numpy(float), chosen.demand)], ignore_index=True)
            forecasts["EN" if fam == "elastic_net" else "HGB"] = CurvePair(ts, "forecast", build_supply_curve(recs), build_demand_curve(recs))
        _plot_forecast_curves(ax, observed, compressed, forecasts, {
            "official": float(row[PRICE]), "P_curve EN": float(row["curve_elastic_net_prediction"]),
            "P_curve HGB": float(row["curve_hgb_prediction"]), "P_adj EN": float(row["adjusted_curve_elastic_net_prediction"]),
            "P_adj HGB": float(row["adjusted_curve_hgb_prediction"]),
        }, f"{ts_local:%Y-%m-%d %H:%M} local ({country}); forecast issued before D-1 12:00")
        if ts_local == fixed[0]:
            plot_curve_comparison(observed, compressed, output_path=FIG / "extension_figure2_price_class_reconstruction.png")
    fig.tight_layout()
    fig.savefig(FIG / "extension_figure3_forecast_vs_observed_curves.png", dpi=150)
    plt.close(fig)

    show = metrics.loc[:, ["period", "model", "n", "mae", "rmse", "bias", "mae_improvement_vs_persistence_pct"]]
    print(show.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
