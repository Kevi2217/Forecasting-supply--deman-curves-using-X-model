"""Out-of-sample evaluation of the curve-independent price benchmarks.

Weekly persistence, direct Elastic Net and direct Histogram Gradient Boosting use only
official historical OMIE Spanish prices and calendar information. They do not depend
on auction curves, so they remain scientifically valid even when the observed-curve
clearing feasibility gate in ``run_project.py`` fails.

Protocol (identical to the direct benchmarks inside ``run_project.py``):
- features: local hour, weekday, month, weekend flag, and official-price lags of
  1, 2 and 7 days aligned on local wall-clock hour (DST-safe);
- hyperparameters selected on the validation period only;
- test period evaluated with monthly expanding-window refits, hyperparameters fixed;
- all models are scored on exactly the same finite test observations.

This script never produces curve-based results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity_forecasting.data.coverage import audit_dir
from electricity_forecasting.data.dataset import iter_price_files, load_prices
from electricity_forecasting.data.parse_omie import parse_price_file
from electricity_forecasting.data.validate import validate_price_records
from electricity_forecasting.evaluation.backtest import chronological_date_split, test_month_windows
from electricity_forecasting.evaluation.diagnostics import conditional_metric_table
from electricity_forecasting.features.build_features import add_calendar_features, calendar_aware_lag
from electricity_forecasting.features.model_matrix import direct_price_feature_columns
from electricity_forecasting.models.benchmarks import weekly_persistence
from electricity_forecasting.models.training import fit_direct_model, select_direct_model
from electricity_forecasting.visualization.plots import plot_metric_comparison, plot_prediction_timeseries

PRICE = "price_es_eur_mwh"


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _elastic_candidates(cfg: dict) -> list[tuple[str, dict[str, object]]]:
    fam = cfg["models"]["elastic_net"]
    return [
        ("elastic_net", {"alpha": float(a), "l1_ratio": float(r), "max_iter": int(fam["max_iter"])})
        for a in fam["alpha_grid"]
        for r in fam["l1_ratio_grid"]
    ]


def _hgb_candidates(cfg: dict) -> list[tuple[str, dict[str, object]]]:
    h = cfg["models"]["hist_gradient_boosting"]
    return [
        (
            "hist_gradient_boosting",
            {
                "learning_rate": float(lr),
                "max_leaf_nodes": int(leaves),
                "max_iter": int(h["max_iter"]),
                "l2_regularization": float(h.get("l2_regularization", 0.0)),
            },
        )
        for lr in h["learning_rate_grid"]
        for leaves in h["max_leaf_nodes_grid"]
    ]


def _walk_forward(frame: pd.DataFrame, test: pd.DataFrame, features: list[str], selection, seed: int) -> pd.Series:
    dates = pd.to_datetime(frame["delivery_date"])
    test_dates = pd.to_datetime(test["delivery_date"])
    out = pd.Series(np.nan, index=test.index, dtype=float)
    for month_start, month_end in test_month_windows(test, date_col="delivery_date"):
        train = frame.loc[dates < month_start].dropna(subset=features + [PRICE])
        month = test.loc[(test_dates >= month_start) & (test_dates <= month_end)].dropna(subset=features)
        if train.empty or month.empty:
            continue
        model = fit_direct_model(
            train, feature_columns=features, target_column=PRICE,
            kind=selection.kind, params=selection.params, random_seed=seed,
        )
        out.loc[month.index] = np.asarray(model.predict(month[features]), dtype=float)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_root = Path(cfg["data"]["raw_dir"])
    out_root = Path("reports/metrics/direct_benchmarks")
    tz = cfg["project"]["timezone"]
    seed = int(cfg["project"]["random_seed"])
    splits = cfg["splits"]
    start = date.fromisoformat(cfg["data"]["start_date"])
    end = date.fromisoformat(cfg["data"]["end_date"])

    coverage = audit_dir(raw_root / "prices", cfg["omie"]["prices_prefix"], start, end)
    if coverage["status"] != "VERIFIED":
        raise SystemExit("BLOCKED: official price-file coverage incomplete; see audit output.")
    files = list(iter_price_files(raw_root / "prices", cfg["data"]["start_date"], cfg["data"]["end_date"]))
    failed = [f.name for f in files if validate_price_records(parse_price_file(f, timezone=tz)).status != "VERIFIED"]
    if failed:
        raise SystemExit(f"FAILED: price files failed validation: {failed[:10]}")

    prices = load_prices(files, timezone=tz)
    prices["interval_start_local"] = prices["interval_start_utc"].dt.tz_convert(tz)
    frame = add_calendar_features(prices)
    for days in (1, 2, 7):
        frame = calendar_aware_lag(frame, value_columns=[PRICE], days=days, align_col="hour")
    features = direct_price_feature_columns()
    modelling = frame.dropna(subset=features + [PRICE]).reset_index(drop=True)
    split = chronological_date_split(
        modelling, date_col="delivery_date", train_end=splits["train_end"],
        validation_end=splits["validation_end"], test_start=splits["test_start"],
    )

    sel_en = select_direct_model(split.train, split.validation, feature_columns=features, target_column=PRICE,
                                 candidates=_elastic_candidates(cfg), random_seed=seed)
    sel_hgb = select_direct_model(split.train, split.validation, feature_columns=features, target_column=PRICE,
                                  candidates=_hgb_candidates(cfg), random_seed=seed)

    test = split.test.copy()
    test["direct_elastic_net_prediction"] = _walk_forward(modelling, test, features, sel_en, seed)
    test["direct_hgb_prediction"] = _walk_forward(modelling, test, features, sel_hgb, seed)
    # Persistence uses the complete official series (before lag-based dropna).
    pers = weekly_persistence(frame[["interval_start_utc", "delivery_date", "period", "hour", PRICE]], align_col="hour")
    test = test.merge(
        pers[["interval_start_utc", "prediction"]].rename(columns={"prediction": "weekly_persistence_prediction"}),
        on="interval_start_utc", how="left", validate="one_to_one",
    )
    pred_cols = ["weekly_persistence_prediction", "direct_elastic_net_prediction", "direct_hgb_prediction"]
    finite = np.isfinite(test[[PRICE, *pred_cols]].to_numpy(dtype=float)).all(axis=1)
    test["evaluation_included"] = finite
    matched = test.loc[finite].sort_values("interval_start_utc").copy()
    metrics = conditional_metric_table(matched, observed_col=PRICE, prediction_cols=pred_cols,
                                       training_prices=split.train[PRICE])

    out_root.mkdir(parents=True, exist_ok=True)
    keep = ["interval_start_utc", "delivery_date", "period", "hour", PRICE, *pred_cols, "evaluation_included"]
    test[keep].to_csv(out_root / "test_predictions.csv", index=False)
    metrics.to_csv(out_root / "forecast_metrics.csv", index=False)
    summary = {
        "status": "VERIFIED (curve-independent benchmarks only)",
        "price_files": len(files),
        "intervals_total": int(len(prices)),
        "intervals_after_lag_dropna": int(len(modelling)),
        "train": [str(split.train["delivery_date"].min()), str(split.train["delivery_date"].max()), len(split.train)],
        "validation": [str(split.validation["delivery_date"].min()), str(split.validation["delivery_date"].max()), len(split.validation)],
        "test": [str(split.test["delivery_date"].min()), str(split.test["delivery_date"].max()), len(split.test)],
        "test_common_finite": int(finite.sum()),
        "test_excluded": int((~finite).sum()),
        "selection": {"direct_elastic_net": asdict(sel_en), "direct_hist_gradient_boosting": asdict(sel_hgb)},
        "headline": metrics.loc[metrics["subset"] == "all", ["model", "n", "mae", "rmse", "bias", "median_absolute_error"]]
        .to_dict(orient="records"),
    }
    _write_json(out_root / "summary.json", summary)
    fig_root = Path("reports/figures")
    plot_prediction_timeseries(
        matched, timestamp_col="interval_start_utc", observed_col=PRICE, prediction_cols=pred_cols,
        output_path=fig_root / "figure4_direct_benchmarks_out_of_sample.png",
    )
    plot_metric_comparison(metrics, output_path=fig_root / "figure5_direct_benchmark_comparison.png")
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
