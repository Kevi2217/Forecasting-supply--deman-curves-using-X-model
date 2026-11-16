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
from electricity_forecasting.data.download import find_local_versions
from electricity_forecasting.data.dataset import (
    aggregate_training_price_weights,
    build_interval_class_table,
    build_observed_clearing_table,
    iter_curve_files,
    iter_price_files,
    load_prices,
    reconstructed_clearing_from_class_row,
    select_country_records_for_interval,
)
from electricity_forecasting.data.parse_omie import parse_curve_file
from electricity_forecasting.data.quality import (
    assess_curve_price_alignment,
    audit_raw_files,
    clearing_validation_summary,
)
from electricity_forecasting.evaluation.backtest import chronological_date_split
from electricity_forecasting.evaluation.diagnostics import conditional_metric_table, error_decomposition_table
from electricity_forecasting.evaluation.metrics import compute_metrics
from electricity_forecasting.evaluation.walk_forward import expanding_monthly_predictions
from electricity_forecasting.features.model_matrix import (
    build_model_matrix,
    curve_feature_columns,
    direct_price_feature_columns,
)
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.compression import (
    CandidateClassSpecs,
    evaluate_class_candidates,
    reconstruct_pair,
    select_class_count,
)
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve
from electricity_forecasting.market.price_classes import PriceClassSpec, fit_price_classes, reconstruct_class_records
from electricity_forecasting.models.benchmarks import weekly_persistence
from electricity_forecasting.models.training import (
    fit_independent_target_models,
    select_curve_model,
    select_direct_model,
)
from electricity_forecasting.visualization.plots import (
    plot_clearing_diagnostics,
    plot_curve_comparison,
    plot_reconstruction_diagnostics,
    plot_metric_comparison,
    plot_observed_curves,
    plot_prediction_timeseries,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute the complete OMIE X-Model-inspired project.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--skip-coverage-gate", action="store_true", help="Development only; never use for empirical claims.")
    return p.parse_args()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _spec_to_dict(spec: PriceClassSpec) -> dict[str, object]:
    return {"side": spec.side, "edges": list(spec.edges), "representatives": list(spec.representatives)}


def _elastic_candidates(cfg: dict) -> list[tuple[str, dict[str, object]]]:
    family = cfg["models"]["elastic_net"]
    return [
        (
            "elastic_net",
            {"alpha": float(alpha), "l1_ratio": float(ratio), "max_iter": int(family["max_iter"])},
        )
        for alpha in family["alpha_grid"]
        for ratio in family["l1_ratio_grid"]
    ]


def _hgb_candidates(cfg: dict) -> list[tuple[str, dict[str, object]]]:
    hgb = cfg["models"]["hist_gradient_boosting"]
    return [
        (
            "hist_gradient_boosting",
            {
                "learning_rate": float(lr),
                "max_leaf_nodes": int(leaves),
                "max_iter": int(hgb["max_iter"]),
                "l2_regularization": float(hgb.get("l2_regularization", 0.0)),
            },
        )
        for lr in hgb["learning_rate_grid"]
        for leaves in hgb["max_leaf_nodes_grid"]
    ]


def _fit_specs_from_weights(weights: dict[str, pd.DataFrame], n_classes: int) -> CandidateClassSpecs:
    return CandidateClassSpecs(
        n_classes=n_classes,
        supply=fit_price_classes(weights["V"], "V", n_classes),
        demand=fit_price_classes(weights["C"], "C", n_classes),
    )


def _add_compressed_clearing(table: pd.DataFrame, supply_spec: PriceClassSpec, demand_spec: PriceClassSpec) -> pd.DataFrame:
    out = table.copy()
    values: list[float | None] = []
    statuses: list[str] = []
    for _, row in out.iterrows():
        value, status = reconstructed_clearing_from_class_row(row, supply_spec, demand_spec)
        values.append(value)
        statuses.append(status)
    out["compressed_curve_clearing_eur_mwh"] = values
    out["compressed_curve_clearing_status"] = statuses
    return out


def _prediction_clearing_frame(
    pred: pd.DataFrame,
    *,
    supply_spec: PriceClassSpec,
    demand_spec: PriceClassSpec,
    value_name: str,
    status_name: str,
    prediction_prefix: str = "pred_",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in pred.iterrows():
        remapped = pd.Series(
            {
                **{
                    f"supply_{i:02d}": row[f"{prediction_prefix}supply_{i:02d}"]
                    for i in range(supply_spec.n_classes)
                },
                **{
                    f"demand_{i:02d}": row[f"{prediction_prefix}demand_{i:02d}"]
                    for i in range(demand_spec.n_classes)
                },
                "interval_start_utc": row["interval_start_utc"],
                "curve_country": "forecast",
            }
        )
        value, status = reconstructed_clearing_from_class_row(remapped, supply_spec, demand_spec)
        rows.append({"interval_start_utc": row["interval_start_utc"], value_name: value, status_name: status})
    return pd.DataFrame(rows)


def _merge_price_and_curves(curve_table: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    price_cols = ["interval_start_utc", "price_es_eur_mwh", "price_pt_eur_mwh"]
    merged = curve_table.merge(prices[price_cols], on="interval_start_utc", how="inner", validate="one_to_one")
    if len(merged) != len(curve_table):
        raise ValueError(
            f"Curve/price alignment lost {len(curve_table) - len(merged)} intervals; investigate timestamps before modelling"
        )
    return merged


def _supported_geography_mask(frame: pd.DataFrame, tolerance: float) -> pd.Series:
    is_es = frame["curve_country"].eq("ES")
    mi_no_split = frame["curve_country"].eq("MI") & (
        frame["price_es_eur_mwh"].astype(float).sub(frame["price_pt_eur_mwh"].astype(float)).abs() <= tolerance
    )
    return is_es | mi_no_split


def _walk_family(
    matrix: pd.DataFrame,
    split,
    *,
    curve_features: list[str],
    direct_features: list[str],
    targets: list[str],
    curve_selection,
    direct_selection,
    seed: int,
):
    return expanding_monthly_predictions(
        matrix,
        split.test,
        date_col="delivery_date",
        curve_feature_columns=curve_features,
        class_target_columns=targets,
        curve_kind=curve_selection.kind,
        curve_params=curve_selection.params,
        direct_feature_columns=direct_features,
        direct_target_column="price_es_eur_mwh",
        direct_kind=direct_selection.kind,
        direct_params=direct_selection.params,
        random_seed=seed,
    )


def _validation_implied_price_mae(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    features: list[str],
    targets: list[str],
    selection,
    supply_spec: PriceClassSpec,
    demand_spec: PriceClassSpec,
    seed: int,
) -> tuple[float, int]:
    fitted = fit_independent_target_models(
        train,
        feature_columns=features,
        target_columns=targets,
        kind=selection.kind,
        params=selection.params,
        random_seed=seed,
    )
    pred, _ = fitted.predict(validation)
    tmp = validation[["interval_start_utc", "price_es_eur_mwh"]].copy()
    tmp = tmp.join(pred.add_prefix("pred_"))
    clearing = _prediction_clearing_frame(
        tmp,
        supply_spec=supply_spec,
        demand_spec=demand_spec,
        value_name="implied_price",
        status_name="status",
    )
    merged = tmp[["interval_start_utc", "price_es_eur_mwh"]].merge(
        clearing, on="interval_start_utc", how="inner", validate="one_to_one"
    )
    metrics = compute_metrics(
        merged["price_es_eur_mwh"].to_numpy(dtype=float), merged["implied_price"].to_numpy(dtype=float)
    )
    return metrics.mae, metrics.n


def _make_result_report(
    path: Path,
    *,
    selected_k: int,
    class_summary: pd.DataFrame,
    model_selection: dict[str, object],
    metric_table: pd.DataFrame,
    error_table: pd.DataFrame,
    common_n: int,
    total_test_n: int,
) -> None:
    lines = [
        "# Empirical Results",
        "",
        "Status: **VERIFIED** for the executed observations and code path described below.",
        "",
        f"Selected price classes per side: **{selected_k}**.",
        f"Matched common test observations used for headline model comparison: **{common_n} / {total_test_n}**.",
        "",
        "## Price-class compression validation",
        "",
        class_summary.to_markdown(index=False),
        "",
        "## Model selection",
        "",
        "```json",
        json.dumps(model_selection, indent=2, default=str),
        "```",
        "",
        "## Out-of-sample metrics",
        "",
        metric_table.to_markdown(index=False),
        "",
        "## Error decomposition",
        "",
        error_table.to_markdown(index=False),
        "",
        "The clearing algorithm is an approximation to the coupled OMIE/EUPHEMIA process and is not presented as exact market replication.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_representative_observed_curves(raw_root: Path, cfg: dict, table: pd.DataFrame, path: Path, *, timezone: str) -> None:
    """Pre-gate data figure: offered (and ex-post matched) curves for a fixed interval.

    The interval is fixed a priori (first test-period day, 20:00 local) rather than chosen
    by error size. Matched curves are plotted only to illustrate the gap; they are never
    used for clearing or modelling.
    """
    day = pd.Timestamp(cfg["splits"]["test_start"]).date()
    target_local = pd.Timestamp(f"{day} 20:00").tz_localize(timezone)
    row = table.loc[table["interval_start_utc"] == target_local.tz_convert("UTC")].iloc[0]
    path_day = find_local_versions(raw_root / "curves", cfg["omie"]["curves_prefix"], day)[-1]
    frame = parse_curve_file(path_day, timezone=timezone)
    rows = frame.loc[frame["interval_start_utc"] == row["interval_start_utc"]]
    country, offered = select_country_records_for_interval(
        rows, preferred_country=cfg["omie"]["preferred_country"], fallback_country=cfg["omie"]["fallback_country"],
        curve_flag=cfg["omie"]["offered_flag"],
    )
    matched_rows = rows.loc[(rows["country"] == country) & (rows["curve_flag"] == "C")]
    pair = CurvePair(row["interval_start_utc"], country, build_supply_curve(offered), build_demand_curve(offered))
    matched = CurvePair(row["interval_start_utc"], country, build_supply_curve(matched_rows), build_demand_curve(matched_rows))
    plot_observed_curves(
        pair, approximate_clearing(pair), official_price=float(row["price_es_eur_mwh"]), matched=matched,
        title=f"OMIE curves, {target_local:%Y-%m-%d %H:%M} local ({country})", output_path=path,
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_root = Path(cfg["data"]["raw_dir"])
    report_root = Path("reports")
    metrics_root = report_root / "metrics"
    fig_root = report_root / "figures"
    processed_root = Path(cfg["data"]["processed_dir"])
    metrics_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(cfg["data"]["start_date"])
    end = date.fromisoformat(cfg["data"]["end_date"])
    coverage = {
        "curves": audit_dir(raw_root / "curves", cfg["omie"]["curves_prefix"], start, end),
        "prices": audit_dir(raw_root / "prices", cfg["omie"]["prices_prefix"], start, end),
    }
    _write_json(metrics_root / "raw_coverage.json", coverage)
    if not args.skip_coverage_gate and any(v["status"] != "VERIFIED" for v in coverage.values()):
        raise SystemExit(
            "BLOCKED: mandatory OMIE raw-file coverage is incomplete. Run scripts/download_omie.py first; "
            "see reports/metrics/raw_coverage.json."
        )

    timezone = cfg["project"]["timezone"]
    preferred = cfg["omie"]["preferred_country"]
    fallback = cfg["omie"]["fallback_country"]
    offered_flag = cfg["omie"]["offered_flag"]
    seed = int(cfg["project"]["random_seed"])
    splits = cfg["splits"]
    all_curve_files = list(iter_curve_files(raw_root / "curves", cfg["data"]["start_date"], cfg["data"]["end_date"]))
    all_price_files = list(iter_price_files(raw_root / "prices", cfg["data"]["start_date"], cfg["data"]["end_date"]))

    # Phase 1 — every raw file is parsed and validated before modelling.
    raw_quality = audit_raw_files(all_curve_files, all_price_files, timezone=timezone)
    _write_json(metrics_root / "data_quality_report.json", raw_quality)
    if raw_quality["status"] != "VERIFIED":
        raise SystemExit("FAILED: one or more mandatory raw OMIE files failed schema/data-quality validation.")

    # Phase 2 — validate observed-curve clearing against the official Spanish price before compression/ML.
    observed = build_observed_clearing_table(
        all_curve_files,
        preferred_country=preferred,
        fallback_country=fallback,
        curve_flag=offered_flag,
        timezone=timezone,
    )
    prices = load_prices(all_price_files, timezone=timezone)
    observed_empirical = _merge_price_and_curves(observed, prices)
    price_tol = float(cfg["market_validation"].get("zonal_price_tolerance_eur_mwh", 1e-6))
    alignment = assess_curve_price_alignment(observed_empirical, price_tolerance_eur_mwh=price_tol)
    _write_json(metrics_root / "curve_price_geography_alignment.json", alignment)
    observed_empirical["geography_supported"] = _supported_geography_mask(observed_empirical, price_tol)
    supported_observed = observed_empirical.loc[observed_empirical["geography_supported"]].copy()
    if supported_observed.empty:
        raise SystemExit("FAILED: no intervals have a curve representation compatible with the Spanish price target.")
    clearing_summary = clearing_validation_summary(supported_observed)
    gate_cfg = cfg["market_validation"]
    clearing_summary["gate"] = {
        "max_clearing_mae_eur_mwh": float(gate_cfg["max_clearing_mae_eur_mwh"]),
        "max_invalid_crossing_share": float(gate_cfg["max_invalid_crossing_share"]),
    }
    clearing_summary["excluded_unsupported_geography_intervals"] = int((~observed_empirical["geography_supported"]).sum())
    _write_json(metrics_root / "clearing_validation.json", clearing_summary)
    observed_empirical.to_csv(processed_root / "observed_clearing_table.csv", index=False)
    plot_clearing_diagnostics(supported_observed, output_path=fig_root / "figure6_clearing_diagnostics.png")
    _plot_representative_observed_curves(
        raw_root, cfg, observed_empirical, fig_root / "figure1_observed_curves.png", timezone=timezone
    )
    max_mae = float(gate_cfg["max_clearing_mae_eur_mwh"])
    max_invalid = float(gate_cfg["max_invalid_crossing_share"])
    if (
        clearing_summary["mae"] is None
        or float(clearing_summary["mae"]) > max_mae
        or float(clearing_summary["invalid_share"]) > max_invalid
    ):
        raise SystemExit(
            "FAILED: approximate observed-curve clearing does not meet the pre-specified feasibility gate; "
            "see reports/metrics/clearing_validation.json."
        )

    # Phase 3 — price-class boundaries come from initial training data only.
    train_curve_files = list(iter_curve_files(raw_root / "curves", cfg["data"]["start_date"], splits["train_end"]))
    weights = aggregate_training_price_weights(
        train_curve_files,
        preferred_country=preferred,
        fallback_country=fallback,
        curve_flag=offered_flag,
        timezone=timezone,
    )
    candidate_specs: list[CandidateClassSpecs] = []
    candidate_failures: dict[str, str] = {}
    for candidate_k in cfg["price_classes"]["candidates"]:
        try:
            candidate_specs.append(_fit_specs_from_weights(weights, int(candidate_k)))
        except ValueError as exc:
            candidate_failures[str(candidate_k)] = str(exc)
    _write_json(metrics_root / "price_class_candidate_failures.json", candidate_failures)
    if not candidate_specs:
        raise SystemExit("FAILED: no requested price-class candidate can be fitted from training data.")

    validation_start = (pd.Timestamp(splits["train_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    validation_curve_files = list(iter_curve_files(raw_root / "curves", validation_start, splits["validation_end"]))
    compression_detail, compression_summary = evaluate_class_candidates(
        validation_curve_files,
        candidate_specs,
        preferred_country=preferred,
        fallback_country=fallback,
        curve_flag=offered_flag,
        timezone=timezone,
    )
    compression_detail.to_csv(metrics_root / "compression_detail.csv", index=False)
    compression_summary.to_csv(metrics_root / "compression_summary.csv", index=False)
    plot_reconstruction_diagnostics(compression_detail, output_path=fig_root / "figure7_reconstruction_diagnostics.png")
    selected_k = select_class_count(
        compression_summary,
        tie_tolerance_eur_mwh=float(cfg["price_classes"].get("tie_tolerance_eur_mwh", 0.05)),
    )
    chosen = next(c for c in candidate_specs if c.n_classes == selected_k)
    _write_json(
        processed_root / "price_class_specs.json",
        {"n_classes": selected_k, "supply": _spec_to_dict(chosen.supply), "demand": _spec_to_dict(chosen.demand)},
    )

    # Compact class-volume table; original clearing comes from the independently validated Phase-2 table.
    class_table = build_interval_class_table(
        all_curve_files,
        supply_spec=chosen.supply,
        demand_spec=chosen.demand,
        preferred_country=preferred,
        fallback_country=fallback,
        curve_flag=offered_flag,
        timezone=timezone,
        include_observed_clearing=False,
    )
    class_table = _add_compressed_clearing(class_table, chosen.supply, chosen.demand)
    empirical = class_table.merge(
        observed[["interval_start_utc", "observed_curve_clearing_eur_mwh", "observed_curve_clearing_status"]],
        on="interval_start_utc",
        how="inner",
        validate="one_to_one",
    )
    empirical = _merge_price_and_curves(empirical, prices)
    empirical["geography_supported"] = _supported_geography_mask(empirical, price_tol)
    empirical.to_csv(processed_root / "interval_dataset.csv", index=False)

    # Phase 4 — leakage-safe class-volume forecasting.
    class_cols = [f"supply_{i:02d}" for i in range(selected_k)] + [f"demand_{i:02d}" for i in range(selected_k)]
    matrix = build_model_matrix(empirical.loc[empirical["geography_supported"]].copy(), class_columns=class_cols, price_column="price_es_eur_mwh")
    c_features = curve_feature_columns(class_cols)
    d_features = direct_price_feature_columns()
    model_required = list(dict.fromkeys(c_features + d_features + class_cols + ["price_es_eur_mwh"]))
    modelling = matrix.dropna(subset=model_required).copy()
    split = chronological_date_split(
        modelling,
        date_col="delivery_date",
        train_end=splits["train_end"],
        validation_end=splits["validation_end"],
        test_start=splits["test_start"],
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("Chronological split produced an empty partition")

    curve_elastic = select_curve_model(
        split.train,
        split.validation,
        feature_columns=c_features,
        target_columns=class_cols,
        candidates=_elastic_candidates(cfg),
        random_seed=seed,
    )
    curve_hgb = select_curve_model(
        split.train,
        split.validation,
        feature_columns=c_features,
        target_columns=class_cols,
        candidates=_hgb_candidates(cfg),
        random_seed=seed,
    )
    direct_elastic = select_direct_model(
        split.train,
        split.validation,
        feature_columns=d_features,
        target_column="price_es_eur_mwh",
        candidates=_elastic_candidates(cfg),
        random_seed=seed,
    )
    direct_hgb = select_direct_model(
        split.train,
        split.validation,
        feature_columns=d_features,
        target_column="price_es_eur_mwh",
        candidates=_hgb_candidates(cfg),
        random_seed=seed,
    )
    elastic_implied_mae, elastic_implied_n = _validation_implied_price_mae(
        split.train,
        split.validation,
        features=c_features,
        targets=class_cols,
        selection=curve_elastic,
        supply_spec=chosen.supply,
        demand_spec=chosen.demand,
        seed=seed,
    )
    hgb_implied_mae, hgb_implied_n = _validation_implied_price_mae(
        split.train,
        split.validation,
        features=c_features,
        targets=class_cols,
        selection=curve_hgb,
        supply_spec=chosen.supply,
        demand_spec=chosen.demand,
        seed=seed,
    )
    selection_payload = {
        "curve_elastic_net": {**asdict(curve_elastic), "validation_implied_price_mae": elastic_implied_mae, "validation_implied_price_n": elastic_implied_n},
        "curve_hist_gradient_boosting": {**asdict(curve_hgb), "validation_implied_price_mae": hgb_implied_mae, "validation_implied_price_n": hgb_implied_n},
        "direct_elastic_net": asdict(direct_elastic),
        "direct_hist_gradient_boosting": asdict(direct_hgb),
    }
    _write_json(metrics_root / "model_selection.json", selection_payload)

    # Phase 5 — monthly expanding-window test forecasts; hyperparameters remain fixed after validation.
    elastic_pred, elastic_diag = _walk_family(
        modelling, split, curve_features=c_features, direct_features=d_features, targets=class_cols,
        curve_selection=curve_elastic, direct_selection=direct_elastic, seed=seed,
    )
    hgb_pred, hgb_diag = _walk_family(
        modelling, split, curve_features=c_features, direct_features=d_features, targets=class_cols,
        curve_selection=curve_hgb, direct_selection=direct_hgb, seed=seed,
    )
    elastic_clear = _prediction_clearing_frame(
        elastic_pred, supply_spec=chosen.supply, demand_spec=chosen.demand,
        value_name="curve_elastic_net_prediction", status_name="curve_elastic_net_status",
    )
    hgb_clear = _prediction_clearing_frame(
        hgb_pred, supply_spec=chosen.supply, demand_spec=chosen.demand,
        value_name="curve_hgb_prediction", status_name="curve_hgb_status",
    )

    base_cols = [
        "interval_start_utc", "delivery_date", "period", "price_es_eur_mwh",
        "observed_curve_clearing_eur_mwh", "compressed_curve_clearing_eur_mwh", "hour",
    ]
    predictions = split.test[base_cols + class_cols].copy()
    predictions = predictions.merge(
        elastic_pred[["interval_start_utc", *[f"pred_{c}" for c in class_cols], "direct_prediction"]],
        on="interval_start_utc", how="inner", validate="one_to_one",
    ).rename(columns={"direct_prediction": "direct_elastic_net_prediction"})
    predictions = predictions.merge(elastic_clear, on="interval_start_utc", how="left", validate="one_to_one")
    predictions = predictions.merge(
        hgb_pred[["interval_start_utc", *[f"pred_{c}" for c in class_cols], "direct_prediction"]].rename(
            columns={**{f"pred_{c}": f"hgb_pred_{c}" for c in class_cols}, "direct_prediction": "direct_hgb_prediction"}
        ),
        on="interval_start_utc", how="inner", validate="one_to_one",
    )
    predictions = predictions.merge(hgb_clear, on="interval_start_utc", how="left", validate="one_to_one")

    # Persistence uses the complete official price series (not the model-filtered subset),
    # aligned on local wall-clock hour so DST days are compared like-for-like.
    price_hist = prices[["interval_start_utc", "delivery_date", "period", "price_es_eur_mwh"]].copy()
    price_hist["hour"] = price_hist["interval_start_utc"].dt.tz_convert(timezone).dt.hour
    persistence = weekly_persistence(price_hist, align_col="hour")
    predictions = predictions.merge(
        persistence[["interval_start_utc", "prediction"]].rename(columns={"prediction": "weekly_persistence_prediction"}),
        on="interval_start_utc", how="left", validate="one_to_one",
    )

    main_curve_name = "curve_elastic_net_prediction" if elastic_implied_mae <= hgb_implied_mae else "curve_hgb_prediction"
    predictions["curve_model_prediction"] = predictions[main_curve_name]
    pred_cols = [
        "weekly_persistence_prediction", "direct_elastic_net_prediction", "direct_hgb_prediction",
        "curve_elastic_net_prediction", "curve_hgb_prediction",
    ]
    finite_mask = np.isfinite(predictions[["price_es_eur_mwh", *pred_cols]].to_numpy(dtype=float)).all(axis=1)
    predictions["evaluation_included"] = finite_mask
    matched = predictions.loc[finite_mask].copy()
    if matched.empty:
        raise SystemExit("FAILED: no common finite test observations across all evaluated models.")
    predictions.to_csv(metrics_root / "test_predictions.csv", index=False)
    _write_json(metrics_root / "constraint_diagnostics.json", {"elastic_net": elastic_diag, "hgb": hgb_diag})

    metrics = conditional_metric_table(
        matched,
        observed_col="price_es_eur_mwh",
        prediction_cols=pred_cols,
        training_prices=split.train["price_es_eur_mwh"],
    )
    metrics.to_csv(metrics_root / "forecast_metrics.csv", index=False)
    errors = error_decomposition_table(matched)
    errors.to_csv(metrics_root / "error_decomposition.csv", index=False)

    # Phase 6 — all figures are generated from actual pipeline outputs only.
    plot_prediction_timeseries(
        matched,
        timestamp_col="interval_start_utc",
        observed_col="price_es_eur_mwh",
        prediction_cols=[main_curve_name, "direct_elastic_net_prediction", "weekly_persistence_prediction"],
        output_path=fig_root / "figure4_out_of_sample_predictions.png",
    )
    plot_metric_comparison(metrics, output_path=fig_root / "figure5_performance_comparison.png")

    representative = matched.loc[matched["period"] == 13]
    rep = (representative.iloc[0] if not representative.empty else matched.iloc[0]).copy()
    raw_day_path = find_local_versions(raw_root / "curves", cfg["omie"]["curves_prefix"], pd.Timestamp(rep["delivery_date"]).date())[-1]
    raw_day = parse_curve_file(raw_day_path, timezone=timezone)
    interval_rows = raw_day.loc[raw_day["interval_start_utc"] == pd.Timestamp(rep["interval_start_utc"])]
    country, observed_records = select_country_records_for_interval(
        interval_rows, preferred_country=preferred, fallback_country=fallback, curve_flag=offered_flag
    )
    observed_pair = CurvePair(
        interval_start_utc=pd.Timestamp(rep["interval_start_utc"]), country=country,
        supply=build_supply_curve(observed_records), demand=build_demand_curve(observed_records),
    )
    observed_clear = approximate_clearing(observed_pair)
    plot_observed_curves(
        observed_pair, observed_clear, official_price=float(rep["price_es_eur_mwh"]),
        output_path=fig_root / "figure1b_observed_curves_test_interval.png",
    )
    compressed_pair, _ = reconstruct_pair(
        observed_records,
        interval_start_utc=pd.Timestamp(rep["interval_start_utc"]), country=country,
        supply_spec=chosen.supply, demand_spec=chosen.demand,
    )
    plot_curve_comparison(observed_pair, compressed_pair, output_path=fig_root / "figure2_price_class_reconstruction.png")

    prefix = "pred_" if main_curve_name.startswith("curve_elastic") else "hgb_pred_"
    predicted_records = pd.concat(
        [
            reconstruct_class_records(
                np.array([rep[f"{prefix}supply_{i:02d}"] for i in range(selected_k)], dtype=float), chosen.supply
            ),
            reconstruct_class_records(
                np.array([rep[f"{prefix}demand_{i:02d}"] for i in range(selected_k)], dtype=float), chosen.demand
            ),
        ],
        ignore_index=True,
    )
    predicted_pair = CurvePair(
        interval_start_utc=pd.Timestamp(rep["interval_start_utc"]), country="forecast",
        supply=build_supply_curve(predicted_records), demand=build_demand_curve(predicted_records),
    )
    plot_curve_comparison(
        observed_pair, predicted_pair, title="Forecasted versus observed auction curves",
        output_path=fig_root / "figure3_forecasted_vs_observed_curves.png",
    )

    _make_result_report(
        report_root / "results.md",
        selected_k=selected_k,
        class_summary=compression_summary,
        model_selection=selection_payload,
        metric_table=metrics.loc[metrics["subset"] == "all"],
        error_table=errors,
        common_n=len(matched),
        total_test_n=len(predictions),
    )
    _write_json(
        metrics_root / "execution_summary.json",
        {
            "status": "VERIFIED",
            "selected_k": selected_k,
            "main_curve_model": main_curve_name,
            "main_curve_model_selection_rule": "lowest validation implied-price MAE after class-level tuning",
            "test_observations_total": len(predictions),
            "test_observations_common": len(matched),
            "excluded_geography_intervals": int((~empirical["geography_supported"]).sum()),
            "figures": [str(p) for p in sorted(fig_root.glob("figure*.png"))],
        },
    )
    print(json.dumps({"status": "VERIFIED", "selected_k": selected_k, "main_curve_model": main_curve_name}, indent=2))


if __name__ == "__main__":
    main()
