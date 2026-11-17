"""Audit the observed-curve clearing discrepancy against OMIE matched curves.

For every delivery interval (selected country, as in the production pipeline) this
computes the gated offered/offered crossing, the matched/matched crossing, the two
mixed crossings, uniform-price consistency checks on the matched curves, offered-but-
unmatched volumes and, where OMIE publishes the 'Tipología de Oferta' column (2025
files), volumes by offer type and counterfactual crossings that remove complex offers
and/or add matched coupling rows.

Outputs
- data/processed/clearing_mechanism_audit.csv : per-interval table (post-auction data)
- reports/metrics/clearing_mechanism_audit.json : summary statistics

This is an ex-post diagnostic. Its per-interval values are post-auction information
and may enter forecasting only as lagged (>= 1 delivery day) historical summaries.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity_forecasting.data.dataset import iter_curve_files, iter_price_files, load_prices
from electricity_forecasting.data.parse_omie import parse_curve_file
from electricity_forecasting.market.clearing_audit import audit_frame
from electricity_forecasting.visualization.plots import plot_clearing_mechanism

_PRICES: pd.Series | None = None


def _init(prices: pd.Series) -> None:
    global _PRICES
    _PRICES = prices


def _work(args: tuple[str, str, str, str]) -> list[dict]:
    path, tz, preferred, fallback = args
    frame = parse_curve_file(Path(path), timezone=tz)
    return audit_frame(frame, _PRICES, preferred_country=preferred, fallback_country=fallback)


def _err_stats(err: pd.Series) -> dict[str, float]:
    err = err.dropna()
    return {
        "n": int(len(err)),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "median_ae": float(err.abs().median()),
        "share_abs_lt_1": float((err.abs() < 1).mean()),
    }


def _ols(y: pd.Series, xs: pd.DataFrame) -> dict[str, object]:
    data = pd.concat([y.rename("y"), xs], axis=1).dropna()
    x = np.column_stack([np.ones(len(data)), data[xs.columns].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(x, data["y"].to_numpy(float), rcond=None)
    resid = data["y"].to_numpy(float) - x @ beta
    r2 = 1 - resid.var() / data["y"].to_numpy(float).var()
    return {"n": int(len(data)), "coef": dict(zip(["intercept", *xs.columns], map(float, beta))), "r2": float(r2)}


def summarise(t: pd.DataFrame) -> dict[str, object]:
    t = t.copy()
    p = t["price_es_eur_mwh"]
    t["err_offered"] = t["p_offered"] - p
    out: dict[str, object] = {"n_intervals": int(len(t)), "period": [str(t["delivery_date"].min()), str(t["delivery_date"].max())]}

    crossings = {
        "offered_supply__offered_demand (gated method)": "p_offered",
        "matched_supply__matched_demand": "p_matched",
        "matched_supply__offered_demand": "p_matched_supply_offered_demand",
        "offered_supply__matched_demand": "p_offered_supply_matched_demand",
    }
    out["crossing_error_vs_official"] = {k: _err_stats(t[c] - p) for k, c in crossings.items()}
    out["crossing_error_by_year"] = {
        str(y): {k: _err_stats(g[c] - g["price_es_eur_mwh"]) for k, c in crossings.items()}
        for y, g in t.groupby(pd.to_datetime(t["delivery_date"]).dt.year)
    }
    month = pd.to_datetime(t["delivery_date"]).dt.to_period("M").astype(str)
    out["offered_crossing_error_by_month"] = {
        m: {**_err_stats(g["p_offered"] - g["price_es_eur_mwh"]),
            "mean_unmatched_supply_mwh": float(g["supply_unmatched_at_or_below_p_mwh"].mean()),
            "mean_official_price": float(g["price_es_eur_mwh"].mean())}
        for m, g in t.groupby(month)
    }
    out["matched_curve_consistency"] = {
        "share_official_price_within_matched_bracket": float(
            ((t["matched_supply_max_price"] <= p + 1e-6) & (p <= t["matched_demand_min_price"] + 1e-6)).mean()
        ),
        "max_abs_matched_supply_minus_demand_mwh": float((t["matched_supply_mwh"] - t["matched_demand_mwh"]).abs().max()),
        "total_matched_supply_above_official_mwh": float(t["matched_supply_above_p_mwh"].sum()),
        "total_matched_demand_below_official_mwh": float(t["matched_demand_below_p_mwh"].sum()),
    }

    u = t["supply_unmatched_at_or_below_p_mwh"]
    u_strict = t["supply_unmatched_strictly_below_p_mwh"]
    dg = t["demand_matched_not_in_offered_mwh"]
    e = t["err_offered"]
    bins = pd.cut(p, [-np.inf, 0, 20, 60, 100, np.inf], labels=["<=0", "0-20", "20-60", "60-100", ">100"])
    out["association"] = {
        "note": "Statistical association between the offered-curve clearing error and post-auction volumes. "
        "Both scale with price level, so within-price-regime correlations are reported as a partial control.",
        "pearson_err_vs_unmatched_supply": float(e.corr(u)),
        "spearman_err_vs_unmatched_supply": float(e.corr(u, method="spearman")),
        "pearson_err_vs_unmatched_supply_strictly_below_p": float(e.corr(u_strict)),
        "pearson_err_vs_demand_gap": float(e.corr(dg)),
        "pearson_err_vs_official_price": float(e.corr(p)),
        "within_price_regime_pearson_err_vs_unmatched_supply": {
            str(k): {"n": int(len(g)), "r": float(g["err_offered"].corr(g["supply_unmatched_at_or_below_p_mwh"]))}
            for k, g in t.groupby(bins, observed=True) if len(g) > 30
        },
        "ols_err_on_volumes": _ols(e, pd.DataFrame({"unmatched_supply_gwh": u / 1000, "demand_gap_gwh": dg / 1000})),
        "ols_err_on_volumes_and_price": _ols(
            e, pd.DataFrame({"unmatched_supply_gwh": u / 1000, "demand_gap_gwh": dg / 1000, "official_price": p})
        ),
        "mean_err_by_unmatched_supply_decile": {
            str(k): float(v) for k, v in e.groupby(pd.qcut(u, 10, duplicates="drop"), observed=True).mean().items()
        },
    }
    gated = out["crossing_error_vs_official"]
    out["mechanical_attribution"] = {
        "note": "Counterfactual replacement of one offered curve by its matched counterpart. This is an "
        "accounting statement about the curves, not a causal identification of why offers were rejected.",
        "mae_offered": gated["offered_supply__offered_demand (gated method)"]["mae"],
        "mae_after_replacing_supply_with_matched": gated["matched_supply__offered_demand"]["mae"],
        "mae_after_replacing_both_with_matched": gated["matched_supply__matched_demand"]["mae"],
    }
    out["volumes_mwh"] = {
        c: {k: float(v) for k, v in t[c].describe(percentiles=[0.05, 0.5, 0.95]).items()}
        for c in ["supply_unmatched_at_or_below_p_mwh", "supply_unmatched_strictly_below_p_mwh", "demand_matched_not_in_offered_mwh"]
    }

    typed = t.loc[t["has_offer_type"].astype(bool)]
    if len(typed):
        tp = typed["price_es_eur_mwh"]
        unm = typed["supply_unmatched_at_or_below_p_mwh"]
        unm_cx = typed["complex_supply_unmatched_at_or_below_p_mwh"]
        out["offer_typology_period"] = {
            "n_intervals": int(len(typed)),
            "first_date": str(typed["delivery_date"].min()),
            "last_date": str(typed["delivery_date"].max()),
            "crossing_error_vs_official": {
                "offered (gated method)": _err_stats(typed["p_offered"] - tp),
                "simple_offers_only_supply": _err_stats(typed["p_simple_supply_offered_demand"] - tp),
                "offered_plus_matched_coupling_rows": _err_stats(typed["p_offered_plus_coupling"] - tp),
                "simple_offers_plus_matched_coupling_rows": _err_stats(typed["p_simple_plus_coupling"] - tp),
                "matched": _err_stats(typed["p_matched"] - tp),
            },
            "mean_unmatched_supply_at_or_below_p_mwh": float(unm.mean()),
            "mean_complex_unmatched_at_or_below_p_mwh": float(unm_cx.mean()),
            "share_of_unmatched_cheap_supply_that_is_complex": float(unm_cx.sum() / unm.sum()) if unm.sum() else None,
            "mean_simple_unmatched_strictly_below_p_mwh": float(typed["simple_supply_unmatched_strictly_below_p_mwh"].mean()),
            "mean_matched_coupling_supply_mwh": float(typed["matched_coupling_supply_mwh"].mean()),
            "mean_matched_coupling_demand_mwh": float(typed["matched_coupling_demand_mwh"].mean()),
            "share_intervals_with_coupling_rows": float(
                ((typed["matched_coupling_supply_mwh"] + typed["matched_coupling_demand_mwh"]) > 0).mean()
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--summary-only", action="store_true", help="Recompute the JSON summary from the existing CSV")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = Path(cfg["data"]["raw_dir"])
    start = args.start or cfg["data"]["start_date"]
    end = args.end or cfg["data"]["end_date"]
    tz = cfg["project"]["timezone"]
    processed = Path(cfg["data"]["processed_dir"])
    if args.summary_only:
        table = pd.read_csv(processed / "clearing_mechanism_audit.csv")
        summary = summarise(table)
        Path("reports/metrics/clearing_mechanism_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot_clearing_mechanism(table, output_path=Path("reports/figures/figure8_clearing_mechanism_audit.png"))
        return
    prices = load_prices(iter_price_files(raw / "prices", start, end), timezone=tz)
    price_map = prices.set_index("interval_start_utc")["price_es_eur_mwh"]
    files = [str(f) for f in iter_curve_files(raw / "curves", start, end)]
    jobs = [(f, tz, cfg["omie"]["preferred_country"], cfg["omie"]["fallback_country"]) for f in files]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(price_map,)) as pool:
        for part in pool.map(_work, jobs, chunksize=4):
            rows.extend(part)
    table = pd.DataFrame(rows).sort_values("interval_start_utc").reset_index(drop=True)
    table = table.merge(prices[["interval_start_utc", "delivery_date", "period"]], on="interval_start_utc", how="left", validate="one_to_one")
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / "clearing_mechanism_audit.csv", index=False)
    summary = summarise(table)
    out = Path("reports/metrics/clearing_mechanism_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_clearing_mechanism(table, output_path=Path("reports/figures/figure8_clearing_mechanism_audit.png"))
    print(json.dumps({k: summary[k] for k in ("n_intervals", "crossing_error_vs_official", "mechanical_attribution")}, indent=2))


if __name__ == "__main__":
    main()
