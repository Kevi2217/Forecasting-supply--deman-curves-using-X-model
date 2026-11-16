"""Explain the observed-curve clearing error using OMIE's matched ('C') curves.

For every interval, the selected country's offered ('O') curves are compared with the
matched ('C') curves *at the official Spanish price* p*:

- ``supply_unmatched_at_or_below_p``: offered sell volume with price <= p* minus matched
  sell volume. Positive values are cheap offers that were *not* accepted, as happens
  when complex-order conditions (e.g. minimum-income, indivisibility, block orders)
  reject bids inside EUPHEMIA. Such volume pushes the naive crossing below p*.
- ``demand_matched_not_in_offered``: matched buy volume minus offered buy volume with
  price >= p*. Positive values are accepted purchases absent from the offered demand
  curve, consistent with market-coupling exports and other flows not represented in
  the published offered curves. Negative values indicate the opposite (e.g. imports
  reducing the need for local matched demand).

This is an ex-post diagnostic of data suitability. Matched curves are never used as
forecasting features or to alter the pre-specified clearing method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity_forecasting.data.dataset import iter_curve_files, iter_price_files, load_prices
from electricity_forecasting.data.parse_omie import parse_curve_file


def _interval_rows(frame: pd.DataFrame, prices: pd.DataFrame, preferred: str, fallback: str) -> list[dict]:
    rows = []
    price_map = prices.set_index("interval_start_utc")["price_es_eur_mwh"]
    for ts, g in frame.groupby("interval_start_utc", sort=True):
        countries = set(g.loc[g["curve_flag"] == "O", "country"])
        country = preferred if preferred in countries else fallback
        g = g.loc[g["country"] == country]
        p = float(price_map.loc[ts])
        o, c = g.loc[g["curve_flag"] == "O"], g.loc[g["curve_flag"] == "C"]
        oS, oD = o.loc[o["side"] == "V"], o.loc[o["side"] == "C"]
        mS = float(c.loc[c["side"] == "V", "energy_mwh"].sum())
        mD = float(c.loc[c["side"] == "C", "energy_mwh"].sum())
        off_s_le = float(oS.loc[oS["price_eur_mwh"] <= p, "energy_mwh"].sum())
        off_d_ge = float(oD.loc[oD["price_eur_mwh"] >= p, "energy_mwh"].sum())
        rows.append(
            {
                "interval_start_utc": ts,
                "curve_country": country,
                "price_es_eur_mwh": p,
                "offered_supply_at_or_below_p_mwh": off_s_le,
                "matched_supply_mwh": mS,
                "offered_demand_at_or_above_p_mwh": off_d_ge,
                "matched_demand_mwh": mD,
                "supply_unmatched_at_or_below_p_mwh": off_s_le - mS,
                "demand_matched_not_in_offered_mwh": mD - off_d_ge,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = Path(cfg["data"]["raw_dir"])
    start = args.start or cfg["data"]["start_date"]
    end = args.end or cfg["data"]["end_date"]
    tz = cfg["project"]["timezone"]
    prices = load_prices(iter_price_files(raw / "prices", start, end), timezone=tz)
    rows: list[dict] = []
    for path in iter_curve_files(raw / "curves", start, end):
        rows.extend(
            _interval_rows(parse_curve_file(path, timezone=tz), prices,
                           cfg["omie"]["preferred_country"], cfg["omie"]["fallback_country"])
        )
    table = pd.DataFrame(rows)
    out = Path("reports/metrics")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "clearing_gap_audit.csv", index=False)

    def _stats(s: pd.Series) -> dict[str, float]:
        return {k: float(v) for k, v in s.describe(percentiles=[0.05, 0.5, 0.95]).items()}

    clearing_path = Path(cfg["data"]["processed_dir"]) / "observed_clearing_table.csv"
    corr = None
    if clearing_path.exists():
        clr = pd.read_csv(clearing_path, parse_dates=["interval_start_utc"])
        merged = table.merge(clr[["interval_start_utc", "observed_curve_clearing_eur_mwh"]], on="interval_start_utc")
        err = merged["observed_curve_clearing_eur_mwh"] - merged["price_es_eur_mwh"]
        corr = {
            "corr_error_vs_supply_unmatched": float(np.corrcoef(err, merged["supply_unmatched_at_or_below_p_mwh"])[0, 1]),
            "corr_error_vs_demand_not_in_offered": float(np.corrcoef(err, merged["demand_matched_not_in_offered_mwh"])[0, 1]),
        }
    summary = {
        "period": [start, end],
        "n_intervals": int(len(table)),
        "supply_unmatched_at_or_below_p_mwh": _stats(table["supply_unmatched_at_or_below_p_mwh"]),
        "demand_matched_not_in_offered_mwh": _stats(table["demand_matched_not_in_offered_mwh"]),
        "share_intervals_supply_unmatched_gt_500mwh": float((table["supply_unmatched_at_or_below_p_mwh"] > 500).mean()),
        "share_intervals_abs_demand_gap_gt_500mwh": float((table["demand_matched_not_in_offered_mwh"].abs() > 500).mean()),
        "clearing_error_correlations": corr,
    }
    (out / "clearing_gap_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
