from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from electricity_forecasting.data.parse_omie import parse_curve_file, parse_price_file
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import build_curve_pair


def main() -> None:
    p = argparse.ArgumentParser(description="Validate approximate clearing on one observed OMIE file pair.")
    p.add_argument("curve_file", type=Path)
    p.add_argument("price_file", type=Path)
    p.add_argument("--country", default="MI", choices=["MI", "ES", "PT"])
    p.add_argument("--period", type=int, default=12)
    args = p.parse_args()

    curves = parse_curve_file(args.curve_file)
    prices = parse_price_file(args.price_file)
    p_row = prices.loc[prices["period"] == args.period]
    if p_row.empty:
        raise SystemExit(f"No official price for period={args.period}")
    timestamp = p_row.iloc[0]["interval_start_utc"]
    pair = build_curve_pair(curves, interval_start_utc=timestamp, country=args.country, curve_flag="O")
    result = approximate_clearing(pair)
    observed = float(p_row.iloc[0]["price_es_eur_mwh"])
    print(
        json.dumps(
            {
                "country_curve": args.country,
                "period": args.period,
                "approximate_clearing": result.__dict__,
                "official_spain_price_eur_mwh": observed,
                "signed_difference_eur_mwh": None
                if result.estimated_price_eur_mwh is None
                else result.estimated_price_eur_mwh - observed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
