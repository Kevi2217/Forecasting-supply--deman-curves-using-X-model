from __future__ import annotations

import argparse
import json
from pathlib import Path

from electricity_forecasting.data.parse_omie import parse_curve_file, parse_price_file
from electricity_forecasting.data.validate import validate_curve_records, validate_price_records


def main() -> None:
    p = argparse.ArgumentParser(description="Parse and inspect representative OMIE raw files.")
    p.add_argument("--curve-file", type=Path)
    p.add_argument("--price-file", type=Path)
    p.add_argument("--report", type=Path, default=Path("reports/metrics/data_quality_sample.json"))
    args = p.parse_args()
    if args.curve_file is None and args.price_file is None:
        p.error("Provide --curve-file and/or --price-file")

    report: dict[str, object] = {}
    if args.curve_file:
        curve = parse_curve_file(args.curve_file)
        report["curves"] = validate_curve_records(curve).to_dict()
        report["curve_head"] = curve.head(10).astype(str).to_dict(orient="records")
    if args.price_file:
        price = parse_price_file(args.price_file)
        report["prices"] = validate_price_records(price).to_dict()
        report["price_head"] = price.head(10).astype(str).to_dict(orient="records")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
