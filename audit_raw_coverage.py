from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from electricity_forecasting.data.coverage import audit_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Audit local OMIE raw-file coverage over a date range.")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--output", type=Path, default=Path("reports/metrics/raw_coverage.json"))
    args = p.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    report = {
        "curves": audit_dir(args.raw_root / "curves", "curva_pbc", start, end),
        "prices": audit_dir(args.raw_root / "prices", "marginalpdbc", start, end),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
