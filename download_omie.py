from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import yaml

from electricity_forecasting.data.download import (
    date_range,
    download_file,
    omie_filename,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download official OMIE day-ahead curve and price files.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--start", help="YYYY-MM-DD; defaults to config")
    p.add_argument("--end", help="YYYY-MM-DD; defaults to config")
    p.add_argument("--dataset", choices=["curves", "prices", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    start = date.fromisoformat(args.start or cfg["data"]["start_date"])
    end = date.fromisoformat(args.end or cfg["data"]["end_date"])
    raw_root = Path(cfg["data"]["raw_dir"])
    base_url = cfg["data"]["omie_base_url"]
    records = []
    selections = ["curves", "prices"] if args.dataset == "both" else [args.dataset]

    for d in date_range(start, end):
        for dataset in selections:
            prefix = cfg["omie"][f"{dataset}_prefix"]
            parent = cfg["omie"][f"{dataset}_parent"]
            filename = omie_filename(prefix, d)
            record = download_file(
                base_url=base_url,
                filename=filename,
                parent=parent,
                destination_dir=raw_root / dataset,
                dataset=dataset,
                timeout_seconds=float(cfg["data"]["request_timeout_seconds"]),
                max_retries=int(cfg["data"]["max_retries"]),
                backoff_seconds=float(cfg["data"]["backoff_seconds"]),
                overwrite=bool(args.overwrite or cfg["data"]["overwrite"]),
            )
            records.append(record)
            logging.info("%s %s (%d bytes)", record.status, filename, record.size_bytes)
            time.sleep(float(cfg["data"]["polite_delay_seconds"]))

    manifest = raw_root / "download_manifest.json"
    write_manifest(records, manifest)
    print(json.dumps({"status": "complete", "files": len(records), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
