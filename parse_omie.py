from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

CURVE_COLUMNS = [
    "period",
    "delivery_date",
    "country",
    "unit",
    "side",
    "energy_mwh",
    "price_eur_mwh",
    "curve_flag",
]
PRICE_COLUMNS = ["year", "month", "day", "period", "price_pt_eur_mwh", "price_es_eur_mwh"]


def decode_omie_bytes(payload: bytes) -> str:
    errors: list[Exception] = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(exc)
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode OMIE payload: {errors}")


def parse_omie_number(value: str) -> float:
    """Parse OMIE numeric strings supporting Spanish and dot-decimal conventions."""
    s = value.strip().replace("\xa0", "")
    if not s:
        raise ValueError("empty numeric field")
    if "," in s and "." in s:
        # Documentation's general convention: 2.651,8 -> 2651.8
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


def _rows(text: str) -> Iterable[list[str]]:
    reader = csv.reader(io.StringIO(text), delimiter=";")
    for raw in reader:
        row = [item.strip() for item in raw]
        if not row or all(not item for item in row):
            continue
        if row[0] == "*" or all(item == "" for item in row):
            continue
        yield row


def _physical_interval_map(delivery_date: date, timezone: str = "Europe/Madrid") -> dict[int, pd.Timestamp]:
    """Map OMIE period numbers to unique physical interval starts.

    The mapping uses Europe/Madrid civil-time transitions. This is the project's
    current implementation assumption and must be checked against raw OMIE files
    before empirical use.
    """
    start = pd.Timestamp(delivery_date).tz_localize(timezone)
    end = pd.Timestamp(delivery_date + pd.Timedelta(days=1)).tz_localize(timezone)
    intervals = pd.date_range(start=start, end=end, inclusive="left", freq="1h")
    return {idx + 1: ts for idx, ts in enumerate(intervals)}


def add_interval_timestamps(df: pd.DataFrame, timezone: str = "Europe/Madrid") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    starts: list[pd.Timestamp] = []
    cache: dict[date, dict[int, pd.Timestamp]] = {}
    for d, period in zip(out["delivery_date"], out["period"], strict=True):
        d_date = pd.Timestamp(d).date()
        mapping = cache.setdefault(d_date, _physical_interval_map(d_date, timezone))
        if int(period) not in mapping:
            raise ValueError(
                f"Period {period} is invalid for {d_date} under timezone {timezone}; "
                f"expected 1..{len(mapping)}"
            )
        starts.append(mapping[int(period)])
    out["interval_start_local"] = pd.DatetimeIndex(starts)
    out["interval_start_utc"] = out["interval_start_local"].dt.tz_convert("UTC")
    out["interval_end_utc"] = out["interval_start_utc"] + pd.Timedelta(hours=1)
    return out


def parse_curve_text(text: str, source_file: str = "<memory>", timezone: str = "Europe/Madrid") -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in _rows(text):
        if len(row) < 8:
            continue
        try:
            period = int(row[0])
        except ValueError:
            continue
        try:
            delivery_date = pd.to_datetime(row[1], format="%d/%m/%Y", errors="raise").date()
            energy = parse_omie_number(row[5])
            price = parse_omie_number(row[6])
        except (ValueError, TypeError):
            continue
        records.append(
            {
                "period": period,
                "delivery_date": delivery_date,
                "country": row[2].upper(),
                "unit": row[3] or None,
                "side": row[4].upper(),
                "energy_mwh": energy,
                "price_eur_mwh": price,
                "curve_flag": row[7].upper(),
                "source_file": source_file,
            }
        )
    if not records:
        raise ValueError(f"No OMIE curve data records found in {source_file}")
    frame = pd.DataFrame.from_records(records)
    return add_interval_timestamps(frame, timezone=timezone)


def parse_curve_file(path: Path, timezone: str = "Europe/Madrid") -> pd.DataFrame:
    return parse_curve_text(decode_omie_bytes(path.read_bytes()), source_file=path.name, timezone=timezone)


def parse_price_text(text: str, source_file: str = "<memory>", timezone: str = "Europe/Madrid") -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in _rows(text):
        if len(row) < 6:
            continue
        try:
            year, month, day, period = map(int, row[:4])
            pt = parse_omie_number(row[4])
            es = parse_omie_number(row[5])
        except (ValueError, TypeError):
            continue
        records.append(
            {
                "year": year,
                "month": month,
                "day": day,
                "period": period,
                "price_pt_eur_mwh": pt,
                "price_es_eur_mwh": es,
                "delivery_date": date(year, month, day),
                "source_file": source_file,
            }
        )
    if not records:
        raise ValueError(f"No OMIE price records found in {source_file}")
    frame = pd.DataFrame.from_records(records)
    return add_interval_timestamps(frame, timezone=timezone)


def parse_price_file(path: Path, timezone: str = "Europe/Madrid") -> pd.DataFrame:
    return parse_price_text(decode_omie_bytes(path.read_bytes()), source_file=path.name, timezone=timezone)
