from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
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

    Period 1 is the first physical hour of the local delivery day; a 23-hour spring
    DST day has periods 1..23 and a 25-hour autumn DST day has periods 1..25. This
    mapping was checked against real 2023-2024 OMIE files (see reports/technical_audit.md).
    """
    start = pd.Timestamp(delivery_date).tz_localize(timezone)
    end = pd.Timestamp(delivery_date + timedelta(days=1)).tz_localize(timezone)
    intervals = pd.date_range(start=start, end=end, inclusive="left", freq="1h")
    return {idx + 1: ts for idx, ts in enumerate(intervals)}


def expected_period_count(delivery_date: date, timezone: str = "Europe/Madrid") -> int:
    return len(_physical_interval_map(delivery_date, timezone))


def add_interval_timestamps(df: pd.DataFrame, timezone: str = "Europe/Madrid") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    # Build the DST-aware mapping once per unique (date, period) pair, then join.
    keys = out[["delivery_date", "period"]].drop_duplicates()
    starts: list[pd.Timestamp] = []
    cache: dict[date, dict[int, pd.Timestamp]] = {}
    for d, period in zip(keys["delivery_date"], keys["period"], strict=True):
        d_date = pd.Timestamp(d).date()
        if d_date not in cache:
            cache[d_date] = _physical_interval_map(d_date, timezone)
        mapping = cache[d_date]
        if int(period) not in mapping:
            raise ValueError(
                f"Period {period} is invalid for {d_date} under timezone {timezone}; "
                f"expected 1..{len(mapping)}"
            )
        starts.append(mapping[int(period)])
    keys = keys.assign(interval_start_local=pd.DatetimeIndex(starts))
    out = out.merge(keys, on=["delivery_date", "period"], how="left", validate="many_to_one", sort=False)
    out["interval_start_utc"] = out["interval_start_local"].dt.tz_convert("UTC")
    out["interval_end_utc"] = out["interval_start_utc"] + pd.Timedelta(hours=1)
    return out


def _parse_number_series(values: pd.Series) -> pd.Series:
    """Vectorised equivalent of :func:`parse_omie_number` (NaN where unparsable)."""
    s = values.astype(str).str.strip().str.replace(" ", "", regex=False)
    has_comma = s.str.contains(",", regex=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _data_rows(text: str, min_fields: int, keep_fields: int | None = None) -> list[list[str]]:
    """Rows whose first field is an integer (data rows), padded/truncated to keep_fields."""
    keep = keep_fields or min_fields
    out: list[list[str]] = []
    for row in _rows(text):
        if len(row) < min_fields:
            continue
        head = row[0]
        if not head or not (head.isdigit() or (head[0] in "+-" and head[1:].isdigit())):
            continue
        out.append((row + [""] * keep)[:keep])
    return out


def _has_offer_type_column(text: str) -> bool:
    """True when the header carries OMIE's 9th column 'Tipología de Oferta' (2025 files).

    Values observed on real files: ``S`` (simple offer), ``C01``/``C02``/``C04`` (offers
    with complex conditions) and coupling rows such as ``Imp FR`` in matched curves.
    """
    for line in text.splitlines()[:6]:
        if line.startswith(("Hora;", "Periodo;")):
            return "tipolog" in line.lower()
    return False


def parse_curve_text(text: str, source_file: str = "<memory>", timezone: str = "Europe/Madrid") -> pd.DataFrame:
    has_type = _has_offer_type_column(text)
    rows = _data_rows(text, 8, 9 if has_type else 8)
    if not rows:
        raise ValueError(f"No OMIE curve data records found in {source_file}")
    raw = pd.DataFrame(rows, columns=CURVE_COLUMNS + (["offer_type"] if has_type else []))
    dates = pd.to_datetime(raw["delivery_date"], format="%d/%m/%Y", errors="coerce")
    energy = _parse_number_series(raw["energy_mwh"])
    price = _parse_number_series(raw["price_eur_mwh"])
    bad = dates.isna() | energy.isna() | price.isna()
    if bad.any():
        # Fail closed: a data row (integer period) that cannot be parsed is a schema problem.
        example = rows[int(np.flatnonzero(bad.to_numpy())[0])]
        raise ValueError(f"{int(bad.sum())} malformed curve data rows in {source_file}; first: {example}")
    frame = pd.DataFrame(
        {
            "period": raw["period"].astype(int),
            "delivery_date": dates.dt.date,
            "country": raw["country"].str.upper(),
            "unit": raw["unit"].where(raw["unit"] != "", None),
            "side": raw["side"].str.upper(),
            "energy_mwh": energy.astype(float),
            "price_eur_mwh": price.astype(float),
            "curve_flag": raw["curve_flag"].str.upper(),
            # Optional: only files whose header has 'Tipología de Oferta'; otherwise None.
            "offer_type": raw["offer_type"].where(raw["offer_type"] != "", None) if has_type else None,
            "source_file": source_file,
        }
    )
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
