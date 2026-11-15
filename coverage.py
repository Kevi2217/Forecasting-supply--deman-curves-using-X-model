from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

PATTERN = re.compile(r"^(?P<prefix>[a-z_]+)_(?P<date>\d{8})\.(?P<version>\d+)$")


def expected_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be on or after start")
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def audit_dir(directory: Path, prefix: str, start: date, end: date) -> dict[str, object]:
    found: dict[str, list[int]] = {}
    malformed: list[str] = []
    if directory.exists():
        for path in directory.iterdir():
            if not path.is_file() or path.name.endswith(".provenance.json"):
                continue
            match = PATTERN.match(path.name)
            if not match or match.group("prefix") != prefix:
                malformed.append(path.name)
                continue
            ymd = match.group("date")
            found.setdefault(ymd, []).append(int(match.group("version")))

    expected = [d.strftime("%Y%m%d") for d in expected_dates(start, end)]
    expected_set = set(expected)
    missing = [ymd for ymd in expected if ymd not in found]
    multi_version = {k: sorted(v) for k, v in found.items() if len(v) > 1}
    out_of_range = sorted(k for k in found if k not in expected_set)
    return {
        "directory": str(directory),
        "prefix": prefix,
        "expected_days": len(expected),
        "days_present": sum(1 for ymd in expected if ymd in found),
        "missing_days": missing,
        "missing_count": len(missing),
        "multiple_versions": multi_version,
        "out_of_range_days": out_of_range,
        "unrecognised_files": sorted(malformed),
        "status": "VERIFIED" if not missing else "FAILED",
    }
