from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

LOGGER = logging.getLogger(__name__)


class OmieFileNotFound(FileNotFoundError):
    """Raised when OMIE has no file under the requested name/version."""


@dataclass(frozen=True)
class DownloadRecord:
    dataset: str
    filename: str
    url: str
    retrieved_at_utc: str
    sha256: str
    size_bytes: int
    content_type: str | None
    status: str


def date_range(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError("end must be on or after start")
    current = start
    while current <= end:
        yield current
        current = date.fromordinal(current.toordinal() + 1)


def omie_filename(prefix: str, delivery_date: date, version: int = 1) -> str:
    if version < 1:
        raise ValueError("version must be >= 1")
    return f"{prefix}_{delivery_date:%Y%m%d}.{version}"


def omie_download_url(base_url: str, filename: str, parent: str) -> str:
    # This query shape is the href used by OMIE's official File Access pages.
    return f"{base_url}?filename={filename}&parents={parent}"


def find_local_versions(directory: Path, prefix: str, delivery_date: date) -> list[Path]:
    """Local raw files for a date, sorted by ascending OMIE version number."""
    stem = f"{prefix}_{delivery_date:%Y%m%d}."
    found = []
    if directory.exists():
        for path in directory.glob(stem + "*"):
            suffix = path.name[len(stem):]
            if suffix.isdigit():
                found.append((int(suffix), path))
    return [p for _, p in sorted(found)]


def download_latest_version(
    *,
    prefix: str,
    delivery_date: date,
    max_version: int = 5,
    **kwargs,
) -> DownloadRecord:
    """Download the published version of an OMIE daily file.

    OMIE republishes corrected files with an incremented suffix and serves only the
    current one. Versions are probed in ascending order; the first that exists is used.
    If any version is already cached locally (and overwrite is False) it is reused.
    """
    destination_dir: Path = kwargs["destination_dir"]
    if not kwargs.get("overwrite", False):
        local = find_local_versions(destination_dir, prefix, delivery_date)
        if local:
            return download_file(filename=local[-1].name, **kwargs)
    last: Exception | None = None
    for version in range(1, max_version + 1):
        try:
            return download_file(filename=omie_filename(prefix, delivery_date, version), **kwargs)
        except RuntimeError as exc:
            if isinstance(exc.__cause__, OmieFileNotFound):
                last = exc
                continue
            raise
    raise RuntimeError(f"No OMIE version 1..{max_version} found for {prefix} {delivery_date}") from last


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _looks_like_html(payload: bytes, content_type: str | None) -> bool:
    ct = (content_type or "").lower()
    head = payload[:300].lstrip().lower()
    return "text/html" in ct or head.startswith(b"<!doctype html") or head.startswith(b"<html")


def download_file(
    *,
    base_url: str,
    filename: str,
    parent: str,
    destination_dir: Path,
    dataset: str,
    timeout_seconds: float = 30.0,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> DownloadRecord:
    """Download one OMIE public file atomically and record provenance.

    The function deliberately rejects HTML error pages masquerading as successful
    downloads. Existing files are cached unless ``overwrite=True``.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / filename
    meta_path = destination_dir / f"{filename}.provenance.json"
    url = omie_download_url(base_url, filename, parent)

    if path.exists() and not overwrite:
        payload = path.read_bytes()
        record = DownloadRecord(
            dataset=dataset,
            filename=filename,
            url=url,
            retrieved_at_utc=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            sha256=_sha256(payload),
            size_bytes=len(payload),
            content_type=None,
            status="cached",
        )
        return record

    sess = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = sess.get(url, timeout=timeout_seconds)
            if getattr(response, "status_code", 200) == 404:
                # Not retryable: OMIE serves only the latest revision (e.g. ``.2``) of a file.
                raise OmieFileNotFound(f"OMIE returned 404 for {filename}")
            response.raise_for_status()
            payload = response.content
            content_type = response.headers.get("Content-Type")
            if len(payload) < 10:
                raise ValueError(f"Downloaded payload is unexpectedly small ({len(payload)} bytes): {url}")
            if _looks_like_html(payload, content_type):
                raise ValueError(f"Received HTML instead of a data file: {url}")

            tmp = path.with_suffix(path.suffix + ".partial")
            tmp.write_bytes(payload)
            tmp.replace(path)
            retrieved = datetime.now(timezone.utc).isoformat()
            record = DownloadRecord(
                dataset=dataset,
                filename=filename,
                url=url,
                retrieved_at_utc=retrieved,
                sha256=_sha256(payload),
                size_bytes=len(payload),
                content_type=content_type,
                status="downloaded",
            )
            meta_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
            return record
        except OmieFileNotFound as exc:
            last_error = exc
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = backoff_seconds * (2**attempt)
            LOGGER.warning("Download attempt %s failed for %s: %s; retrying in %.1fs", attempt + 1, filename, exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"Failed to download {filename} from OMIE after {max_retries + 1} attempts") from last_error


def write_manifest(records: list[DownloadRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2, sort_keys=True), encoding="utf-8")
