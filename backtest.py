from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    train_end: str,
    validation_end: str,
    test_start: str,
) -> ChronologicalSplit:
    if timestamp_col not in df.columns:
        raise KeyError(timestamp_col)
    out = df.sort_values(timestamp_col).copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    train_end_ts = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    validation_end_ts = pd.Timestamp(validation_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    if not (train_end_ts < validation_end_ts < test_start_ts):
        raise ValueError("Require train_end < validation_end < test_start")
    train = out.loc[ts <= train_end_ts].copy()
    validation = out.loc[(ts > train_end_ts) & (ts <= validation_end_ts)].copy()
    test = out.loc[ts >= test_start_ts].copy()
    return ChronologicalSplit(train=train, validation=validation, test=test)


def chronological_date_split(
    df: pd.DataFrame,
    *,
    date_col: str,
    train_end: str,
    validation_end: str,
    test_start: str,
) -> ChronologicalSplit:
    """Split on local delivery dates, avoiding UTC boundary leakage on DST/offset days."""
    if date_col not in df.columns:
        raise KeyError(date_col)
    out = df.copy()
    dates = pd.to_datetime(out[date_col]).dt.date
    train_end_d = pd.Timestamp(train_end).date()
    validation_end_d = pd.Timestamp(validation_end).date()
    test_start_d = pd.Timestamp(test_start).date()
    if not (train_end_d < validation_end_d < test_start_d):
        raise ValueError("Require train_end < validation_end < test_start")
    return ChronologicalSplit(
        train=out.loc[dates <= train_end_d].copy(),
        validation=out.loc[(dates > train_end_d) & (dates <= validation_end_d)].copy(),
        test=out.loc[dates >= test_start_d].copy(),
    )


def test_month_windows(test: pd.DataFrame, *, date_col: str = "delivery_date") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(test[date_col])
    if dates.empty:
        return []
    periods = dates.dt.to_period("M").drop_duplicates().sort_values()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for period in periods:
        start = period.start_time
        end = period.end_time.normalize()
        windows.append((start, end))
    return windows
