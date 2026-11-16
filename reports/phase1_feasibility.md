# Phase 1 feasibility report

Date: 2026-11-14

## Decision

**PARTIALLY VERIFIED / BLOCKED in this execution environment.**

The official OMIE archive exposes the required day-ahead aggregate-curve and marginal-price file families over the target 2023–2024 period, and the official format documentation supplies the schema needed for a parser. The project is therefore technically feasible in a normal internet-connected environment.

The present runtime cannot materialise OMIE's binary `.1` downloads: the browsing layer can inspect the official source pages/documentation but does not expose `application/octet-stream` file bytes, while the local Python/container runtime has no external DNS/network access.

## Implemented Phase-1 controls

- deterministic official filename construction;
- configurable date ranges;
- timeout/retry/exponential backoff;
- atomic download and caching;
- SHA-256 provenance records;
- local 2023–2024 coverage audit;
- robust encoding/locale-number parsing;
- 23/24/25-hour DST timestamp mapping;
- raw schema/data-quality validation;
- source-file retention in parsed records;
- fail-closed empirical runner.

## Acceptance gate behaviour

`python scripts/run_project.py` currently terminates before any modelling with:

```text
BLOCKED: mandatory OMIE raw-file coverage is incomplete. Run scripts/download_omie.py first; see reports/metrics/raw_coverage.json.
```

This is the intended outcome. Empirical forecasting is not considered complete until real historical OMIE curves have been successfully processed.
