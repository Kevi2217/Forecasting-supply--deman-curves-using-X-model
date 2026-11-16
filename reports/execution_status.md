# Execution status

Date: 2026-11-16

| Stage | Status |
|---|---|
| Raw data download, 2023-01-01 – 2024-12-31 (1 462 files, 2.3 GB) | VERIFIED |
| Coverage audit | VERIFIED: 731/731 curves, 731/731 prices |
| Parsing, schema and data-quality validation (DST-exact) | VERIFIED: 1 462/1 462 files |
| Timestamp / DST handling | VERIFIED |
| Observed supply/demand curve construction | VERIFIED |
| Approximate clearing vs official Spanish price | EXECUTED. **Gate FAILED** (MAE 26.14 > 10) |
| Price classes, reconstruction, curve features, curve models | NOT EXECUTED: blocked by the gate |
| Weekly persistence, direct Elastic Net, direct HGB | VERIFIED: out-of-sample, N = 2 209 |
| Error decomposition | Component A only (clearing approximation); B and C blocked |
| Tests | 35 passed, 0 failed |

The runner (`scripts/run_project.py`) exits with:

```text
FAILED: approximate observed-curve clearing does not meet the pre-specified feasibility gate; see reports/metrics/clearing_validation.json.
```

This is the intended fail-closed behaviour. See `results.md` for the numbers and `technical_audit.md` for the diagnosis.
