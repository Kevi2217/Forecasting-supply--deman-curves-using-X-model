# Execution status

Date: 2026-11-16 (reconciled after the curve extension run).

## What has actually been executed

| Stage | Script | Status |
|---|---|---|
| Raw download 2023-01-01 – 2024-12-31 (1 462 files) | `download_omie.py` | VERIFIED |
| Raw download 2025-01-01 – 2025-09-30 (546 files) | `download_omie.py` | VERIFIED: 273/273 days per dataset, all revision `.1` |
| Coverage, schema and DST validation | `run_project.py`, `run_curve_extension.py` | VERIFIED: 2 008/2 008 files |
| Observed-curve clearing gate (production, 2023–24) | `run_project.py` | EXECUTED, **FAILED** (MAE 26.14 > 10). Unchanged; the runner still stops here. |
| Clearing-discrepancy audit (2023-01 – 2025-09) | `audit_clearing_mechanism.py` | EXECUTED (`clearing_discrepancy_audit.md`) |
| Direct benchmarks, 2024-Q4 | `run_direct_benchmarks.py` | VERIFIED; reproduced exactly inside the extension |
| Price classes (k = 40, validation-selected) | `run_curve_extension.py` | EXECUTED (extension only) |
| Unadjusted curve EN / HGB | `run_curve_extension.py` | EXECUTED, **EXPLORATORY** |
| Adjusted curve EN / HGB (out-of-fold-trained correction) | `run_curve_extension.py` | EXECUTED |
| Reference corrections (constant bias, observed-trained ĝ) | `run_curve_extension.py` | EXECUTED |
| Error decomposition 1–4, conditional and monthly metrics | `run_curve_extension.py` | EXECUTED |
| Diebold–Mariano comparisons (supplementary) | `extension_significance.py` | EXECUTED |
| Tests | `pytest -q` | 45 passed, 0 failed |

## Why `results.md` previously said price-class selection was not executed

That statement described the **production** pipeline, which stops at the failed clearing gate before Phase 3, and it remains true for that pipeline. Price-class selection (k = 40) was executed later, only in the separate exploratory path. The earlier report also wrongly listed figures 1, 4 and 5 as not produced; they were produced by the pre-gate step and by the direct-benchmark script. `results.md` has been rewritten to reconcile both points.

## Periods

| Period | Dates | Intervals in the common mask |
|---|---|---|
| Training | 2023-01-08 – 2024-06-30 | 12 953 |
| Validation | 2024-07-01 – 2024-09-30 | 2 208 |
| Exploratory evaluation (previously examined) | 2024-10-01 – 2024-12-31 | 2 209 |
| Confirmation (quasi-holdout) | 2025-01-01 – 2025-09-30 | 6 548 of 6 551 |

Out-of-fold walk-forward: monthly refits for every month 2023-07 – 2025-09 (54 family-month refits). Each uses only data dated before its month.
