# Independent technical audit

Date: 2026-11-16. Supersedes the earlier audits of 2026-11-14 and of the morning of 2026-11-16.

Scope:
- Production pipeline on 2023-01-01 – 2024-12-31.
- Clearing-discrepancy audit and exploratory/adjusted curve extension on 2023-01-01 – 2025-09-30.
- Environment: Python 3.12.10 virtual environment (numpy 2.5.5, pandas 2.3.3, scikit-learn 1.9.1).

## Summary verdict

- **Data.** All 2 008 OMIE files are real, complete and validated, including the 2025-03-19 schema change.
- **Production gate.** The observed-curve clearing gate **fails** (MAE 26.14 vs ≤ 10) and was **not** weakened. `run_project.py` and `configs/default.yaml` are unchanged in logic and thresholds.
- **Cause of the failure.** The audit locates it in the content of OMIE's offered curves, not in the code: the same clearing algorithm reproduces the price on matched curves (MAE 0.43).
- **Curve models.** They were run only in a separate, pre-registered exploratory path.
- **Leakage.** No look-ahead was found in features, classes, hyperparameters or the correction.
- **Labelling.** 2024-Q4 is labelled exploratory; 2025-01 – 09 is a quasi-holdout (see results §4).

## Audit table

| Question | Status | Evidence |
|---|---|---|
| Data real and correctly sourced? | VERIFIED | 1 004 + 1 004 official files with SHA-256 provenance sidecars. One price day exists only as revision `.2`. |
| Schema verified on real bytes? | VERIFIED | ISO-8859-1, `;`, Spanish number format. A 9-column schema (`Periodo`, `Potencia`, `Tipología de Oferta`) from 2025-03-19 is parsed as optional `offer_type`. Quarter-hour files (from 2025-10-01) fail closed. |
| Spanish price column | VERIFIED | Column 6, checked on a split hour |
| DST handling | VERIFIED | Exact 23/24/25 periods; 24 095 unique UTC intervals |
| Clearing implementation correct? | VERIFIED | Matched × matched crossing MAE 0.43, median AE 0. p\* lies within the matched bracket in 99.5 % of intervals. The audit's `p_offered` equals the production observed-clearing table exactly (max difference 0.0, 17 544 intervals). |
| Why does the gate fail? | EXPLAINED (association + accounting) | Swapping offered supply for matched supply cuts MAE from 24.6 to 5.6. Error vs unaccepted cheap supply: r = −0.71. From 2025-03-19, all unaccepted cheap supply is complex-condition-labelled. Causality is not identified (`clearing_discrepancy_audit.md` §5). |
| Gate weakened or bypassed? | NO | The production runner still exits `FAILED`. Curve models ran only in `run_curve_extension.py`, with every output labelled exploratory. |
| Price classes training-only; k selected on validation only? | VERIFIED | Classes are fitted on 2023-01-01 – 2024-06-30 files. k = 40 is chosen on 2024-Q3 compression error (14.07 / 6.15 / 3.60 for k = 10 / 20 / 40). A recomputation reproduced the cached specification exactly before modelling. |
| Hyperparameters chosen without evaluation data | VERIFIED | Selected on 2024-Q3 only (`extension/model_selection.json`). Direct selections are identical to the standalone benchmark run. |
| Correction uses only pre-auction information | VERIFIED + TESTED | Predictors: forecast-curve state, ≥ 1-day lags of discrepancy / unaccepted volume / demand gap, calendar. The D−1 files are issued around 12:55 on D−2. A unit test asserts that no post-auction column enters unlagged. |
| Correction trained only on the past | VERIFIED | Refitted monthly on out-of-fold rows dated before the forecast month. The constant baseline uses 2023-07 – 2024-06 only. |
| Out-of-fold curve forecasts genuinely out-of-fold | VERIFIED | Each family-month refit trains on dates strictly before the month (`expanding_monthly_predictions`, `dates < month_start`). |
| Hyperparameters informed by validation used for training-period out-of-fold forecasts | DISCLOSED | No evaluation data involved; slight optimism possible in the correction's training errors |
| Same evaluation intervals and forecast origins for all models | VERIFIED | Common finite mask: 2 209/2 209 (2024-Q4), 6 548/6 551 (2025). The 3 exclusions are lag-source hours missing on the 23-hour DST day, excluded for every model. |
| Economically valid forecast curves | VERIFIED | Non-negative class volumes (5.4 % EN and 1.0 % HGB raw negatives clipped, reported), monotone step curves, 19 746/19 746 crossings |
| Decomposition correct | TESTED | Signed identities (1+2+3; 3+4) are asserted numerically in code and in a unit test. MAE/RMSE are reported per component and never summed. |
| MAE/RMSE/DM correct | TESTED | Deterministic metric tests; a DM test-statistic sign/power test |
| Scalers fitted on past data only | VERIFIED | `StandardScaler` sits inside each sklearn `Pipeline` |
| Independence / confidentiality | VERIFIED | No references to employers, collaborations or confidential research |

## Defects found and fixed

1. **Repository layout lost.** The commit was a flattened upload and the working tree was a filename-scrambled copy (preserved in `git stash@{0}`). Layout rebuilt.
2. **Parser performance.** About 40× speedup by vectorising.
3. **Silent row skipping.** Malformed rows now fail closed.
4. **DST validation gap.** Period counts must now equal the exact day length.
5. **File revisions.** `.1`–`.5` are probed; the latest revision is used.
6. **DST misalignment of lags and persistence.** Lags now join on local wall-clock hour.
7. **O(n²) step-curve evaluation.** Now O(n log n).
8. **2025 schema: offer-typology column silently dropped.** The parser kept only 8 fields. It now keeps the optional `offer_type` column and pads short rows instead of skipping them.
9. **Walk-forward cache contamination risk.** Found in a dry run: a toy configuration wrote into the shared cache. The cache was deleted, and refits are now keyed by a configuration hash and refused on mismatch.

## Tests

`pytest -q`: **45 passed, 0 failed.** That is 35 previous tests plus 10 new ones:
- offer-type parsing (2);
- the clearing audit on synthetic offered/matched curves (3);
- trailing-median look-ahead;
- forecast vs observed curve state;
- no unlagged post-auction predictors;
- DM test;
- decomposition identities.

## Remaining limitations

- **The curve approach does not beat direct forecasting** on the 2025 quasi-holdout (MAE +2.6 to +3.1, p ≤ 0.001).
- **2025 is not a perfectly untouched holdout.** The clearing audit inspected 2025 observed curves before Amendment 1. The clearing discrepancy also shifts structurally around the 2025-03-19 offer-typology change, so 2025 tests robustness to a regime change, not stationarity.
- **The correction is regime-blind.** It harms the curve models in negative-price hours, where clearing is near-exact. A regime-dependent correction would need fresh untouched data to evaluate.
- **The production curve-model selection criterion is dominated by near-empty classes.** It scales class MAE by max(mean |y|, 1), so the curve validation scores of 7.25 / 8.45 are dominated by classes with tiny volumes. It was kept for fidelity to the production design.
- **Independent class-volume models** ignore multivariate curve dependence.
- **No public pre-auction load, wind or solar forecasts** are used by any model.
- **The `C01`/`C02`/`C04` codes are not decoded** from official documentation.
