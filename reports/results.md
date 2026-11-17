# Empirical results

Run date: 2026-11-16. Data: official OMIE files for delivery dates **2023-01-01 – 2025-09-30**:
- 1 004 curve files and 1 004 price files, all of which pass the coverage, schema and DST gates;
- 24 095 hourly intervals.

**Status in one paragraph.** The production pipeline (`run_project.py`, 2023–2024) still **stops at its pre-specified clearing gate**: naive clearing of OMIE offered curves has MAE 26.14 EUR/MWh against a limit of 10. That gate was not changed. In a separate, pre-registered execution path (`run_curve_extension.py`), the curve-based models were trained anyway and labelled **exploratory**, and a dynamic clearing correction was added. The adjusted curve models clearly beat the unadjusted curve models and a constant-bias baseline. They do **not** beat direct price forecasting on the 2025 confirmation period, and are statistically tied with it on 2024-Q4.

## 1. Data and quality gates

| Check | Result |
|---|---|
| Curves (`curva_pbc`) | 731/731 days (2023–24) + 273/273 days (2025-01 – 09); all VERIFIED |
| Prices (`marginalpdbc`) | Same coverage; 2023-05-28 exists only as revision `.2` |
| Schema / DST validation | 2 008 / 2 008 files VERIFIED; DST days 23 h / 25 h exact |
| Schema change | From 2025-03-19 OMIE adds a `Tipología de Oferta` column and labels power as `Potencia`; periods stay hourly. Parsed and validated. |
| Geography | `MI` curve in unsplit hours, `ES` curve in every split hour; 0 exclusions |
| Out of scope | From 2025-10-01 the market uses 15-minute periods |

## 2. Clearing gate (production) and clearing-discrepancy audit

| Production gate (2023–2024) | n | MAE | RMSE | Bias | Gate |
|---|---|---|---|---|---|
| Naive crossing of offered curves | 17 544 | 26.14 | 35.65 | −24.67 | MAE ≤ 10: **FAILED** (unchanged) |

Audit summary (2023-01 – 2025-09, `clearing_discrepancy_audit.md`):

| Crossing (same algorithm) | MAE | Bias |
|---|---|---|
| offered × offered | 24.64 | −23.21 |
| **matched × matched** | **0.43** | −0.39 |
| matched supply × offered demand | 5.63 | −5.37 |
| offered supply × matched demand | 35.07 | −33.37 |

- **The algorithm and parsing are correct.** On the matched curves, the method reproduces the official price.
- **The gap is mainly cheap supply that was offered but not accepted.** Correlation with the error is r = −0.71 overall and −0.73 to −0.76 within price regimes above 20 EUR/MWh.
- **From 2025-03-19 the typology labels settle where the unaccepted volume comes from.** Every unaccepted cheap MWh carries a complex-condition label. Simple offers below the price are always accepted.
- **Coupling flows do not explain the gap.** Adding the realised coupling rows does not reduce it.
- **This is association and accounting, not identified causality.**
- **The discrepancy is non-stationary.** Monthly MAE ranges from 8.5 to 51.1, and it falls to about 10–13 from April 2025.

## 3. Price classes (validation only)

Classes are fitted on the training files only, 2023-01-01 – 2024-06-30. The count k is selected by compression error on the validation files (2024-Q3) as the smallest k within 0.05 EUR/MWh of the best.

| k per side | Validation MAE (compressed vs observed-curve clearing) | Bias |
|---|---|---|
| 10 | 14.07 | +8.57 |
| 20 | 6.15 | +2.41 |
| **40 (selected)** | **3.60** | +2.18 |

Hence 80 class-volume targets: 40 supply plus 40 demand.

## 4. Protocol and labels

| Period | Dates | N in the common mask | Role |
|---|---|---|---|
| Training | 2023-01-08 – 2024-06-30 | 12 953 | Fitting; the first 7 days are lost to the 7-day lags |
| Validation | 2024-07-01 – 2024-09-30 | 2 208 | All selection: k, hyperparameters, correction specification |
| **Exploratory** | 2024-10-01 – 2024-12-31 | 2 209 / 2 209 | Previously examined (direct benchmarks). **Not** an untouched test. |
| **Confirmation** | 2025-01-01 – 2025-09-30 | 6 548 / 6 551 | Evaluated once with the frozen protocol. 3 intervals are excluded (see below). |

The three excluded intervals are 2025-03-31, 2025-04-01 and 2025-04-06 at 02:00. Their lag source hour does not exist on the 23-hour DST day 2025-03-30, and they are excluded from **every** model.

How "untouched" is 2025?
- No 2025 forecast or evaluation result was computed before the protocol and Amendment 1 were frozen.
- Hyperparameters, k and the correction specification were all chosen on 2024-Q3.
- However, the clearing-discrepancy **audit** (observed curves, no forecasts) was run on 2025 data before Amendment 1, which was requested by the project owner.
- 2025 is therefore best described as a *quasi-holdout*: no model or methodology choice was tuned on 2025 forecast performance.

Every evaluation month is forecast by models refitted on all data dated before that month (monthly expanding window). Price-class boundaries and hyperparameters are frozen.

## 5. Main comparison

EUR/MWh. Bias = mean(prediction − actual). Improvement = 1 − MAE / MAE(persistence). All rows in a period share identical intervals and forecast origins.

### Confirmation period 2025-01-01 – 2025-09-30 (N = 6 548)

| Model | MAE | RMSE | Bias | N | vs persistence |
|---|---|---|---|---|---|
| Weekly Persistence | 28.18 | 40.73 | +1.60 | 6 548 | — |
| Direct Elastic Net | 18.85 | 25.38 | +1.29 | 6 548 | +33.1 % |
| **Direct Gradient Boosting** | **17.76** | **24.79** | +2.35 | 6 548 | **+37.0 %** |
| Unadjusted Curve-Based Elastic Net *(exploratory)* | 28.38 | 40.60 | −21.29 | 6 548 | −0.7 % |
| Unadjusted Curve-Based Gradient Boosting *(exploratory)* | 25.91 | 37.39 | −19.23 | 6 548 | +8.1 % |
| Adjusted Curve-Based Elastic Net | 22.00 | 29.89 | −5.47 | 6 548 | +21.9 % |
| Adjusted Curve-Based Gradient Boosting | 20.41 | 27.50 | −1.17 | 6 548 | +27.6 % |

### Exploratory period 2024-10-01 – 2024-12-31 (N = 2 209)

| Model | MAE | RMSE | Bias | N | vs persistence |
|---|---|---|---|---|---|
| Weekly Persistence | 37.29 | 48.93 | −5.82 | 2 209 | — |
| **Direct Elastic Net** | **22.63** | **29.53** | −2.81 | 2 209 | **+39.3 %** |
| Direct Gradient Boosting | 23.17 | 29.99 | −4.06 | 2 209 | +37.9 % |
| Unadjusted Curve-Based Elastic Net *(exploratory)* | 33.73 | 43.64 | −24.06 | 2 209 | +9.6 % |
| Unadjusted Curve-Based Gradient Boosting *(exploratory)* | 34.70 | 43.60 | −25.49 | 2 209 | +7.0 % |
| Adjusted Curve-Based Elastic Net | 24.44 | 31.79 | −10.84 | 2 209 | +34.5 % |
| Adjusted Curve-Based Gradient Boosting | 23.49 | 31.22 | −5.99 | 2 209 | +37.0 % |

The direct and persistence rows on 2024-Q4 match the earlier standalone benchmark run exactly.

### Reference variants (not headline models)

| Variant | 2024-Q4 MAE | 2025 MAE | 2025 Bias |
|---|---|---|---|
| Curve EN + constant training bias (+13.86) | 28.85 | 25.97 | −7.42 |
| Curve GB + constant training bias (+19.12) | 27.71 | 24.69 | −0.10 |
| Curve EN + observed-curve-trained ĝ (original pre-registration) | 21.80 | 20.46 | −5.93 |
| Curve GB + observed-curve-trained ĝ (original pre-registration) | 23.16 | 19.51 | −2.39 |

The observed-curve-trained correction (the originally pre-registered variant) does slightly *better* than the out-of-fold-trained primary in both periods. The amendment did not improve results, and the primary is kept as the headline because it was frozen before evaluation.

## 6. Correction selection (validation 2024-Q3, curve forecasts from the walk-forward)

| Curve family | No correction | Constant bias | Trailing 28-day hourly median | ĝ out-of-fold (chosen) | ĝ observed-trained |
|---|---|---|---|---|---|
| Elastic Net | 23.95 | 22.66 | 20.53 | **18.72**: EN α = 1, l1 = 0.5 | 18.46: HGB (≥ 5 % better than EN 19.44) |
| HGB | 24.41 | 20.80 | 19.09 | **17.61**: EN α = 1, l1 = 0.1 | 17.60: EN |

For the primary correction, HGB did not clear the 5 % improvement threshold for either family (18.17 vs 18.72; 17.48 vs 17.61), so the regularised linear model was retained.

## 7. Statistical comparison (supplementary, not pre-registered)

Diebold–Mariano tests on absolute error. Hourly loss differentials are averaged per delivery day; Newey–West with 7 lags and the HLN correction (`dm_tests.csv`).

| Comparison (A vs B) | 2024-Q4: ΔMAE, p | 2025: ΔMAE, p |
|---|---|---|
| Adjusted EN vs unadjusted EN | −9.29, < 0.001 | −6.38, < 0.001 |
| Adjusted GB vs unadjusted GB | −11.21, < 0.001 | −5.50, 0.001 |
| Adjusted EN vs constant-bias EN | −4.41, < 0.001 | −3.97, < 0.001 |
| Adjusted GB vs constant-bias GB | −4.23, < 0.001 | −4.29, < 0.001 |
| Adjusted EN vs direct EN | +1.81, 0.136 | **+3.15, 0.001** |
| Adjusted GB vs direct GB | +0.32, 0.745 | **+2.64, < 0.001** |
| Observed-trained-ĝ EN vs direct EN | −0.83, 0.402 | +1.61, 0.078 |
| Observed-trained-ĝ GB vs direct GB | −0.01, 0.987 | +1.75, 0.003 |

## 8. Error decomposition (signed identities exact; MAE/RMSE not additive)

Components:
1. P_obs − P_official
2. P_comp − P_obs
3. P_curve − P_comp
4. ĝ − d, where d = P_official − P_comp

Identities: unadjusted error = 1 + 2 + 3; adjusted error = 3 + 4.

| Component | 2024-Q4 EN: mean / MAE | 2024-Q4 GB: mean / MAE | 2025 EN: mean / MAE | 2025 GB: mean / MAE |
|---|---|---|---|---|
| 1 Clearing approximation | −23.64 / 25.65 | −23.64 / 25.65 | −19.31 / 20.63 | −19.31 / 20.63 |
| 2 Price-class compression | +2.53 / 4.08 | +2.53 / 4.08 | +2.16 / 4.02 | +2.16 / 4.02 |
| 3 Curve forecasting | −2.94 / 24.34 | −4.37 / 25.87 | −4.14 / 19.58 | −2.08 / 17.35 |
| **Total unadjusted** (1+2+3) | −24.06 / 33.73 | −25.49 / 34.70 | −21.29 / 28.38 | −19.23 / 25.91 |
| 4 Adjustment estimation (out-of-fold ĝ) | −7.90 / 15.19 | −1.62 / 14.43 | −1.34 / 14.06 | +0.91 / 13.53 |
| **Total adjusted** (3+4) | −10.84 / 24.44 | −5.99 / 23.49 | −5.47 / 22.00 | −1.17 / 20.41 |

Reading:
- The unadjusted bias is almost entirely component 1, the clearing approximation.
- Curve forecasting (3) is nearly unbiased but has a large absolute error, about the size of the direct models' whole MAE.
- The correction removes most of the bias. Component 4 still has MAE ≈ 14: the day-to-day discrepancy is only partly predictable.
- Per-variant rows are in `extension/error_decomposition.csv`.

## 9. Conditional performance

High-price and large-change thresholds are training-only 95th percentiles: 140.21 EUR/MWh and 26.41 EUR/MWh.

| Subset (n 2024-Q4 / 2025) | Persistence | Direct EN | Direct GB | Unadj. EN | Unadj. GB | Adj. EN | Adj. GB |
|---|---|---|---|---|---|---|---|
| All, 2025 (6 548) | 28.2 | 18.9 | 17.8 | 28.4 | 25.9 | 22.0 | 20.4 |
| High price, 2025 (370) | 39.8 | 26.9 | **24.5** | 66.6 | 67.8 | 37.0 | 31.6 |
| Large change, 2025 (783) | 36.7 | **24.3** | **24.3** | 47.3 | 44.2 | 29.9 | 27.7 |
| **Negative price, 2025 (535)** | 3.6 | 7.4 | 3.8 | **2.0** | 2.2 | 11.7 | 13.7 |
| High price, 2024-Q4 (308) | 31.4 | 22.2 | 26.3 | 44.0 | 47.3 | 27.5 | **21.2** |
| Large change, 2024-Q4 (135) | 42.0 | 27.8 | 28.6 | 49.5 | 45.1 | 30.4 | **26.5** |

In negative-price (renewable-marginal) hours the **unadjusted** curve models are the most accurate of all, consistent with the near-exact clearing in that regime. The linear correction spoils this by adding a regime-blind positive shift. This is an observation from the confirmation data and was **not** used to change the model.

Monthly MAE (`extension/monthly_metrics.csv`): the curve models do worst in 2025-02 (adjusted EN 36.5, GB 31.4, direct GB 22.6) and 2025-06 (adjusted EN 29.5). Both months show clearing discrepancy shifts (`clearing_discrepancy_audit.md` §6). In 2024-12 and 2025-09, the adjusted curve models are marginally ahead of direct EN.

## 10. Curve validity

- Every forecast curve is a monotone step function at training-derived class prices, with non-negative volumes. Every implied price exists: 19 746 / 19 746 crossings.
- Negative raw class forecasts were clipped to zero for 5.4 % (EN) and 1.0 % (HGB) of 1 579 680 class predictions.
- Implied prices land on the 40 class representative prices, so different curve forecasts often give the same price. This discreteness is part of components 2 and 3.

## 11. Figures

| File | Content |
|---|---|
| `figure1_observed_curves.png` | Offered vs matched curves, 2024-10-01 20:00 |
| `figure6_clearing_diagnostics.png` | Gate diagnostics (2023–24) |
| `figure8_clearing_mechanism_audit.png` | Monthly crossing error by curve combination, and error vs unaccepted volume |
| `extension_discrepancy_stability.png` | Monthly distribution of d = P_official − P_comp, and mean by hour/year |
| `extension_figure2_price_class_reconstruction.png` | Observed vs 40-class reconstructed curve |
| `extension_figure7_reconstruction_diagnostics.png` | Compression error by k (validation) |
| `extension_figure3_forecast_vs_observed_curves.png` | Forecast vs observed curves for two fixed intervals (2024-10-01 and 2025-06-02, 20:00) |
| `extension_predictions_{exploratory_2024Q4,confirmation_2025}.png` | Actual vs predicted prices, last two weeks of each period |
| `extension_model_comparison.png` | MAE of all models and reference variants, both periods |
| `extension_conditional_mae_by_hour.png` | MAE by local hour |
| `figure4_…`, `figure5_…` | Direct benchmarks (2024-Q4) |

## 12. Conclusions

1. Naive clearing of OMIE's public offered curves cannot reproduce the official price (production gate failed, unchanged). The cause is in the data, not the code: cheap complex-condition offers are selectively rejected and invisible before the auction.
2. Unadjusted curve-based forecasts inherit a bias of about −20 EUR/MWh. They are no better than weekly persistence in 2025 (MAE 28.4 / 25.9 vs 28.2).
3. A dynamic, regularised correction trained on out-of-fold curve-forecast errors reduces MAE by 5.5–6.4 EUR/MWh in 2025 (p ≤ 0.001). It also beats a constant training-bias correction (p < 0.001), so the improvement is not just a level shift.
4. **The curve-based approach does not add value beyond direct price forecasting here.** It is tied on 2024-Q4 and significantly worse on 2025 (+2.6 to +3.1 EUR/MWh MAE).
   - The curve forecasts are only competitive where clearing is near-exact (negative prices).
   - Elsewhere, the unpredictable part of the clearing discrepancy (component 4 MAE ≈ 14) costs more than the curves' structural information gains.
