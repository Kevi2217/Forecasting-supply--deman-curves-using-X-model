# Empirical results

Run date: 2026-11-16. Data: official OMIE files for delivery dates **2023-01-01 – 2024-12-31**.

**Overall status: INCOMPLETE — the pre-specified observed-curve clearing feasibility gate FAILED.**
The curve-based (X-Model-inspired) models were therefore **not trained or evaluated**, by design.
The three curve-independent benchmarks were evaluated out of sample on real data.

## 1. Data and quality gates

| Check | Result |
|---|---|
| Raw coverage, curves (`curva_pbc`) | 731 / 731 days, VERIFIED |
| Raw coverage, prices (`marginalpdbc`) | 731 / 731 days, VERIFIED (2023-05-28 is published only as revision `.2`) |
| Schema / data-quality validation | 1 462 / 1 462 files VERIFIED, including DST-exact period counts (23 h on 2023-03-26, 2024-03-31; 25 h on 2023-10-29, 2024-10-27) |
| Delivery intervals | 17 544 (= 731 × 24; DST days offset) |
| Curve geography | `MI` 16 486 intervals, `ES` 1 058 (all Iberian market-split hours); `MI` never used during an ES/PT price split → 0 exclusions |

Sources: `reports/metrics/raw_coverage.json`, `data_quality_report.json`, `curve_price_geography_alignment.json`.

## 2. Clearing approximation vs official Spanish price (gate)

Naive stepwise crossing of the published *offered* aggregate curves, all 17 544 intervals:

| n | MAE | RMSE | Bias | Median AE | p95 AE | Invalid crossings |
|---|---|---|---|---|---|---|
| 17 544 | **26.14** | 35.65 | **−24.67** | 20.28 | 74.34 | 0 (0.0 %) |

Gate (pre-specified in `configs/default.yaml`): MAE ≤ 10 EUR/MWh and invalid share ≤ 1 %. **MAE criterion failed.**

Error by price regime (EUR/MWh):

| Official price | n | MAE | Bias | Share \|err\| < 1 |
|---|---|---|---|---|
| < 0 | 247 | 0.41 | −0.24 | 0.89 |
| = 0 | 646 | 0.10 | −0.07 | 0.97 |
| 0 – 20 | 2 198 | 5.29 | −4.76 | 0.19 |
| 20 – 60 | 3 093 | 26.41 | −24.45 | 0.03 |
| 60 – 100 | 5 465 | 30.66 | −28.86 | 0.04 |
| > 100 | 5 895 | 33.51 | −32.05 | 0.03 |

By year: 2023 MAE 32.27 (bias −30.91); 2024 MAE 20.02 (bias −18.45). Test quarter (2024-Q4): MAE 25.65.
By hour, the error peaks at 18:00–22:00 local (MAE ≈ 40–47) and is lowest at night (≈ 10). See `figures/figure6_clearing_diagnostics.png`.

### Why the approximation fails (ex-post audit, `scripts/audit_clearing_gap.py`)

Comparing offered (`O`) and matched (`C`) curves at the official price p\*:

| Quantity (MWh per interval) | Mean | Median | p95 | Share > 500 MWh |
|---|---|---|---|---|
| Offered supply at ≤ p\* that was **not** matched | 4 552 | 4 252 | 12 075 | 84 % |
| Matched demand **absent** from offered demand at ≥ p\* | 1 016 | 85 | 3 068 | 47 % (\|gap\|) |

Correlation with the clearing error: −0.71 (unmatched cheap supply) and −0.22 (demand gap).

Interpretation: the published offered curves contain cheap supply that EUPHEMIA rejects (consistent with complex-order conditions such as minimum-income, indivisibility and block orders) and omit market-coupling flows. A naive crossing of these curves therefore clears systematically too low whenever thermal units with such conditions are marginal, and is nearly exact when renewables set zero/negative prices. This is a property of the public data and the market design, not a parsing error.

## 3. Out-of-sample benchmark comparison

Test set: 2024-10-01 – 2024-12-31, monthly expanding-window refits, hyperparameters selected on 2024-07-01 – 2024-09-30. All rows share the same **N = 2 209** intervals; 0 test intervals excluded. Metrics in EUR/MWh, bias = mean(prediction − actual).

| Model | MAE | RMSE | Bias | N |
|---|---|---|---|---|
| Weekly Persistence | 37.29 | 48.93 | −5.82 | 2 209 |
| Direct Elastic Net | **22.63** | **29.53** | −2.81 | 2 209 |
| Direct Gradient Boosting | 23.17 | 29.99 | −4.06 | 2 209 |
| Curve-Based Elastic Net | not evaluated — clearing gate failed | | | |
| Curve-Based Gradient Boosting | not evaluated — clearing gate failed | | | |

Selected hyperparameters (validation MAE): Direct Elastic Net α = 0.001, l1_ratio = 0.9 (18.40); Direct HGB learning rate 0.05, 31 leaves, 150 iterations, L2 0.1 (16.44).

Conditional subsets (thresholds from training data only: high price ≥ 140.21, large hourly change ≥ 26.41):

| Subset | n | Persistence MAE | Direct EN MAE | Direct HGB MAE |
|---|---|---|---|---|
| High price | 308 | 31.44 (bias −28.67) | 22.16 (bias −21.08) | 26.28 (bias −26.08) |
| Large absolute change | 135 | 42.01 | 27.83 | 28.59 |
| Negative price | 5 | 0.36 | 3.73 | 5.25 |

The negative-price subset (n = 5) is too small for inference. All models under-predict high-price hours.

Sources: `reports/metrics/direct_benchmarks/{summary.json, forecast_metrics.csv, test_predictions.csv}`.

## 4. Stages not executed (blocked by the gate)

Price-class fitting and selection, curve reconstruction, class-volume Elastic Net / HGB, implied-price forecasting, error decomposition B/C, and figures 1–5 and 7. No values or figures are reported for them.
