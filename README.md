# Day-Ahead Electricity Price Forecasting from Auction Curves

An independent quantitative research and machine-learning portfolio project inspired by the X-Model literature. It asks whether forecasting the *structure* of the Spanish day-ahead auction (supply and demand curves) yields better price forecasts than forecasting the price directly.

## Headline findings (real OMIE data, 2023-01 – 2025-09)

1. **Public offered curves do not reproduce the official price.** A naive stepwise crossing has MAE 26.14 EUR/MWh (bias −24.67), so the project's pre-specified feasibility gate (MAE ≤ 10) **fails**. The gate is kept and the production pipeline still stops there.
2. **The cause is in the data, not the code.** The same algorithm reproduces the price on OMIE's *matched* curves (MAE 0.43). The gap comes mainly from cheap complex-condition offers that the market algorithm rejects and that are invisible before the auction. This is shown by association and accounting, not by causal identification.
3. **Curve models were run as a separately labelled exploratory experiment**, with a pre-registered dynamic correction P_adj = P_curve + ĝ(X).
4. **The correction works relative to the unadjusted curve models.** On the 2025 quasi-holdout it cuts curve-model MAE by 5.5–6.4 EUR/MWh (p ≤ 0.001) and beats a constant-bias baseline (p < 0.001).
5. **The curve approach does not beat direct price forecasting.** It is tied on 2024-Q4 and significantly worse on 2025 (+2.6 to +3.1 EUR/MWh MAE).

## Model comparison

All models in a period are scored on identical intervals and forecast origins, with monthly expanding refits. EUR/MWh; bias = mean(prediction − actual).

| Model | 2024-Q4 MAE | RMSE | Bias | **2025 MAE** | RMSE | Bias | 2025 vs persistence |
|---|---|---|---|---|---|---|---|
| Weekly Persistence | 37.29 | 48.93 | −5.82 | 28.18 | 40.73 | +1.60 | — |
| Direct Elastic Net | **22.63** | 29.53 | −2.81 | 18.85 | 25.38 | +1.29 | +33.1 % |
| Direct Gradient Boosting | 23.17 | 29.99 | −4.06 | **17.76** | 24.79 | +2.35 | +37.0 % |
| Unadjusted Curve-Based Elastic Net *(exploratory)* | 33.73 | 43.64 | −24.06 | 28.38 | 40.60 | −21.29 | −0.7 % |
| Unadjusted Curve-Based Gradient Boosting *(exploratory)* | 34.70 | 43.60 | −25.49 | 25.91 | 37.39 | −19.23 | +8.1 % |
| Adjusted Curve-Based Elastic Net | 24.44 | 31.79 | −10.84 | 22.00 | 29.89 | −5.47 | +21.9 % |
| Adjusted Curve-Based Gradient Boosting | 23.49 | 31.22 | −5.99 | 20.41 | 27.50 | −1.17 | +27.6 % |

N = 2 209 (2024-Q4) and 6 548 of 6 551 (2025). The three 2025 exclusions are lag-source hours missing on the 23-hour DST day, and they are excluded from every model.

**Labels.**
- *2024-Q4* is **exploratory**: the direct benchmarks had already been examined on it.
- *2025-01 – 09* is a **quasi-holdout**. All choices were frozen on 2024-Q3 before any 2025 forecast existed. However, 2025 observed curves were inspected in the clearing audit, and the clearing discrepancy shifts structurally after OMIE's 2025-03-19 offer-typology change.

Full detail: [`reports/results.md`](reports/results.md). Audit: [`reports/clearing_discrepancy_audit.md`](reports/clearing_discrepancy_audit.md). Protocol: [`reports/preregistration_curve_extension.md`](reports/preregistration_curve_extension.md). Status: [`reports/execution_status.md`](reports/execution_status.md), [`reports/technical_audit.md`](reports/technical_audit.md).

## Research question and relation to the X-Model

> Can forecasting day-ahead auction supply and demand curves produce more accurate Spanish day-ahead price predictions than direct price models?

The X-Model (Ziel & Steinert, 2016):
1. compresses auction curves into price classes;
2. forecasts the class volumes;
3. reconstructs the curves;
4. derives the implied clearing price.

This repository follows that idea: training-only, energy-weighted price classes (k = 40 per side, chosen on validation) and Elastic Net / Histogram Gradient Boosting class-volume models. The adjusted model adds a statistical clearing correction. It is an **X-Model-inspired extension**, not a reproduction of the paper.

## How the adjusted model works

P_adj,t = P_curve,t + ĝ(X_t)

- **Out-of-fold curve forecasts.** The curve models are refitted monthly (expanding window) and forecast every month from 2023-07 onwards.
- **Target.** ĝ is trained on the realised errors of these out-of-fold forecasts.
- **Predictors, all available before the D−1 12:00 gate closure:**
  - the forecast curves' implied price and total volumes;
  - 1-, 2- and 7-day lags of the clearing discrepancy, the unaccepted cheap supply and the demand gap;
  - hour and weekday indicators.
- **Never used as predictors:** same-day matched curves, rejected volumes, offer typology or official prices.
- **Model choice.** A regularised Elastic Net was selected on validation. HGB was allowed only with a ≥ 5 % validation gain, which it did not achieve.

## Error decomposition (2025, Gradient Boosting family)

Signed means add exactly; MAEs do not.

| Component | Signed mean | MAE |
|---|---|---|
| 1 Clearing approximation (P_obs − P_official) | −19.31 | 20.63 |
| 2 Price-class compression (P_comp − P_obs) | +2.16 | 4.02 |
| 3 Curve forecasting (P_curve − P_comp) | −2.08 | 17.35 |
| 4 Adjustment estimation (ĝ − d) | +0.91 | 13.53 |
| Total unadjusted (1+2+3) | −19.23 | 25.91 |
| Total adjusted (3+4) | −1.17 | 20.41 |

## Interpretation

- The correction removes the clearing bias, but the day-to-day discrepancy is only partly predictable (component 4 MAE ≈ 14 EUR/MWh). That residual costs more than the structural information the curves add.
- The curve models are best only where clearing is near-exact. In 2025 negative-price hours (n = 535), the unadjusted curve models have MAE 2.0–2.2, against 3.8–7.4 for the direct models.
- The regime-blind linear correction loses that advantage. A regime-dependent correction is the obvious next step, but it needs fresh untouched data to be evaluated honestly.

## Figures (`reports/figures/`)

| Figure | Content |
|---|---|
| `figure1_observed_curves.png` | Offered vs matched curves, 2024-10-01 20:00 |
| `figure6_clearing_diagnostics.png`, `figure8_clearing_mechanism_audit.png` | Clearing gate diagnostics; monthly error by curve combination; error vs unaccepted volume |
| `extension_figure2_price_class_reconstruction.png`, `extension_figure7_reconstruction_diagnostics.png` | 40-class reconstruction and compression error by k |
| `extension_figure3_forecast_vs_observed_curves.png` | Forecast vs observed supply/demand curves (two fixed intervals) |
| `extension_predictions_exploratory_2024Q4.png`, `extension_predictions_confirmation_2025.png` | Actual vs predicted prices |
| `extension_model_comparison.png`, `extension_conditional_mae_by_hour.png` | Benchmark comparison; MAE by hour |
| `extension_discrepancy_stability.png` | Stability of the clearing discrepancy |
| `figure4_…`, `figure5_…` | Direct benchmarks (2024-Q4) |

## Repository structure

```text
├── configs/default.yaml          # production pipeline (2023–2024, gates)
├── configs/extension.yaml        # exploratory/adjusted extension (2023-01 – 2025-09)
├── src/electricity_forecasting/
│   ├── data/                     # download, parse (incl. 2025 offer typology), coverage, validation
│   ├── market/                   # curves, clearing, clearing_audit, price classes, compression
│   ├── features/  models/  evaluation/  visualization/
├── scripts/
│   ├── download_omie.py          # official raw files (+ provenance)
│   ├── audit_raw_coverage.py     # completeness audit
│   ├── run_project.py            # production pipeline; stops at the failed clearing gate
│   ├── run_direct_benchmarks.py  # curve-independent benchmarks (2024-Q4)
│   ├── audit_clearing_mechanism.py  # offered vs matched clearing audit (2023–2025)
│   ├── run_curve_extension.py    # exploratory + adjusted curve models (cached stages)
│   └── extension_significance.py # Diebold–Mariano comparisons
├── tests/                        # 45 tests
├── data/{raw,interim,processed}/ # not committed
└── reports/                      # results, audits, protocol, metrics/, extension/, figures/
```

## Reproduce

Python 3.11+ (executed with 3.12.10).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                                                              # 45 passed

python scripts/download_omie.py --start 2023-01-01 --end 2025-09-30 --dataset both    # ~35 min, ~3.3 GB
python scripts/run_project.py                     # production: exits FAILED at the clearing gate (by design)
python scripts/run_direct_benchmarks.py           # direct benchmarks, 2024-Q4
python scripts/audit_clearing_mechanism.py --config configs/extension.yaml           # ~6 min
python scripts/run_curve_extension.py --config configs/extension.yaml                # ~3.2 h on 10 cores; stages cached
python scripts/extension_significance.py
```

## Limitations

- The curve approach is not better than direct forecasting on the available holdout. All curve results are conditional on a failed clearing prerequisite.
- 2025 is a quasi-holdout and includes a structural change in the clearing discrepancy.
- Independent class-volume models ignore multivariate curve dependence. The inherited curve-model selection score is dominated by near-empty classes.
- No public pre-auction load, wind or solar forecasts are used. The benchmarks are modest baselines, not state of the art.
- OMIE's `C01`/`C02`/`C04` offer-typology codes are not decoded from official documentation.

## Data attribution and references

Data: **OMIE (Operador del Mercado Ibérico de Energía)**, public day-ahead files; see [`reports/data_sources.md`](reports/data_sources.md). Raw files are not committed.

Ziel, F. & Steinert, R. (2016). *Electricity Price Forecasting using Sale and Purchase Curves: The X-Model*. Energy Economics.

## Independence

This is a standalone, publicly shareable portfolio project built from public market data and independently developed code. It is not presented as sponsored by or developed for an electricity company.
