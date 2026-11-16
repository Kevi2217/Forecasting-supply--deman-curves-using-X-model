# Day-Ahead Electricity Price Forecasting from Auction Curves

An independent quantitative research and machine-learning portfolio project inspired by the X-Model literature. The project asks whether forecasting the *structure* of a day-ahead electricity auction can produce useful Spanish electricity-price forecasts compared with forecasting the price directly.

## Status (executed 2026-11-16 on real OMIE data, 2023-01-01 – 2024-12-31)

**Data pipeline: VERIFIED. Curve-clearing feasibility gate: FAILED. Overall empirical project: INCOMPLETE.**

- All 1 462 official OMIE files (731 curve days, 731 price days) were downloaded, parsed and validated, including DST-exact period counts.
- Naive clearing of the published *offered* supply/demand curves misses the official Spanish price by **MAE 26.14 EUR/MWh (bias −24.67)**. The pre-specified gate is MAE ≤ 10.
- The pipeline therefore stops before price-class compression and curve-based forecasting, as designed. **The curve-based (X-Model-inspired) models were not trained, and no performance is claimed for them.**
- The curve-independent benchmarks were evaluated out of sample on Q4-2024 (N = 2 209).

Details: [`reports/results.md`](reports/results.md), [`reports/technical_audit.md`](reports/technical_audit.md), [`reports/execution_status.md`](reports/execution_status.md).

## Research question

> Can forecasting the underlying day-ahead electricity auction supply and demand curves produce more accurate Spanish day-ahead electricity-price predictions than conventional direct price forecasting models?

The price is an outcome of the interaction between supply and demand curves. A curve model might therefore preserve information about market elasticity and nonlinear price sensitivity that a direct model does not represent.

## Relationship to the X-Model

The X-Model (Ziel & Steinert, 2016) reduces auction curves to price classes, forecasts class-level volumes, reconstructs the curves and derives an implied clearing price. This repository follows that structural idea (it is *X-Model-inspired*, not a replication) with **Elastic Net** and **Histogram Gradient Boosting** class-volume models.

## Empirical results

### Data

| Item | Value |
|---|---|
| Period | 2023-01-01 – 2024-12-31 (731 delivery days) |
| Intervals | 17 544 hourly (DST days: 23 h on 2023-03-26, 2024-03-31; 25 h on 2023-10-29, 2024-10-27) |
| Curve geography | Iberian `MI` curves in 16 486 hours; Spanish `ES` curves in all 1 058 market-split hours; 0 intervals excluded |
| Data-quality gate | 1 462 / 1 462 files VERIFIED |
| Split | train 2023-01-08 – 2024-06-30 (12 953; the first 7 days are lost to the 7-day lag) · validation 2024-07-01 – 09-30 (2 208) · test 2024-10-01 – 12-31 (2 209) |

### Clearing approximation vs official price (gate)

| n | MAE | RMSE | Bias | Median AE | p95 AE | Invalid |
|---|---|---|---|---|---|---|
| 17 544 | 26.14 | 35.65 | −24.67 | 20.28 | 74.34 | 0 % |

The approximation is almost exact when the price is zero or negative (MAE 0.10 / 0.41). The MAE rises to 26–34 when the price is above 20 EUR/MWh, and to about 45 in the 19:00–20:00 evening peak.

An ex-post audit against OMIE's *matched* curves explains the gap:
- In 84 % of hours, more than 500 MWh of supply offered at or below the official price was **not** accepted (mean 4.6 GWh). Its correlation with the clearing error is −0.71.
- Matched demand also includes volume absent from the offered demand curve, consistent with market-coupling exports.

The published offered curves therefore do not encode EUPHEMIA's complex-order rejections or interconnection flows, so a naive crossing systematically clears too low.

### Out-of-sample benchmark comparison (2024-10-01 – 2024-12-31)

Monthly expanding-window refits; hyperparameters chosen on the validation quarter only; identical test observations for every model; EUR/MWh; bias = mean(prediction − actual).

| Model | MAE | RMSE | Bias | N |
|---|---|---|---|---|
| Weekly Persistence | 37.29 | 48.93 | −5.82 | 2 209 |
| Direct Elastic Net | **22.63** | **29.53** | −2.81 | 2 209 |
| Direct Gradient Boosting | 23.17 | 29.99 | −4.06 | 2 209 |
| Curve-Based Elastic Net | — not evaluated (clearing gate failed) | | | |
| Curve-Based Gradient Boosting | — not evaluated (clearing gate failed) | | | |

Both direct models beat persistence by about 39 % MAE. Elastic Net and HGB are practically tied with these lag/calendar features. All models under-predict high-price hours (≥ 140.21 EUR/MWh, the training-set 95th percentile; n = 308). The test quarter has only 5 negative-price hours, too few for conclusions.

### Figures (`reports/figures/`)

| Figure | Content |
|---|---|
| `figure1_observed_curves.png` | Offered (and ex-post matched) supply/demand curves, 2024-10-01 20:00, with approximate and official prices |
| `figure4_direct_benchmarks_out_of_sample.png` | Official vs predicted prices, last two test weeks |
| `figure5_direct_benchmark_comparison.png` | MAE/RMSE benchmark comparison |
| `figure6_clearing_diagnostics.png` | Clearing approximation vs official: scatter, error distribution, error by hour |

The price-class reconstruction, forecast-vs-observed curve and reconstruction-error figures (figures 2, 3, 7) are produced only after a passed clearing gate, so none exist. No synthetic substitute is provided.

## Interpretation

1. **The curve-based hypothesis remains untested here, not refuted.** Its prerequisite, that observed public curves reproduce the price, fails for OMIE offered curves under a naive crossing. Any curve forecast would inherit an expected bias of about −25 EUR/MWh before forecasting error, which is larger than the direct models' entire MAE.
2. **This is a data/market-design finding, not a code defect.** Demand-side volumes reconcile with matched curves, the error disappears in renewable-marginal hours, and it is one-sided where thermal units bidding with complex conditions are marginal.
3. **A fix would be a methodology change.** For example, the rejected complex-order volume and net coupling flow could be treated as additional, forecastable curve shifts. Such a change should be specified in advance, before it is run against the gate.

## Workflow

```text
OMIE raw files → coverage + schema + DST gates → UTC/local intervals
→ observed cumulative curves → approximate stepwise clearing
→ [GATE: clearing vs official Spanish price]  ← stops here on 2023–2024 data
→ training-only price classes → compression validation
→ leakage-safe class-volume features → Elastic Net / HGB
→ forecast curves → implied price → common-sample walk-forward evaluation
```

Curve-independent benchmarks (`scripts/run_direct_benchmarks.py`) follow the same split, features, selection and walk-forward protocol using official prices only.

## Market and data conventions

- Curve files (`curva_pbc_YYYYMMDD.N`): ISO-8859-1, `;`-separated, Spanish number format. `C` = purchase, `V` = sale; `O` = offered, `C` = matched. OMIE serves only the latest revision `N` (one price day exists only as `.2`).
- Price files (`marginalpdbc_YYYYMMDD.N`): column 5 = Portugal, **column 6 = Spain** (verified on a market-split hour).
- Supply is accumulated low → high price and demand high → low. Same-price bids are aggregated.
- Intervals are `Europe/Madrid` periods mapped to UTC. Lags and weekly persistence align on **local wall-clock hour**, so DST days are compared like-for-like.
- Forecast features use only data from earlier delivery days. The day-D curve file is published around 12:55 on D−1, so D−1 results are public before the D−1 12:00 gate closure.

## Repository structure

```text
├── configs/default.yaml          # dates, splits, gates, model grids
├── src/electricity_forecasting/
│   ├── data/                     # download, parse, coverage, validation
│   ├── market/                   # curves, clearing, price classes, compression
│   ├── features/  models/  evaluation/  visualization/
├── scripts/
│   ├── download_omie.py          # official raw files (+ provenance)
│   ├── audit_raw_coverage.py     # completeness audit
│   ├── inspect_data.py           # schema/quality of single files
│   ├── run_project.py            # full fail-closed pipeline
│   ├── run_direct_benchmarks.py  # curve-independent benchmarks
│   └── audit_clearing_gap.py     # offered-vs-matched diagnostic
├── tests/                        # 35 tests
├── data/{raw,interim,processed}/ # not committed
└── reports/                      # results, audit, metrics/, figures/
```

## Reproduce

Python 3.11+ (executed with 3.12.10).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

pytest -q                                                                  # 35 passed
python scripts/download_omie.py --start 2023-01-01 --end 2024-12-31 --dataset both   # ~25 min, 2.3 GB
python scripts/audit_raw_coverage.py --start 2023-01-01 --end 2024-12-31
python scripts/run_project.py          # ~20 min; exits FAILED at the clearing gate
python scripts/run_direct_benchmarks.py
python scripts/audit_clearing_gap.py   # needs data/processed/observed_clearing_table.csv from run_project.py
```

Outputs: `reports/metrics/` (`raw_coverage.json`, `data_quality_report.json`, `curve_price_geography_alignment.json`, `clearing_validation.json`, `clearing_gap_audit.{json,csv}`, `direct_benchmarks/`), `reports/figures/`, `data/processed/observed_clearing_table.csv`.

## Limitations

- The clearing function is a naive step-curve crossing. It does not model EUPHEMIA complex orders, block orders or market coupling, and on this data it fails the project's own feasibility gate.
- The curve-based models are implemented and unit-tested but were not evaluated on real data.
- Benchmarks use calendar and price lags only (no load, wind or solar forecasts), so they are modest baselines rather than state-of-the-art.
- The evaluation covers one market and one test quarter, with very few negative-price hours.
- Independent class-volume forecasts would not preserve multivariate curve dependence. This is a design limitation even if the gate passes.

## Data attribution and references

Data: **OMIE (Operador del Mercado Ibérico de Energía)**, public day-ahead files; see [`reports/data_sources.md`](reports/data_sources.md). Raw files are not committed.

Ziel, F. & Steinert, R. (2016). *Electricity Price Forecasting using Sale and Purchase Curves: The X-Model*. Energy Economics.

## Independence

This is a standalone, publicly shareable portfolio project built from public market data and independently developed code. It is not presented as sponsored by or developed for an electricity company.
