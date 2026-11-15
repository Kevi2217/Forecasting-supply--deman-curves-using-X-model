# Day-Ahead Electricity Price Forecasting from Auction Curves

An independent quantitative research and machine-learning portfolio project inspired by the X-Model literature. The project asks whether forecasting the *structure* of a day-ahead electricity auction can produce useful Spanish electricity-price forecasts compared with forecasting the price directly.

## Status

**Software implementation: TESTED. Empirical project: BLOCKED at the real-data materialisation gate in the current build environment.**

The complete MVP pipeline has been implemented: OMIE download/provenance, raw-data validation, timezone/DST handling, observed-curve clearing validation, price-class compression, Elastic Net and nonlinear Histogram Gradient Boosting class-volume forecasts, direct-price benchmarks, expanding-window backtesting, error decomposition, figures, and technical audit.

The official OMIE archive and file schemas have been verified, but this build environment cannot retrieve OMIE's binary `.1` files because its local Python/container runtime has no external DNS access. The code therefore **does not report fabricated empirical metrics or figures**. Running the repository in a normal internet-connected Python environment is required before the empirical status can become VERIFIED.

See [`reports/execution_status.md`](reports/execution_status.md), [`reports/data_sources.md`](reports/data_sources.md), and [`reports/technical_audit.md`](reports/technical_audit.md).

## Research question

> Can forecasting the underlying day-ahead electricity auction supply and demand curves produce more accurate Spanish day-ahead electricity-price predictions than conventional direct price forecasting models?

The economic motivation is that the market price is an outcome of the interaction between supply and demand curves. A curve model may therefore preserve information about market elasticity and nonlinear price sensitivity that a direct price model does not explicitly represent.

## Relationship to the X-Model

The original X-Model models auction-curve information, reduces its dimensionality into price classes, forecasts class-level volumes, reconstructs supply and demand curves, and derives an implied clearing price. This repository follows that structural idea but is deliberately described as **X-Model-inspired**, not an exact replication.

The main methodological change is the estimator. Instead of reproducing the original LASSO specification, this project evaluates:

- **Elastic Net** as a transparent regularised class-volume model.
- **Histogram Gradient Boosting** as the nonlinear challenger.

The nonlinear model is included to test whether tree-based boosting captures interactions and nonlinearities in auction-volume dynamics that a linear regularised model may miss.

## Mandatory datasets

Only official public OMIE market data are used for the empirical MVP:

- **Aggregate supply and demand curves:** `curva_pbc_YYYYMMDD.1`
- **Day-ahead marginal prices:** `marginalpdbc_YYYYMMDD.1`
- **Target period:** 2023-01-01 through 2024-12-31
- **Primary price target:** Spanish day-ahead marginal price

The project prefers an `ES` curve representation when published for an interval and otherwise uses `MI` only as a documented fallback. An `MI` curve is not assumed to reproduce a Spanish zonal price when the Spanish and Portuguese marginal prices differ; such intervals are explicitly identified and excluded from the supported curve-to-Spanish-price modelling sample.

Raw source files are intentionally not committed. Download provenance, retrieval timestamps and checksums are generated locally.

## End-to-end workflow

```text
Official OMIE raw files
        ↓
Coverage + schema + data-quality gates
        ↓
Timezone-aware physical delivery intervals
        ↓
Observed cumulative supply/demand curves
        ↓
Approximate stepwise market clearing
        ↓
Validation against official Spanish marginal price
        ↓
Training-only price-class construction
        ↓
Compression/reconstruction validation
        ↓
Lagged, leakage-safe class-volume features
        ↓
Elastic Net + Histogram Gradient Boosting
        ↓
Forecasted class volumes
        ↓
Reconstructed forecast curves
        ↓
Implied clearing price
        ↓
Weekly persistence + direct-price benchmarks
        ↓
Monthly expanding-window out-of-sample evaluation
```

## Market and data conventions

The parser follows the documented OMIE public-file representation for the target period. Purchase records (`C`) and sale records (`V`) contain incremental MWh quantities at bid prices in EUR/MWh. Same-price quantities are aggregated before cumulative curves are constructed.

Supply is accumulated from low to high price. Demand is accumulated from high to low price. Local delivery intervals are interpreted in `Europe/Madrid`; UTC timestamps are used internally so the 23-hour and 25-hour daylight-saving transition days remain unambiguous.

## Approximate clearing

The implemented clearing function intentionally does **not** claim to reproduce EUPHEMIA. It evaluates the observed step curves on the union of bid-price levels and returns the first price at which cumulative supply meets or exceeds cumulative demand, with explicit diagnostics for no crossing, ambiguity and domain boundaries.

Before any ML model is allowed to run, this approximation must pass a pre-specified project feasibility gate against the official Spanish price. The current defaults are stored in `configs/default.yaml`; they are project acceptance thresholds, not market rules.

Errors are kept separate:

1. **Clearing approximation error:** official price vs. clearing from original observed curves.
2. **Compression error:** original-curve clearing vs. compressed/reconstructed-curve clearing.
3. **Forecast error:** compressed observed clearing vs. clearing from forecasted class volumes.

## Price-class representation

Candidate class counts are 10, 20 and 40 per market side. Boundaries are fitted **only on the initial training period** using energy-weighted price quantiles. Candidate counts that cannot form distinct bins because of price clustering are rejected and documented rather than silently modified.

The final class count is selected on chronological validation data by balancing clearing-price reconstruction error and model complexity. Validation and test observations always reuse the training-derived boundaries. Volume outside those boundaries is measured explicitly.

## Forecasting models

### Curve models

One model is fitted per supply/demand class for the MVP. Candidate inputs include calendar variables and genuinely historical class-volume and price lags. No contemporaneous target-class volume is used as a predictor.

The two curve families are:

- Elastic Net
- `HistGradientBoostingRegressor`

Hyperparameters are selected using the validation period only. Within each family, class-volume tuning uses mean scale-adjusted MAE; the final choice between the linear and nonlinear curve families uses the **validation implied-price MAE** after reconstruction. The final test period is not used for model-family selection.

### Price benchmarks

- Calendar-aware **weekly persistence**.
- Direct **Elastic Net** price forecast.
- Direct **Histogram Gradient Boosting** price forecast.

All headline comparisons use exactly the same finite test observations.

## Backtesting

The default chronological split is:

- Training: 2023-01-01 to 2024-06-30
- Validation: 2024-07-01 to 2024-09-30
- Test: 2024-10-01 to 2024-12-31

The final test is evaluated with monthly expanding-window refits. Hyperparameters and price-class definitions remain fixed after validation; only newly realised historical observations become available to later test-month refits.

Reported metrics, when the empirical run is possible, are MAE, RMSE, mean bias, median absolute error and observation count, plus diagnostics for negative prices, high prices, large price changes and hour of day.

## Repository structure

```text
electricity-curve-forecasting/
├── README.md
├── pyproject.toml
├── configs/default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/electricity_forecasting/
│   ├── data/
│   ├── market/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
├── scripts/
├── tests/
└── reports/
```

## Installation

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Reproduce the project

### 1. Download the official 2023–2024 OMIE files

```bash
python scripts/download_omie.py --start 2023-01-01 --end 2024-12-31 --dataset both
```

### 2. Audit local coverage

```bash
python scripts/audit_raw_coverage.py --start 2023-01-01 --end 2024-12-31
```

### 3. Inspect a representative source file

```bash
python scripts/inspect_data.py \
  --curve-file data/raw/curves/curva_pbc_20240101.1 \
  --price-file data/raw/prices/marginalpdbc_20240101.1
```

### 4. Run tests

```bash
pytest -q
```

Current build checkpoint: **29 passed, 0 failed, 0 skipped**.

### 5. Execute all empirical phases

```bash
python scripts/run_project.py
```

The runner is intentionally fail-closed. It stops if raw coverage is incomplete, a source file fails validation, the observed-curve clearing approximation fails its feasibility gate, or no valid test comparison can be produced.

A successful empirical run writes processed data, model-selection records, predictions, metrics, error decomposition, execution summary, `reports/results.md`, and all five required empirical figures.

## Required figures

On a successful real-data run the pipeline generates:

1. Observed supply/demand curves with approximate and official price markers.
2. Original versus price-class reconstructed curves.
3. Forecasted versus observed curves for a held-out interval.
4. Out-of-sample official and model price forecasts.
5. MAE/RMSE model comparison.

No synthetic chart is substituted for a missing empirical chart.

## Testing and safeguards

The test suite covers parser behaviour, locale-formatted numbers, 23/24/25-hour days, physical interval identity, curve accumulation, multiple bids at identical prices, clearing edge cases, price classes, volume conservation, leakage-safe calendar lags, chronological splits, model constraints, model selection, walk-forward fitting, country-selection rules, geography diagnostics and pipeline integration.

The project additionally enforces:

- no random train/test split;
- no fitting scalers/classes/models on future data;
- no blind 168-row lag across DST changes;
- no silent interpolation of the market clearing price;
- no silent clipping of out-of-range class input volumes;
- explicit diagnostics for non-negative forecast constraints;
- separate structural and forecasting error components;
- identical evaluation observations across benchmark models;
- no empirical claim unless real OMIE curves have actually been processed.

## Current empirical result

**BLOCKED in this build environment.** The official source pages and documentation are accessible, but OMIE serves raw data as binary downloads and the local Python/container runtime has no external DNS/network access. Consequently the repository does not contain real 2023–2024 raw files here and the empirical stages have not been labelled VERIFIED.

This is an execution-environment limitation, not a substitution of the research design. See [`reports/results.md`](reports/results.md).

## Limitations

Even after a successful run, the project has important limitations:

- The clearing function is only an approximation to coupled European market clearing.
- Complex/block orders and network constraints are not explicitly solved.
- Price-class compression intentionally loses local bid detail.
- Independent class forecasts do not preserve the full multivariate dependence structure.
- `MI` curves are not treated as exact Spanish zonal curves during ES/PT price separation.
- Optional historical forecast vintages for load/wind/solar are excluded from the MVP unless their publication timing can be verified.
- The experiment covers one market and a limited historical period.
- Forecasting superiority must be demonstrated empirically; it is not assumed from the model design.

## Future improvements

After the MVP is empirically validated, useful extensions include exogenous day-ahead load/wind/solar forecast vintages, multivariate class forecasting, probabilistic curve forecasts, time-varying class definitions tested without leakage, richer market-coupling information and alternative monotone curve representations.

## Data attribution and references

Primary empirical data provider: **OMIE (Operador del Mercado Ibérico de Energía)**. Source URLs, dataset names, schemas, provenance policy and reuse notes are documented in [`reports/data_sources.md`](reports/data_sources.md).

Methodological inspiration: Ziel, F. & Steinert, R. (2016), *Electricity Price Forecasting using Sale and Purchase Curves: The X-Model*, Energy Economics.

## Independence

This is a standalone, publicly shareable portfolio project built from public market data and independently developed code. It is not presented as sponsored by or developed for an electricity company.
