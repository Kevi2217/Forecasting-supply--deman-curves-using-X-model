# Methodology

## Research design

The project tests whether forecasting a lower-dimensional representation of day-ahead auction supply and demand curves can improve Spanish day-ahead price prediction relative to direct price models. It is X-Model-inspired rather than an exact replication.

## Phase 1 — source and raw-data gate

The empirical sample is fixed to 2023-01-01 through 2024-12-31. Official OMIE daily aggregate-curve files and official marginal-price files are mandatory. OMIE serves only the latest revision of a file (e.g. `.2`); the downloader probes revisions `.1`–`.5` and the pipeline uses the highest local revision. Local coverage must be complete before the main runner proceeds. Every raw file is then parsed and validated; a single failed mandatory source file causes the empirical run to stop.

Physical delivery intervals use `Europe/Madrid` local market time and are mapped to timezone-aware UTC identifiers. This preserves distinct physical intervals on 23-hour and 25-hour DST dates.

## Phase 2 — market representation and clearing gate

For each interval, the project prefers published Spanish (`ES`) offered curves and falls back to MIBEL-wide (`MI`) offered curves only when `ES` is absent. If an `MI` curve is used while the official Spanish and Portuguese marginal prices differ, that interval is not considered a supported curve-to-Spanish-price observation.

Sale (`V`) MWh increments are aggregated at equal price and accumulated from low to high price. Purchase (`C`) increments are aggregated at equal price and accumulated from high to low price.

Approximate clearing uses the union of observed price levels and the discrete step functions. The returned estimate is the first observed price at which cumulative supply meets or exceeds cumulative demand. The function also returns no-crossing/ambiguity/boundary diagnostics. Linear interpolation is intentionally not introduced without evidence that it improves fidelity for this market representation.

Before compression or machine learning, the original observed curves are compared with the official Spanish marginal price. MAE, RMSE, bias, median AE, p95 absolute error and invalid-crossing share are recorded. The default feasibility thresholds in `configs/default.yaml` are pre-specified project acceptance gates, not statements about official market rules.

## Phase 3 — dimensionality reduction

Price classes are trained only on the initial training period. Candidate counts are 10, 20 and 40 per side. Energy-weighted price quantiles define candidate boundaries. A requested candidate that cannot form distinct bins because of price clustering is rejected and logged.

For validation observations, original curves are compressed into class volumes and reconstructed at training-derived representative prices. The project reports original-vs-compressed clearing error, invalid clearing counts and unrepresented out-of-range volume. The smallest class count within the configured MAE tolerance of the best candidate is selected.

## Phase 4 — forecasting

The target is the next delivery interval's class-volume vector. The MVP uses independent models per class and documents that this does not preserve the full multivariate dependence structure.

Features include local calendar variables and calendar-aligned historical class-volume/price lags. Lags are joined by local delivery date and **local wall-clock hour**, not by fixed row offsets or OMIE period numbers. Period numbers shift by one hour on 23/25-hour DST days. On a 25-hour source day the first occurrence of the repeated hour is used. Weekly persistence uses the same alignment on the complete official price series. All model inputs must be available before the relevant auction result is known.

Two curve families are evaluated:

1. Elastic Net.
2. Histogram Gradient Boosting.

Within each family, hyperparameters are selected on validation data using mean class-level MAE scaled by typical target magnitude. After each family's class hyperparameters are fixed, its reconstructed validation price is calculated. The final curve family is selected by the lower **validation implied-price MAE**, which aligns model-family selection with the research objective without using the test set.

Negative raw class-volume predictions are constrained to zero and the number/share/sum of violations is reported. The intervention is never hidden.

## Phase 5 — benchmarks and test evaluation

Benchmarks are calendar-aware weekly persistence, direct Elastic Net and direct Histogram Gradient Boosting. All models predict the same official Spanish price and headline metrics use exactly the same finite test intervals.

The test period is evaluated with monthly expanding-window refits. Hyperparameters, class definitions and feature specification remain fixed after validation. Later test months may use already realised observations from earlier test months as historical training information, which is valid walk-forward forecasting.

Metrics are MAE, RMSE, mean bias, median AE and `n`. Additional retrospective diagnostics cover negative prices, training-defined high prices, training-defined large price changes and hour of day.

## Phase 6 — error decomposition and outputs

Three mechanisms are evaluated separately:

- A: official price vs original observed-curve approximate clearing;
- B: original observed-curve clearing vs compressed/reconstructed observed-curve clearing;
- C: compressed observed-curve clearing vs forecast-curve clearing.

The total official-price-to-forecast error is reported separately. MAEs/RMSEs are not added together as if they formed an algebraic decomposition.

Five required figures are generated only from real pipeline outputs. If the empirical run is blocked, no synthetic chart is passed off as an empirical figure.

## Execution outcome (2026-11-16)

The Phase-2 gate failed on real data (MAE 26.14 EUR/MWh vs a limit of 10), so Phases 3–4 and the curve parts of Phases 5–6 were not executed. The three curve-independent benchmarks were evaluated with the Phase-5 protocol by `scripts/run_direct_benchmarks.py`. That protocol uses identical features, validation-only hyperparameter selection, monthly expanding refits and a common observation mask. See `results.md`.
