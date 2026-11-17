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

## Execution outcome of the production pipeline (2026-11-16)

The Phase-2 gate failed on real data (MAE 26.14 EUR/MWh vs a limit of 10), so the production runner did not execute Phases 3–4 or the curve parts of Phases 5–6. It still does not. These phases were executed only in the separate exploratory extension described below. The three curve-independent benchmarks were evaluated with the Phase-5 protocol by `scripts/run_direct_benchmarks.py`. That protocol uses identical features, validation-only hyperparameter selection, monthly expanding refits and a common observation mask. See `results.md`.

## Extension (2026-11-16): exploratory curve models and an adjusted X-Model-inspired variant

The production pipeline above is unchanged: `run_project.py` with `default.yaml` still stops at the failed clearing gate. The extension is a **separate execution path**, `scripts/run_curve_extension.py` with `configs/extension.yaml`. Its protocol was frozen in `preregistration_curve_extension.md` before any curve-model output existed.

### Why the clearing approximation fails

`clearing_discrepancy_audit.md` covers the full audit. In short:
- The stepwise crossing reproduces the official price on OMIE's **matched** curves (MAE 0.43).
- On the **offered** curves it is biased low (MAE 24.6, 2023-01 – 2025-09) because cheap offers carrying complex conditions are selectively rejected.
- From 2025-03-19 the published offer typology shows that every unmatched cheap MWh is a complex-condition offer. Removing *all* complex offers overshoots, and adding realised coupling flows does not help.
- The discrepancy varies strongly over time (monthly MAE 8.5 to 51) and falls sharply from April 2025.

### Why an extension

A pre-auction forecaster cannot observe which complex offers will be rejected. The discrepancy is, however, persistent from day to day and strongly associated with lagged rejected volume. It can therefore be forecast as a *statistical* correction rather than modelled mechanically. This departs from the X-Model, which assumes the curves determine the price. The extension is described as **X-Model-inspired** and is not a reproduction of the original paper.

### Model A: unadjusted curve-based (EXPLORATORY)

This uses production phases 3–6 unchanged:
- training-only price classes, with k selected on validation compression error;
- independent class-volume Elastic Net / HGB models, with hyperparameters selected on validation;
- implied price from the naive crossing of the forecast curves.

It is exploratory because its prerequisite (the clearing gate) failed.

### Model B: adjusted curve-based (primary correction per Amendment 1)

P_adj,t = P_curve,t + ĝ(X_t).

1. **Out-of-fold curve forecasts.** The curve models, with hyperparameters frozen after validation, are refitted monthly on an expanding window. They forecast every month from 2023-07 to 2025-09; 2023-01 – 06 is a burn-in. Each month uses only data dated before it.
2. **Target.** r_t = P_official,t − P_curve,t^OOF, the realised error of the out-of-fold curve forecast. ĝ therefore learns from realistic forecasting errors, not only from observed-curve reconstruction errors.
3. **Fitting.** ĝ is fitted on out-of-fold rows dated before the forecast month and refitted monthly with a frozen specification.
4. **Selection** (validation 2024-Q3). Fit on out-of-fold rows from 2023-07 – 2024-06 and score MAE(P_official, P_curve + ĝ) on the walk-forward validation forecasts. Elastic Net (α ∈ {0.01, 0.1, 1, 10}, l1_ratio ∈ {0.1, 0.5, 0.9}) is primary. HGB is adopted only if it lowers validation MAE by at least 5 %; it did not, for either family.

| Predictor (all known before the D−1 12:00 gate closure) | Source |
|---|---|
| Curve-implied price, total supply volume, total demand volume | The **forecast** curves, in both training and prediction rows |
| d = P_official − P_comp, lagged 1, 2, 7 days (same local hour) | Official prices and curves of past days |
| Unaccepted cheap supply, lagged 1, 2, 7 days | Matched curves of past days |
| Matched demand absent from offered demand, lagged 1, 2, 7 days | Matched curves of past days |
| Hour-of-day and weekday indicators | Calendar |

The day-D−1 curve and price files are issued around 12:55 on D−2, so 1-day lags are public before the D−1 auction.

**Never used as predictors:** same-day matched curves, unaccepted volumes, offer typology, official prices, or realised coupling flows. Unit tests check that no post-auction column enters the feature list unlagged.

**Reference variants** (reported, not headlined):
- *constant training bias*: the mean out-of-fold residual over 2023-07 – 2024-06, never updated;
- *observed-curve-trained ĝ*: the originally pre-registered target d_t = P_official − P_comp on observed curves, with observed curve state in training rows;
- trailing 28-day same-hour median (validation only).

### Model C: direct models and persistence

These are re-run in the same frame on identical evaluation intervals.

### Caching and reproducibility

`run_curve_extension.py` caches the interval dataset, the price-class specification, the hyperparameter selection and each family-month walk-forward refit. The refit cache is keyed by a hash of the configuration, classes and selected hyperparameters; a mismatching cache is refused unless `--rebuild` is passed.

### Periods and evaluation

| Period | Dates | Use |
|---|---|---|
| Training | 2023-01 – 2024-06 | Fitting |
| Validation | 2024-07 – 2024-09 | All selection |
| Exploratory evaluation | 2024-Q4 | Already examined for the direct models |
| Confirmation | 2025-01 – 2025-09 | Evaluated once, frozen protocol, monthly expanding refits |

Metrics are MAE, RMSE, bias, N and MAE improvement over weekly persistence, all on a common finite mask.

**Error decomposition.** Each component is reported with its signed mean, MAE and RMSE. Signed errors add exactly; MAE and RMSE do not.
1. Clearing approximation: P_obs − P_official.
2. Compression: P_comp − P_obs.
3. Curve forecasting: P_curve − P_comp.
4. Adjustment estimation: ĝ − d.

The identities are: unadjusted error = 1 + 2 + 3, and adjusted error = 3 + 4.

Supplementary inference (not pre-registered): Diebold–Mariano tests on absolute error. Hourly differentials are averaged per delivery day; Newey–West variance with 7 lags and the Harvey–Leybourne–Newbold correction (`scripts/extension_significance.py`).
