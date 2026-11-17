# Pre-registration: exploratory curve models and the adjusted curve-based extension

Written 2026-11-16, **before** any curve-based forecast, validation score or 2025 evaluation result was computed. The protocol below is frozen: no element of it may be changed after the confirmation period is evaluated. Configuration: `configs/extension.yaml`. Runner: `scripts/run_curve_extension.py`.

## Status of the production pipeline

`scripts/run_project.py` with `configs/default.yaml` is unchanged. Its observed-curve clearing gate (MAE ≤ 10 EUR/MWh) still fails (MAE 26.14) and still stops it. Everything below runs in a separate execution path. Every result from that path is labelled either *exploratory* or *confirmation*.

## Periods

| Role | Dates | Use |
|---|---|---|
| Training | 2023-01-01 – 2024-06-30 | Price classes, model fitting (first 7 days lost to lags) |
| Validation | 2024-07-01 – 2024-09-30 | Price-class count, hyperparameters, adjustment specification |
| Exploratory evaluation | 2024-10-01 – 2024-12-31 | Already examined for direct benchmarks, so **not** an untouched test |
| Confirmation | 2025-01-01 – 2025-09-30 | Evaluated once with the frozen protocol. This is the last hourly period before the 15-minute market time unit went live on 2025-10-01. |

The evaluation periods use monthly expanding-window refits: each month is predicted by models refitted on all data dated before that month. Hyperparameters, price-class boundaries and the adjustment *specification* stay fixed. Only parameters are re-estimated.

## A. Unadjusted curve-based model (exploratory)

This is the existing production code path, phases 3–6, unchanged.

- Training-only energy-weighted quantile price classes, k ∈ {10, 20, 40}, chosen by validation compression MAE.
- Class-volume Elastic Net and HGB models, with hyperparameters chosen by validation scaled multi-target MAE.
- Implied price from the naive stepwise crossing of the forecast curves.

It is labelled **EXPLORATORY**, because its clearing prerequisite failed.

## B. Adjusted curve-based model (pre-registered extension)

P_adj,t = P_curve,t + ĝ(X_t), where P_curve,t is the implied price of the forecast curves from model A.

**Training target.** d_t = P_official,t − P_comp,t. Here P_comp,t is the clearing price of the price-class-compressed **observed offered** curve, which is the object model A forecasts. The target is computed only for past delivery days.

**Predictors.** All are available before the 12:00 D−1 gate closure for delivery day D.

| Predictor | Training rows | Prediction rows |
|---|---|---|
| `curve_price` | P_comp (observed, compressed) | P_curve (forecast curves) |
| `curve_supply_total`, `curve_demand_total` | Observed class-volume sums | Forecast class-volume sums |
| d lagged 1, 2, 7 days (same local hour) | Past days only | Past days only |
| Offered-but-unmatched supply ≤ p\*, lagged 1, 2, 7 days | Past matched curves | Past matched curves |
| Matched demand not in the offered curve, lagged 1, 2, 7 days | Past matched curves | Past matched curves |
| Hour-of-day and weekday indicators | Calendar | Calendar |

The day-D−1 curve and price files are issued around 12:55 on D−2, so 1-day lags are public before the D−1 auction.

**Never used as predictors:**
- same-day matched curves, rejected volumes or offer-typology volumes;
- same-day official prices;
- realised coupling flows.

Mapping training rows from observed to forecast curve state is a deliberate errors-in-variables choice: ĝ is a calibration from the public-curve price space to the official-price space.

**Correction candidates**, fitted on training and scored on validation by the MAE of P_curve,val + ĝ against the official price, where P_curve,val comes from the curve models fitted on training only:

1. **Elastic Net** (primary, regularised). StandardScaler inside the pipeline. α ∈ {0.01, 0.1, 1, 10}, l1_ratio ∈ {0.1, 0.5, 0.9}.
2. **HGB.** learning_rate ∈ {0.05, 0.1}, max_leaf_nodes ∈ {15, 31}, 150 iterations, L2 0.1. Adopted only if its best validation MAE is at least **5 % lower** than the best Elastic Net validation MAE.
3. **References only, never selectable:**
   - no correction;
   - the training-mean constant d;
   - the trailing 28-day same-hour median of d.

   These show whether a constant shift would be stable.

Selection is made separately for the Elastic Net curve family and the HGB curve family. If the chosen correction does not beat "no correction" on validation, the adjusted model is still reported, together with that fact.

## C. Direct models and persistence

These follow the existing protocol and features (calendar plus official-price lags of 1, 2 and 7 days), re-run within the same frame so that all seven models share forecast origins and evaluation intervals.

## Evaluation

- **Common mask.** All seven models are scored on the identical set of finite predictions in each period. Exclusions are counted and reported.
- **Metrics.** MAE, RMSE, bias = mean(prediction − actual), N, and MAE improvement relative to weekly persistence.
- **Confirmation claim.** The adjustment counts as confirmed only if adjusted beats unadjusted on the 2025 confirmation period for the same curve family. A claim that the curve approach adds value requires an adjusted curve model to beat the corresponding direct model on 2025.
- **Error decomposition.** Signed and absolute statistics are reported; components are not assumed to be additive in MAE or RMSE.
  1. Clearing approximation: P_obs − P_official.
  2. Compression: P_comp − P_obs.
  3. Curve forecasting: P_curve − P_comp.
  4. Adjustment estimation: ĝ − d.

  Signed identities hold exactly: unadjusted error = 1 + 2 + 3, and adjusted error = 3 + 4.

## Data-quality gates (fail closed)

- Full coverage of curves and prices for 2023-01-01 – 2025-09-30.
- Schema and DST validation of every file.
- No interval excluded silently.

## Amendment 1 (2026-11-16, user-directed, before any evaluation-period prediction)

**Timing.** The first full run was stopped during hyperparameter selection. By then it had produced data gates, k = 40 and the interval dataset, but **no validation-period curve forecasts and no evaluation-period predictions**. The changes below were requested by the project owner and fixed before any such output existed.

1. **Out-of-fold curve forecasts.** Curve models (frozen hyperparameters) produce chronological out-of-fold forecasts with monthly expanding refits for every month from **2023-07** to 2025-09. Each month uses only data dated before it; 2023-01 – 2023-06 is a burn-in.
   - Validation-period curve forecasts for adjustment selection now come from this walk-forward, not from a single train-only fit.
   - Caveat: the curve hyperparameters were selected on the validation quarter, so the training-period out-of-fold forecasts use hyperparameters informed by validation data. No evaluation-period data is involved.
2. **Primary adjustment (OOF-trained).** The target is r_t = P_official,t − P_curve,t^OOF, the realised error of the out-of-fold curve forecast. The predictors are as specified above, with curve state taken from the out-of-fold forecast curves in both training and prediction.
   - Selection on validation: ĝ is fitted on out-of-fold rows from 2023-07 – 2024-06 and scored on the walk-forward validation forecasts.
   - Elastic Net is primary; HGB is adopted only with a ≥ 5 % validation improvement.
   - In evaluation, ĝ is refitted monthly on all out-of-fold rows dated before the month.
3. **Secondary adjustment (originally pre-registered).** The observed-curve-trained correction is kept unchanged as a sensitivity analysis. It is reported, not headlined.
4. **Bias baseline.** A constant correction equal to the mean out-of-fold residual over 2023-07 – 2024-06 (training only, never updated), reported for each curve family.
5. **Unchanged:** periods, data gates, price classes (k = 40, selected on validation compression error), model grids, metrics, the common evaluation mask, the decomposition identities, and the production clearing gate (still failed, still enforced).
