from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from electricity_forecasting.evaluation.backtest import test_month_windows
from electricity_forecasting.models.training import fit_direct_model, fit_independent_target_models


def expanding_monthly_predictions(
    full_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    date_col: str,
    curve_feature_columns: list[str],
    class_target_columns: list[str],
    curve_kind: str,
    curve_params: dict[str, Any],
    direct_feature_columns: list[str],
    direct_target_column: str,
    direct_kind: str,
    direct_params: dict[str, Any],
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_rows: list[pd.DataFrame] = []
    constraint_records: list[dict[str, object]] = []
    full_dates = pd.to_datetime(full_frame[date_col])
    test_dates = pd.to_datetime(test_frame[date_col])

    for month_start, month_end in test_month_windows(test_frame, date_col=date_col):
        train_mask = full_dates < month_start
        month_mask = (test_dates >= month_start) & (test_dates <= month_end)
        train = full_frame.loc[train_mask].dropna(
            subset=curve_feature_columns + direct_feature_columns + class_target_columns + [direct_target_column]
        )
        month = test_frame.loc[month_mask].dropna(subset=curve_feature_columns + direct_feature_columns)
        if train.empty or month.empty:
            continue

        curve_models = fit_independent_target_models(
            train,
            feature_columns=curve_feature_columns,
            target_columns=class_target_columns,
            kind=curve_kind,
            params=curve_params,
            random_seed=random_seed,
        )
        curve_pred, diagnostics = curve_models.predict(month)
        direct_model = fit_direct_model(
            train,
            feature_columns=direct_feature_columns,
            target_column=direct_target_column,
            kind=direct_kind,
            params=direct_params,
            random_seed=random_seed,
        )
        direct_pred = np.asarray(direct_model.predict(month[direct_feature_columns]), dtype=float)

        out = month[["interval_start_utc", date_col, "period", direct_target_column]].copy()
        out = out.join(curve_pred.add_prefix("pred_"))
        out["direct_prediction"] = direct_pred
        out["refit_month"] = month_start.strftime("%Y-%m")
        prediction_rows.append(out)
        for target, diag in diagnostics.items():
            constraint_records.append({"refit_month": month_start.strftime("%Y-%m"), "target": target, **diag})

    if not prediction_rows:
        raise ValueError("No walk-forward predictions produced")
    predictions = pd.concat(prediction_rows).sort_values("interval_start_utc").reset_index(drop=True)
    diagnostics = {
        "constraint_diagnostics": constraint_records,
        "n_refit_months": int(predictions["refit_month"].nunique()),
    }
    return predictions, diagnostics
