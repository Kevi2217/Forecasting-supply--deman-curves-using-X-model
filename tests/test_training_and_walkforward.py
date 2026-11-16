import numpy as np
import pandas as pd

from electricity_forecasting.evaluation.backtest import chronological_date_split
from electricity_forecasting.evaluation.walk_forward import expanding_monthly_predictions
from electricity_forecasting.features.model_matrix import build_model_matrix, curve_feature_columns, direct_price_feature_columns
from electricity_forecasting.models.training import select_curve_model, select_direct_model


def _synthetic_daily_frame():
    dates = pd.date_range("2024-01-01", "2024-05-31", freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 40.0 + 0.05 * i + 3.0 * np.sin(i / 7)
        rows.append(
            {
                "interval_start_utc": pd.Timestamp(d, tz="UTC"),
                "delivery_date": d,
                "period": 1,
                "price_es_eur_mwh": price,
                "supply_00": 100.0 + 0.2 * i,
                "demand_00": 120.0 + 0.15 * i,
            }
        )
    return pd.DataFrame(rows)


def test_model_selection_and_expanding_walk_forward_run_without_future_features():
    base = _synthetic_daily_frame()
    matrix = build_model_matrix(base, class_columns=["supply_00", "demand_00"]).dropna()
    split = chronological_date_split(
        matrix,
        date_col="delivery_date",
        train_end="2024-03-15",
        validation_end="2024-04-15",
        test_start="2024-04-16",
    )
    c_features = curve_feature_columns(["supply_00", "demand_00"])
    d_features = direct_price_feature_columns()
    curve_sel = select_curve_model(
        split.train,
        split.validation,
        feature_columns=c_features,
        target_columns=["supply_00", "demand_00"],
        candidates=[("ridge", {"alpha": 1.0})],
    )
    direct_sel = select_direct_model(
        split.train,
        split.validation,
        feature_columns=d_features,
        target_column="price_es_eur_mwh",
        candidates=[("ridge", {"alpha": 1.0})],
    )
    pred, diag = expanding_monthly_predictions(
        matrix,
        split.test,
        date_col="delivery_date",
        curve_feature_columns=c_features,
        class_target_columns=["supply_00", "demand_00"],
        curve_kind=curve_sel.kind,
        curve_params=curve_sel.params,
        direct_feature_columns=d_features,
        direct_target_column="price_es_eur_mwh",
        direct_kind=direct_sel.kind,
        direct_params=direct_sel.params,
    )
    assert len(pred) == len(split.test)
    assert pred["pred_supply_00"].ge(0).all()
    assert pred["pred_demand_00"].ge(0).all()
    assert diag["n_refit_months"] >= 1
