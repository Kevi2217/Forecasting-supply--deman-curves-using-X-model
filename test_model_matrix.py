import pandas as pd

from electricity_forecasting.features.model_matrix import build_model_matrix, curve_feature_columns


def test_model_matrix_uses_only_prior_delivery_dates_for_lags():
    ts = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-02T00:00Z", "2024-01-03T00:00Z", "2024-01-08T00:00Z"], utc=True)
    df = pd.DataFrame(
        {
            "interval_start_utc": ts,
            "delivery_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08"]),
            "period": [1, 1, 1, 1],
            "price_es_eur_mwh": [10.0, 11.0, 12.0, 13.0],
            "supply_00": [100.0, 101.0, 102.0, 107.0],
            "demand_00": [90.0, 91.0, 92.0, 97.0],
        }
    )
    out = build_model_matrix(df, class_columns=["supply_00", "demand_00"])
    jan8 = out.loc[out["delivery_date"] == pd.Timestamp("2024-01-08")].iloc[0]
    assert jan8["supply_00_lag_7d"] == 100.0
    assert jan8["price_es_eur_mwh_lag_7d"] == 10.0
    assert "supply_00" not in curve_feature_columns(["supply_00", "demand_00"])
