import pandas as pd

from electricity_forecasting.features.build_features import calendar_aware_lag


def test_calendar_lag_uses_date_and_period_not_row_offset():
    df = pd.DataFrame(
        {
            "delivery_date": pd.to_datetime(["2024-03-24", "2024-03-31"]),
            "period": [5, 5],
            "x": [10.0, 20.0],
        }
    )
    out = calendar_aware_lag(df, value_columns=["x"], days=7)
    assert pd.isna(out.loc[0, "x_lag_7d"])
    assert out.loc[1, "x_lag_7d"] == 10.0


def test_calendar_lag_accepts_python_date_objects():
    from datetime import date

    df = pd.DataFrame({"delivery_date": [date(2024, 1, 1), date(2024, 1, 2)], "period": [1, 1], "x": [3.0, 5.0]})
    out = calendar_aware_lag(df, value_columns=["x"], days=1)
    assert out.loc[1, "x_lag_1d"] == 3.0
