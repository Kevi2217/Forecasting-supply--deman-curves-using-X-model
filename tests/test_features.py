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


def test_hour_aligned_lag_is_dst_safe_on_25_hour_day():
    # 2024-10-20 (24h) -> 2024-10-27 (25h, 02:00 repeated).
    src = pd.date_range("2024-10-20 00:00", periods=24, freq="h")
    tgt_phys = pd.date_range(
        pd.Timestamp("2024-10-27", tz="Europe/Madrid"),
        pd.Timestamp("2024-10-28", tz="Europe/Madrid"),
        freq="h",
        inclusive="left",
    )
    assert len(tgt_phys) == 25
    a = pd.DataFrame(
        {
            "delivery_date": src.normalize(),
            "period": range(1, 25),
            "hour": src.hour,
            "x": [float(h) for h in src.hour],
        }
    )
    b = pd.DataFrame(
        {
            "delivery_date": tgt_phys.tz_localize(None).normalize(),
            "period": range(1, 26),
            "hour": tgt_phys.hour,
            "x": 0.0,
        }
    )
    out = calendar_aware_lag(pd.concat([a, b]), value_columns=["x"], days=7, align_col="hour")
    day = out.loc[out["delivery_date"] == pd.Timestamp("2024-10-27")]
    assert day["x_lag_7d"].notna().all()  # all 25 physical hours receive a lag
    # Every interval is lagged from the same wall-clock hour a week earlier.
    assert (day["x_lag_7d"].to_numpy() == day["hour"].to_numpy()).all()
    # Period-based lag would misalign (period 25 has no source and period 5 maps to 04:00).
    by_period = calendar_aware_lag(pd.concat([a, b]), value_columns=["x"], days=7)
    pday = by_period.loc[by_period["delivery_date"] == pd.Timestamp("2024-10-27")]
    assert pday["x_lag_7d"].isna().any()


def test_weekly_persistence_hour_alignment_after_spring_dst():
    from electricity_forecasting.models.benchmarks import weekly_persistence

    phys = pd.date_range(
        pd.Timestamp("2024-03-31", tz="Europe/Madrid"), pd.Timestamp("2024-04-01", tz="Europe/Madrid"),
        freq="h", inclusive="left",
    )  # 23 hours, 02:00 missing
    nxt = pd.date_range("2024-04-07 00:00", periods=24, freq="h")
    a = pd.DataFrame({"delivery_date": phys.tz_localize(None).normalize(), "period": range(1, 24),
                      "hour": phys.hour, "price_es_eur_mwh": [float(h) for h in phys.hour]})
    b = pd.DataFrame({"delivery_date": nxt.normalize(), "period": range(1, 25),
                      "hour": nxt.hour, "price_es_eur_mwh": 0.0})
    out = weekly_persistence(pd.concat([a, b]), align_col="hour")
    day = out.loc[out["delivery_date"] == pd.Timestamp("2024-04-07")].set_index("hour")
    assert pd.isna(day.loc[2, "prediction"])  # 02:00 did not exist a week earlier
    assert day.drop(index=2)["prediction"].tolist() == [float(h) for h in day.drop(index=2).index]
