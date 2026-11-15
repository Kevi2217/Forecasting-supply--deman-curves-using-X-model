import numpy as np
import pandas as pd

from electricity_forecasting.data.quality import assess_curve_price_alignment, clearing_validation_summary


def test_mi_price_split_is_reported():
    df = pd.DataFrame(
        {
            "curve_country": ["MI", "MI", "ES"],
            "price_es_eur_mwh": [50.0, 60.0, 70.0],
            "price_pt_eur_mwh": [50.0, 61.0, 75.0],
        }
    )
    report = assess_curve_price_alignment(df)
    assert report["status"] == "PARTIALLY_VERIFIED"
    assert report["mi_intervals_with_es_pt_price_split"] == 1
    assert np.isclose(report["mi_split_share"], 0.5)


def test_es_only_alignment_is_verified_even_if_pt_differs():
    df = pd.DataFrame(
        {
            "curve_country": ["ES", "ES"],
            "price_es_eur_mwh": [50.0, 60.0],
            "price_pt_eur_mwh": [55.0, 61.0],
        }
    )
    report = assess_curve_price_alignment(df)
    assert report["status"] == "VERIFIED"
    assert report["mi_intervals"] == 0


def test_clearing_summary_keeps_invalids_separate():
    df = pd.DataFrame(
        {
            "price_es_eur_mwh": [50.0, 60.0, 70.0],
            "observed_curve_clearing_eur_mwh": [51.0, np.nan, 68.0],
            "observed_curve_clearing_status": ["unique", "no_crossing", "unique"],
        }
    )
    report = clearing_validation_summary(df)
    assert report["n_matched"] == 2
    assert report["invalid_count"] == 1
    assert np.isclose(report["mae"], 1.5)
