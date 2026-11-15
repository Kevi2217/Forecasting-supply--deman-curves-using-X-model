import pandas as pd

from electricity_forecasting.data.dataset import select_country_records_for_interval


def _frame(countries):
    rows = []
    for country in countries:
        rows.extend(
            [
                {"country": country, "curve_flag": "O", "side": "V", "energy_mwh": 1.0, "price_eur_mwh": 10.0},
                {"country": country, "curve_flag": "O", "side": "C", "energy_mwh": 1.0, "price_eur_mwh": 20.0},
            ]
        )
    return pd.DataFrame(rows)


def test_country_selection_prefers_spain_when_present():
    country, selected = select_country_records_for_interval(_frame(["MI", "ES"]), preferred_country="ES", fallback_country="MI")
    assert country == "ES"
    assert set(selected["country"]) == {"ES"}


def test_country_selection_uses_mibel_fallback():
    country, selected = select_country_records_for_interval(_frame(["MI"]), preferred_country="ES", fallback_country="MI")
    assert country == "MI"
    assert set(selected["country"]) == {"MI"}
