import pandas as pd
import pytest

from electricity_forecasting.market.curves import build_demand_curve, build_supply_curve


def fixture_records():
    return pd.DataFrame(
        {
            "side": ["V", "V", "V", "C", "C", "C"],
            "price_eur_mwh": [10.0, 20.0, 20.0, 50.0, 40.0, 40.0],
            "energy_mwh": [5.0, 7.0, 3.0, 4.0, 6.0, 2.0],
        }
    )


def test_supply_aggregates_same_price_and_cumulates():
    curve = build_supply_curve(fixture_records())
    assert list(curve["price_eur_mwh"]) == [10.0, 20.0]
    assert list(curve["increment_mwh"]) == [5.0, 10.0]
    assert list(curve["cumulative_mwh"]) == [5.0, 15.0]


def test_demand_aggregates_and_descends():
    curve = build_demand_curve(fixture_records())
    assert list(curve["price_eur_mwh"]) == [50.0, 40.0]
    assert list(curve["increment_mwh"]) == [4.0, 8.0]
    assert list(curve["cumulative_mwh"]) == [4.0, 12.0]


def test_negative_increment_rejected():
    df = fixture_records()
    df.loc[0, "energy_mwh"] = -1.0
    with pytest.raises(ValueError):
        build_supply_curve(df)
