import pandas as pd

from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve


def pair(records):
    df = pd.DataFrame(records, columns=["side", "price_eur_mwh", "energy_mwh"])
    return CurvePair(pd.Timestamp("2024-01-01T00:00:00Z"), "MI", build_supply_curve(df), build_demand_curve(df))


def test_unique_step_crossing():
    p = pair([
        ("V", 10.0, 5.0),
        ("V", 20.0, 10.0),
        ("C", 30.0, 8.0),
        ("C", 15.0, 3.0),
    ])
    result = approximate_clearing(p)
    assert result.status.startswith("OK")
    assert result.estimated_price_eur_mwh == 20.0


def test_no_crossing():
    p = pair([
        ("V", 10.0, 1.0),
        ("C", 30.0, 100.0),
    ])
    result = approximate_clearing(p)
    assert result.status == "NO_CROSSING"
    assert result.estimated_price_eur_mwh is None
