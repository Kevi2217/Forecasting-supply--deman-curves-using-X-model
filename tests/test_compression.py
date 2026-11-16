import pandas as pd

from electricity_forecasting.market.compression import CandidateClassSpecs, reconstruct_pair
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.price_classes import fit_price_classes


def test_reconstructed_class_pair_is_economically_monotone_and_clears():
    records = pd.DataFrame(
        {
            "side": ["V", "V", "V", "C", "C", "C"],
            "price_eur_mwh": [0.0, 20.0, 50.0, 80.0, 40.0, 10.0],
            "energy_mwh": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    s = fit_price_classes(records, "V", 2)
    d = fit_price_classes(records, "C", 2)
    pair, diag = reconstruct_pair(
        records,
        interval_start_utc=pd.Timestamp("2024-01-01T00:00Z"),
        country="MI",
        supply_spec=s,
        demand_spec=d,
    )
    result = approximate_clearing(pair)
    assert pair.supply["cumulative_mwh"].is_monotonic_increasing
    assert result.status.startswith("OK")
    assert sum(diag.values()) == 0.0
