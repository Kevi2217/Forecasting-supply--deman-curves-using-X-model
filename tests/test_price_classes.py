import numpy as np
import pandas as pd
import pytest

from electricity_forecasting.market.price_classes import compress_to_classes, fit_price_classes, reconstruct_class_records


def training():
    prices = np.arange(1.0, 11.0)
    return pd.DataFrame({"side": ["V"] * 10, "price_eur_mwh": prices, "energy_mwh": np.ones(10)})


def test_class_compression_conserves_in_range_volume():
    spec = fit_price_classes(training(), "V", 2)
    volumes, diag = compress_to_classes(training(), spec)
    assert volumes.sum() == pytest.approx(10.0)
    assert diag["underflow_mwh"] == 0
    assert diag["overflow_mwh"] == 0
    reconstructed = reconstruct_class_records(volumes, spec)
    assert (reconstructed["energy_mwh"] >= 0).all()


def test_future_out_of_range_is_reported_not_silently_absorbed():
    spec = fit_price_classes(training(), "V", 2)
    future = pd.DataFrame({"side": ["V", "V"], "price_eur_mwh": [0.0, 20.0], "energy_mwh": [2.0, 3.0]})
    _, diag = compress_to_classes(future, spec)
    assert diag["underflow_mwh"] == pytest.approx(2.0)
    assert diag["overflow_mwh"] == pytest.approx(3.0)
