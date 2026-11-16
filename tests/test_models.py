import numpy as np

from electricity_forecasting.models.curve_forecast import enforce_nonnegative


def test_nonnegative_constraint_reports_impact():
    constrained, diag = enforce_nonnegative(np.array([2.0, -1.5, 0.0]))
    assert constrained.tolist() == [2.0, 0.0, 0.0]
    assert diag.n_negative_raw == 1
    assert diag.negative_volume_sum == 1.5
