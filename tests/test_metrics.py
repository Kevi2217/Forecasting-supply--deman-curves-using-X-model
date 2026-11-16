import numpy as np
import pytest

from electricity_forecasting.evaluation.metrics import compute_metrics


def test_metrics_known_values():
    result = compute_metrics(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert result.n == 2
    assert result.mae == pytest.approx(0.5)
    assert result.rmse == pytest.approx(np.sqrt(0.5))
    assert result.bias == pytest.approx(0.5)
