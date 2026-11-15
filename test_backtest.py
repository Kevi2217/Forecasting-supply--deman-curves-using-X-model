import pandas as pd

from electricity_forecasting.evaluation.backtest import chronological_split


def test_chronological_split_no_overlap():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-06-30T12:00Z", "2024-07-01T12:00Z", "2024-09-30T12:00Z", "2024-10-01T12:00Z"],
                utc=True,
            ),
            "x": [1, 2, 3, 4],
        }
    )
    split = chronological_split(
        df,
        timestamp_col="ts",
        train_end="2024-06-30",
        validation_end="2024-09-30",
        test_start="2024-10-01",
    )
    assert split.train["x"].tolist() == [1]
    assert split.validation["x"].tolist() == [2, 3]
    assert split.test["x"].tolist() == [4]
