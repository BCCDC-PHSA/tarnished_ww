import numpy as np
import pandas as pd

from tarnished_ww.io import forecast_tests_per_capita
from tarnished_ww.schemas import ColumnSpec


def _testing_frame(start, periods, regions=("A", "B")):
    dates = pd.date_range(start, periods=periods, freq="D")
    rows = []
    for date in dates:
        for region_index, region in enumerate(regions):
            rows.append(
                {
                    "surveillance_date": date,
                    "wwtp": region,
                    "total_tests": 100 + 10 * date.dayofweek + 20 * region_index,
                }
            )
    return pd.DataFrame(rows)


def test_testing_forecast_uses_recent_weekday_history_in_region_order():
    cols = ColumnSpec(region="wwtp")
    train = _testing_frame("2025-01-01", periods=70)
    future = _testing_frame("2025-03-12", periods=21)
    population = np.array([1_000.0, 2_000.0])

    result = forecast_tests_per_capita(
        train,
        future,
        population,
        regions=["A", "B"],
        cols=cols,
    )

    expected_counts = np.array(
        [[100 + 10 * date.dayofweek, 120 + 10 * date.dayofweek]
         for date in pd.date_range("2025-03-12", periods=21, freq="D")]
    )
    np.testing.assert_allclose(result, expected_counts / population[None, :])


def test_testing_forecast_ignores_future_observed_testing_values():
    cols = ColumnSpec(region="wwtp")
    train = _testing_frame("2025-01-01", periods=70)
    future = _testing_frame("2025-03-12", periods=21)

    baseline = forecast_tests_per_capita(
        train,
        future,
        np.array([1_000.0, 2_000.0]),
        regions=["A", "B"],
        cols=cols,
    )
    future["total_tests"] = 10_000_000
    changed = forecast_tests_per_capita(
        train,
        future,
        np.array([1_000.0, 2_000.0]),
        regions=["A", "B"],
        cols=cols,
    )

    np.testing.assert_array_equal(changed, baseline)
