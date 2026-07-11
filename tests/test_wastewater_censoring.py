import numpy as np
import pandas as pd

from tarnished_ww.io import getting_cases_and_ww_logged, standardize_input
from tarnished_ww.schemas import ColumnSpec


def test_getting_cases_and_ww_logged_reads_missing_and_left_censored_flags():
    cols = ColumnSpec(region="wwtp")
    df = pd.DataFrame(
        {
            "surveillance_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "wwtp": ["A", "B", "A", "B"],
            "total_ed_visits": [0, 0, 0, 0],
            "total_tests": [10, 10, 10, 10],
            "total_cases_rsv": [1, 2, 3, 4],
            "load_trillion_rsv": [9.0, 4.0, 99.0, 2.0],
            "wastewater_missing_rsv": [False, False, True, False],
            "is_left_censored_rsv": [False, True, False, False],
        }
    )

    standardize_input(df, ["rsv"], cols)
    y_cases, log_y, left_censored, pivot = getting_cases_and_ww_logged(
        df,
        "rsv",
        cols,
        return_censoring=True,
    )

    assert pivot["load_trillion_rsv"].columns.tolist() == ["A", "B"]
    np.testing.assert_array_equal(y_cases, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.ma.getmaskarray(log_y).tolist() == [[False, False], [True, False]]
    assert np.isclose(log_y[0, 0], np.log(10.0))
    assert np.isclose(log_y[0, 1], np.log(5.0))
    assert left_censored.tolist() == [[False, True], [False, False]]
