import pandas as pd
import numpy as np
from .schemas import ColumnSpec, PopulationSpec

def _coerce_bool_flag(values):
    flag = pd.Series(values).fillna(False)
    if flag.dtype == bool:
        return flag.to_numpy(dtype=bool)
    if np.issubdtype(flag.dtype, np.number):
        return flag.to_numpy(dtype=float) != 0
    return flag.astype(str).str.strip().str.lower().isin(
        {"true", "t", "yes", "y", "1"}
    ).to_numpy(dtype=bool)

def standardize_input(df, diseases, cols: ColumnSpec):
    df = df.copy()
    df[cols.date] = pd.to_datetime(df[cols.date])

    # rename region -> internal name expected by build_functions
    if cols.region != cols.region_internal:
        df = df.rename(columns={cols.region: cols.region_internal})

    # required columns (dynamic)
    required = {cols.date, cols.region_internal, cols.ed_visits, cols.tests}
    for d in diseases:
        required.add(cols.cases_tpl.format(disease=d))
        required.add(cols.wwload_tpl.format(disease=d))

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input dataframe is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nTip: pass a ColumnSpec with your column names if they differ."
        )
    return df

def validate_population(pop_df, wwtps, cols: PopulationSpec):
    missing = {cols.region, cols.population} - set(pop_df.columns)
    if missing:
        raise ValueError(f"Population table missing columns: {missing}")
    if cols.region != cols.region_internal:
        pop_df = pop_df.rename(columns={cols.region: cols.region_internal})
    pop_df = pop_df.set_index(cols.region_internal)

    missing_regions = set(wwtps) - set(pop_df.index)
    if missing_regions:
        raise ValueError(
            "Population missing for the following wwtps:\n"
            + ", ".join(sorted(missing_regions))
        )

    return pop_df.loc[wwtps, cols.population].values


def train_test_split_by_forecast_horizon(df, cols: ColumnSpec, horizon_days: int = 21):
    df["rank_desc"] = df.groupby(cols.region_internal)[cols.date].rank(method="first", ascending=False)
    df_test = df[df["rank_desc"] <= horizon_days].copy() # last 21 days for testing
    df_train = df[df["rank_desc"] > horizon_days].copy()
    df_train.drop(columns="rank_desc", inplace=True)
    df_test.drop(columns="rank_desc", inplace=True)
    return df_train, df_test

def get_label(df, cols: ColumnSpec):
    pivot_df = df.pivot(index = cols.date, columns = cols.region_internal, values=[cols.ed_visits])
    pivot_df = pivot_df.sort_index()
    y_ed = pivot_df[cols.ed_visits].fillna(0).to_numpy(dtype=float)
    wwtps = pivot_df[cols.ed_visits].columns.tolist()
    return y_ed, wwtps

def get_tests_per_capita(df, population, cols: ColumnSpec):
    df = df.copy()
    total_tests_df = df.pivot(index = cols.date, columns = cols.region_internal, values = cols.tests)
    total_tests_df = total_tests_df.sort_index()
    total_tests_df = total_tests_df.interpolate(method="linear", axis=0, limit_direction="both")
    tests_per_capita = total_tests_df.to_numpy(dtype=float)/population
    return tests_per_capita


def forecast_tests_per_capita(
    df_train,
    df_future,
    population,
    regions,
    cols: ColumnSpec,
    lookback_weeks: int = 8,
):
    """Forecast future testing rates from recent same-weekday observations.

    Only dates and region labels are read from ``df_future``. Future testing
    values are deliberately ignored to prevent leakage in retrospective runs.
    """
    if lookback_weeks < 1:
        raise ValueError("lookback_weeks must be at least 1")

    population = np.asarray(population, dtype=float)
    regions = list(regions)
    if population.ndim != 1 or population.size != len(regions):
        raise ValueError("population and regions must have the same length")
    if np.any(~np.isfinite(population)) or np.any(population <= 0):
        raise ValueError("population values must be finite and positive")

    train = df_train.copy()
    train[cols.date] = pd.to_datetime(train[cols.date])
    history = train.pivot(
        index=cols.date,
        columns=cols.region_internal,
        values=cols.tests,
    ).sort_index()
    history = history.reindex(columns=regions)

    if history.empty:
        raise ValueError("Cannot forecast testing rates without training history")

    daily_index = pd.date_range(history.index.min(), history.index.max(), freq="D")
    history = history.reindex(daily_index)
    history = history.interpolate(method="time", limit_direction="both")

    if history.isna().any().any():
        missing_regions = history.columns[history.isna().any()].tolist()
        raise ValueError(
            "Training testing history is missing for regions: "
            + ", ".join(map(str, missing_regions))
        )

    recent = history.tail(7 * lookback_weeks)
    weekday_levels = recent.groupby(recent.index.dayofweek).median()
    fallback_levels = recent.median(axis=0)

    future_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(df_future[cols.date]).dropna().unique())
    )
    if future_dates.empty:
        raise ValueError("df_future does not contain any forecast dates")

    forecast_counts = np.vstack(
        [
            weekday_levels.loc[date.dayofweek].fillna(fallback_levels).to_numpy(
                dtype=float
            )
            for date in future_dates
        ]
    )

    # The cases likelihood takes log(tests_per_capita), so keep forecasts positive.
    forecast_counts = np.maximum(forecast_counts, 1.0)
    return forecast_counts / population[None, :]

def getting_cases_and_ww_logged(
    df: pd.DataFrame,
    disease: str,
    cols: ColumnSpec,
    return_censoring: bool = False,
):
    # Prepare pivoted data
    cases_col = cols.cases_tpl.format(disease=disease)
    ww_col = cols.wwload_tpl.format(disease=disease)
    missing_col = cols.ww_missing_tpl.format(disease=disease)
    left_censored_col = cols.ww_left_censored_tpl.format(disease=disease)
    value_cols = [cases_col, ww_col]
    if missing_col in df.columns:
        value_cols.append(missing_col)
    if left_censored_col in df.columns:
        value_cols.append(left_censored_col)

    pivot_df = df.pivot(index= cols.date, 
                        columns=cols.region_internal, 
                        values=value_cols)
    
    pivot_df = pivot_df.sort_index()
    pivot_df[cases_col] = pivot_df[cases_col].fillna(0)
    y_cases = pivot_df[cases_col].to_numpy(dtype=float)

    y_signal = pivot_df[ww_col].to_numpy(dtype=float)  # shape (T, R)
    y_signal_array = np.array(y_signal, dtype=np.float64)
    if missing_col in df.columns:
        missing_flag = _coerce_bool_flag(
            pivot_df[missing_col].to_numpy(dtype=object).ravel()
        ).reshape(y_signal_array.shape)
        y_signal_array = np.where(missing_flag, np.nan, y_signal_array)
    y_signal_array = np.where(np.isnan(y_signal_array), np.nan, y_signal_array + 1)
    log_y_signal = np.log(y_signal_array)
    log_y_signal_masked = np.ma.masked_invalid(log_y_signal)

    if not return_censoring:
        return y_cases, log_y_signal_masked, pivot_df

    if left_censored_col in df.columns:
        left_censored = _coerce_bool_flag(
            pivot_df[left_censored_col].to_numpy(dtype=object).ravel()
        ).reshape(y_signal_array.shape)
        left_censored = left_censored & ~np.ma.getmaskarray(log_y_signal_masked)
    else:
        left_censored = np.zeros(y_signal_array.shape, dtype=bool)
    return y_cases, log_y_signal_masked, left_censored, pivot_df

def make_rolling_forecast_splits(
    df,
    cols: ColumnSpec,
    horizon_days: int = 21,
    step_days: int = 21,
    min_train_days: int = 180,
    start_date=None,
    end_date=None,
):
    """
    Create expanding-window rolling forecast splits.

    For each forecast origin:
        train = dates <= origin
        test  = origin < dates <= origin + horizon_days

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataframe.
    cols : ColumnSpec
        Column specification.
    horizon_days : int
        Forecast horizon, e.g. 21 days.
    step_days : int
        Distance between forecast origins.
        Use 21 for non-overlapping windows, 7 for weekly rolling windows.
    min_train_days : int
        Minimum amount of training history before first forecast origin.
    start_date, end_date : str or pd.Timestamp, optional
        Optional bounds for forecast origins.

    Returns
    -------
    list[dict]
        Each dict has window_id, forecast_origin, forecast_start,
        forecast_end, train_df, test_df.
    """
    df = df.copy()
    df[cols.date] = pd.to_datetime(df[cols.date])
    df = df.sort_values([cols.region_internal, cols.date])

    all_dates = pd.Index(sorted(df[cols.date].dropna().unique()))

    first_possible_origin = all_dates.min() + pd.Timedelta(days=min_train_days)
    last_possible_origin = all_dates.max() - pd.Timedelta(days=horizon_days)

    if start_date is not None:
        first_possible_origin = max(first_possible_origin, pd.Timestamp(start_date))
    if end_date is not None:
        last_possible_origin = min(last_possible_origin, pd.Timestamp(end_date))

    origins = pd.date_range(
        start=first_possible_origin,
        end=last_possible_origin,
        freq=f"{step_days}D",
    )

    splits = []

    for i, origin in enumerate(origins):
        forecast_start = origin + pd.Timedelta(days=1)
        forecast_end = origin + pd.Timedelta(days=horizon_days)

        train_df = df[df[cols.date] <= origin].copy()
        test_df = df[
            (df[cols.date] >= forecast_start)
            & (df[cols.date] <= forecast_end)
        ].copy()

        # keep only windows with data
        if train_df.empty or test_df.empty:
            continue

        splits.append(
            {
                "window_id": i,
                "forecast_origin": origin,
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "train_df": train_df,
                "test_df": test_df,
            }
        )

    return splits
