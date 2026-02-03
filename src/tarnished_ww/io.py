import pandas as pd
import numpy as np
from .schemas import ColumnSpec, PopulationSpec

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

def getting_cases_and_ww_logged(df: pd.DataFrame, disease: str, cols: ColumnSpec):
    # Prepare pivoted data
    pivot_df = df.pivot(index= cols.date, 
                        columns=cols.region_internal, 
                        values=[cols.cases_tpl.format(disease=disease), 
                               cols.wwload_tpl.format(disease=disease)])
    
    pivot_df = pivot_df.sort_index()
    pivot_df[cols.cases_tpl.format(disease=disease)] = pivot_df[cols.cases_tpl.format(disease=disease)].fillna(0)
    y_cases = pivot_df[cols.cases_tpl.format(disease=disease)].to_numpy(dtype=float)

    y_signal = pivot_df[cols.wwload_tpl.format(disease=disease)].to_numpy(dtype=float)  # shape (T, R)
    y_signal_array = np.array(y_signal, dtype=np.float64)
    y_signal_array = np.where(np.isnan(y_signal_array), np.nan, y_signal_array + 1)
    log_y_signal = np.log(y_signal_array)
    log_y_signal_masked = np.ma.masked_invalid(log_y_signal)
    return y_cases, log_y_signal_masked, pivot_df