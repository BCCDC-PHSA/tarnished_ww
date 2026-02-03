import pandas as pd
import numpy as np
from .schemas import ColumnSpec

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

def validate_population(pop_df, wwtps):
    required = {"wwtp", "population"}
    missing = required - set(pop_df.columns)
    if missing:
        raise ValueError(f"Population table missing columns: {missing}")

    pop_df = pop_df.set_index("wwtp")

    missing_regions = set(wwtps) - set(pop_df.index)
    if missing_regions:
        raise ValueError(
            "Population missing for the following wwtps:\n"
            + ", ".join(sorted(missing_regions))
        )

    return pop_df.loc[wwtps, "population"].values


def train_test_split_by_forecast_horizon(df, cols: ColumnSpec, horizon_days: int = 21):
        df["rank_desc"] = df.groupby("wwtp")["surveillance_date"].rank(method="first", ascending=False)
        df_test = df[df["rank_desc"] <= horizon_days].copy() # last 21 days for testing
        df_train = df[df["rank_desc"] > horizon_days].copy()
        df_train.drop(columns="rank_desc", inplace=True)
        df_test.drop(columns="rank_desc", inplace=True)
        return df_train, df_test

def get_label(df, cols: ColumnSpec):
    pivot_df = df.pivot(index='surveillance_date', columns='wwtp', values=['total_ed_visits'])
    pivot_df['total_ed_visits'] = pivot_df['total_ed_visits'].fillna(0)
    pivot_df = pivot_df.sort_index()
    y_ed = pivot_df['total_ed_visits'].values 
    y_ed = np.asarray(y_ed)
    return y_ed

def get_tests_per_capita(df, population, cols: ColumnSpec):
    df = df.copy()
    total_tests_df = df.pivot(index='surveillance_date', columns='wwtp', values=['total_tests_all_ages'])
    total_tests_df = total_tests_df.sort_index()
    total_tests_df = total_tests_df.interpolate(method="linear", axis=0, limit_direction="both")
    tests_per_capita = np.asarray(total_tests_df['total_tests_all_ages'].values)/population
    return tests_per_capita