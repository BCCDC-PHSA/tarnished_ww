import pandas as pd
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