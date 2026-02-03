import pymc as pm
import numpy as np
import pandas as pd
from .io import ( standardize_input, 
                 validate_population, 
                 train_test_split_by_forecast_horizon, 
                 get_label, 
                 get_tests_per_capita,
)
from .schemas import ColumnSpec
from .build_functions import build_joint_model

def fit_joint_model(df,
                pop_df,
                diseases = ["covid","rsv","flua"],
                random_seed = 123,
                draws = 2000,
                tune = 1000,
                target_accept = 0.9,
                chains = 4,
                cols = ColumnSpec()):
    # Preprocess data
    df = standardize_input(df, diseases, cols)
    population = validate_population(pop_df, df_train[cols.region_internal].unique())
    df_train, df_test = train_test_split_by_forecast_horizon(df, cols, horizon_days=21) 
    y_ed = get_label(df_train, cols)
    tests_per_capita = get_tests_per_capita(df_train, population, cols)

    #fit model
    with pm.Model() as model:
        build_joint_model(diseases, 
                        df_train, 
                        y_ed, 
                        population, 
                        population.shape[0], 
                        tests_per_capita = tests_per_capita)
        results = pm.sample(draws = draws,
                            tune = tune,
                            target_accept = target_accept,
                            chains = chains,
                            random_seed = random_seed,
                            return_inferencedata=True, 
                            )
    
        return {"model": model,
                "idata": results,
                "train_df": df_train,
                "test_df": df_test
                }
