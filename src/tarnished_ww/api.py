import pymc as pm
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from .io import ( standardize_input, 
                 validate_population, 
                 train_test_split_by_forecast_horizon, 
                 get_label, 
                 get_tests_per_capita,
)
from .schemas import ColumnSpec, SamplerSpec, PopulationSpec
from .build_functions import build_joint_model
from .forecast_functions import build_forecast_model

def fit_joint_model(df,
                pop_df,
                diseases = ["covid","rsv","flua"],
                cols: ColumnSpec = ColumnSpec(),
                pop_cols: PopulationSpec = PopulationSpec(),
                sampler: SamplerSpec = SamplerSpec()):
    # Preprocess data
    df = standardize_input(df, diseases, cols)
    df_train, df_test = train_test_split_by_forecast_horizon(df, cols, horizon_days=21) 
    y_ed, wwtps = get_label(df_train, cols)
    population = validate_population(pop_df, wwtps, pop_cols)
    tests_per_capita = get_tests_per_capita(df_train, population, cols)

    #fit model
    with pm.Model() as model:
        build_joint_model(diseases, 
                        df_train, 
                        y_ed, 
                        population, 
                        population.shape[0], 
                        cols,
                        tests_per_capita = tests_per_capita)
        sample_kwargs = dict(draws = sampler.draws,
                         tune = sampler.tune,
                         target_accept = sampler.target_accept,
                         chains = sampler.chains,
                         cores = sampler.cores,
                         random_seed = sampler.random_seed,
                         return_inferencedata =True,
        )
        
        if sampler.extra:
            sample_kwargs.update(sampler.extra)
        results = pm.sample(**sample_kwargs)
        results.extend(pm.sample_posterior_predictive(results,
                                                      random_seed=sampler.random_seed))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    output_path = Path(os.path.dirname(Path.cwd())) / "models" / "model_traces"
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(os.path.join(output_path, f"fit_results_{timestamp}.nc"))
    
    return {"model": model,
            "trace": results,
            "train_df": df_train,
            "test_df": df_test,
            "wwtps": wwtps,
            "population": population,
            "diseases": diseases,
            "cols": cols,
            "sampler": sampler,
            "timestamp": timestamp,
            }

def predict_joint_model(fit_results):
    df_test = fit_results["test_df"]
    population = fit_results["population"]
    diseases = fit_results["diseases"]
    cols = fit_results["cols"]
    sampler = fit_results["sampler"]
    timestamp = fit_results["timestamp"]
    tests_per_capita = get_tests_per_capita(df_test, population, cols)
    with pm.Model() as forecast_model:
        var_names = build_forecast_model(diseases, 
                                         fit_results["trace"], 
                                         df_test,
                                         population, 
                                         population.shape[0],
                                         cols,
                                         tests_per_capita = tests_per_capita, 
                                         )
        predictions = pm.sample_posterior_predictive(fit_results["trace"], 
                                                     model=forecast_model, 
                                                     predictions=True, 
                                                     var_names=var_names + ["ed_contribution_residual", 
                                                                            "ed_contribution_covid",
                                                                            "ed_contribution_rsv",
                                                                            "ed_contribution_flua"],
                                                    random_seed=sampler.random_seed
                                                    )
    output_path = Path(os.path.dirname(Path.cwd())) / "models" / "model_traces"
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_netcdf(os.path.join(output_path,f"predictions_{timestamp}.nc"))
    return {"model": forecast_model,
            "trace": predictions,}