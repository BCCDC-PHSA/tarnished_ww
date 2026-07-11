import arviz as az
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import arviz as az


def summarize_forecast_ed_window(
    fit_results,
    prediction_results,
    ci: float = 0.95,
):
    """
    Convert forecast_ed_visits posterior predictive samples into tidy rows.

    Returns one row per forecast date x WWTP.
    """
    cols = fit_results["cols"]
    df_test = fit_results["test_df"]
    wwtps = fit_results["wwtps"]

    pivot_obs = (
        df_test
        .pivot(index=cols.date, columns=cols.region_internal, values=cols.ed_visits)
        .sort_index()
        .fillna(0)
    )

    samples = prediction_results["trace"].predictions["forecast_ed_visits"].values
    # shape: chains, draws, horizon, regions

    mean_pred = samples.mean(axis=(0, 1))
    median_pred = np.median(samples, axis=(0, 1))
    hdi = az.hdi(samples, hdi_prob=ci)

    rows = []

    for t, date in enumerate(pivot_obs.index):
        for r, region in enumerate(pivot_obs.columns):
            rows.append(
                {
                    "window_id": fit_results.get("window_id"),
                    "forecast_origin": fit_results.get("forecast_origin"),
                    "forecast_date": date,
                    "horizon": t + 1,
                    "wwtp": region,
                    "observed": pivot_obs.loc[date, region],
                    "pred_mean": mean_pred[t, r],
                    "pred_median": median_pred[t, r],
                    "pred_lower": hdi[t, r, 0],
                    "pred_upper": hdi[t, r, 1],
                    "covered": (
                        pivot_obs.loc[date, region] >= hdi[t, r, 0]
                        and pivot_obs.loc[date, region] <= hdi[t, r, 1]
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_rolling_forecast_metrics(rolling_predictions_df):
    """
    Simple rolling-window metrics by region and horizon.
    """
    df = rolling_predictions_df.copy()
    point_pred_col = "pred_median" if "pred_median" in df.columns else "pred_mean"
    df["abs_error"] = (df["observed"] - df[point_pred_col]).abs()

    metrics_by_region = (
        df.groupby("wwtp")
        .agg(
            mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            n=("observed", "size"),
        )
        .reset_index()
    )

    metrics_by_horizon = (
        df.groupby("horizon")
        .agg(
            mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            n=("observed", "size"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        {
            "mae": [df["abs_error"].mean()],
            "coverage": [df["covered"].mean()],
            "n": [len(df)],
        }
    )

    return {
        "overall": overall,
        "by_region": metrics_by_region,
        "by_horizon": metrics_by_horizon,
    }

def filtering_best_models(df,errors_df,wwtps):
    df_all =[df[df['regions']==wwtp].explode(['date','predictions','observed','features','wastewater','offset']) for wwtp in wwtps]
    errors_all = errors_df[errors_df['regions'].isin(wwtps)]
    return df_all, errors_all

def create_lagged_features_by_region(df, target_column, lags_dict):
    """
    Create lagged features for the target variable by region with different lag intervals.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing 'region' and time series data.
        target_column (str): The target column to create lags for.
        lags_dict (dict): A dictionary where keys are lag names and values are the lag intervals (in days).
        
    Returns:
        pd.DataFrame: The dataframe with lagged features.
    """
    df_copy = df.copy()

    # Apply lagging process by region
    for target in target_column:
        for region in df_copy['region'].unique():
            region_data = df_copy[df_copy['region'] == region].copy()
            for lag_name, lag_days in lags_dict.items():
                df_copy.loc[df_copy['region'] == region, f'{target}_{lag_name}'] = region_data[target].shift(lag_days)
        # Drop NaNs by group (region) to avoid data loss
        df_copy = df_copy.dropna()

    return df_copy

def compute_mae_test(final_predictions_per_region, new_data, region_column, label):
    """
    Compute MAE for test data per region.

    Parameters:
        final_predictions_per_region (dict): Dictionary containing test predictions per region.
        new_data (DataFrame): Test dataset containing actual values.
        region_column (str): Column name for regions.
        label (str): Target variable (actual values in test set).

    Returns:
        dict: MAE per region for test data.
    """

    mae_test_per_region = {}

    for region in final_predictions_per_region.keys():
        # Get test predictions
        test_predictions = final_predictions_per_region[region]['test']

        # Get actual test values from new_data for this region
        y_test = new_data[new_data[region_column] == region][label].values

        # Compute MAE only if we have valid data
        if len(y_test) == len(test_predictions):
            mae_test_per_region[region] = round(mean_absolute_error(y_test, test_predictions), 4)
        else:
            print(f"Warning: Mismatch in test data size for region {region}")

    return mae_test_per_region


def averaging_model_wwtps(df_all):
    
    columns_to_be_added= ['predictions','observed']
    columns_to_be_kept = ['date','features','wastewater','offset','testing_date','comb']
    
    df_all[0].reset_index(inplace=True, drop=True)
    df_final = df_all[0][columns_to_be_added + columns_to_be_kept]
    index = columns_to_be_kept
    for df_temp in df_all[1:]:
        df_temp.reset_index(inplace=True, drop=True)
        df_temp.drop('regions',axis=1,inplace=True)
        df_final = df_final.merge(df_temp, on = index, how='left')
        for col_final in columns_to_be_added:
            list_cols = [col for col in df_final.columns if col_final in col]
            df_final[col_final] = df_final[list_cols].fillna(0.0).sum(axis=1)
            for col_del in list_cols:
                del df_final[col_del]
    return df_final

def with_or_without(wastewater):
    if wastewater:
        return 'with wastewater'
    else:
        return 'without wastewater'
    

def get_summaries_posterior(trace, vars_name, predictions = True):
    if predictions == True:
        forecast_trace = trace.predictions[vars_name]  # shape: (chains, draws, H, regions)
        forecast_mean = forecast_trace.values.mean(axis = (0,1)) # shape: (H, R)
        forecast_hdi = az.hdi(forecast_trace.values, hdi_prob=0.95)  # shape: (H, R, 2)
    else:
        forecast_trace = trace.posterior_predictive[vars_name]  # shape: (chains, draws, H, regions)
        forecast_mean = forecast_trace.values.mean(axis = (0,1)) # shape: (H, R)
        forecast_hdi = az.hdi(forecast_trace.values, hdi_prob=0.95)  # shape: (H, R, 2)

    return forecast_mean, forecast_hdi


def coverage_from_samples(xgb_samples,obs_array):
    lower_xgb = xgb_samples.quantile(0.025, dim="sample")
    upper_xgb = xgb_samples.quantile(0.975, dim="sample")
    covered_xgb = (obs_array >= lower_xgb) & (obs_array <= upper_xgb)
    coverage_xgb = covered_xgb.mean().item()   # single float
    coverage_per_region_xgb = covered_xgb.mean(dim="time")
    coverage_per_time_xgb = covered_xgb.mean(dim="wwtp")
    return coverage_per_region_xgb, coverage_per_time_xgb

def compute_mape(xgb_samples, obs_array):
  xgb_pred_mean = xgb_samples.mean(dim="sample")
  obs_eval = obs_array.sel(time=xgb_pred_mean.time, wwtp=xgb_pred_mean.wwtp)
  mape_xgb = ((abs(obs_eval - xgb_pred_mean) / obs_eval)*100).mean(dim="time")
  return mape_xgb
