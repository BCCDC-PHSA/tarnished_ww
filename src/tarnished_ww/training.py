import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid

def rolling_window_xgboost(df, region_column, label, features, window_size, step_size, param_grid, lags_dict):
    """
    Perform rolling window cross-validation with XGBoost for time series forecasting per region.
    Hyperparameters are tuned using grid search across all rolling windows and the best parameters 
    are selected based on the lowest average MAE across regions.
    
    Parameters:
        df (DataFrame): Time-series data with a region column.
        region_column (str): The column name representing different regions.
        label (str): The target variable name.
        features (list): List of feature column names.
        window_size (int): Size of the rolling window for training.
        step_size (int): Step size for rolling validation.
        param_grid (dict): Hyperparameter grid for tuning.

    Returns:
        dict: Best parameters based on average MAE across all rolling windows and regions.
        dict: MAE results per region.
    """

    param_mae_dict = {}  # To store MAE for each parameter set
    region_results = {}  # To store MAE results per region

    unique_regions = df[region_column].unique()
    features_lagged = [f'{feat}_{lag_name}' for lag_name, lag_days in lags_dict.items() for feat in features]
    for region in unique_regions:
        print(f"\nProcessing region: {region}")
        df_region = df[df[region_column] == region].copy()
        results = []

        # Rolling window for each region
        for start in range(0, len(df_region) - window_size, step_size):
            train_data = df_region.iloc[start:start + window_size]
            val_data = df_region.iloc[start + window_size:start + window_size + step_size]

            # Prepare features and target
            X_train = train_data[features_lagged + [f'{label}_{lag_name}' for lag_name, lag_days in lags_dict.items()]]
            y_train = train_data[label]
            X_val = val_data[features_lagged + [f'{label}_{lag_name}' for lag_name, lag_days in lags_dict.items()]]
            y_val = val_data[label]

            # Grid search over hyperparameters
            for params in ParameterGrid(param_grid):
                model = XGBRegressor(objective='reg:absoluteerror', random_state=42, **params)
                model.fit(X_train, y_train)

                # Evaluate performance
                y_val_pred = model.predict(X_val)
                mae_val = mean_absolute_error(y_val, y_val_pred)

                # Store MAE per parameter set
                param_tuple = tuple(params.items())  # Convert params to tuple key
                if param_tuple not in param_mae_dict:
                    param_mae_dict[param_tuple] = []
                param_mae_dict[param_tuple].append(mae_val)

                # Store results for this region and fold
                results.append({'start': start, 'mae': mae_val})

        # Store region results
        region_results[region] = results

    # Compute average MAE for each parameter set across all regions
    avg_mae_per_param = {params: np.mean(maes) for params, maes in param_mae_dict.items()}

    # Select best hyperparameters
    best_param_set = min(avg_mae_per_param, key=avg_mae_per_param.get)

    return best_param_set, region_results

def train_and_predict_per_region_xgboost(df, new_data, region_column, label, features, best_params, lags_dict):
    """
    Train final XGBoost models per region using the best parameters and predict on new unseen data.

    Parameters:
        df (DataFrame): Historical training data with regions and time series.
        new_data (DataFrame): New unseen data to predict on.
        region_column (str): Column name for regions.
        label (str): Target variable.
        features (list): Feature columns.
        best_params (dict): Best hyperparameters from rolling window tuning.

    Returns:
        dict: Final MAE per region (on training data).
        dict: Final predictions per region (for training and test data).
    """

    final_mae_per_region = {}
    final_predictions_per_region = {}

    unique_regions = df[region_column].unique()
    all_train_dfs = []
    all_test_dfs = []
    features_lagged = [f'{feat}_{lag_name}' for lag_name, lag_days in lags_dict.items() for feat in features]

    for region in unique_regions:
        print(f"\nTraining final model for region: {region}")

        # Filter data for the region
        df_region = df[df[region_column] == region].copy()
        new_data_region = new_data[new_data[region_column] == region].copy()

        # Train final model using all available training data
        X_train = df_region[features_lagged + [f'{label}_{lag_name}' for lag_name, lag_days in lags_dict.items()]]
        y_train = df_region[label]

        final_model = XGBRegressor(objective='reg:absoluteerror', random_state=42, **dict(best_params))
        final_model.fit(X_train, y_train)

        # Predict on training data (for visualization)
        train_predictions = final_model.predict(X_train)

        # Predict on new unseen data
        X_test = new_data_region[features_lagged + [f'{label}_{lag_name}' for lag_name, lag_days in lags_dict.items()]]
        y_test = new_data_region[label]
        test_predictions = final_model.predict(X_test)

        # Compute MAE on training data
        train_mae = mean_absolute_error(y_train, train_predictions)
        final_mae_per_region[region] = round(train_mae, 4)

        # Store predictions (both training and test)
        final_predictions_per_region[region] = {
            'train': train_predictions,
            'test': test_predictions
        }

        print(f"Final MAE on training data for {region}: {train_mae}")
        test_df = pd.DataFrame({'Pred': test_predictions, 'Actual': y_test,
                                     'surveillance_date': new_data_region['surveillance_date']})
        train_df = pd.DataFrame({'Pred': train_predictions, 'Actual': y_train,
                                     'surveillance_date': df_region['surveillance_date']})
        train_df['surveillance_date'] = pd.to_datetime(train_df['surveillance_date'])
        test_df['surveillance_date'] = pd.to_datetime(test_df['surveillance_date'])
        train_df['region'] = test_df['region'] = region
        importance_dict = final_model.get_booster().get_score(importance_type='gain')

        importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values()),
        "Region":region}).sort_values(by='Importance', ascending=False)

        # Append to lists
        all_train_dfs.append(train_df)
        all_test_dfs.append(test_df)

    final_train_df = pd.concat(all_train_dfs, ignore_index=True)
    final_test_df = pd.concat(all_test_dfs, ignore_index=True)

    return final_mae_per_region, final_predictions_per_region,  final_train_df, final_test_df, importance_df

def moving_block_bootstrap(df, block_size, target_length, seed=None):
    """
    Sample blocks of time series data with replacement.

    Parameters:
        df (DataFrame): Input time-ordered data.
        block_size (int): Length of each block.
        target_length (int): Desired number of rows in the output.
        seed (int): Random seed.

    Returns:
        DataFrame: Bootstrapped time series sample.
    """
    if seed is not None:
        np.random.seed(seed)

    num_blocks = int(np.ceil(target_length / block_size))
    starts = np.random.randint(0, len(df) - block_size + 1, size=num_blocks)
    blocks = [df.iloc[start:start + block_size] for start in starts]
    bootstrapped_df = pd.concat(blocks, ignore_index=True).iloc[:target_length]
    return bootstrapped_df


def train_and_predict_per_region_xgboost_bootstrapping(
    df,
    new_data,
    region_column,
    label,
    features,
    best_params,
    lags_dict,
    n_bootstraps: int = 1,
    ci: float = 0.95,
    block_size: int = 14  # default block size for MBB
):
    """
    Train final XGBoost models per region using best parameters, with MBB bootstrapping.

    Returns:
        final_mae_per_region (dict): MAE on train set (averaged across bootstraps).
        final_train_df (DataFrame): Train predictions with columns [region, surveillance_date, Actual, Pred, bootstrap].
        final_test_df (DataFrame): Test predictions with columns [region, surveillance_date, Actual, Pred, bootstrap].
    """
    all_train_dfs = []
    all_test_dfs = []
    final_mae_per_region = {}

    unique_regions = df[region_column].unique()
    features_lagged = [f'{feat}_{lag_name}' for lag_name, lag_days in lags_dict.items() for feat in features]

    for region in unique_regions:
        maes = []
        df_region_full = df[df[region_column] == region].copy()
        new_data_region = new_data[new_data[region_column] == region].copy()
        target_len = len(df_region_full)

        for b in range(n_bootstraps):
            # Moving block bootstrap
            df_region = moving_block_bootstrap(df_region_full, block_size, target_len, seed=42 + b)

            X_train = df_region[features_lagged + [f"{label}_{lag_name}" for lag_name in lags_dict]]
            y_train = df_region[label]

            model = XGBRegressor(
                objective="reg:absoluteerror",
                random_state=42 + b,
                **best_params
            )
            model.fit(X_train, y_train)

            # Train predictions
            train_preds = model.predict(X_train)
            maes.append(mean_absolute_error(y_train, train_preds))

            # Test predictions
            X_test = new_data_region[features_lagged + [f"{label}_{lag_name}" for lag_name in lags_dict]]
            y_test = new_data_region[label]
            test_preds = model.predict(X_test)

            # Build DataFrames
            train_df = pd.DataFrame({
                "region": region,
                "surveillance_date": pd.to_datetime(df_region["surveillance_date"]),
                "Actual": y_train.values,
                "Pred": train_preds,
                "bootstrap": b
            })
            test_df = pd.DataFrame({
                "region": region,
                "surveillance_date": pd.to_datetime(new_data_region["surveillance_date"]),
                "Actual": y_test.values,
                "Pred": test_preds,
                "bootstrap": b
            })

            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)

        final_mae_per_region[region] = round(np.mean(maes), 4)

    final_train_df = pd.concat(all_train_dfs, ignore_index=True)
    final_test_df = pd.concat(all_test_dfs, ignore_index=True)

    return final_mae_per_region, final_train_df, final_test_df
