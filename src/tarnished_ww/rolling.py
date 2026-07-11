import pandas as pd
from datetime import datetime
from pathlib import Path

from .io import standardize_input, make_rolling_forecast_splits
from .api import fit_joint_model_from_split, predict_joint_model
from .schemas import ColumnSpec, PopulationSpec, SamplerSpec
from .helper_functions import summarize_forecast_ed_window


def rolling_forecast_joint_model(
    df,
    pop_df,
    diseases=["covid", "rsv", "flua"],
    cols: ColumnSpec = ColumnSpec(),
    pop_cols: PopulationSpec = PopulationSpec(),
    sampler: SamplerSpec = SamplerSpec(),
    horizon_days: int = 21,
    step_days: int = 21,
    min_train_days: int = 180,
    start_date=None,
    end_date=None,
    keep_models: bool = False,
    center_case_signal: bool = True,
    center_ed_signal: bool = True,
    sigma_ed_prior_scale: float = 0.0005,
):
    """
    Run rolling 21-day forecast evaluation.

    By default:
      - expanding training window
      - 21-day forecast horizon
      - non-overlapping 21-day windows

    Set step_days=7 for weekly rolling origins.
    """
    df_std = standardize_input(df, diseases, cols)

    splits = make_rolling_forecast_splits(
        df_std,
        cols=cols,
        horizon_days=horizon_days,
        step_days=step_days,
        min_train_days=min_train_days,
        start_date=start_date,
        end_date=end_date,
    )

    all_prediction_rows = []
    window_results = []
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path.cwd().parent / "models" / f"model_traces_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving rolling traces to: {output_dir}")

    for split in splits:
        print(
            f"Window {split['window_id']}: "
            f"train <= {split['forecast_origin'].date()}, "
            f"forecast {split['forecast_start'].date()} to {split['forecast_end'].date()}"
        )

        fit_results = fit_joint_model_from_split(
            df_train=split["train_df"],
            df_test=split["test_df"],
            pop_df=pop_df,
            diseases=diseases,
            cols=cols,
            pop_cols=pop_cols,
            sampler=sampler,
            window_id=split["window_id"],
            forecast_origin=split["forecast_origin"],
            output_dir=output_dir,
            center_case_signal=center_case_signal,
            center_ed_signal=center_ed_signal,
            sigma_ed_prior_scale=sigma_ed_prior_scale,
        )

        pred_results = predict_joint_model(fit_results)

        pred_df = summarize_forecast_ed_window(
            fit_results=fit_results,
            prediction_results=pred_results,
        )

        all_prediction_rows.append(pred_df)

        if keep_models:
            window_results.append(
                {
                    "split": split,
                    "fit_results": fit_results,
                    "prediction_results": pred_results,
                }
            )

    rolling_predictions = pd.concat(all_prediction_rows, ignore_index=True)
    rolling_predictions.to_csv(output_dir / "rolling_predictions.csv", index=False)

    return {
        "predictions": rolling_predictions,
        "windows": splits,
        "window_results": window_results if keep_models else None,
        "run_id": run_id,
        "output_dir": output_dir,
    }
