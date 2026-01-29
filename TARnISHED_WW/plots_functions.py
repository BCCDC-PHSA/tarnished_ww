import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def resampling_weekly(all_train_df,all_test_df):
       all_train_w = []
       all_test_w = []
       for region in all_train_df['region'].unique():
              # Filter data
              train_df = all_train_df[all_train_df['region'] == region]
              test_df = all_test_df[all_test_df['region'] == region]
              # Set 'dates' as the index
              train_df.set_index('surveillance_date', inplace=True)
              test_df.set_index('surveillance_date', inplace=True)
              # Add an indicator to separate train and test later
              train_df['is_train'] = True
              test_df['is_train'] = False

              # Combine for unified resampling
              combined_df = pd.concat([train_df, test_df])

              # Resample the combined data
              combined_resampled = combined_df.resample('W')[['Actual', 'Pred']].sum()
              combined_resampled = combined_resampled.reset_index()

              # Reassign train/test after resampling
              train_resampled = combined_resampled[combined_df['is_train'].resample('W').first().values]
              test_resampled = combined_resampled[~combined_df['is_train'].resample('W').first().values]
              train_resampled['region'] = test_resampled['region'] = region

              # Append to lists
              all_train_w.append(train_resampled)
              all_test_w.append(test_resampled)

       final_train_resampled = pd.concat(all_train_w, ignore_index=True)
       final_test_resampled = pd.concat(all_test_w, ignore_index=True)

       return final_train_resampled, final_test_resampled

def plot_train_test_predictions(all_train_df, all_test_df, rows, cols, savepath): 
    # Plot actual vs predicted values
       fig, axs = plt.subplots(rows, cols, figsize=(18, 12), sharex=False, sharey=False)
       axs = axs.flatten()  
       bottom_row_indices = list(range((rows - 1) * cols, rows * cols))

       for i, region in enumerate(all_train_df['region'].unique()):
              ax = axs[i]
              # Filter data
              region_train = all_train_df[all_train_df['region'] == region]
              region_test = all_test_df[all_test_df['region'] == region]
              region_all = pd.concat([region_train, region_test]).sort_values("surveillance_date")

              # Plot train
              ax.plot(region_all['surveillance_date'], region_all['Actual'], label='Observed (Train)', linestyle='-', color='red', alpha=0.7, linewidth=5)
              ax.plot(region_all['surveillance_date'], region_all['Pred'], label='Predictions (Train)', linestyle='--', color='blue', alpha=0.7, linewidth=5)

              # Plot test
              if not region_test.empty:
                    ax.plot(region_test['surveillance_date'], region_test['Actual'], label='Actual (Test)', linestyle='-', color='orange', alpha=0.7, linewidth=5)
                    ax.plot(region_test['surveillance_date'], region_test['Pred'], label='Predicted (Test)', linestyle='--', color='cyan', alpha=0.7, linewidth=5)

              # Formatting
              ax.set_title(f"Region: {region}", fontsize=32)
              #ax.set_xlabel("Date", fontsize=30)
              #ax.set_ylabel("Respiratory Related ED Visits", fontsize=30)
              ax.tick_params(axis='x', rotation=45, labelsize=28, width=1.5, length=6)
              if i not in bottom_row_indices:
                     ax.set_xticklabels([])
              else:
                     ax.tick_params(axis='x', rotation=45, labelsize=28, width=1.5, length=6)
              ax.tick_params(axis='y', labelsize=28, width=1.5, length=6)
              ax.grid(True, linestyle='--', alpha=0.5)

       # Put legend in the last subplot (or use fig.legend for a global one)
       axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
       #ax.legend(loc='upper left', fontsize= 26)
       # Adjust layout
       plt.tight_layout()
       plt.subplots_adjust(left=0.1, right=0.92, bottom=0.12)
       plt.suptitle("Observed vs Predicted Respiratory Related ED visits per Region", fontsize=32, y=1.02)
       fig.text(0.02, 0.5, 'Respiratory Related ED Visits', va='center', rotation='vertical', fontsize=32)
       fig.text(0.5, 0.005, 'Date', ha='center', fontsize=32)

       # Show or save
       plt.savefig(os.path.join(savepath,"actual_vs_predicted_regions.png"), bbox_inches="tight")
       plt.show()

def plot_posterior_data(wwtps,dates, y_cases, pred_cases, pred_cases_interval,
                        y_ww, pred_ww, pred_ww_interval, latent_mean, latent_hdi, 
                        suffix, forecasting=False, forecast_start = None,
                        region_ids=None, latent_plot=False):
        
        if region_ids is None:
            region_ids = list(range(len(wwtps)))  # plot all by default

        for r in region_ids:
            if latent_plot == True:
                fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            else:
                fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            # --- CASES ---
            ax1 = axs[1]

            ax1.plot(dates[1:], y_cases[1:, r], label='Observed Cases', color='black')
            ax1.plot(dates[1:], pred_cases[1:, r], label='Predicted Mean', color='blue')
            ax1.fill_between(
                    dates[1:],
                    pred_cases_interval[1:, r, 0],
                    pred_cases_interval[1:, r, 1],
                    color='blue',
                    alpha=0.3,
                    label='95% Credible Interval'
            )
            ax1.set_ylabel("Cases Counts", color='black', fontsize=18)
            ax1.set_title(f"Case Counts", fontsize = 24)
            ax1.grid(True, linestyle="--", alpha=0.5)
            #ax1.set_xlabel("Date", fontsize = 22)
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            axs[1].legend(lines_1, labels_1, loc="lower left")
            axs[1].tick_params(axis='x', rotation=45, labelsize=16)

            # --- WASTEWATER ---
            ax3 = axs[0]

            ax3.plot(dates[1:], y_ww[1:, r], label='Observed Wastewater', color='black')
            ax3.plot(dates[1:], pred_ww[1:, r], label='Predicted Mean', color='green')
            ax3.fill_between(
                    dates[1:],
                    pred_ww_interval[1:, r, 0],
                    pred_ww_interval[1:, r, 1],
                    color='green',
                    alpha=0.3,
                    label='95% Credible Interval'
            )
            ax3.set_ylabel("Wastewater", color='black', fontsize=18)
            ax3.set_title(f"Wastewater (Log Scale)", fontsize = 24)
            #ax3.set_xlabel("Date", fontsize = 22)
            ax3.grid(True, linestyle="--", alpha=0.5)

            lines_3, labels_3 = ax3.get_legend_handles_labels()
            axs[0].legend(lines_3, labels_3, loc="lower left")

            if latent_plot==True:
                # --- Latent---
                ax5 = axs[2]

                ax5.plot(dates[1:], latent_mean[1:, r], label='Latent Process (Log Scale)', color='black')
                ax5.fill_between(dates[1:],
                                latent_hdi[1:, r, 0],
                                latent_hdi[1:, r, 1],
                                color='grey',
                                alpha=0.3,
                                label='95% Credible Interval'
                )
                ax5.set_ylabel("Latent Process", color='black', fontsize=18)
                ax5.set_title(f"Model Community Infection Level (Log scale)", fontsize = 24)
                #ax5.set_xlabel("Date", fontsize = 22)
                ax5.grid(True, linestyle="--", alpha=0.5)

            if forecasting==True:
                    ax1.axvline(x=forecast_start, color='red', linestyle='--', label='Forecasting')
                    ax3.axvline(x=forecast_start, color='red', linestyle='--', label='Forecasting')
                    if latent_plot == True:
                        ax5.axvline(x=forecast_start, color='red', linestyle='--', label='Forecasting')
            plt.suptitle(f"{wwtps[r]} {suffix.upper()}", fontsize = 26)
            fig.supxlabel("Date", fontsize=22);
            plt.tight_layout()
            plt.show()

def plot_posterior_diseases_region(diseases, dates, y_cases, pred_cases, pred_cases_interval,
                                    y_ww, pred_ww, pred_ww_interval, wwtps, region_id,
                                    forecasting=False, forecast_start=None, size =[14,4]):
    """
    Plot posterior predictions for cases and wastewater for a single region, across multiple diseases.

    Parameters
    ----------
    diseases : list of str
        List of suffixes for each disease.
    dates : np.ndarray
        Dates array of shape (T + forecast,)
    y_cases : dict
        Dict of observed cases arrays (T, R) per disease.
    pred_cases : dict
        Dict of predicted case mean arrays (T + forecast, R) per disease.
    pred_cases_interval : dict
        Dict of predicted case intervals (T + forecast, R, 2) per disease.
    y_ww : dict
        Dict of observed wastewater arrays (T, R) per disease.
    pred_ww : dict
        Dict of predicted wastewater mean arrays (T + forecast, R) per disease.
    pred_ww_interval : dict
        Dict of predicted wastewater intervals (T + forecast, R, 2) per disease.
    wwtps : list
        List of region names.
    region_id : int
        Index of region to plot.
    forecasting : bool
        If True, a red vertical line will mark the forecast start.
    forecast_start : datetime or str
        Value in `dates` to use as forecast start marker.
    """

    n_diseases = len(diseases)
    fig, axs = plt.subplots(nrows=n_diseases, ncols=2, figsize=(size[0], size[1] * n_diseases),
                        sharex=True, gridspec_kw={"hspace": 0.23, "wspace": 0.25})
    for i, disease in enumerate(diseases):
        # --- CASES ---
        ax_cases = axs[i, 0]
        ax_cases.plot(dates[disease][1:], y_cases[disease][1:, region_id], label="Observed", color="black")
        ax_cases.plot(dates[disease][1:], pred_cases[disease][1:, region_id], label="Predicted Mean", color="aquamarine")
        ax_cases.fill_between(
            dates[disease][1:], 
            pred_cases_interval[disease][1:, region_id, 0], 
            pred_cases_interval[disease][1:, region_id, 1],
            color="aquamarine", alpha=0.8, label="95% CI"
        )
        ax_cases.set_ylabel("Cases", fontsize=14)
        ax_cases.set_title(f"{disease.upper()}", fontsize=12)
        ax_cases.grid(True, linestyle="--", alpha=0.5)
        if i ==0:
            ax_cases.legend(
                loc='upper center',          # place above the plot
                bbox_to_anchor=(0.5, 1.5),  # x=center, y=just above plot
                ncol=2,                      # show legend items side by side
                frameon=False                # remove legend box
            )
        if forecasting:
            ax_cases.axvline(x=forecast_start[disease], color='red', linestyle='--', label='Forecast')

        # --- WASTEWATER ---
        ax_ww = axs[i, 1]
        ax_ww.plot(dates[disease][1:], y_ww[disease][1:, region_id], label="Observed", color="black")
        ax_ww.plot(dates[disease][1:], pred_ww[disease][1:, region_id], label="Predicted Mean", color="salmon")
        
        ax_ww.fill_between(
            dates[disease][1:], 
            np.maximum(0, pred_ww_interval[disease][1:, region_id, 0]), 
            pred_ww_interval[disease][1:, region_id, 1],
            color="salmon", alpha=0.6, label="95% CI"
        )
        ax_ww.set_ylabel("Wastewater", fontsize=12)
        ax_ww.set_title(f"{disease.upper()}", fontsize=12)
        ax_ww.grid(True, linestyle="--", alpha=0.5)
        if i ==0:
            ax_ww.legend(
                loc='upper center',          # place above the plot
                bbox_to_anchor=(0.5, 1.5),  # x=center, y=just above plot
                ncol=2,                      # show legend items side by side
                frameon=False                # remove legend box
            )
        if forecasting:
            ax_ww.axvline(x=forecast_start[disease], color='red', linestyle='--', label='Forecast')

    #fig.text(0.05, 0.5, 'Cases', va='center', rotation='vertical', fontsize=14)
    #fig.text(0.49, 0.5, 'Wastewater', va='center', rotation='vertical', fontsize=14)
    fig.supxlabel("Date", fontsize=16, y=0.02)
    for ax_row in axs:
        for ax in ax_row:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
    #plt.suptitle(f"Posterior Predictive for WWTP {region_id + 1}", fontsize=18);
    # silencing warnings
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]);
    plt.show()


def plot_train_test_predictions_bootstrapping(all_train_df,all_test_df,rows,cols,savepath, start = 0, ci: float = 0.95):
    """
    Plot observed vs predictions per region, with test-set CI band.

    Parameters:
        all_train_df (DataFrame): output from train function.
        all_test_df (DataFrame): output from train function.
        rows, cols (int): subplot grid.
        savepath (str): folder to save figure.
        ci (float): confidence interval width (e.g. 0.95).
    """
    fig, axs = plt.subplots(rows, cols, figsize=(18, 12), sharex=False, sharey=False)
    axs = axs.flatten()
    bottom_row_indices = list(range((rows - 1) * cols, rows * cols))

    alpha = (1.0 - ci) / 2

    for i, region in enumerate(sorted(all_train_df["region"].unique())):
        ax = axs[i]
        # train + test merged
        df_train = all_train_df[all_train_df["region"] == region]
        df_test  = all_test_df[ all_test_df["region"] == region]

        # Plot train actual vs pred (single line)
        df_train_sorted = df_train.sort_values("surveillance_date")
        ax.plot(
            df_train_sorted["surveillance_date"][start:],
            df_train_sorted["Actual"][start:],
            label="Observed",
            color="black",
            linestyle=":", linewidth=2, alpha=0.7
        )
        ax.plot(
            df_train_sorted["surveillance_date"][start:],
            df_train_sorted["Pred"][start:],
            label= "Predicted",
            linestyle="-", linewidth=1, alpha=0.7
        )

        # For test, compute CI bands
        df_test_grouped = (
            df_test
            .groupby("surveillance_date")["Pred"]
            .agg(["mean", lambda x: np.quantile(x, alpha), lambda x: np.quantile(x, 1 - alpha)])
            .rename(columns={
                "mean": "Pred_mean",
                "<lambda_0>": "Pred_lower",
                "<lambda_1>": "Pred_upper"
            })
            .reset_index()
        )

        # plot actual test
        ax.plot(
            df_test_grouped["surveillance_date"],
            all_test_df[all_test_df["region"] == region]
                  .drop_duplicates("surveillance_date")["Actual"],color="black",
            linestyle=":", linewidth=2, alpha=0.7
        )#label="Observed (Test)",

        # plot mean prediction
        ax.plot(
            df_test_grouped["surveillance_date"],
            df_test_grouped["Pred_mean"],
            #label="Predicted",
            color="#4C72B0",
            linestyle="--", linewidth=2, alpha=0.7
        )

        # fill between CI
        ax.fill_between(
            df_test_grouped["surveillance_date"],
            df_test_grouped["Pred_lower"],
            df_test_grouped["Pred_upper"],
            color="lightblue",  # 👈 choose a soft color here
            alpha=0.7,
            label=f"{int(ci*100)}% CI"
        )

        # formatting
        ax.set_title(f"{region}", fontsize=20)
        if i not in bottom_row_indices:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis="x", rotation=45, labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)

    # global legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle("Observed vs Predicted ED Visits with Test CI", fontsize=24)

    # save + show
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(os.path.join(savepath, "actual_vs_predicted_regions_ci.png"), bbox_inches="tight")
    plt.show()
