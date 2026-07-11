import pymc as pm
import arviz as az
import numpy as np
import pytensor.tensor as pt
from .build_functions import applying_convolution, gamma_kernel
from .io import getting_cases_and_ww_logged
from .schemas import ColumnSpec

def sampling_latent_forecast(num_regions, T_train, T_forecast, latent_disease, suffix):
    """
    Builds a new latent process for forecasting starting from a fixed init_value.
    """
    chol = pm.Flat(f"chol_cov_{suffix}", shape=(num_regions*(num_regions+1)//2))
    chol_matrix = pm.Deterministic(f"chol_matrix_{suffix}",pm.expand_packed_triangular(num_regions, chol))
    #latent_disease = pm.Flat(f"latent_{suffix}", shape=(T_train, num_regions))
    epsilon = 1e-10
    init_dist = pm.MvNormal.dist(mu=pt.mean(latent_disease[-7:, :], axis=0), cov=epsilon * np.eye(num_regions))
    latent_forecast = pm.MvGaussianRandomWalk(
        f"latent_forecast_{suffix}",
        mu=np.zeros(num_regions),  # mean zero step
        chol=chol_matrix,
        init_dist=init_dist,
        shape=(T_forecast, num_regions)
    )
    return latent_forecast

def conv1d_pytensor_causal_with_history(future, kernel, history_tail, lag):
    """
    future:  (Tf, R)  latent for forecast horizon
    kernel:  (L,)     causal taps, already normalized
    history_tail: (L-1, R) last L-1 rows from TRAINING latent (time order)

    returns: (Tf, R) causal conv over [history_tail ; future], sliced to future
    """
    if lag > 1:
        full = pt.concatenate([history_tail, future], axis=0)   # (L-1+Tf, R)
    else:
        full = future

    T_full = full.shape[0]
    # sliding windows (causal): conv[t] = sum_i k[i] * full[t - i]
    windows = pt.stack([full[i:i + (T_full - lag + 1)] for i in range(lag)][::-1], axis=1)  # (T_full-L+1, L, R)
    k = kernel[:, None]                                  # (L,1)
    conv_full = pt.sum(windows * k[None, :, :], axis=1)  # (T_full-L+1, R)

    Tf = future.shape[0]
    return conv_full[-Tf:, :]                            # forecast slice (Tf, R)


def forecasting_cases(
    lag,
    population,
    latent,
    alpha,
    history_tail,
    kernel,
    suffix,
    tests_per_capita=None,
    observed=None,
    case_signal_center=None,
):
    cases_signal = conv1d_pytensor_causal_with_history(latent, kernel, history_tail, lag)
    if case_signal_center is not None:
        cases_signal_for_likelihood = cases_signal - case_signal_center[None, :]
    else:
        cases_signal_for_likelihood = cases_signal
    # Previous non-centered version:
    # log_mu_c = cases_signal + alpha + pm.math.log(population[None,:])
    log_mu_c = cases_signal_for_likelihood + alpha + pm.math.log(population[None,:])

    if tests_per_capita is not None:
        beta = pm.Flat(f"beta_tests_{suffix}")
        mu_c = pm.Deterministic(f"reporting_rate_{suffix}", pm.math.exp(log_mu_c + beta*pm.math.log(tests_per_capita) ) )
    else:
        mu_c = pm.Deterministic(f"reporting_rate_{suffix}", pm.math.exp(log_mu_c) )
        print("No test seeking behavior adjustment")

    kwargs = dict(mu=mu_c)
    if observed is not None:
        kwargs["observed"] = observed

    return pm.Poisson(f"forecast_observed_cases_{suffix}", **kwargs)

def forecasting_wastewater(lag, latent, arrival_rate, sigma_rw, history_tail, kernel, suffix, observed = None):
    wastewater_conv =  conv1d_pytensor_causal_with_history(latent, kernel, history_tail, lag)
    mu_w = arrival_rate[None,:]*wastewater_conv
    kwargs = dict(mu=mu_w, sigma=sigma_rw)
    if observed is not None:
        kwargs["observed"] = observed

    return pm.Normal(f"forecast_observed_wastewater_{suffix}", **kwargs)

def ed_forecast(
    T_forecast,
    T_train,
    population,
    num_regions,
    nu,
    latent_dict,
    history_tail,
    lag,
    shape,
    scale,
    delay_ed,
    diseases,
    ed_signal_center=None,
):
    latent_x = pm.Flat(f"latent_ed", shape=(num_regions, T_train))
    x_last = latent_x[:,-1]

    sigma_rw = pm.Flat("sigma_ed") # or 0.03-0.06
    latent_forecast_ed = pm.GaussianRandomWalk(
        "latent_forecast_ed", mu=0.0, sigma=sigma_rw,
        init_dist=pm.DiracDelta.dist(x_last, shape = (num_regions,)),
        shape=(num_regions, T_forecast),
    )
    # Previous static residual forecast:
    # log_residual_rate = pm.Flat("log_residual_rate", shape=(num_regions,))
    # residual_rate = pm.Deterministic(
    #     "ed_contribution_residual",
    #     pm.math.exp(log_residual_rate)[None, :],
    # )

    ed_kernel = {disease: gamma_kernel(lag[disease], 
                                       shape[disease], 
                                       scale[disease], 
                                       delay_ed[disease]) for disease in diseases}
    disease_contributions = {}
    for disease in diseases:
        convolved =  conv1d_pytensor_causal_with_history(latent_dict[disease], ed_kernel[disease], history_tail[disease], lag[disease])
        if ed_signal_center is not None and ed_signal_center.get(disease) is not None:
            convolved_for_likelihood = convolved - ed_signal_center[disease][None, :]
        else:
            convolved_for_likelihood = convolved
        # Previous non-centered version:
        # contribution = pm.math.exp(nu[disease][None,:] + convolved)
        contribution = pm.math.exp(nu[disease][None,:] + convolved_for_likelihood)
        disease_contributions[disease] = pm.Deterministic(f"ed_contribution_{disease}", contribution)
    convolved_list = pt.stack(list(disease_contributions.values()), axis=0)
    convolved_sum = pm.math.sum(convolved_list, axis=0)

    ed_rate_per_capita = convolved_sum + pm.math.exp(latent_forecast_ed.T)
    disease_contributions["residual"] = pm.Deterministic("ed_contribution_residual", pm.math.exp(latent_forecast_ed.T))
    # Previous static residual forecast:
    # ed_rate_per_capita = convolved_sum + residual_rate
    # disease_contributions["residual"] = residual_rate
    return pm.Poisson("forecast_ed_visits",  mu = ed_rate_per_capita*population[None,:]) 

def adding_disease_forecast(pop, num_regions, T_train, T_forecast, latent_disease_dict,
                            alpha, offset, arrival_rate,
                            lag_reporting, shape_reporting, scale_reporting,
                            lag_ww, shape_ww, scale_ww, latent_dict, suffix,
                            tests_per_capita=None, case_signal_center=None):
    # Parameters
    sigma_rw = pm.Flat(f"sigma_wastewater_{suffix}")
    latent_disease = pm.Flat(f"latent_{suffix}", shape=(T_train, num_regions))
    latent_disease_dict[suffix] = latent_disease[-(lag_reporting-1):, :]

    # Kernels
    kernel_c = gamma_kernel(lag_reporting, shape = shape_reporting, scale = scale_reporting, delay=offset)
    kernel_w = gamma_kernel(lag_ww, shape = shape_ww, scale = scale_ww, delay=0)

    # Latent process and likelihoods
    latent_forecast = sampling_latent_forecast(num_regions, T_train, T_forecast, latent_disease, suffix)

    latent_dict[suffix] = latent_forecast

    forecasting_cases(
        lag_reporting,
        pop,
        latent_forecast,
        alpha,
        latent_disease_dict[suffix],
        kernel_c,
        suffix=f"{suffix}",
        tests_per_capita=tests_per_capita,
        case_signal_center=case_signal_center,
    )
    forecasting_wastewater(lag_ww, latent_forecast, arrival_rate, sigma_rw,latent_disease_dict[suffix], kernel_w, suffix=f"{suffix}")


    return latent_dict

def build_forecast_model(diseases, 
                         trace_ed, 
                         df_test, 
                         population, 
                         num_regions, 
                         cols: ColumnSpec,
                         tests_per_capita = None,
                         ):
    latent_dict = {}
    latent_disease_dict={}
    pivot_dfs, y_cases_all, log_y_signals_all = {}, {}, {}
    lag_ww = {'covid': 12, 'rsv': 12, "flua": 10}
    shape_ww = {'covid': 4.0, "rsv": 4.0, "flua":2.5} #mode:(5.0-1)*1.0=4.0, (4.0-1)*2.3=6.9, (2.5-1)*1.5=2.25
    scale_ww = {'covid': 1.3, "rsv": 1.2, "flua":1.5}

    lag_reporting = {'covid': 12, 'rsv': 12, 'flua': 10}
    shape_reporting = {'covid': 6.0, 'rsv':   6.0, 'flua':  4.0}  #mode:(6.0-1)*0.80=4.0, (5.0-1)*1.725=6.9, (4.0-1)*0.75=2.25
    scale_reporting = {'covid': 0.80, 'rsv':   0.72, 'flua':  0.75}         

    lag_ed   = {'covid': 12, 'rsv': 12, 'flua': 10}
    shape_ed = {'covid': 6.0, 'rsv':   6.0,  'flua':  4.0}  #mode:(6.0-1)*0.80=4.0, (5.0-1)*1.725=6.9, (4.0-1)*0.75=2.25
    scale_ed = {'covid': 0.80,  'rsv':   0.72, 'flua':  0.75}       

    arrival_rate_free = {d:None for d in diseases}
    arrival_rate = {d:None for d in diseases}
    alpha = {d:None for d in diseases}
    offset_ed = {d:None for d in diseases}
    nu = {d:None for d in diseases}
    case_signal_center = {d: None for d in diseases}
    ed_signal_center = {d: None for d in diseases}
    var_names = []
    T_train = trace_ed.posterior["latent_ed"].shape[-1]
    # Previous static residual forecast expected no latent_ed in the fitted trace:
    # T_train = trace_ed.posterior[f"latent_{diseases[0]}"].shape[-2]
    for disease in diseases:
        print(f"Adding model for {disease}")
        y_cases, log_y, pivot= getting_cases_and_ww_logged(df_test, disease, cols)
        pivot_dfs[disease] = pivot
        y_cases_all[disease] = y_cases
        log_y_signals_all[disease] = log_y
        T_forecast = pivot.shape[0]
        arrival_rate_free[disease] = pm.Flat(f"arrival_rate_free_{disease}", shape=(num_regions-1,))
        arrival_rate[disease] = pt.concatenate([pt.constant([1.0]), arrival_rate_free[disease]])
        alpha[disease] = pm.Flat(f"alpha_{disease}")
        case_center_name = f"case_signal_center_{disease}"
        if case_center_name in trace_ed.posterior:
            case_signal_center[disease] = pm.Flat(case_center_name, shape=(num_regions,))
        offset_name = f"offset_ed_{disease}"
        if offset_name in trace_ed.posterior:
            offset_ed[disease] = pm.Flat(offset_name)
        else:
            offset_ed[disease] = 0

        nu[disease] = pm.Flat(f"nu_{disease}", shape=(num_regions,))
        ed_center_name = f"ed_signal_center_{disease}"
        if ed_center_name in trace_ed.posterior:
            ed_signal_center[disease] = pm.Flat(ed_center_name, shape=(num_regions,))
        latent_dict= adding_disease_forecast(population, num_regions, T_train, T_forecast, latent_disease_dict,
                                            alpha[disease], offset_ed[disease], arrival_rate[disease],
                                            lag_reporting[disease], shape_reporting[disease], scale_reporting[disease],
                                            lag_ww[disease], shape_ww[disease], scale_ww[disease], latent_dict, disease,
                                            tests_per_capita=tests_per_capita,
                                            case_signal_center=case_signal_center[disease])
        var_names += [f"forecast_observed_cases_{disease}",
                      f"forecast_observed_wastewater_{disease}",
                      f"latent_forecast_{disease}",]
    var_names += ["forecast_ed_visits", "latent_forecast_ed"]
    # Previous static residual forecast returned only forecast_ed_visits:
    # var_names += ["forecast_ed_visits"]
    ed_forecast(T_forecast, T_train, population, num_regions, nu, 
                latent_dict,  latent_disease_dict, lag_ed, shape_ed, scale_ed, offset_ed, diseases,
                ed_signal_center=ed_signal_center)
    return var_names
