import pymc as pm
import numpy as np
import pytensor.tensor as pt
from .io import getting_cases_and_ww_logged
from .schemas import ColumnSpec

def mask_convolution_boundary(observed, lag, left_censored=None):
    """Mask the leading causal-convolution boundary where pre-history is unknown."""
    if observed is None or lag <= 1:
        return observed, left_censored

    observed_masked = np.ma.array(observed, copy=True)
    observed_masked[: lag - 1, ...] = np.ma.masked

    if left_censored is None:
        return observed_masked, left_censored

    left_censored_masked = np.array(left_censored, copy=True)
    left_censored_masked[: lag - 1, ...] = False
    return observed_masked, left_censored_masked

def conv1d_pytensor_causal(signal, kernel, lag, mode="same"):
    """
    Causal temporal conv: conv[t] = sum_{i=0..lag-1} kernel[i] * signal[t-i]
    signal: (T, R)
    kernel: (lag,)
    returns (T, R) for mode='same'
    """
    if mode != "same":
        raise NotImplementedError("Causal version implemented for mode='same' only.")

    T = signal.shape[0]
    # Left-pad with (lag-1) zeros in time; no right pad
    pad_width = pt.stack([pt.stack([lag-1, 0]), pt.stack([0, 0])])
    padded = pt.pad(signal, pad_width, mode="constant")

    # Sliding windows: for each t in [0..T-1], window = padded[t : t+lag]
    # shape: (T, lag, R)
    windows = pt.stack([padded[i:i+T] for i in range(lag)][::-1], axis=1)
    # Explanation: reversing the stack is equivalent to reversing kernel later

    # Use kernel as-is (no need to flip now because we reversed windows)
    k = kernel[:, None]   # (lag, 1)
    conv = pt.sum(windows * k[None, :, :], axis=1)  # (T, R)
    return conv

def gamma_kernel(lag, shape, scale, delay):
    i = pt.arange(lag)
    shifted_i = i - delay
    #gamma is only defined for shifted_i >= 0
    mask = pt.ge(shifted_i, 0)
    # Evaluate gamma pdf only where valid, set zero elsewhere
    valid_kernel = (shifted_i**(shape - 1) * pt.exp(-shifted_i / scale) * mask)
    kernel = valid_kernel / pt.sum(valid_kernel)

    return kernel

def build_initial_latent_mean(
    y_cases,
    log_y_signal_masked,
    population,
    init_window=12,
    wastewater_weight=0.7,
):
    """Blend early wastewater and case signal into a disease-specific latent start."""
    init_window = max(1, min(init_window, y_cases.shape[0]))

    early_cases = np.asarray(y_cases[:init_window, :], dtype=float)
    early_case_rate = np.log1p(np.maximum(early_cases, 0.0) / population[None, :])
    case_summary = np.nanmedian(early_case_rate, axis=0)

    early_waste = np.ma.array(log_y_signal_masked[:init_window, :], copy=False)
    waste_summary = np.ma.median(early_waste, axis=0).filled(np.nan)

    if np.all(np.isnan(waste_summary)):
        return case_summary

    waste_center = np.nanmedian(waste_summary)
    case_center = np.nanmedian(case_summary)
    case_on_waste_scale = case_summary + (waste_center - case_center)

    blended = np.where(
        np.isnan(waste_summary),
        case_on_waste_scale,
        wastewater_weight * waste_summary + (1.0 - wastewater_weight) * case_on_waste_scale,
    )

    fallback = np.nanmedian(case_on_waste_scale)
    return np.where(np.isnan(blended), fallback, blended)

def sampling_latent_process(T, num_regions, eta, sd_latent, suffix=None, init_mean=None):
    # Priors on Gaussian random walks
    chol = sampling_covariance_matrix(num_regions, eta, sd_latent, suffix)
    if init_mean is None:
        init_mean = np.zeros(num_regions)
    init_mean = np.asarray(init_mean, dtype=float)
    # Previous generic start:
    # unamed_dist = pm.MvNormal.dist(mu=np.zeros(num_regions), cov=0.1*np.eye(num_regions))
    unamed_dist =  pm.MvNormal.dist(mu=init_mean, cov=0.1*np.eye(num_regions))
    latent = pm.MvGaussianRandomWalk(f"latent_{suffix}", 
                                    mu = np.zeros(num_regions), 
                                    chol=chol, 
                                    init_dist = unamed_dist, 
                                    shape=(T, num_regions))

    return latent

def sampling_covariance_matrix(num_regions, eta, sd_latent, suffix):
    # Hyperpriors on Cholesky matrices
    chol,_,_ = pm.LKJCholeskyCov(f"chol_cov_{suffix}", 
                                 n=num_regions, 
                                 eta=eta, 
                                 sd_dist=pm.HalfNormal.dist(sd_latent), 
                                 compute_corr=True, store_in_trace=True)
    return chol

def applying_convolution(lag, kernel, latent, suffix="", forecasting = False):
    if forecasting:
        convolved = conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag, mode='same')
        return convolved
    else:
        convolved = conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag,mode='same')
        return pm.Deterministic(f"conv_{suffix}", convolved)

def case_likelihood(
    population,
    latent,
    alpha,
    lag,
    kernel,
    suffix,
    tests_per_capita=None,
    observed=None,
    center_signal=True,
):
    cases_signal =  conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag, mode='same')
    if center_signal:
        cases_signal_center = pm.Deterministic(
            f"case_signal_center_{suffix}",
            pt.mean(cases_signal, axis=0),
        )
        cases_signal_for_likelihood = cases_signal - cases_signal_center[None, :]
    else:
        cases_signal_for_likelihood = cases_signal

    # Previous non-centered version:
    # log_mu_c = cases_signal + alpha + pm.math.log(population[None,:])
    log_mu_c = cases_signal_for_likelihood + alpha + pm.math.log(population[None,:])
    
    if tests_per_capita is not None:
        beta = pm.HalfNormal(f"beta_tests_{suffix}", 0.5)
        mu_c = pm.Deterministic(f"reporting_rate_{suffix}", pm.math.exp(log_mu_c + beta*pm.math.log(tests_per_capita) ) )
    else:
        mu_c = pm.Deterministic(f"reporting_rate_{suffix}", pm.math.exp(log_mu_c) )
        print("No test seeking behavior adjustment")

    kwargs = dict(mu = mu_c)
    if observed is not None:
        kwargs["observed"] = observed

    return pm.Poisson(f"observed_cases_{suffix}", **kwargs)

def wastewater_likelihood(
    latent,
    arrival_rate,
    sigma_rw,
    lag,
    kernel,
    suffix,
    observed=None,
    left_censored=None,
    mask_initial_boundary=True,
):
    wastewater_conv = conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag, mode='same')
    mu_w = pm.Deterministic(f"mu_waste_{suffix}", arrival_rate[None,:] * wastewater_conv)
    if mask_initial_boundary:
        observed, left_censored = mask_convolution_boundary(
            observed,
            lag,
            left_censored=left_censored,
        )
    kwargs = dict(mu=mu_w, sigma=sigma_rw)
    if observed is not None:
        kwargs["observed"] = observed

    if observed is not None and left_censored is not None and np.any(left_censored):
        censor_limit = np.ma.filled(observed, np.nan)
        censored_mask = left_censored & ~np.isnan(censor_limit)
        if np.any(censored_mask):
            upper = np.where(censored_mask, censor_limit, np.inf)
            return pm.Censored(
                f"observed_wastewater_{suffix}",
                pm.Normal.dist(mu=mu_w, sigma=sigma_rw),
                lower=None,
                upper=upper,
                observed=observed,
            )

    return pm.Normal(f"observed_wastewater_{suffix}", **kwargs)

def ed_likelihood(
        initial_log_rate,
        T,
        population,
        num_regions,
        latent_dict,
        nu,
        lag,
        shape,
        scale,
        delay_ed,
        diseases,
        observed=None,
        center_signal=True,
        sigma_ed_prior_scale=0.0005,
    ):
    sigma_rw = pm.HalfNormal("sigma_ed", sigma_ed_prior_scale)
    latent_ed = pm.GaussianRandomWalk(
        "latent_ed", mu=0.0, sigma=sigma_rw,
        init_dist=pm.Normal.dist(initial_log_rate, 0.05, shape=(num_regions,)),
        shape=(num_regions, T),
    )
    ed_kernel = {disease: gamma_kernel(lag[disease], 
                                       shape[disease], 
                                       scale[disease], 
                                       delay_ed[disease]) for disease in diseases}
    disease_contributions = {}
    for disease in diseases:
        convolved = applying_convolution(lag[disease],
                                        ed_kernel[disease], 
                                        latent_dict[disease], 
                                        forecasting=True)
        if center_signal:
            ed_signal_center = pm.Deterministic(
                f"ed_signal_center_{disease}",
                pt.mean(convolved, axis=0),
            )
            convolved_for_likelihood = convolved - ed_signal_center[None, :]
        else:
            convolved_for_likelihood = convolved
        contribution = pm.math.exp(nu[disease][None,:] + convolved_for_likelihood)
        disease_contributions[disease] = pm.Deterministic(f"ed_contribution_{disease}", contribution)
    convolved_list = pt.stack(list(disease_contributions.values()), axis=0)
    convolved_sum = pm.math.sum(convolved_list, axis=0)
    disease_contributions["residual"] = pm.Deterministic("ed_contribution_residual", pm.math.exp(latent_ed.T))
    ed_rate_per_capita = convolved_sum + pm.math.exp(latent_ed.T) #(alpha_resid[None, :] + 
    return pm.Poisson("ed_visits",  mu = ed_rate_per_capita*population[None,:], observed = observed) #ed_rate_per_capita*population[None,:]

def adding_disease_model(y_cases, log_y_signal_masked, pop, T, num_regions,
                         alpha, offset, arrival_rate, eta, sd_latent,
                         lag_reporting, shape_reporting, scale_reporting,
                         lag_ww, shape_ww, scale_ww, latent_dict, suffix,
                         tests_per_capita=None, ww_left_censored=None,
                         center_case_signal=True):
    # Parameters
    sigma_rw = pm.HalfNormal(f"sigma_wastewater_{suffix}", sigma=0.5)

    # Kernels
    kernel_c = gamma_kernel(lag_reporting, shape = shape_reporting, scale = scale_reporting, delay=offset)
    kernel_w = gamma_kernel(lag_ww, shape = shape_ww, scale = scale_ww, delay=0)

    # Latent process and likelihoods
    init_mean = build_initial_latent_mean(y_cases, log_y_signal_masked, pop)
    latent = sampling_latent_process(
        T,
        num_regions,
        eta,
        sd_latent,
        suffix=suffix,
        init_mean=init_mean,
    )
    latent_dict[suffix] = latent

    case_likelihood(
        pop,
        latent[:T],
        alpha,
        lag_reporting,
        kernel_c,
        suffix=suffix,
        observed=y_cases,
        tests_per_capita=tests_per_capita,
        center_signal=center_case_signal,
    )
    
    wastewater_likelihood(
        latent[:T],
        arrival_rate,
        sigma_rw,
        lag_ww,
        kernel_w,
        suffix=suffix,
        observed=log_y_signal_masked,
        left_censored=ww_left_censored,
    )

    return latent_dict

def build_joint_model(
    diseases,
    df_train,
    y_ed,
    population,
    num_regions,
    cols: ColumnSpec,
    tests_per_capita=None,
    center_case_signal=True,
    center_ed_signal=True,
    sigma_ed_prior_scale=0.0005,
):
    latent_dict = {}
    pivot_dfs, y_cases_all, log_y_signals_all = {}, {}, {}
    lag_ww = {'covid': 12, 'rsv': 12, "flua": 10}
    shape_ww = {'covid': 4.0, "rsv": 4.0, "flua":2.5} #mode:(4.0-1)*1.3=3.9, (4.0-1)*1.2=3.6, (2.5-1)*1.5=2.25
    scale_ww = {'covid': 1.3, "rsv": 1.2, "flua":1.5}

    lag_reporting = {'covid': 12, 'rsv': 12, 'flua': 10}
    shape_reporting = {'covid': 6.0, 'rsv':   6.0, 'flua':  4.0}  #mode:(6.0-1)*0.80=4.0, (6.0-1)*0.72=3.6, (4.0-1)*0.75=2.25
    scale_reporting = {'covid': 0.80, 'rsv':   0.72, 'flua':  0.75}         

    lag_ed   = {'covid': 12, 'rsv': 12, 'flua': 10}
    shape_ed = {'covid': 6.0, 'rsv':   6.0,  'flua':  4.0}  #mode:(6.0-1)*0.80=4.0, (6.0-1)*0.72=3.6, (4.0-1)*0.75=2.25
    scale_ed = {'covid': 0.80,  'rsv':   0.72, 'flua':  0.75}       

    arrival_rate_free = {d:None for d in diseases}
    arrival_rate = {d:None for d in diseases}
    alpha = {d:None for d in diseases}
    offset_ed = {d:None for d in diseases}
    nu = {d:None for d in diseases}
    sd_latent = {d:None for d in diseases}
    eta = 15.0
    alpha_prior = {
        "covid": -10,
        "rsv": -7, 
        "flua": -5,
    }
    nu_prior = {
        "covid": -16,
        "rsv": -14,
        "flua": -12,
    }
    nu_prior_sd = {
        "covid": 0.5,
        "rsv": 0.5,
        "flua": 1,
    }
    nu_sigma_sd = {
        "covid": 0.1,
        "rsv": 0.2,
        "flua": 1,
    }
    sd_latent_prior = {
        "covid": 0.01,
        "rsv": 0.01,
        "flua": 0.005,
    }
    typical_daily_ed_visits = np.nanmedian(y_ed, axis=0)
    initial_rate = typical_daily_ed_visits / population
    initial_log_rate = np.log(initial_rate)

    for disease in diseases:
        print(f"Adding model for {disease}")
        y_cases, log_y, ww_left_censored, pivot = getting_cases_and_ww_logged(
            df_train,
            disease,
            cols,
            return_censoring=True,
        )
        pivot_dfs[disease] = pivot
        y_cases_all[disease] = y_cases
        log_y_signals_all[disease] = log_y
        T = pivot.shape[0]
        arrival_rate_free[disease] = pm.HalfNormal(f"arrival_rate_free_{disease}", 1.0, shape=(num_regions-1,))
        arrival_rate[disease] = pt.concatenate([pt.constant([1.0]), arrival_rate_free[disease]])

        alpha[disease] = pm.Normal(f"alpha_{disease}", mu = alpha_prior[disease], sigma=1.0)
        offset_ed[disease] = pm.HalfNormal(f"offset_ed_{disease}", sigma=1) #forcing positive
        sd_latent[disease]= pm.HalfNormal(f"sd_latent_{disease}", sigma=sd_latent_prior[disease])#0.01)  # small global scale

        nu_mu = pm.Normal(f"nu_mu_{disease}", mu=nu_prior[disease], sigma=nu_prior_sd[disease])
        nu_sigma = pm.HalfNormal(f"nu_sigma_{disease}", nu_sigma_sd[disease])
        nu_offset = pm.Normal(f"nu_offset_{disease}", mu=0, sigma=1, shape=(num_regions,))
        nu[disease] = pm.Deterministic(f"nu_{disease}", nu_mu + nu_offset * nu_sigma)

        latent_dict = adding_disease_model(y_cases, log_y, population, T, num_regions, alpha[disease], 
                                           offset_ed[disease], arrival_rate[disease], eta, sd_latent[disease],
                                           lag_reporting[disease], shape_reporting[disease], scale_reporting[disease],
                                           lag_ww[disease], shape_ww[disease], scale_ww[disease], latent_dict, disease,
                                           tests_per_capita=tests_per_capita,
                                           ww_left_censored=ww_left_censored,
                                           center_case_signal=center_case_signal)
    ed_likelihood(initial_log_rate, T, population, num_regions, latent_dict, nu, lag_ed, shape_ed, scale_ed, 
                  offset_ed, diseases, observed = y_ed, center_signal=center_ed_signal,
                  sigma_ed_prior_scale=sigma_ed_prior_scale)
