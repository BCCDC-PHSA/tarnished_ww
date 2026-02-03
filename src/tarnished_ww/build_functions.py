import pymc as pm
import arviz as az
import numpy as np
import pytensor.tensor as pt

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

def sampling_latent_process(T, num_regions, eta, sd_latent, suffix=None):
    # Priors on Gaussian random walks
    chol = sampling_covariance_matrix(num_regions, eta, sd_latent, suffix)
    unamed_dist =  pm.MvNormal.dist(mu=np.zeros(num_regions), cov=0.1*np.eye(num_regions))
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

def case_likelihood(population, latent, alpha, lag, kernel, suffix, tests_per_capita = None, observed = None):
    cases_signal =  conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag, mode='same')

    log_mu_c = cases_signal + alpha + pm.math.log(population[None,:])
    
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

def wastewater_likelihood(latent, arrival_rate, sigma_rw, lag, kernel, suffix, observed = None):
    wastewater_conv = conv1d_pytensor_causal(signal=latent, kernel = kernel,lag = lag, mode='same')
    mu_w = pm.Deterministic(f"mu_waste_{suffix}", arrival_rate[None,:] * wastewater_conv)
    kwargs = dict(mu=mu_w, sigma=sigma_rw)
    if observed is not None:
        kwargs["observed"] = observed

    return pm.Normal(f"observed_wastewater_{suffix}", **kwargs)

def ed_likelihood(T, population, num_regions, latent_dict, nu, lag, shape, scale, delay_ed, diseases, observed = None):
    sigma_rw = pm.HalfNormal("sigma_ed", 0.0005) # or 0.03–0.06
    x0_sd = 0.5
    latent_ed = pm.GaussianRandomWalk(
        "latent_ed", mu=0.0, sigma=sigma_rw,
        init_dist=pm.Normal.dist(0.0, x0_sd, shape=(num_regions,)),
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
        #convolved = conv_raw - conv_raw.mean(axis=0, keepdims=True)
        contribution = pm.math.exp(nu[disease][None,:] + convolved)
        disease_contributions[disease] = pm.Deterministic(f"ed_contribution_{disease}", contribution)
    convolved_list = pt.stack(list(disease_contributions.values()), axis=0)
    convolved_sum = pm.math.sum(convolved_list, axis=0)
    disease_contributions["residual"] = pm.Deterministic("ed_contribution_residual", pm.math.exp(latent_ed.T))
    ed_rate_per_capita = convolved_sum + pm.math.exp(latent_ed.T) #(alpha_resid[None, :] + 
    return pm.Poisson("ed_visits",  mu = ed_rate_per_capita*population[None,:], observed = observed) #ed_rate_per_capita*population[None,:]

def getting_df_data_logged(df, suffix):
    # Prepare pivoted data
    pivot_df = df.pivot(index='surveillance_date', columns='wwtp', 
                        values=[f'total_cases_{suffix}', f'load_trillion_{suffix}'])
    pivot_df[f'total_cases_{suffix}'] = pivot_df[f'total_cases_{suffix}'].fillna(0)
    pivot_df = pivot_df.sort_index()
    y_cases = pivot_df[f'total_cases_{suffix}'].values
    y_signal = pivot_df[f'load_trillion_{suffix}'].values  # shape (T, R)
    y_signal_array = np.array(y_signal, dtype=np.float64)
    y_signal_array = np.where(np.isnan(y_signal_array), np.nan, y_signal_array + 1)
    log_y_signal = np.log(y_signal_array)
    log_y_signal_masked = np.ma.masked_invalid(log_y_signal)
    return y_cases, log_y_signal_masked, pivot_df

def getting_df_data(df, suffix):
    # Prepare pivoted data
    pivot_df = df.pivot(index='surveillance_date', columns='wwtp', 
                        values=[f'total_cases_{suffix}', f'load_trillion_{suffix}'])
    pivot_df[f'total_cases_{suffix}'] = pivot_df[f'total_cases_{suffix}'].fillna(0)
    pivot_df = pivot_df.sort_index()
    y_cases = pivot_df[f'total_cases_{suffix}'].values
    y_signal = pivot_df[f'load_trillion_{suffix}'].values  # shape (T, R)
    y_signal_array = np.array(y_signal, dtype=np.float64)
    return y_cases, y_signal_array, pivot_df

def adding_disease_model(y_cases, log_y_signal_masked, pop, T, num_regions,
                         alpha, offset, arrival_rate, eta, sd_latent,
                         lag_reporting, shape_reporting, scale_reporting,
                         lag_ww, shape_ww, scale_ww, latent_dict, suffix, tests_per_capita = None):
    # Parameters
    sigma_rw = pm.HalfNormal(f"sigma_wastewater_{suffix}", sigma=0.5)

    # Kernels
    kernel_c = gamma_kernel(lag_reporting, shape = shape_reporting, scale = scale_reporting, delay=offset)
    kernel_w = gamma_kernel(lag_ww, shape = shape_ww, scale = scale_ww, delay=0)

    # Latent process and likelihoods
    latent = sampling_latent_process(T, num_regions, eta, sd_latent, suffix=suffix)
    latent_dict[suffix] = latent

    case_likelihood(pop, latent[:T], alpha, lag_reporting, kernel_c, suffix=suffix, observed=y_cases, tests_per_capita  = tests_per_capita)
    
    wastewater_likelihood(latent[:T], arrival_rate, sigma_rw, lag_ww, kernel_w, suffix=suffix, observed=log_y_signal_masked)

    return latent_dict

def build_joint_model(diseases, df_train, y_ed, population, num_regions, tests_per_capita = None):
    latent_dict = {}
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
    eta = 15.0
    sd_latent = pm.HalfNormal(f"sd_latent", sigma=0.01)  # small global scale
    for disease in diseases:
        print(f"Adding model for {disease}")
        y_cases, log_y, pivot = getting_df_data_logged(df_train, disease)
        pivot_dfs[disease] = pivot
        y_cases_all[disease] = y_cases
        log_y_signals_all[disease] = log_y
        T = pivot.shape[0]
        arrival_rate_free[disease] = pm.HalfNormal(f"arrival_rate_free_{disease}", 1.0, shape=(num_regions-1,))
        arrival_rate[disease] = pt.concatenate([pt.constant([1.0]), arrival_rate_free[disease]])
        alpha[disease] = pm.Normal(f"alpha_{disease}", mu = -12, sigma=1.0)
        offset_ed[disease] = pm.HalfNormal(f"offset_ed_{disease}", sigma=1) #forcing positive
        if disease == "flua":
            nu_mu = pm.Normal(f"nu_mu_{disease}", mu=-12, sigma=1)
            nu_sigma = pm.HalfNormal(f"nu_sigma_{disease}", 1)
            nu[disease] = pm.Normal(f"nu_{disease}", mu=nu_mu, sigma=nu_sigma, shape=(num_regions,))
        else:
            nu[disease] =  pm.Normal(f"nu_{disease}", mu = -12, sigma=0.5, shape = (num_regions,))
        latent_dict = adding_disease_model(y_cases, log_y, population, T, num_regions, alpha[disease], 
                                           offset_ed[disease], arrival_rate[disease], eta, sd_latent,
                                           lag_reporting[disease], shape_reporting[disease], scale_reporting[disease],
                                           lag_ww[disease], shape_ww[disease], scale_ww[disease], latent_dict, disease,
                                           tests_per_capita  = tests_per_capita )
    ed_likelihood(T, population, num_regions, latent_dict, nu, lag_ed, shape_ed, scale_ed, 
                  offset_ed, diseases, observed = y_ed)
