<p align="center">
  <img src="assets/logo.png" alt="Tarnished-WW logo" width="160">
</p>

<h1 align="center">TARNISHED-WW</h1>

<p align="center">
  Time-series Analysis of Random Walkers for Infection Surveillance using Wastewater and Hospital ED Visits
</p>

<p align="center">
  Joint Bayesian model for cases, wastewater viral load, and ED visits.
</p>

## TARnISHED-WW

Bayesian joint modeling of wastewater, reported cases, and ED visits across multiple regions and respiratory viral infections - RSV, Influenza A and Covid-19.` tarnished-ww` is built on **PyMC** and is designed for public health surveillance + forecasting workflows.

## Project Organization

```         
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data               <- The original, raw data and processed data
│
├── models             <- model results, model checkpoints, or model summaries
│
├── notebooks          <- Main pipeline of models.
│
├── pyproject.toml     <- Project configuration file with package metadata for TARnSIHED_WW and configuration for tools like black
|
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── paper            <- Generated manuscript as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src   <- Source code for use in this project.
    │
    └── tarnished_ww
        ├── __init__.py             <- Makes TARnISHED_WW a Python module
        │
        ├── api.py                  <- Wrappers to fit model and predict using posteriors
        │
        ├── io.py                   <- Functions to process data
        │
        ├── schemas.py              <- Stores columns names specifications for input data
        │       
        ├── config.py               <- Store useful variables and configuration
        │
        ├── build_functions.py              <- Code to create bayesian modules for TARnISHED_WW
        │
        ├── helper_functions.py             <- Code to create features, metrics and others for models
        │
        ├── forecast_functions.py        <- Code to create TARnISHED forecasts
        │      
        ├── plots_functions.py                <- Code to create visualizations
        │
        └── training.py                <- Code to train XGBoost or other ML models
```

------------------------------------------------------------------------

## Description

This repository contains the code and resources for the TARnISHED-WW package. This package is designed to analyze time-series data of infections surveillance, such as reported cases, test positivity rate and wastewater viral load of major respiratory infections -  COVID-19, Influenza A, RSv - and respiratory emergency department (ED) visits using a hierarchical Bayesian framework.

## Project status

In progress: 90% complete, missing documentation and wrapper functions for some modules.

**Project Progress: 90%**

🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜  90%

## Install

### From PyPI (still awaiting submission)

```bash
pip install tarnished-ww
```

### From GitHub 

```bash
pip install git+https://github.com/<your-org-or-user>/tarnished-ww.git
```

or

```bash
pip install git+https://github.com/<your-org-or-user>/tarnished-ww.git@main
# or
pip install git+https://github.com/<your-org-or-user>/tarnished-ww.git@v0.1.0
```

## Quick start

This tutorial simulates synthetic data and runs a smoke test. 

### Simulating synthetic data and fitting tarnished-ww

```python
import pandas as pd
from tarnished_ww import fit_joint_model, predict_joint_model
from tarnished_ww import ColumnSpec, PopulationSpec, SamplerSpec
from tarnished_ww.simulate import simulate_synthetic

# ----------------------------
# 1) Simulate example data
# ----------------------------
df, pop = simulate_synthetic(n_regions=3, n_days=60, seed=1)

# ----------------------------
# 2) Define column mappings
# ----------------------------
cols = ColumnSpec(region="wwtp_id")
pop_cols = PopulationSpec(region="wwtp_id")

# ----------------------------
# 3) Sampler configuration (smoke test)
# ----------------------------
sampler = SamplerSpec(
    draws=100,
    tune=100,
    chains=4,
    cores=4,
    target_accept=0.9,
    random_seed=123,
    extra={
        "progressbar": True,
        "compute_convergence_checks": False,  # faster for smoke test
    }
)

# ----------------------------
# 4) Fit model
# ----------------------------
res = fit_joint_model(
    df,
    pop,
    diseases=["covid", "rsv", "flua"],
    cols=cols,
    pop_cols=pop_cols,
    sampler=sampler,
)

print("✅ Model built and sampled.")
```

### Forecasting

```python
forecast = predict_joint_model(res)
print(forecast.head())
```


### Inputs

* df (observations)

A long-form dataframe containing:

1. region identifier - `wwtp_id`
2. date/time index - `surveillance_date`
3. disease-specific observed signals
   1. cases - `total_cases_{disease}`
   2. wastewater viral load - `load_trillion_{disease}`
   3. ED visits - `total_ed_visits`
   4. Total respiratory tests - `total_tests`

* pop (population)
  
A dataframe with:

1. region identifier - `wwtp_id`
2. population size - `population`
