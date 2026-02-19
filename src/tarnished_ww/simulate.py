# src/tarnished_ww/simulate.py
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .schemas import ColumnSpec, PopulationSpec

def make_equicorr_sigma(step_sds: np.ndarray, rho: float) -> np.ndarray:
    """
    Sigma = diag(step_sds) @ Corr @ diag(step_sds)
    Corr has 1 on diagonal and rho off-diagonal (equicorrelation).
    """
    R = len(step_sds)
    if not (-1.0 / (R - 1) < rho < 1.0):
        raise ValueError(f"rho must be in (-1/(R-1), 1). Got {rho} for R={R}")
    corr = np.full((R, R), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    D = np.diag(step_sds)
    return D @ corr @ D


def simulate_mvg_rw(
    rng: np.random.Generator,
    n_steps: int,
    sigma: np.ndarray,          # (R, R)
    init: np.ndarray,           # (R,)
) -> np.ndarray:
    """
    Multivariate Gaussian random walk over regions.
    Returns Z with shape (n_steps, R).
    """
    R = sigma.shape[0]
    L = np.linalg.cholesky(sigma)  # assumes PSD
    Z = np.empty((n_steps, R), dtype=float)
    Z[0] = init

    for t in range(1, n_steps):
        eps = L @ rng.normal(size=R)
        Z[t] = Z[t - 1] + eps
    return Z

def simulate_synthetic(
    diseases: Sequence[str] = ("covid", "rsv", "flua"),
    n_regions: int = 3,
    n_days: int = 90,
    start_date: str = "2024-01-01",
    seed: int = 123,
    # population & scaling
    population_range: Tuple[int, int] = (50_000, 300_000),
    tests_per_1k_mean: float = 2.0,     # average tests per 1k per day
    tests_overdispersion: float = 0.2,  # multiplicative noise on tests
    # latent process
    disease_baseline: Optional[Dict[str, float]] = None,  # log intensity baseline per disease
    region_effect_sd: float = 0.3,      # region-level offset sd on log scale
    # observation models
    case_noise_scale: float = 1.0,      # multiplier on Poisson mean (keep 1.0 unless you want bias)
    ww_noise_sd: float = 0.35,          # noise on log wastewater
    ww_scale: float = 1.0,              # scales wastewater relative to exp(latent)
    # ED visits
    include_ed: bool = True,
    ed_baseline_per_100k: float = 20.0,  # baseline ED visits per 100k per day
    ed_from_disease_weight: Optional[Dict[str, float]] = None,  # how much each disease contributes
    # output schemas
    col: ColumnSpec = ColumnSpec(),
    pop: PopulationSpec = PopulationSpec(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a small synthetic dataset following a simplified generative story.

    Returns
    -------
    df : pd.DataFrame
        Main time-series table with one row per (date, region).
    pop_df : pd.DataFrame
        Population table (region -> population).
    """
    rng = np.random.default_rng(seed)

    if disease_baseline is None:
        # baselines on log scale (roughly sets typical case magnitudes)
        disease_baseline = {d: -6.5 for d in diseases}  # conservative default

    if ed_from_disease_weight is None:
        # how strongly each disease increases ED visits
        ed_from_disease_weight = {d: 0.15 for d in diseases}

    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # Regions + population
    region_ids = [i+1 for i in range(n_regions)]
    populations = rng.integers(population_range[0], population_range[1] + 1, size=n_regions)

    pop_df = pd.DataFrame(
        {
            pop.region: region_ids,
            pop.population: populations,
        }
    )

    # Build base frame (date x region)
    grid = pd.MultiIndex.from_product([dates, region_ids], names=[col.date, col.region]).to_frame(index=False)

    # Tests: driven by population with multiplicative lognormal-ish noise
    pop_map = dict(zip(region_ids, populations))
    grid[col.tests] = [
        max(
            0.0,
            (tests_per_1k_mean / 1000.0) * pop_map[r] * float(np.exp(rng.normal(0.0, tests_overdispersion)))
        )
        for r in grid[col.region].values
    ]
    # Convert tests to integer counts
    grid[col.tests] = rng.poisson(lam=np.maximum(grid[col.tests].values, 0.0)).astype(int)

    # Latent RW per (disease, region)
    latent = {d: {} for d in diseases}
    R = n_regions
    step_sds = rng.uniform(0.06, 0.15, size=R)     # region-specific RW step sd
    Sigma = make_equicorr_sigma(step_sds, rho=0.6) # correlated regions
    for d in diseases:
        init = disease_baseline[d] + rng.normal(0, 0.3, size=R)
        latent[d]= simulate_mvg_rw(rng, n_days, Sigma, init)  # shape (T, R)

    # Generate cases + wastewater
    for d in diseases:
        case_col = col.cases_tpl.format(disease=d)
        ww_col = col.wwload_tpl.format(disease=d)

        case_means = np.empty(len(grid), dtype=float)
        ww_log_means = np.empty(len(grid), dtype=float)

        # fill by region blocks
        idx = 0
        for r_idx, r in enumerate(region_ids):
            pop_r = pop_map[r]
            tests_r = grid.loc[grid[col.region] == r, col.tests].values.astype(float)
            z =latent[d][:, r_idx]

            # Simple reporting intensity:
            # - exp(z) is latent rate per person-day
            # - scale by population and tests (small elasticity)
            # adjust elasticity to mimic reporting rate.
            tests_elasticity = 0.2
            lam_cases = np.exp(z) * pop_r * (np.maximum(tests_r, 1.0) ** tests_elasticity)
            lam_cases = case_noise_scale * lam_cases

            case_means[idx: idx + n_days] = lam_cases
            ww_log_means[idx: idx + n_days] = np.log(np.exp(z) * pop_r * ww_scale + 1e-12)
            idx += n_days

        grid[case_col] = rng.poisson(lam=np.maximum(case_means, 0.0)).astype(int)
        # wastewater observed on log-scale with Normal noise; output in "trillion" units-ish
        ww_log_obs = ww_log_means + rng.normal(0.0, ww_noise_sd, size=len(grid))
        grid[ww_col] = np.exp(ww_log_obs)

    # ED visits: baseline + disease contributions
    if include_ed:
        lam_ed = np.zeros(len(grid), dtype=float)
        # baseline proportional to population
        lam_ed += [
            (ed_baseline_per_100k / 100_000.0) * pop_map[r]
            for r in grid[col.region].values
        ]

        # add contributions from disease cases (simple linear link)
        for d in diseases:
            w = ed_from_disease_weight.get(d, 0.0)
            case_col = col.cases_tpl.format(disease=d)
            lam_ed += w * grid[case_col].values.astype(float)

        grid[col.ed_visits] = rng.poisson(lam=np.maximum(lam_ed, 0.0)).astype(int)

    return grid, pop_df