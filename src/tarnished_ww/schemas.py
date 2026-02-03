
from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnSpec:
    date: str = "surveillance_date"
    region: str = "wwtp_id"              # user column
    region_internal: str = "wwtp"        # what your functions expect
    ed_visits: str = "total_ed_visits"
    tests: str = "total_tests_all_ages"
    cases_tpl: str = "total_cases_{disease}"
    wwload_tpl: str = "load_trillion_{disease}"

@dataclass(frozen=True)
class PopulationSpec:
    region: str = "wwtp_id"              # user column
    region_internal: str = "wwtp"        # what your functions expect
    population: str = "population"


from typing import Optional, Any, Dict

@dataclass(frozen=True)
class SamplerSpec:
    """
    Configuration for PyMC sampling.

    Notes
    -----
    - draws: posterior draws (samples kept after tuning)
    - tune: warmup/adaptation steps (usually not kept)
    """
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: Optional[int] = None
    target_accept: float = 0.9
    random_seed: int = 123

    # Optional: keep it flexible for advanced users
    # e.g., init="adapt_diag", progressbar=False, compute_convergence_checks=False, etc.
    extra: Optional[Dict[str, Any]] = None
