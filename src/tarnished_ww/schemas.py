
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
