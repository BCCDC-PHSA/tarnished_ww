"""
tarnished_ww: TARnISHED wastewater + ED joint Bayesian model.
"""

from .schemas import ColumnSpec, PopulationSpec, SamplerSpec
from .api import fit_joint_model, predict_joint_model

__all__ = [
    "fit_joint_model",
    "predict_joint_model",
    "ColumnSpec",
    "SamplerSpec",
    "PopulationSpec",
    "__version__",
]

# Keep version in one place.
__version__ = "0.1.0"