"""
Inverse PINN for Richards Equation

Given surface flux as BC and observed surface moisture,
find water table depth z_b(t) via spline optimization.
"""

from .normalization_helper import NormalizationHelper
from .gradient_based_sampling import gradient_based_sampling

from .pinn_inverse import RichardsInversePINN, WaterTableSpline, PressureHeadNet
from .train_inverse import train_inverse_pinn, train_inverse_pinn_from_config
from .training_inverse import (
    InverseSamplingHelpers,
    compute_inverse_losses,
    InverseWeightManager,
)
from .config_loader_inverse import InversePINNConfig, load_inverse_config
from .visualization_inverse import (
    plot_inverse_results,
    plot_inverse_training_losses,
    save_inverse_figures,
)
from .training_logger_inverse import InverseTrainingLogger

__all__ = [
    'NormalizationHelper',
    'gradient_based_sampling',
    'RichardsInversePINN',
    'WaterTableSpline',
    'PressureHeadNet',
    'train_inverse_pinn',
    'train_inverse_pinn_from_config',
    'InverseSamplingHelpers',
    'compute_inverse_losses',
    'InverseWeightManager',
    'InversePINNConfig',
    'load_inverse_config',
    'plot_inverse_results',
    'plot_inverse_training_losses',
    'save_inverse_figures',
    'InverseTrainingLogger',
]
