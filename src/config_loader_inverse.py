"""
Configuration loader for Inverse PINN training.

Loads hyperparameters from YAML files for the inverse problem.
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch


class InversePINNConfig:
    """Configuration class for Inverse PINN training."""

    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.config_path = config_path
        self._validate_config()

    def _validate_config(self):
        """Validate that all required sections are present."""
        required_sections = [
            'data', 'soil', 'training', 'network',
            'sampling', 'optimization', 'inverse'
        ]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section '{section}' in config")

    # ========== Data Configuration ==========
    @property
    def flux_data_path(self) -> str:
        """Path to surface flux data file."""
        return self._config['data']['flux_path']

    @property
    def theta_obs_data_path(self) -> str:
        """Path to observed surface moisture data file."""
        return self._config['data']['theta_obs_path']

    @property
    def start_date(self) -> str:
        return self._config['data']['start_date']

    @property
    def end_date(self) -> str:
        return self._config['data']['end_date']

    @property
    def interpolate(self) -> bool:
        return self._config['data'].get('interpolate', True)

    @property
    def max_gap_hours(self) -> int:
        return self._config['data'].get('max_gap_hours', 6)

    # ========== Soil Parameters ==========
    @property
    def soil_params(self) -> Dict[str, float]:
        return self._config['soil']

    # ========== Training Configuration ==========
    @property
    def n_epochs(self) -> int:
        return int(self._config['training']['n_epochs'])

    @property
    def learning_rate(self) -> float:
        return float(self._config['training']['learning_rate'])

    @property
    def device(self) -> str:
        device_str = self._config['training'].get('device', 'auto')
        if device_str == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_str

    @property
    def seed(self) -> int:
        return self._config['training'].get('seed', 42)

    # ========== Network Configuration ==========
    @property
    def h_net_config(self) -> Dict[str, int]:
        return self._config['network']['h_net']

    # ========== Inverse Problem Configuration ==========
    @property
    def n_spline_knots(self) -> int:
        """Number of spline knots for water table parameterization."""
        return int(self._config['inverse']['n_spline_knots'])

    @property
    def lambda_smooth(self) -> float:
        """Smoothness regularization weight."""
        return float(self._config['inverse'].get('lambda_smooth', 1e-4))

    @property
    def lambda_curv(self) -> float:
        """Curvature regularization weight."""
        return float(self._config['inverse'].get('lambda_curv', 1e-6))

    @property
    def zb_initial(self) -> float:
        """Initial water table depth guess [m]."""
        return float(self._config['inverse']['zb_initial'])

    @property
    def zb_max(self) -> Optional[float]:
        """Maximum allowed water table depth [m], None for no constraint."""
        val = self._config['inverse'].get('zb_max', None)
        return float(val) if val is not None else None

    # ========== Sampling Configuration ==========
    @property
    def batch_size(self) -> int:
        return int(self._config['sampling']['batch_size'])

    @property
    def boundary_ratio(self) -> float:
        return float(self._config['sampling']['boundary_ratio'])

    @property
    def batch_size_bc(self) -> int:
        """Legacy: used as default for both flux and theta if not specified."""
        return int(self._config['sampling'].get('batch_size_bc', 100))

    @property
    def batch_size_flux(self) -> int:
        """Batch size for flux BC sampling (q_surf gradient-based)."""
        return int(self._config['sampling'].get('batch_size_flux',
                   self._config['sampling'].get('batch_size_bc', 500)))

    @property
    def batch_size_theta(self) -> int:
        """Batch size for theta data matching (theta gradient-based)."""
        return int(self._config['sampling'].get('batch_size_theta',
                   self._config['sampling'].get('batch_size_bc', 500)))

    @property
    def interp_ratio(self) -> float:
        return float(self._config['sampling'].get('interp_ratio', 0.80))

    @property
    def neighbor_ratio(self) -> float:
        return float(self._config['sampling'].get('neighbor_ratio', 0.05))

    @property
    def baseline_ratio(self) -> float:
        return float(self._config['sampling'].get('baseline_ratio', 0.15))

    @property
    def gradient_neighbor_expansion(self) -> int:
        return int(self._config['sampling'].get('neighbor_expansion', 2))

    @property
    def gradient_threshold(self) -> float:
        return float(self._config['sampling'].get('gradient_threshold', 0.7))

    @property
    def gradient_power(self) -> float:
        return float(self._config['sampling'].get('power', 2.0))

    # ========== Optimization Configuration ==========
    @property
    def weight_update_freq(self) -> float:
        return float(self._config['optimization']['weight_update_freq'])

    @property
    def weight_lr(self) -> float:
        return float(self._config['optimization'].get('weight_lr', 0.5))

    @property
    def use_initial_scales(self) -> bool:
        return self._config['optimization'].get('use_initial_scales', True)

    @property
    def initial_pde_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_pde_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_flux_bc_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_flux_bc_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_wt_head_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_wt_head_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_data_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_data_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_ic_h_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_ic_h_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_ic_zb_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_ic_zb_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_zb_smooth_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_zb_smooth_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_zb_curv_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_zb_curv_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_zb_max_depth_weight(self) -> Optional[float]:
        val = self._config['optimization'].get('initial_zb_max_depth_weight', None)
        return float(val) if val is not None else None

    @property
    def initial_zb_guide_weight(self) -> Optional[float]:
        """Physics-based z_b guidance weight"""
        val = self._config['optimization'].get('initial_zb_guide_weight', None)
        return float(val) if val is not None else None

    @property
    def use_amp(self) -> bool:
        return self._config['optimization'].get('use_amp', False)

    @property
    def use_multi_gpu(self) -> bool:
        return self._config['optimization'].get('use_multi_gpu', True)

    @property
    def grad_accumulation_steps(self) -> int:
        return int(self._config['optimization'].get('grad_accumulation_steps', 1))

    @property
    def switch_to_lbfgs_epoch(self) -> Optional[int]:
        epoch = self._config['optimization'].get('switch_to_lbfgs_epoch', None)
        return int(epoch) if epoch is not None else None

    @property
    def lbfgs_lr(self) -> float:
        return float(self._config['optimization'].get('lbfgs_lr', 1.0))

    @property
    def lbfgs_max_iter(self) -> int:
        return int(self._config['optimization'].get('lbfgs_max_iter', 20))

    @property
    def lbfgs_history_size(self) -> int:
        return int(self._config['optimization'].get('lbfgs_history_size', 50))

    @property
    def lbfgs_tolerance_grad(self) -> float:
        return float(self._config['optimization'].get('lbfgs_tolerance_grad', 1e-7))

    @property
    def lbfgs_tolerance_change(self) -> float:
        return float(self._config['optimization'].get('lbfgs_tolerance_change', 1e-9))

    @property
    def use_lbfgs(self) -> bool:
        return self._config['optimization'].get('use_lbfgs', False)

    @property
    def lbfgs_start_epoch(self) -> Optional[int]:
        val = self._config['optimization'].get('lbfgs_start_epoch', None)
        return int(val) if val is not None else None

    @property
    def lbfgs_n_epochs(self) -> int:
        return int(self._config['optimization'].get('lbfgs_n_epochs', 1000))

    # ========== Performance Optimization Configuration ==========
    @property
    def use_all_data_points(self) -> bool:
        """Use all BC data points directly instead of gradient-based sampling."""
        return self._config['optimization'].get('use_all_data_points', True)

    @property
    def use_torch_compile(self) -> bool:
        """Use torch.compile for model optimization (PyTorch 2.0+)."""
        return self._config['optimization'].get('use_torch_compile', False)

    @property
    def print_freq(self) -> int:
        """Frequency of printing progress (epochs)."""
        return int(self._config['optimization'].get('print_freq', 500))

    @property
    def bc_resample_freq(self) -> int:
        """Frequency of resampling BC points when using gradient-based sampling (epochs).
        Set to 0 to resample every epoch (original behavior).
        Higher values cache samples longer for better performance."""
        return int(self._config['optimization'].get('bc_resample_freq', 100))

    # ========== Physics Configuration ==========
    @property
    def zr(self) -> float:
        return float(self._config.get('physics', {}).get('zr', 0.5))

    @property
    def L(self) -> float:
        return float(self._config.get('physics', {}).get('L', 4.0))

    @property
    def S_max(self) -> float:
        return float(self._config.get('physics', {}).get('S_max', 1e-7))

    @property
    def ic_type(self) -> str:
        return self._config.get('physics', {}).get('ic_type', 'hydrostatic')

    # ========== Checkpointing Configuration ==========
    @property
    def checkpoint_dir(self) -> str:
        return self._config.get('checkpointing', {}).get('dir', 'checkpoints_inverse')

    @property
    def checkpoint_freq(self) -> Optional[int]:
        freq = self._config.get('checkpointing', {}).get('freq', None)
        return int(freq) if freq is not None else None

    @property
    def keep_last_n_checkpoints(self) -> Optional[int]:
        keep_n = self._config.get('checkpointing', {}).get('keep_last_n', 3)
        return int(keep_n) if keep_n is not None else None

    # ========== Output Configuration ==========
    @property
    def output_dir(self) -> str:
        return self._config.get('output', {}).get('dir', 'outputs_inverse')

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("INVERSE PINN CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Config file: {self.config_path}")
        print(f"\nData:")
        print(f"  Flux path: {self.flux_data_path}")
        print(f"  Theta obs path: {self.theta_obs_data_path}")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        print(f"\nInverse Problem:")
        print(f"  Spline knots: {self.n_spline_knots}")
        print(f"  Initial z_b: {self.zb_initial} m")
        print(f"  Lambda smooth: {self.lambda_smooth}")
        print(f"  Lambda curv: {self.lambda_curv}")
        print(f"\nTraining:")
        print(f"  Epochs: {self.n_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"\nNetwork:")
        print(f"  h_net: {self.h_net_config}")
        print(f"\nSampling:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Boundary ratio: {self.boundary_ratio}")
        print(f"  BC batch size: {self.batch_size_bc}")
        print(f"\nOptimization:")
        print(f"  Weight update freq: {self.weight_update_freq}")
        print(f"  Use AMP: {self.use_amp}")
        print(f"\nPhysics:")
        print(f"  zr: {self.zr}, L: {self.L}")
        print(f"  S_max: {self.S_max}")
        print(f"\nCheckpointing:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Frequency: {self.checkpoint_freq}")
        print("=" * 70)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()


def load_inverse_config(config_path: str) -> InversePINNConfig:
    """
    Convenience function to load inverse configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        InversePINNConfig object
    """
    return InversePINNConfig(config_path)
