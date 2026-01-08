"""
Inverse PINN for Richards Equation

Given:
    - Surface flux q(0,t) as Neumann BC
    - Observed surface soil moisture θ_obs(0,t) as fitting target

Find:
    - Water table depth z_b(t) via spline parameterization
    - Pressure head h(z,t) via neural network

Such that:
    - Richards equation is satisfied in the domain
    - h = 0 at water table (z = -z_b)
    - Predicted surface moisture matches observations

Coordinate convention:
    - z = 0 at surface
    - z < 0 below surface (z points upward)
    - z_b > 0 is water table depth (positive value)
    - Water table location: z = -z_b

Physics:
    - Richards equation: ∂S_e/∂t̃ + ∂q̃/∂z̃ + S̃ = 0
    - Darcy flux: q̃ = -K̃(∂h̃/∂z̃ + 1)
    - van Genuchten retention: S_e = [1 + (α̃|h̃|)^n]^(-m)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureHeadNet(nn.Module):
    """Neural network for dimensionless pressure head h̃(z̃,t̃,z̃_b)

    Modified for inverse problem: includes z_b as explicit input to enable
    gradient flow from surface moisture data loss to z_b parameters.
    """

    def __init__(self, hidden_dim, num_layers, t_ref_tilde=None, z_max_tilde=1.0,
                 include_zb_input=True):
        """
        Args:
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            t_ref_tilde: Fixed reference time for scaling (default: 15 days / T)
            z_max_tilde: Maximum expected dimensionless depth for scaling (typically 1.0)
            include_zb_input: If True, adds z_b as a 3rd input to the network.
                              This enables gradient flow to z_b from surface data loss.
        """
        super().__init__()

        # Use fixed reference time of 15 days if not specified
        if t_ref_tilde is None:
            t_ref_tilde = 15.0 * 86400  # 15 days in seconds, will be normalized by T

        # Store dimensionless normalization parameters for NN input scaling
        self.t_ref_tilde = t_ref_tilde
        self.z_max_tilde = z_max_tilde
        self.include_zb_input = include_zb_input

        # Input dimension: 2 (z, t) or 3 (z, t, zb)
        input_dim = 3 if include_zb_input else 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z_tilde, t_tilde, zb_tilde=None):
        """
        Args:
            z_tilde: Dimensionless spatial coordinate (typically in [-z_b_max/L, 0])
            t_tilde: Dimensionless time
            zb_tilde: Dimensionless water table depth (optional, for inverse problem)

        Returns:
            h_tilde: Dimensionless pressure head (O(1) values)
        """
        # Scale inputs using FIXED reference time (duration-independent)
        t_scaled = t_tilde / self.t_ref_tilde
        z_scaled = (z_tilde + self.z_max_tilde) / self.z_max_tilde  # [-z_max_tilde, 0] -> [0, 1]

        if self.include_zb_input and zb_tilde is not None:
            # Scale zb to ~[0, 1] range (assuming max zb ~ z_max_tilde)
            zb_scaled = zb_tilde / self.z_max_tilde
            inputs = torch.cat([z_scaled, t_scaled, zb_scaled], dim=1)
        else:
            inputs = torch.cat([z_scaled, t_scaled], dim=1)

        h_tilde = self.net(inputs)

        return h_tilde

    def forward_with_prescaled_z(self, z_scaled, t_tilde, zb_tilde=None):
        """
        Forward pass with pre-scaled z coordinate.

        Used when z is dynamically scaled relative to current z_b,
        so z_scaled is already in [0, 1] regardless of z_b value.

        CRITICAL: zb_tilde must be passed for inverse problem to enable
        gradient flow from surface data loss to z_b parameters.

        Args:
            z_scaled: Pre-scaled spatial coordinate in [0, 1]
                      (0 = water table, 1 = surface)
            t_tilde: Dimensionless time
            zb_tilde: Dimensionless water table depth (required for inverse)

        Returns:
            h_tilde: Dimensionless pressure head
        """
        t_scaled = t_tilde / self.t_ref_tilde

        if self.include_zb_input and zb_tilde is not None:
            # Scale zb to ~[0, 1] range
            zb_scaled = zb_tilde / self.z_max_tilde
            inputs = torch.cat([z_scaled, t_scaled, zb_scaled], dim=1)
        else:
            inputs = torch.cat([z_scaled, t_scaled], dim=1)

        h_tilde = self.net(inputs)

        return h_tilde


class WaterTableSpline(nn.Module):
    """
    Learnable spline parameterization for water table depth z_b(t).

    z_b(t) is parameterized as piecewise linear interpolation between
    learnable knot values at fixed time locations.

    z_b > 0 always (water table depth below surface is positive).
    """

    def __init__(self, n_knots, t_min, t_max, zb_initial, device='cpu', zb_max=None,
                 L=1.0, T=1.0):
        """
        Args:
            n_knots: Number of knot points for spline
            t_min: Minimum time (dimensional, seconds)
            t_max: Maximum time (dimensional, seconds)
            zb_initial: Initial water table depth (dimensional, meters)
            device: Torch device
            zb_max: Maximum allowed water table depth (meters), None for no constraint
            L: Length scale for non-dimensionalization [m]
            T: Time scale for non-dimensionalization [s]
        """
        super().__init__()

        self.n_knots = n_knots
        self.t_min = t_min
        self.t_max = t_max
        self.device = device
        self.zb_max = zb_max
        self.L = L  # Length scale for dimensionless regularization
        self.T = T  # Time scale for dimensionless regularization

        # Fixed knot times (evenly spaced)
        # Register as buffer so it's not a learnable parameter
        t_knots = torch.linspace(t_min, t_max, n_knots, device=device)
        self.register_buffer('t_knots', t_knots)

        # Learnable knot values for z_b (in log space to ensure z_b > 0)
        # Using log parameterization: z_b = exp(log_zb_knots) ensures z_b > 0
        if isinstance(zb_initial, (list, tuple, np.ndarray, torch.Tensor)):
            # Initialize from array of values (physics-based initialization)
            zb_init_array = torch.tensor(zb_initial, dtype=torch.float32, device=device)
            if len(zb_init_array) != n_knots:
                raise ValueError(f"zb_initial has {len(zb_init_array)} values but n_knots={n_knots}")
            zb_init_array = torch.clamp(zb_init_array, min=0.1)  # Ensure positive
            log_zb_init = torch.log(zb_init_array)
        else:
            # Initialize all knots to the same scalar value
            log_zb_init = torch.log(torch.tensor(zb_initial, device=device))
            log_zb_init = torch.full((n_knots,), log_zb_init.item(), device=device)
        self.log_zb_knots = nn.Parameter(log_zb_init)

    def forward(self, t):
        """
        Interpolate water table depth at given times.

        Args:
            t: Time tensor (dimensional, seconds) of shape (N, 1) or (N,)

        Returns:
            zb: Water table depth (dimensional, meters) of shape (N, 1), always > 0
        """
        t_flat = t.flatten()

        # Get z_b values at knots (ensure positive via exp)
        zb_knots = torch.exp(self.log_zb_knots)

        # Clamp query times to valid range
        t_clamped = torch.clamp(t_flat, self.t_knots[0], self.t_knots[-1])

        indices = torch.searchsorted(self.t_knots, t_clamped)
        indices = torch.clamp(indices, 1, self.n_knots - 1)

        # Get surrounding knot times and values
        t0 = self.t_knots[indices - 1]
        t1 = self.t_knots[indices]
        zb0 = zb_knots[indices - 1]
        zb1 = zb_knots[indices]

        alpha = (t_clamped - t0) / (t1 - t0 + 1e-12)
        zb_interp = zb0 + alpha * (zb1 - zb0)

        # NOTE: Hard clamp removed - use soft penalty (max_depth_penalty) instead
        # This preserves gradients when z_b exceeds zb_max

        return zb_interp.view(-1, 1)

    def get_dzb_dt(self, t):
        """
        Compute time derivative of water table depth dz_b/dt.

        Uses automatic differentiation through the spline.

        Args:
            t: Time tensor (dimensional, seconds) of shape (N, 1)

        Returns:
            dzb_dt: Time derivative of z_b (m/s) of shape (N, 1)
        """
        t_grad = t.detach().requires_grad_(True)
        zb = self.forward(t_grad)
        dzb_dt = torch.autograd.grad(
            zb.sum(), t_grad, create_graph=True
        )[0]
        return dzb_dt

    def smoothness_penalty(self):
        """
        Compute smoothness regularization term in DIMENSIONLESS form.

        Penalizes |dz̃_b/dt̃|^2 where z̃_b = z_b/L and t̃ = t/T.
        This ensures the penalty is O(1) and comparable to other losses.

        Returns:
            penalty: Scalar smoothness penalty (mean of squared differences)
        """
        zb_knots = torch.exp(self.log_zb_knots)

        # Convert to dimensionless: z̃_b = z_b / L
        zb_knots_tilde = zb_knots / self.L

        dzb_tilde = zb_knots_tilde[1:] - zb_knots_tilde[:-1]

        # Dimensionless time differences: dt̃ = dt / T
        dt = self.t_knots[1:] - self.t_knots[:-1]
        dt_tilde = dt / self.T

        # Squared rate of change in dimensionless form
        dzb_dt_tilde_squared = (dzb_tilde / (dt_tilde + 1e-12)) ** 2

        # Return mean (not sum) for consistency across different n_knots
        return dzb_dt_tilde_squared.mean()

    def second_derivative_penalty(self):
        """
        Compute second derivative regularization (curvature penalty) in DIMENSIONLESS form.

        Penalizes |d²z̃_b/dt̃²|^2 to encourage linear variations.

        Returns:
            penalty: Scalar curvature penalty (mean)
        """
        zb_knots = torch.exp(self.log_zb_knots)

        if self.n_knots < 3:
            return torch.tensor(0.0, device=self.device)

        zb_knots_tilde = zb_knots / self.L

        # Second differences (discrete second derivative)
        d2zb_tilde = zb_knots_tilde[2:] - 2 * zb_knots_tilde[1:-1] + zb_knots_tilde[:-2]
        dt = (self.t_knots[1] - self.t_knots[0])  # Assume uniform spacing
        dt_tilde = dt / self.T

        d2zb_dt2_tilde_squared = (d2zb_tilde / (dt_tilde ** 2 + 1e-12)) ** 2

        return d2zb_dt2_tilde_squared.mean()

    def max_depth_penalty(self):
        """
        Soft penalty for z_b exceeding zb_max.

        Uses quadratic penalty: mean(max(0, z_b - zb_max)^2) / L^2
        This provides gradient signal to push z_b back below the limit.

        Returns:
            penalty: Scalar penalty (0 if all z_b <= zb_max)
        """
        if self.zb_max is None:
            return torch.tensor(0.0, device=self.device)

        zb_knots = torch.exp(self.log_zb_knots)

        # Quadratic penalty for exceeding max (dimensionless)
        excess = torch.clamp(zb_knots - self.zb_max, min=0.0)
        excess_tilde = excess / self.L

        return (excess_tilde ** 2).mean()


class RichardsInversePINN(nn.Module):
    """
    Inverse PINN for Richards equation.

    Given surface flux q(0,t) and observed surface moisture θ_obs(0,t),
    find water table depth z_b(t) such that predicted surface moisture
    matches observations.

    Physics:
        - Richards equation: ∂S_e/∂t̃ + ∂q̃/∂z̃ + S̃ = 0
        - Surface BC: q(0,t) = q_prescribed (Neumann)
        - Water table BC: h(-z_b,t) = 0 (Dirichlet)
        - Data matching: θ_pred(0,t) ≈ θ_obs(0,t)

    Coordinate convention:
        - z = 0 at surface
        - z < 0 below surface (z points upward)
        - z_b > 0 is water table depth (positive)
        - Water table at z = -z_b
    """

    def __init__(
        self,
        soil_params,
        q_surf_data,          # Surface flux data: (times, flux_values) in [s, m/s]
        theta_obs_data,       # Observed surface moisture: (times, theta_values)
        et_data=None,         # ET data for root uptake (optional)
        zr=0.5,               # Root zone depth [m]
        h_net_config=None,    # Pressure head network config
        n_spline_knots=20,    # Number of spline knots for z_b
        normalizer=None,      # NormalizationHelper instance
        zb_initial=1.5,       # Initial water table depth [m]
        zb_max=None,          # Maximum water table depth constraint [m]
        t_ref_days=15.0,      # Reference time for network scaling [days]
        z_max_tilde=1.0,      # Max dimensionless depth for network
        ic_profile=None,      # Initial condition profile
        ic_type='hydrostatic',
        device='cpu'
    ):
        """
        Args:
            soil_params: Soil parameters dict (theta_s, theta_r, alpha, n, Ks, l)
            q_surf_data: Tuple of (times, flux) for surface flux BC [s, m/s]
                         Positive flux = upward (evaporation)
                         Negative flux = downward (infiltration)
            theta_obs_data: Tuple of (times, theta) for observed surface moisture
            et_data: Optional (times, ET_rate) for root uptake [s, 1/s]
            zr: Root zone depth [m]
            h_net_config: Dict with 'hidden_dim' and 'num_layers'
            n_spline_knots: Number of knots for z_b spline
            normalizer: NormalizationHelper instance
            zb_initial: Initial water table depth [m]
            t_ref_days: Reference time for network scaling [days]
            z_max_tilde: Maximum dimensionless depth
            ic_profile: Initial condition profile dict
            ic_type: Type of IC ('obs', 'linear', 'hydrostatic')
            device: Computation device
        """
        super().__init__()

        if q_surf_data is None:
            raise ValueError("q_surf_data is required for inverse problem")
        if theta_obs_data is None:
            raise ValueError("theta_obs_data is required for inverse problem")
        if normalizer is None:
            raise ValueError("normalizer is required")

        self.normalizer = normalizer
        self.device = device

        # Compute fixed reference time in dimensionless units
        t_ref_dim = t_ref_days * 86400  # Convert days to seconds
        t_ref_tilde = t_ref_dim / normalizer.T

        self.t_ref_tilde = t_ref_tilde
        self.z_max_tilde = z_max_tilde

        if h_net_config is None:
            h_net_config = {'hidden_dim': 64, 'num_layers': 4}

        self.h_net = PressureHeadNet(
            h_net_config["hidden_dim"],
            h_net_config["num_layers"],
            t_ref_tilde=t_ref_tilde,
            z_max_tilde=z_max_tilde
        )

        t_min = min(min(q_surf_data[0]), min(theta_obs_data[0]))
        t_max = max(max(q_surf_data[0]), max(theta_obs_data[0]))
        self.t_min = t_min
        self.t_max = t_max

        # Initialize water table spline (pass L and T for dimensionless regularization)
        self.zb_spline = WaterTableSpline(
            n_knots=n_spline_knots,
            t_min=t_min,
            t_max=t_max,
            zb_initial=zb_initial,
            device=device,
            L=normalizer.L,
            T=normalizer.T,
            zb_max=zb_max
        )

        self.zr_tilde = normalizer.normalize_z(zr)  # This will be negative since zr > 0
        self.zr_tilde = abs(self.zr_tilde)  # Make it positive for root zone depth
        # Handle array or scalar zb_initial - use first value for IC target
        if isinstance(zb_initial, (list, tuple, np.ndarray, torch.Tensor)):
            zb_initial_scalar = float(zb_initial[0])
        else:
            zb_initial_scalar = float(zb_initial)
        self.zb_initial_tilde = normalizer.normalize_zb(zb_initial_scalar)

        # Store and normalize surface flux data
        self.q_surf_times_dim = torch.tensor(q_surf_data[0], dtype=torch.float32, device=device)
        self.q_surf_values_dim = torch.tensor(q_surf_data[1], dtype=torch.float32, device=device)

        # Normalize flux: q̃ = q / K_*
        self.q_surf_times_tilde = normalizer.normalize_t(self.q_surf_times_dim)
        self.q_surf_values_tilde = normalizer.normalize_q(self.q_surf_values_dim)

        # Store and normalize observed surface moisture data
        self.theta_obs_times_dim = torch.tensor(theta_obs_data[0], dtype=torch.float32, device=device)
        self.theta_obs_values_dim = torch.tensor(theta_obs_data[1], dtype=torch.float32, device=device)

        # Normalize moisture: θ̃ = (θ - θ_r) / θ_* = S_e
        self.theta_obs_times_tilde = normalizer.normalize_t(self.theta_obs_times_dim)
        self.theta_obs_values_tilde = (self.theta_obs_values_dim - normalizer.theta_r) / normalizer.theta_star

        # Store and normalize ET data (if provided)
        if et_data is not None:
            self.et_times_dim = torch.tensor(et_data[0], dtype=torch.float32, device=device)
            self.et_values_dim = torch.tensor(et_data[1], dtype=torch.float32, device=device)
            self.et_values_tilde = normalizer.normalize_S(self.et_values_dim)
            self.et_times_tilde = normalizer.normalize_t(self.et_times_dim)
            self.use_et_data = True
        else:
            self.S_max_tilde = normalizer.S_max_tilde
            self.use_et_data = False

        self.ic_type = ic_type

        # Store measured IC profile if provided
        self.ic_profile = None
        if ic_profile is not None:
            depths = torch.tensor(ic_profile['depths'], dtype=torch.float32, device=device)
            z_ic = -depths  # Convert positive depth to negative z
            theta_ic = torch.tensor(ic_profile['theta'], dtype=torch.float32, device=device)

            Se_ic = (theta_ic - normalizer.theta_r) / normalizer.theta_star
            Se_ic = torch.clamp(Se_ic, 1e-6, 1.0 - 1e-6)

            # Convert Se to pressure head using van Genuchten inversion
            m = 1.0 - 1.0 / normalizer.n
            inv_term = torch.pow(Se_ic, -1.0 / m) - 1.0
            h_ic = -torch.pow(torch.clamp(inv_term, min=0.0), 1.0 / normalizer.n) / normalizer.alpha

            self.ic_profile = {
                'z': z_ic,
                'h': h_ic
            }

        self.call_count = 0

    def forward(self, z_tilde, t_tilde):
        """
        Forward pass: compute pressure head and water table depth.

        Args:
            z_tilde: Dimensionless spatial coordinate (z̃ = z/L)
            t_tilde: Dimensionless time (t̃ = t/T)

        Returns:
            h_tilde: Dimensionless pressure head
            zb_tilde: Dimensionless water table depth (from spline)
        """
        # Get water table depth first (needed for z scaling)
        t_dim = self.normalizer.denormalize_t(t_tilde)
        zb_dim = self.zb_spline(t_dim)
        zb_tilde = self.normalizer.normalize_zb(zb_dim)

        # Scale z relative to current z_b so z_scaled is always in [0, 1]
        # z_scaled = (z_tilde + z_b_tilde) / z_b_tilde
        # At z=0 (surface): z_scaled = 1
        # At z=-z_b (water table): z_scaled = 0
        z_scaled_dynamic = (z_tilde + zb_tilde) / (zb_tilde + 1e-8)

        # CRITICAL: Pass zb_tilde to network to enable gradient flow
        # Without this, at z=0 (surface), z_scaled=1 regardless of z_b,
        # so data loss gradient cannot flow to z_b parameters.
        h_tilde = self.h_net.forward_with_prescaled_z(z_scaled_dynamic, t_tilde, zb_tilde)

        return h_tilde, zb_tilde

    # ==================== INTERPOLATION HELPERS ====================

    def _interp_1d(self, x, xp, fp):
        """GPU-optimized 1D linear interpolation."""
        x_flat = x.flatten()
        x_clamped = torch.clamp(x_flat, xp[0], xp[-1])

        indices = torch.searchsorted(xp, x_clamped)
        indices = torch.clamp(indices, 1, len(xp) - 1)

        x0 = xp[indices - 1]
        x1 = xp[indices]
        y0 = fp[indices - 1]
        y1 = fp[indices]

        alpha = (x_clamped - x0) / (x1 - x0 + 1e-12)
        y_interp = y0 + alpha * (y1 - y0)

        return y_interp.reshape_as(x)

    def surface_flux_tilde(self, t_tilde):
        """
        Get prescribed dimensionless surface flux at given times.

        Convention: positive flux = upward (out of soil)

        Args:
            t_tilde: Dimensionless time

        Returns:
            q_surf_tilde: Dimensionless surface flux
        """
        return self._interp_1d(t_tilde, self.q_surf_times_tilde, self.q_surf_values_tilde)

    def theta_obs_tilde(self, t_tilde):
        """
        Get observed dimensionless surface moisture at given times.

        Args:
            t_tilde: Dimensionless time

        Returns:
            theta_obs_tilde: Observed dimensionless surface moisture (S_e)
        """
        return self._interp_1d(t_tilde, self.theta_obs_times_tilde, self.theta_obs_values_tilde)

    def et_rate_tilde(self, t_tilde):
        """Get dimensionless ET rate for root uptake."""
        if self.use_et_data:
            return self._interp_1d(t_tilde, self.et_times_tilde, self.et_values_tilde)
        else:
            return torch.full_like(t_tilde, self.S_max_tilde)

    # ==================== PHYSICS COMPUTATIONS ====================

    def root_uptake_tilde(self, z_tilde, t_tilde):
        """
        Dimensionless root uptake S̃(z̃,t̃) = β(z̃) × T̃_p(t̃)

        Exponential distribution:
            β(z̃) = (1/z̃_r) exp(z̃/z̃_r)  for z̃ ∈ [-z̃_r, 0]

        Note: z̃ <= 0, z̃_r > 0 (positive root zone depth)
        """
        Tp_tilde = self.et_rate_tilde(t_tilde)

        # Root distribution: β(z̃) = (1/z̃_r) exp(z̃/z̃_r)
        # Since z̃ <= 0 and z̃_r > 0, exp(z̃/z̃_r) decays with depth
        beta = (1.0 / self.zr_tilde) * torch.exp(z_tilde / self.zr_tilde)

        S_uptake = beta * Tp_tilde

        # Apply only in root zone (z̃ >= -z̃_r), zero below
        return torch.where(
            z_tilde >= -self.zr_tilde,
            S_uptake,
            torch.zeros_like(z_tilde)
        )

    def compute_derivatives_tilde(self, h_tilde, z_tilde, t_tilde):
        """
        Compute dimensionless derivatives for Richards equation.

        Returns:
            dSe_dt_tilde: ∂S_e/∂t̃
            dq_dz_tilde: ∂q̃/∂z̃
            q_tilde: Dimensionless flux q̃ = -K̃(∂h̃/∂z̃ + 1)
            K_tilde: Dimensionless conductivity
        """
        Se = self.normalizer.Se_tilde(h_tilde)

        # ∂S_e/∂t̃
        dSe_dt_tilde = torch.autograd.grad(Se.sum(), t_tilde, create_graph=True)[0]

        # ∂h̃/∂z̃
        dh_dz_tilde = torch.autograd.grad(h_tilde.sum(), z_tilde, create_graph=True)[0]

        K_tilde = self.normalizer.K_tilde(h_tilde)

        # Dimensionless Darcy flux: q̃ = -K̃(∂h̃/∂z̃ + 1)
        # The +1 term comes from the gravity component (z points upward)
        q_tilde = -K_tilde * (dh_dz_tilde + 1.0)

        # ∂q̃/∂z̃
        dq_dz_tilde = torch.autograd.grad(q_tilde.sum(), z_tilde, create_graph=True)[0]

        return dSe_dt_tilde, dq_dz_tilde, q_tilde, K_tilde

    # ==================== RESIDUAL COMPUTATIONS ====================

    def pde_residual(self, z, t):
        """
        Dimensionless Richards equation residual.

        PDE: ∂S_e/∂t̃ + ∂q̃/∂z̃ + S̃ = 0

        Args:
            z: DIMENSIONAL spatial coordinate [m] (z <= 0)
            t: DIMENSIONAL time [s]

        Returns:
            Dimensionless PDE residual (should be ~0)
        """
        z_tilde = self.normalizer.normalize_z(z).requires_grad_(True)
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)

        h_tilde, _ = self(z_tilde, t_tilde)
        dSe_dt_tilde, dq_dz_tilde, _, _ = self.compute_derivatives_tilde(h_tilde, z_tilde, t_tilde)
        S_tilde = self.root_uptake_tilde(z_tilde, t_tilde)

        # Dimensionless PDE: ∂S_e/∂t̃ + ∂q̃/∂z̃ + S̃ = 0
        return dSe_dt_tilde + dq_dz_tilde + S_tilde

    def surface_flux_bc_residual(self, t):
        """
        Surface flux (Neumann) BC residual.

        At z = 0: q_pred = q_prescribed

        Args:
            t: DIMENSIONAL time [s]

        Returns:
            Dimensionless BC residual (q̃_pred - q̃_prescribed)
        """
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)
        z0_tilde = torch.zeros_like(t_tilde, requires_grad=True, device=t_tilde.device)

        # Get predicted pressure head at surface
        h0_tilde, _ = self(z0_tilde, t_tilde)

        # Compute predicted flux at surface
        # ∂h̃/∂z̃ at z=0
        dh_dz_tilde = torch.autograd.grad(h0_tilde.sum(), z0_tilde, create_graph=True)[0]

        K0_tilde = self.normalizer.K_tilde(h0_tilde)

        # Predicted flux: q̃ = -K̃(∂h̃/∂z̃ + 1)
        q_pred_tilde = -K0_tilde * (dh_dz_tilde + 1.0)

        # Prescribed flux (from data)
        q_prescribed_tilde = self.surface_flux_tilde(t_tilde)

        # Residual: predicted - prescribed
        return q_pred_tilde - q_prescribed_tilde

    def water_table_head_residual(self, t):
        """
        Water table head BC residual.

        At z = -z_b(t): h = 0

        Args:
            t: DIMENSIONAL time [s]

        Returns:
            Dimensionless BC residual (h̃ at water table, should be ~0)
        """
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)

        # Get water table depth from spline
        zb_dim = self.zb_spline(t)
        zb_tilde = self.normalizer.normalize_zb(zb_dim)

        # Water table location: z = -z_b (negative since below surface)
        z_wt_tilde = -zb_tilde

        # Get pressure head at water table
        h_wt_tilde, _ = self(z_wt_tilde, t_tilde)

        # Residual: h at water table should be 0
        return h_wt_tilde

    def surface_moisture_data_residual(self, t):
        """
        Surface moisture data-matching residual.

        This is the inverse problem objective: predicted surface moisture
        should match observed surface moisture.

        Args:
            t: DIMENSIONAL time [s]

        Returns:
            Dimensionless residual (θ̃_pred - θ̃_obs)
        """
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)
        z0_tilde = torch.zeros_like(t_tilde, requires_grad=True, device=t_tilde.device)

        # Get predicted pressure head at surface
        h0_tilde, _ = self(z0_tilde, t_tilde)

        # Convert to effective saturation (= dimensionless moisture)
        Se_pred = self.normalizer.Se_tilde(h0_tilde)

        # Get observed moisture (already normalized)
        Se_obs = self.theta_obs_tilde(t_tilde)

        # Residual: predicted - observed
        return Se_pred - Se_obs

    def initial_conditions_residual(self, z, t0):
        """
        Initial conditions residual.

        Args:
            z: DIMENSIONAL spatial coordinate [m]
            t0: DIMENSIONAL initial time [s]

        Returns:
            Tuple of (h residual, zb residual)
        """
        z_tilde = self.normalizer.normalize_z(z)
        t0_tilde = self.normalizer.normalize_t(t0)

        h_tilde, zb_tilde = self(z_tilde, t0_tilde)

        # Determine target IC based on ic_type
        if self.ic_type == 'obs' and self.ic_profile is not None:
            z_measured = self.ic_profile['z']
            h_measured = self.ic_profile['h']

            zb_ic_dim = self.zb_spline(t0)
            z_wt = -zb_ic_dim.view(-1)
            h_wt = torch.zeros_like(z_wt)

            z_extended = torch.cat([z_measured, z_wt])
            h_extended = torch.cat([h_measured, h_wt])

            sort_idx = torch.argsort(z_extended)
            z_extended_sorted = z_extended[sort_idx]
            h_extended_sorted = h_extended[sort_idx]

            h_ic = self._interp_1d(z, z_extended_sorted, h_extended_sorted)
            h_ic_tilde = self.normalizer.normalize_h(h_ic)

        elif self.ic_type == 'linear' and self.ic_profile is not None:
            h_surface = self.ic_profile['h'][self.ic_profile['z'].argmax()]
            zb_ic_dim = self.zb_spline(t0)
            h_ic = h_surface * (1.0 + z / zb_ic_dim)
            h_ic_tilde = self.normalizer.normalize_h(h_ic)

        else:  # hydrostatic
            # Get surface moisture at t=0 from observed data
            t0_tensor = torch.tensor([[self.t_min]], device=self.device, dtype=torch.float32)
            Se_0 = self.theta_obs_tilde(self.normalizer.normalize_t(t0_tensor))
            Se_0 = torch.clamp(Se_0, 1e-6, 1.0 - 1e-6)

            m = 1.0 - 1.0 / self.normalizer.n
            inv_term = torch.pow(Se_0, -1.0 / m) - 1.0
            h_bc0 = -torch.pow(torch.clamp(inv_term, min=0.0), 1.0 / self.normalizer.n) / self.normalizer.alpha

            # Get water table depth at t=0
            zb_ic_dim = self.zb_spline(t0)

            # Quadratic profile: h(0)=h_bc0, h(-zb)=0, dh/dz|_{-zb}=-1
            a = (zb_ic_dim + h_bc0) / (zb_ic_dim ** 2)
            b = 1.0 + 2.0 * h_bc0 / zb_ic_dim
            c = h_bc0

            h_ic = a * z**2 + b * z + c
            h_ic_tilde = self.normalizer.normalize_h(h_ic)

        # Initial water table depth - compare against initial guess, NOT spline value
        # This provides an anchor for z_b at t=0
        zb_ic_target_tilde = self.zb_initial_tilde  # Use initial guess as target

        res_h = h_tilde - h_ic_tilde
        res_zb = zb_tilde - zb_ic_target_tilde

        return res_h, res_zb

    def zb_smoothness_residual(self):
        """
        Water table smoothness regularization.

        Returns:
            Scalar smoothness penalty
        """
        return self.zb_spline.smoothness_penalty()

    def zb_curvature_residual(self):
        """
        Water table curvature regularization (second derivative).

        Returns:
            Scalar curvature penalty
        """
        return self.zb_spline.second_derivative_penalty()

    def zb_guidance_residual(self, t):
        """
        Physics-based guidance loss: encourage z_b(t) to follow estimate from θ_obs.

        The physics-based estimate uses the intuition that:
        - Wet surface (high θ, high Se) → shallow water table (small z_b)
        - Dry surface (low θ, low Se) → deeper water table (larger z_b)

        Formula: z_b_target = L * (1 - Se) * scale + offset

        Args:
            t: Dimensional time tensor [s]

        Returns:
            Tensor of residuals (zb_predicted - zb_target) / L
        """
        # Get model's z_b prediction
        zb_pred = self.zb_spline(t)
        L = self.normalizer.L

        # Physics-based estimate from observed θ
        # Use 0.4 scaling factor (was 0.8) to give shallower estimates
        t_tilde = self.normalizer.normalize_t(t)
        Se_obs = self.theta_obs_tilde(t_tilde)
        Se_obs = torch.clamp(Se_obs, 0.01, 0.99)
        zb_target = L * (1.0 - Se_obs) * 0.4 + 0.2
        zb_max = self.zb_spline.zb_max if self.zb_spline.zb_max is not None else L
        zb_target = torch.clamp(zb_target, 0.2, zb_max)

        return (zb_pred - zb_target) / L

    # ==================== PREDICTION METHODS ====================

    def predict_water_table(self, t):
        """
        Predict dimensional water table depth.

        Args:
            t: Dimensional time [s]

        Returns:
            zb: Dimensional water table depth [m]
        """
        return self.zb_spline(t)

    def predict_head(self, z, t):
        """
        Predict dimensional pressure head.

        Args:
            z: Dimensional spatial coordinate [m]
            t: Dimensional time [s]

        Returns:
            h: Dimensional pressure head [m]
            zb: Dimensional water table depth [m]
        """
        z_tilde = self.normalizer.normalize_z(z)
        t_tilde = self.normalizer.normalize_t(t)

        with torch.no_grad():
            h_tilde, zb_tilde = self(z_tilde, t_tilde)

        h = self.normalizer.denormalize_h(h_tilde)
        zb = self.normalizer.denormalize_zb(zb_tilde)

        return h, zb

    def predict_surface_moisture(self, t):
        """
        Predict dimensional surface soil moisture.

        Args:
            t: Dimensional time [s]

        Returns:
            theta: Dimensional surface soil moisture [m³/m³]
        """
        t_tilde = self.normalizer.normalize_t(t)
        z0_tilde = torch.zeros_like(t_tilde, device=t_tilde.device)

        with torch.no_grad():
            h0_tilde, _ = self(z0_tilde, t_tilde)
            Se = self.normalizer.Se_tilde(h0_tilde)

        theta = Se * self.normalizer.theta_star + self.normalizer.theta_r

        return theta

    def predict_surface_flux(self, t):
        """
        Predict dimensional surface flux.

        Args:
            t: Dimensional time [s]

        Returns:
            q: Dimensional surface flux [m/s]
        """
        # Use enable_grad context to ensure gradients are computed
        with torch.enable_grad():
            t_tilde = self.normalizer.normalize_t(t).detach().clone()
            t_tilde.requires_grad_(True)

            z0_tilde = torch.zeros_like(t_tilde, device=t_tilde.device)
            z0_tilde.requires_grad_(True)

            # Forward pass - manually compute to ensure grad tracking
            # Get water table depth
            t_dim = self.normalizer.denormalize_t(t_tilde)
            zb_dim = self.zb_spline(t_dim)
            zb_tilde = self.normalizer.normalize_zb(zb_dim)

            # Scale z relative to z_b
            z_scaled = (z0_tilde + zb_tilde) / (zb_tilde + 1e-8)

            # Get h from network (pass zb_tilde to enable gradient flow)
            h0_tilde = self.h_net.forward_with_prescaled_z(z_scaled, t_tilde, zb_tilde)

            dh_dz_tilde = torch.autograd.grad(
                h0_tilde.sum(), z0_tilde,
                create_graph=False, retain_graph=True
            )[0]

            K0_tilde = self.normalizer.K_tilde(h0_tilde)
            q_tilde = -K0_tilde * (dh_dz_tilde + 1.0)

            q = self.normalizer.denormalize_q(q_tilde)

        return q.detach()
