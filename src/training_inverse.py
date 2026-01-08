"""
Training utilities for Inverse PINN.

Loss components:
    1. PDE residual: Richards equation satisfied in domain
    2. Surface flux BC: q(0,t) = q_prescribed (Neumann)
    3. Water table head BC: h(-z_b,t) = 0
    4. Data matching: θ_pred(0,t) ≈ θ_obs(0,t)
    5. Initial conditions: h(z,0), z_b(0)
    6. z_b regularization: smoothness + curvature penalties

No kinematic BC (unlike forward problem).
"""

import torch
import math
import numpy as np


class InverseSamplingHelpers:
    """Helper functions for sampling training points in inverse problem."""

    @staticmethod
    def sample_initial_condition_points(model, batch_size, t_min, device):
        """
        Sample initial condition points.

        Args:
            model: Inverse PINN model
            batch_size: Number of IC points
            t_min: Initial time [s]
            device: Torch device

        Returns:
            z_ic: Spatial coordinates [m] (negative, below surface)
            t_ic: Time coordinates [s] (all at t_min)
        """
        ic_batch_size = min(batch_size // 4, 50)
        t_ic = torch.full((ic_batch_size, 1), t_min, device=device)

        # Get water table depth at initial time
        with torch.no_grad():
            zb0 = model.predict_water_table(t_ic)

        # Sample z uniformly in [-zb(t_min), 0]
        u_ic = torch.rand_like(t_ic)
        z_ic = -zb0 * u_ic  # z in [-zb, 0]

        return z_ic, t_ic

    @staticmethod
    def sample_pde_points(model, batch_size, t_max, device, boundary_ratio=0.7):
        """
        Sample PDE collocation points.

        Similar to forward problem but uses spline for water table.

        Args:
            model: Inverse PINN model
            batch_size: Number of collocation points
            t_max: Maximum time [s]
            device: Torch device
            boundary_ratio: Fraction of points near boundaries

        Returns:
            z_col: Spatial coordinates [m]
            t_col: Time coordinates [s]
        """
        t_col = torch.rand(batch_size, 1, device=device) * t_max
        t_col.requires_grad_(True)

        # Sample normalized depth u ∈ [0,1]
        n_boundary = int(batch_size * boundary_ratio)
        n_interior = batch_size - n_boundary

        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        u_surface = torch.distributions.Beta(
            torch.tensor(1.0, device=device),
            torch.tensor(3.0, device=device)
        ).sample((n_surface,))
        u_bottom = torch.distributions.Beta(
            torch.tensor(3.0, device=device),
            torch.tensor(1.0, device=device)
        ).sample((n_bottom,))
        u_interior = torch.rand(n_interior, device=device)

        u_col = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        # Map to physical depth using water table from spline
        with torch.no_grad():
            zb_col = model.predict_water_table(t_col)

        # z in [-zb, 0]
        z_col = (-u_col * zb_col).requires_grad_(True)

        return z_col, t_col


def compute_inverse_losses(model, z_col, t_col, t_bc_flux, t_bc_theta, z_ic, t_ic,
                           lambda_smooth=1e-4, lambda_curv=1e-6, lambda_max_depth=10.0,
                           lambda_guide=1.0):
    """
    Compute all loss components for inverse problem.

    Args:
        model: RichardsInversePINN model
        z_col: PDE collocation z-coordinates
        t_col: PDE collocation times
        t_bc_flux: Times for flux BC (sampled based on q_surf gradients)
        t_bc_theta: Times for theta data matching (sampled based on theta gradients)
        z_ic: Initial condition z-coordinates
        t_ic: Initial condition times
        lambda_smooth: Weight for smoothness penalty
        lambda_curv: Weight for curvature penalty
        lambda_max_depth: Weight for max depth penalty (soft constraint)
        lambda_guide: Weight for physics-based z_b guidance

    Returns:
        Dict of loss components:
            - pde: Richards equation residual
            - flux_bc: Surface flux BC (Neumann)
            - wt_head: Water table head BC (h=0)
            - data: Surface moisture data matching
            - ic_h: Initial condition for h
            - ic_zb: Initial condition for z_b
            - zb_smooth: Smoothness penalty
            - zb_curv: Curvature penalty
            - zb_max_depth: Max depth soft constraint penalty
            - zb_guide: Physics-based z_b guidance loss
    """
    res_pde = model.pde_residual(z_col, t_col)
    loss_pde = (res_pde**2).mean()

    # Surface flux BC (Neumann) - uses flux-gradient sampled times
    res_flux = model.surface_flux_bc_residual(t_bc_flux)
    loss_flux = (res_flux**2).mean()

    # Water table head BC - uses flux-gradient sampled times (coupled with flux)
    res_wt_head = model.water_table_head_residual(t_bc_flux)
    loss_wt_head = (res_wt_head**2).mean()

    # Data matching (inverse objective) - uses theta-gradient sampled times
    res_data = model.surface_moisture_data_residual(t_bc_theta)
    loss_data = (res_data**2).mean()

    res_ic_h, res_ic_zb = model.initial_conditions_residual(z_ic, t_ic)
    loss_ic_h = (res_ic_h**2).mean()
    loss_ic_zb = (res_ic_zb**2).mean()

    loss_zb_smooth = model.zb_smoothness_residual() * lambda_smooth
    loss_zb_curv = model.zb_curvature_residual() * lambda_curv

    # Max depth soft constraint (provides gradient when z_b exceeds zb_max)
    loss_zb_max_depth = model.zb_spline.max_depth_penalty() * lambda_max_depth

    # Physics-based z_b guidance loss - SKIP computation when weight=0 for speed
    if lambda_guide > 0:
        res_zb_guide = model.zb_guidance_residual(t_bc_theta)
        loss_zb_guide = (res_zb_guide**2).mean() * lambda_guide
    else:
        # Skip expensive computation when weight is zero
        loss_zb_guide = torch.tensor(0.0, device=z_col.device)

    return {
        "pde": loss_pde,
        "flux_bc": loss_flux,
        "wt_head": loss_wt_head,
        "data": loss_data,
        "ic_h": loss_ic_h,
        "ic_zb": loss_ic_zb,
        "zb_smooth": loss_zb_smooth,
        "zb_curv": loss_zb_curv,
        "zb_max_depth": loss_zb_max_depth,
        "zb_guide": loss_zb_guide,
    }


def compute_grad_norm(loss, model):
    """
    Compute L2 norm of gradients for a loss term.

    GPU-optimized: returns GPU tensor.
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    grad_norms_squared = []
    for grad in grads:
        if grad is not None:
            grad_norms_squared.append(grad.norm(2) ** 2)

    if len(grad_norms_squared) == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    total_norm_squared = torch.stack(grad_norms_squared).sum()
    return total_norm_squared.sqrt()


def apply_weights_and_compute_gradients(losses, weights, model):
    """Apply weights and compute gradient norms."""
    weighted_losses = {key: weights[key] * losses[key] for key in losses}

    gradients = {}
    for key in losses:
        gradients[key] = compute_grad_norm(weighted_losses[key], model)

    total_loss = sum(weighted_losses.values())

    return weighted_losses, gradients, total_loss


def apply_weights_fixed_mode(losses, weights):
    """Apply weights without computing gradients (fixed weight mode)."""
    weighted_losses = {key: weights[key] * losses[key] for key in losses}
    total_loss = sum(weighted_losses.values())
    gradients = {key: 0.0 for key in losses}
    gradients["total"] = 0.0

    return weighted_losses, gradients, total_loss


class InverseWeightManager:
    """
    Adaptive weights for inverse PINN loss components.

    Loss components:
        - pde: PDE residual
        - flux_bc: Surface flux BC
        - wt_head: Water table head BC
        - data: Data matching (inverse objective)
        - ic_h: Initial h
        - ic_zb: Initial z_b
        - zb_smooth: Smoothness regularization
        - zb_curv: Curvature regularization
    """

    def __init__(
        self,
        use_initial_scales: bool = True,
        weight_lr: float = 0.2,
        ema_alpha: float = 0.9,
        min_w: float = 1e-6,
        max_w: float = 1e3,  # Reduced from 1e10 to prevent weight explosion
        max_step_factor: float = 2.0,
        eps: float = 1e-12,
        initial_pde_weight: float = None,
        initial_flux_bc_weight: float = None,
        initial_wt_head_weight: float = None,
        initial_data_weight: float = None,
        initial_ic_h_weight: float = None,
        initial_ic_zb_weight: float = None,
        initial_zb_smooth_weight: float = None,
        initial_zb_curv_weight: float = None,
        initial_zb_max_depth_weight: float = None,
        initial_zb_guide_weight: float = None,  # Physics-based z_b guidance
        use_fixed_weights: bool = False,
    ):
        self.weight_lr = float(weight_lr)
        self.ema_alpha = float(ema_alpha)
        self.min_w, self.max_w = float(min_w), float(max_w)
        self.max_step_factor = float(max_step_factor)
        self.eps = float(eps)
        self.use_fixed_weights = use_fixed_weights

        if initial_pde_weight is not None:
            base = {
                "pde": initial_pde_weight,
                "flux_bc": initial_flux_bc_weight,
                "wt_head": initial_wt_head_weight,
                "data": initial_data_weight,
                "ic_h": initial_ic_h_weight,
                "ic_zb": initial_ic_zb_weight,
                "zb_smooth": initial_zb_smooth_weight,
                "zb_curv": initial_zb_curv_weight,
                "zb_max_depth": initial_zb_max_depth_weight,
                "zb_guide": initial_zb_guide_weight if initial_zb_guide_weight is not None else 10.0,
            }
        else:
            # Default weights for inverse problem
            # Data matching is key objective - give it high weight
            base = {
                "pde": 1.0,
                "flux_bc": 10.0 if use_initial_scales else 1.0,
                "wt_head": 10.0 if use_initial_scales else 1.0,
                "data": 100.0 if use_initial_scales else 1.0,  # High weight for data matching
                "ic_h": 1.0,
                "ic_zb": 1.0,
                "zb_smooth": 1.0,  # Regularization
                "zb_curv": 1.0,   # Regularization
                "zb_max_depth": 1.0,  # Max depth soft constraint
                "zb_guide": 10.0,  # Physics-based z_b guidance
            }

        self.weights = {k: float(v) for k, v in base.items()}
        self.weight_history = {k: [self.weights[k]] for k in self.weights}
        self.grad_ema = {k: None for k in self.weights}

    def _to_float(self, x):
        """Convert to float, handling GPU tensors."""
        if isinstance(x, torch.Tensor):
            return x.detach()
        try:
            return float(x)
        except Exception:
            return np.float64(x).item()

    def update(self, current_grads, counts=None, already_weighted: bool = True):
        """Update weights based on gradient norms."""
        if self.use_fixed_weights:
            return

        g_unw_tensors = {}
        for k in self.weights:
            if k not in current_grads:
                continue
            g = self._to_float(current_grads[k])

            if isinstance(g, torch.Tensor):
                if already_weighted:
                    g = g / max(self.weights[k], self.eps)
                if counts is not None and k in counts and counts[k] and counts[k] > 0:
                    g = g / math.sqrt(float(counts[k]))
                g_unw_tensors[k] = torch.clamp(g, min=0.0)
            else:
                if already_weighted:
                    g = g / max(self.weights[k], self.eps)
                if counts is not None and k in counts and counts[k] and counts[k] > 0:
                    g = g / math.sqrt(float(counts[k]))
                g_unw_tensors[k] = max(g, 0.0)

        g_unw = {}
        for k, v in g_unw_tensors.items():
            if isinstance(v, torch.Tensor):
                g_unw[k] = v.cpu().item()
            else:
                g_unw[k] = v

        for k in self.weights:
            if k in g_unw:
                if self.grad_ema[k] is None:
                    self.grad_ema[k] = g_unw[k]
                else:
                    self.grad_ema[k] = (
                        self.ema_alpha * self.grad_ema[k]
                        + (1.0 - self.ema_alpha) * g_unw[k]
                    )

        # Target = median of valid EMA grads
        valid = [
            v for v in (self.grad_ema[k] for k in self.weights)
            if v is not None and v > self.eps
        ]
        if not valid:
            return
        g_target = float(np.median(valid))

        log_cap = math.log(self.max_step_factor)
        for k in self.weights:
            v = self.grad_ema[k]
            if v is None or v <= self.eps:
                continue
            ratio = v / g_target
            log_w = math.log(self.weights[k])
            log_w_new = log_w - self.weight_lr * math.log(ratio)
            log_w_new = max(log_w - log_cap, min(log_w + log_cap, log_w_new))
            w_new = math.exp(log_w_new)
            self.weights[k] = float(min(self.max_w, max(self.min_w, w_new)))

    def is_using_fixed_weights(self):
        return self.use_fixed_weights

    def record_history(self):
        for k in self.weights:
            self.weight_history[k].append(self.weights[k])

    def get_weights(self):
        return dict(self.weights)

    def print_update(self, epoch):
        print(f"\n[Epoch {epoch+1}] Updated weights:")
        for k in self.weights:
            g = self.grad_ema[k]
            gtxt = f"{g:.3e}" if (g is not None) else "nan"
            print(f"  {k}: {self.weights[k]:.3e} (grad_unw_ema: {gtxt})")


def compute_total_grad_norm(model):
    """Compute total gradient norm after backward pass."""
    grad_norms_squared = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms_squared.append(param.grad.norm(2) ** 2)

    if len(grad_norms_squared) == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    total_norm_squared = torch.stack(grad_norms_squared).sum()
    return total_norm_squared.sqrt()


def compute_full_sample_loss_inverse(model, flux_times, theta_times, t_min, device,
                                     t_max=None, boundary_ratio=0.7,
                                     sample_size=5000, chunk_size=1000,
                                     lambda_smooth=1e-4, lambda_curv=1e-6):
    """
    Compute loss over full dataset for smooth loss tracking.

    Args:
        model: RichardsInversePINN model
        flux_times: BC time points for flux evaluation
        theta_times: BC time points for theta data matching
        t_min: Minimum time
        device: Torch device
        t_max: Maximum time
        boundary_ratio: Boundary sampling ratio
        sample_size: Number of PDE samples
        chunk_size: Chunk size for processing
        lambda_smooth: Smoothness regularization weight
        lambda_curv: Curvature regularization weight

    Returns:
        Dict of loss components
    """
    model.eval()

    loss_accum_gpu = {
        "pde": torch.tensor(0.0, device=device),
        "flux_bc": torch.tensor(0.0, device=device),
        "wt_head": torch.tensor(0.0, device=device),
        "data": torch.tensor(0.0, device=device),
        "ic_h": torch.tensor(0.0, device=device),
        "ic_zb": torch.tensor(0.0, device=device),
    }

    if t_max is None:
        t_max = model.t_max

    n_pde_points = 0
    for i in range(0, sample_size, chunk_size):
        chunk_batch_size = min(chunk_size, sample_size - i)

        t_chunk = torch.rand(chunk_batch_size, 1, device=device) * t_max

        n_boundary = int(chunk_batch_size * boundary_ratio)
        n_interior = chunk_batch_size - n_boundary
        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        u_surface = torch.distributions.Beta(
            torch.tensor(1.0, device=device),
            torch.tensor(3.0, device=device)
        ).sample((n_surface,))
        u_bottom = torch.distributions.Beta(
            torch.tensor(3.0, device=device),
            torch.tensor(1.0, device=device)
        ).sample((n_bottom,))
        u_interior = torch.rand(n_interior, device=device)
        u_chunk = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        zb_chunk = model.predict_water_table(t_chunk)
        z_chunk = (-u_chunk * zb_chunk).detach().requires_grad_(True)
        t_chunk_grad = t_chunk.detach().requires_grad_(True)

        res_pde = model.pde_residual(z_chunk, t_chunk_grad)
        loss_accum_gpu["pde"] += (res_pde**2).sum().detach()
        n_pde_points += len(z_chunk)

    loss_accum_gpu["pde"] /= n_pde_points

    # Flux BC losses (using flux times)
    n_flux_points = 0
    batch_size_bc = 100
    for i in range(0, len(flux_times), batch_size_bc):
        j = min(i + batch_size_bc, len(flux_times))
        t_flux = flux_times[i:j]

        res_flux = model.surface_flux_bc_residual(t_flux)
        loss_accum_gpu["flux_bc"] += (res_flux**2).sum().detach()

        res_wt_head = model.water_table_head_residual(t_flux)
        loss_accum_gpu["wt_head"] += (res_wt_head**2).sum().detach()

        n_flux_points += len(t_flux)

    loss_accum_gpu["flux_bc"] /= n_flux_points
    loss_accum_gpu["wt_head"] /= n_flux_points

    # Data matching loss (using theta times)
    n_theta_points = 0
    for i in range(0, len(theta_times), batch_size_bc):
        j = min(i + batch_size_bc, len(theta_times))
        t_theta = theta_times[i:j]

        res_data = model.surface_moisture_data_residual(t_theta)
        loss_accum_gpu["data"] += (res_data**2).sum().detach()

        n_theta_points += len(t_theta)

    loss_accum_gpu["data"] /= n_theta_points

    sampling = InverseSamplingHelpers()
    z_ic, t_ic = sampling.sample_initial_condition_points(model, 200, t_min, device)
    res_ic_h, res_ic_zb = model.initial_conditions_residual(z_ic, t_ic)
    loss_accum_gpu["ic_h"] = (res_ic_h**2).mean().detach()
    loss_accum_gpu["ic_zb"] = (res_ic_zb**2).mean().detach()

    model.train()

    loss_accum = {k: v.cpu().item() for k, v in loss_accum_gpu.items()}

    loss_accum["zb_smooth"] = model.zb_smoothness_residual().item() * lambda_smooth
    loss_accum["zb_curv"] = model.zb_curvature_residual().item() * lambda_curv

    return loss_accum
