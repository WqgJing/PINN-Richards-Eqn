"""
Training loop for Inverse PINN.

Solves the inverse problem: given surface flux q(0,t) and observed
surface moisture θ_obs(0,t), find water table depth z_b(t).
"""

import numpy as np
import torch
from torch.optim import Adam
import os

from .normalization_helper import NormalizationHelper
from .pinn_inverse import RichardsInversePINN
from .training_inverse import (
    InverseSamplingHelpers,
    compute_inverse_losses,
    apply_weights_and_compute_gradients,
    apply_weights_fixed_mode,
    InverseWeightManager,
    compute_total_grad_norm,
    compute_full_sample_loss_inverse,
)
from .training_logger_inverse import InverseTrainingLogger
from .gradient_based_sampling import gradient_based_sampling


def _precompute_gradient_weights(times_t, values_t, device, power=2.0):
    """
    Precompute gradient-based sampling weights ONCE at start.

    This avoids recomputing gradients every epoch for static data.
    """
    times_flat = times_t.flatten()

    dv = torch.diff(values_t)
    dt = torch.diff(times_flat)
    gradients = dv / (dt + 1e-12)
    gradient_magnitude = torch.abs(gradients)

    weights = torch.zeros(len(values_t), device=device)
    weights[:-1] = gradient_magnitude
    weights[-1] = gradient_magnitude[-1]

    baseline = weights.max() * 0.1
    weights = (weights + baseline).pow(power)
    weights = weights / weights.sum()

    return weights, gradients


def train_inverse_pinn_from_config(config, q_surf_data, theta_obs_data, et_data=None,
                                    ic_profile=None, device='auto'):
    """
    Train inverse PINN using configuration object.

    Args:
        config: InversePINNConfig object (loaded from YAML)
        q_surf_data: Surface flux (times, values) in [s, m/s]
        theta_obs_data: Observed moisture (times, values) in [s, m³/m³]
        et_data: Optional ET data for root uptake
        ic_profile: Optional initial condition profile
        device: Device override ('auto', 'cuda', 'cpu')

    Returns:
        model, losses, comps, sample_losses, sample_comps, sample_epochs
    """
    import torch

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return train_inverse_pinn(
        soil_params=config.soil_params,
        q_surf_data=q_surf_data,
        theta_obs_data=theta_obs_data,
        et_data=et_data,
        zr=config.zr,
        h_net_config=config.h_net_config,
        n_spline_knots=config.n_spline_knots,
        L=config.L,
        S_max=config.S_max,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        zb_initial=config.zb_initial,
        zb_max=config.zb_max,
        lambda_smooth=config.lambda_smooth,
        lambda_curv=config.lambda_curv,
        weight_update_freq=config.weight_update_freq,
        weight_lr=config.weight_lr,
        use_initial_scales=config.use_initial_scales,
        batch_size=config.batch_size,
        batch_size_flux=config.batch_size_flux,
        batch_size_theta=config.batch_size_theta,
        boundary_ratio=config.boundary_ratio,
        ic_profile=ic_profile,
        ic_type=config.ic_type,
        device=device,
        interp_ratio=config.interp_ratio,
        neighbor_ratio=config.neighbor_ratio,
        baseline_ratio=config.baseline_ratio,
        gradient_neighbor_expansion=config.gradient_neighbor_expansion,
        gradient_threshold=config.gradient_threshold,
        gradient_power=config.gradient_power,
        use_multi_gpu=config.use_multi_gpu,
        use_amp=config.use_amp,
        grad_accumulation_steps=config.grad_accumulation_steps,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_freq=config.checkpoint_freq,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        initial_pde_weight=config.initial_pde_weight,
        initial_flux_bc_weight=config.initial_flux_bc_weight,
        initial_wt_head_weight=config.initial_wt_head_weight,
        initial_data_weight=config.initial_data_weight,
        initial_ic_h_weight=config.initial_ic_h_weight,
        initial_ic_zb_weight=config.initial_ic_zb_weight,
        initial_zb_smooth_weight=config.initial_zb_smooth_weight,
        initial_zb_curv_weight=config.initial_zb_curv_weight,
        initial_zb_max_depth_weight=config.initial_zb_max_depth_weight,
        initial_zb_guide_weight=config.initial_zb_guide_weight,
        use_lbfgs=config.use_lbfgs,
        lbfgs_start_epoch=config.lbfgs_start_epoch,
        lbfgs_max_iter=config.lbfgs_max_iter,
        lbfgs_lr=config.lbfgs_lr,
        lbfgs_history_size=config.lbfgs_history_size,
        lbfgs_tolerance_grad=config.lbfgs_tolerance_grad,
        lbfgs_tolerance_change=config.lbfgs_tolerance_change,
        lbfgs_n_epochs=config.lbfgs_n_epochs,
        use_all_data_points=config.use_all_data_points,
        use_torch_compile=config.use_torch_compile,
        print_freq=config.print_freq,
    )


def train_inverse_pinn(
    soil_params,
    q_surf_data,          # Surface flux: (times, flux_values) in [s, m/s]
    theta_obs_data,       # Observed surface moisture: (times, theta_values)
    et_data=None,         # ET data for root uptake (optional)
    zr=0.5,               # Root zone depth [m]
    h_net_config=None,    # Pressure head network config
    n_spline_knots=20,    # Number of spline knots for z_b
    L=5.0,                # Characteristic length [m]
    S_max=1e-7,           # Maximum sink term [1/s]
    n_epochs=10000,
    learning_rate=1e-3,
    zb_initial=1.5,       # Initial water table depth [m]
    zb_max=None,          # Maximum water table depth constraint [m]
    lambda_smooth=1e-4,   # Smoothness regularization weight
    lambda_curv=1e-6,     # Curvature regularization weight
    weight_update_freq=100,
    weight_lr=0.5,
    use_initial_scales=True,
    batch_size=500,
    batch_size_flux=500,   # For flux BC (q_surf gradient-based sampling)
    batch_size_theta=500,  # For theta data matching (theta gradient-based sampling)
    boundary_ratio=0.7,
    ic_profile=None,
    ic_type='hydrostatic',
    device='cpu',
    interp_ratio=0.80,
    neighbor_ratio=0.05,
    baseline_ratio=0.15,
    gradient_neighbor_expansion=2,
    gradient_threshold=0.7,
    gradient_power=2.0,
    use_multi_gpu=True,
    use_amp=False,
    grad_accumulation_steps=1,
    checkpoint_dir='checkpoints_inverse',
    checkpoint_freq=None,
    resume_from_checkpoint=None,
    keep_last_n_checkpoints=3,
    initial_pde_weight=None,
    initial_flux_bc_weight=None,
    initial_wt_head_weight=None,
    initial_data_weight=None,
    initial_ic_h_weight=None,
    initial_ic_zb_weight=None,
    initial_zb_smooth_weight=None,
    initial_zb_curv_weight=None,
    initial_zb_max_depth_weight=None,
    initial_zb_guide_weight=None,  # Physics-based z_b guidance weight
    use_lbfgs=False,
    lbfgs_start_epoch=None,  # When to switch to L-BFGS (None = after Adam epochs)
    lbfgs_max_iter=50,
    lbfgs_lr=1.0,
    lbfgs_history_size=50,
    lbfgs_tolerance_grad=1e-7,
    lbfgs_tolerance_change=1e-9,
    lbfgs_n_epochs=1000,  # Number of L-BFGS iterations
    use_all_data_points=True,  # Skip gradient sampling, use all BC points directly
    use_torch_compile=False,   # Use torch.compile for model optimization
    print_freq=500,            # Reduce print frequency for speed
):
    """
    Train inverse PINN to find water table depth from surface observations.

    Given:
        - Surface flux q(0,t) as boundary condition
        - Observed surface moisture θ_obs(0,t) as fitting target

    Find:
        - Water table depth z_b(t) via spline optimization

    Args:
        soil_params: Soil parameters dict
        q_surf_data: Surface flux (times, values) in [s, m/s]
        theta_obs_data: Observed moisture (times, values) in [s, m³/m³]
        et_data: Optional ET data for root uptake
        zr: Root zone depth [m]
        h_net_config: Network config for pressure head
        n_spline_knots: Number of knots for z_b spline
        L: Characteristic length scale [m]
        S_max: Maximum sink term [1/s]
        n_epochs: Training epochs
        learning_rate: Adam learning rate
        zb_initial: Initial water table depth [m]
        lambda_smooth: Smoothness regularization weight
        lambda_curv: Curvature regularization weight
        ... (other training parameters)

    Returns:
        model: Trained RichardsInversePINN
        losses: Total loss history
        comps: Loss component histories
        sample_losses: Sample-based losses
        sample_comps: Sample-based components
        sample_epochs: Sample epochs
    """

    if q_surf_data is None:
        raise ValueError("q_surf_data is required for inverse problem")
    if theta_obs_data is None:
        raise ValueError("theta_obs_data is required for inverse problem")

    t_min_q = min(q_surf_data[0])
    t_max_q = max(q_surf_data[0])
    t_min_theta = min(theta_obs_data[0])
    t_max_theta = max(theta_obs_data[0])

    t_min = min(t_min_q, t_min_theta)
    t_max = max(t_max_q, t_max_theta)

    # Create tensors for gradient-based BC sampling
    # Theta data (for data matching loss)
    theta_times_t = torch.tensor(theta_obs_data[0], dtype=torch.float32, device=device).view(-1, 1)
    theta_values_t = torch.tensor(theta_obs_data[1], dtype=torch.float32, device=device)

    # q_surf data (for flux BC loss)
    qsurf_times_t = torch.tensor(q_surf_data[0], dtype=torch.float32, device=device).view(-1, 1)
    qsurf_values_t = torch.tensor(q_surf_data[1], dtype=torch.float32, device=device)

    normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)

    if h_net_config is None:
        h_net_config = {'hidden_dim': 64, 'num_layers': 4}

    # Constant z_b initialization for all spline knots
    print(f"z_b initialization: {zb_initial} m (constant for all {n_spline_knots} knots)")

    model = RichardsInversePINN(
        soil_params=soil_params,
        q_surf_data=q_surf_data,
        theta_obs_data=theta_obs_data,
        et_data=et_data,
        zr=zr,
        h_net_config=h_net_config,
        n_spline_knots=n_spline_knots,
        normalizer=normalizer,
        zb_initial=zb_initial,
        zb_max=zb_max,
        z_max_tilde=1.0,
        ic_profile=ic_profile,
        ic_type=ic_type,
        device=device,
    ).to(device)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = use_multi_gpu and n_gpus > 1

    if use_data_parallel:
        print(f"Multi-GPU mode: Using {n_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        model_core = model.module
    else:
        model_core = model

    scaler = None
    if use_amp:
        if not torch.cuda.is_available():
            print("Warning: AMP requested but CUDA not available. Disabling AMP.")
            use_amp = False
        else:
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training (AMP) enabled")

    # torch.compile optimization (PyTorch 2.0+)
    if use_torch_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")

    # Optimizer with different learning rates for h_net vs z_b
    # Use slightly lower learning rate for z_b (was 0.1x, now 0.5x for faster adaptation)
    zb_lr = learning_rate * 0.5  # 2x smaller for z_b

    h_net_params = list(model.h_net.parameters())
    zb_params = list(model.zb_spline.parameters())

    # Try fused Adam if available (PyTorch 2.0+ on CUDA) for better performance
    use_fused_adam = device == 'cuda' and hasattr(torch.optim.Adam, 'fused')
    try:
        if use_fused_adam:
            optimizer = Adam([
                {'params': h_net_params, 'lr': learning_rate},
                {'params': zb_params, 'lr': zb_lr}
            ], fused=True)
            print("Using fused Adam optimizer (faster on CUDA)")
        else:
            optimizer = Adam([
                {'params': h_net_params, 'lr': learning_rate},
                {'params': zb_params, 'lr': zb_lr}
            ])
    except TypeError:
        # Fused not available in this PyTorch version
        optimizer = Adam([
            {'params': h_net_params, 'lr': learning_rate},
            {'params': zb_params, 'lr': zb_lr}
        ])

    print(f"Using separate learning rates: h_net={learning_rate:.2e}, z_b={zb_lr:.2e}")

    use_fixed_weights = weight_update_freq >= n_epochs
    if use_fixed_weights:
        print(f"Fixed weights mode enabled")

    weight_manager = InverseWeightManager(
        use_initial_scales=use_initial_scales,
        weight_lr=weight_lr,
        use_fixed_weights=use_fixed_weights,
        initial_pde_weight=initial_pde_weight,
        initial_flux_bc_weight=initial_flux_bc_weight,
        initial_wt_head_weight=initial_wt_head_weight,
        initial_data_weight=initial_data_weight,
        initial_ic_h_weight=initial_ic_h_weight,
        initial_ic_zb_weight=initial_ic_zb_weight,
        initial_zb_smooth_weight=initial_zb_smooth_weight,
        initial_zb_curv_weight=initial_zb_curv_weight,
        initial_zb_max_depth_weight=initial_zb_max_depth_weight,
        initial_zb_guide_weight=initial_zb_guide_weight,
    )

    logger = InverseTrainingLogger()
    sampling = InverseSamplingHelpers()

    start_epoch = 0
    checkpoint_path_list = []

    if checkpoint_freq is not None and checkpoint_freq > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpointing enabled: saving every {checkpoint_freq} epochs to {checkpoint_dir}/")

    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)
            model_core.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            if 'weight_manager_state' in checkpoint:
                wm_state = checkpoint['weight_manager_state']
                weight_manager.weights = wm_state.get('weights', weight_manager.weights)
                weight_manager.weight_history = {k: [weight_manager.weights[k]] for k in weight_manager.weights}
                weight_manager.grad_ema = wm_state.get('grad_ema', weight_manager.grad_ema)

            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint not found: {resume_from_checkpoint}")

    print("\n" + "="*70)
    print("INVERSE PINN TRAINING")
    print("="*70)
    print(f"Objective: Find z_b(t) from surface flux and moisture observations")
    print(f"Surface flux data: {len(q_surf_data[0])} points")
    print(f"Observed moisture data: {len(theta_obs_data[0])} points")
    print(f"Time range: [{t_min:.1f}, {t_max:.1f}] s ({(t_max-t_min)/86400:.2f} days)")
    print(f"Spline knots: {n_spline_knots}")
    print(f"Initial z_b: {zb_initial:.2f} m")
    print(f"\nRegularization:")
    print(f"  Smoothness (lambda_smooth): {lambda_smooth:.2e}")
    print(f"  Curvature (lambda_curv): {lambda_curv:.2e}")
    print(f"\nTraining: {n_epochs} epochs, lr={learning_rate:.2e}")
    print(f"Device: {device}")
    print(f"Initial weights: {weight_manager.get_weights()}")
    print("="*70 + "\n")

    # Print initial z_b knots for debugging
    zb_knots_initial = torch.exp(model.zb_spline.log_zb_knots).detach().cpu().numpy()
    print(f"\nInitial z_b knots (first 5): {zb_knots_initial[:5]}")
    print(f"Initial z_b knots (last 5): {zb_knots_initial[-5:]}")
    print(f"Initial z_b range: [{zb_knots_initial.min():.3f}, {zb_knots_initial.max():.3f}] m\n")

    # Performance optimization: prepare BC data based on strategy
    if use_all_data_points:
        # Use ALL data points directly - faster for small datasets (e.g., 228 points)
        # Clone and enable gradients for each epoch (will be reused)
        t_bc_flux_all = qsurf_times_t.clone().requires_grad_(True)
        t_bc_theta_all = theta_times_t.clone().requires_grad_(True)
        print(f"PERFORMANCE: Using ALL data points directly ({len(qsurf_times_t)} flux, {len(theta_times_t)} theta)")
    else:
        # Precompute gradient weights ONCE (not every epoch)
        qsurf_grad_weights, _ = _precompute_gradient_weights(qsurf_times_t, qsurf_values_t, device, power=gradient_power)
        theta_grad_weights, _ = _precompute_gradient_weights(theta_times_t, theta_values_t, device, power=gradient_power)
        print(f"PERFORMANCE: Using gradient-based sampling (precomputed weights)")

    # Cache Beta distributions for PDE sampling (avoid recreation every epoch)
    beta_surface = torch.distributions.Beta(
        torch.tensor(1.0, device=device),
        torch.tensor(3.0, device=device)
    )
    beta_bottom = torch.distributions.Beta(
        torch.tensor(3.0, device=device),
        torch.tensor(1.0, device=device)
    )

    # Main training loop (Adam only)
    import time as time_module
    train_start_time = time_module.time()

    for epoch in range(start_epoch, n_epochs):
        if epoch % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        # Sample PDE training points (optimized with cached Beta distributions)
        t_col = torch.rand(batch_size, 1, device=device) * t_max
        t_col.requires_grad_(True)

        n_boundary = int(batch_size * boundary_ratio)
        n_interior = batch_size - n_boundary
        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        u_surface = beta_surface.sample((n_surface,))
        u_bottom = beta_bottom.sample((n_bottom,))
        u_interior = torch.rand(n_interior, device=device)
        u_col = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        with torch.no_grad():
            zb_col = model_core.predict_water_table(t_col)
        z_col = (-u_col * zb_col).requires_grad_(True)

        # BC time points - use all data or sampled
        if use_all_data_points:
            # Use all data points directly (fastest for small datasets)
            t_bc_flux = qsurf_times_t.clone().requires_grad_(True)
            t_bc_theta = theta_times_t.clone().requires_grad_(True)
        else:
            t_bc_flux, _ = gradient_based_sampling(
                qsurf_times_t, qsurf_values_t, batch_size_flux,
                device=device, use_interpolation=True,
                interp_ratio=interp_ratio, neighbor_ratio=neighbor_ratio,
                baseline_ratio=baseline_ratio, neighbor_expansion=gradient_neighbor_expansion,
                gradient_threshold=gradient_threshold, power=gradient_power
            )
            t_bc_theta, _ = gradient_based_sampling(
                theta_times_t, theta_values_t, batch_size_theta,
                device=device, use_interpolation=True,
                interp_ratio=interp_ratio, neighbor_ratio=neighbor_ratio,
                baseline_ratio=baseline_ratio, neighbor_expansion=gradient_neighbor_expansion,
                gradient_threshold=gradient_threshold, power=gradient_power
            )

        z_ic, t_ic = sampling.sample_initial_condition_points(
            model_core, batch_size, t_min, device
        )

        # Compute losses (Adam only)
        if use_amp:
            with torch.cuda.amp.autocast():
                losses = compute_inverse_losses(
                    model, z_col, t_col, t_bc_flux, t_bc_theta, z_ic, t_ic,
                    lambda_smooth=lambda_smooth, lambda_curv=lambda_curv
                )
        else:
            losses = compute_inverse_losses(
                model, z_col, t_col, t_bc_flux, t_bc_theta, z_ic, t_ic,
                lambda_smooth=lambda_smooth, lambda_curv=lambda_curv
            )

        weights = weight_manager.get_weights()
        if weight_manager.is_using_fixed_weights():
            weighted_losses, gradients, total_loss = apply_weights_fixed_mode(losses, weights)
        else:
            weighted_losses, gradients, total_loss = apply_weights_and_compute_gradients(
                losses, weights, model
            )

        total_loss = total_loss / grad_accumulation_steps

        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (epoch + 1) % grad_accumulation_steps == 0:
            if not weight_manager.is_using_fixed_weights():
                total_grad_norm = compute_total_grad_norm(model)
                gradients["total"] = total_grad_norm
            else:
                gradients["total"] = 0.0

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            gradients["total"] = 0.0

        if (epoch + 1) % weight_update_freq == 0 and epoch > 0:
            weight_manager.update(gradients)
            weight_manager.print_update(epoch)

        logger.record_losses(total_loss, weighted_losses)
        logger.record_gradients(gradients)
        weight_manager.record_history()

        if checkpoint_freq is not None and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model_core.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'weight_manager_state': {
                    'weights': weight_manager.weights,
                    'grad_ema': weight_manager.grad_ema,
                },
                'training_config': {
                    'n_epochs': n_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'n_spline_knots': n_spline_knots,
                    'lambda_smooth': lambda_smooth,
                    'lambda_curv': lambda_curv,
                },
                'normalization_params': {
                    'soil_params': soil_params,
                    'L': L,
                    'S_max': S_max,
                    'zr': zr,
                },
                'rng_state': torch.get_rng_state(),
            }

            if use_amp and scaler is not None:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            if torch.cuda.is_available():
                checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

            torch.save(checkpoint_dict, checkpoint_path)
            checkpoint_path_list.append(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            if keep_last_n_checkpoints is not None and len(checkpoint_path_list) > keep_last_n_checkpoints:
                old_checkpoint = checkpoint_path_list.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

        # Print progress (use configurable frequency)
        if (epoch + 1) % print_freq == 0 or epoch == 0:
            # Get current z_b range
            with torch.no_grad():
                t_test = torch.tensor([[t_min], [t_max]], device=device)
                zb_test = model_core.predict_water_table(t_test).flatten()
                zb_min, zb_max_val = zb_test[0].item(), zb_test[1].item()

            elapsed = time_module.time() - train_start_time
            epochs_done = epoch - start_epoch + 1
            epochs_per_sec = epochs_done / elapsed if elapsed > 0 else 0
            eta_sec = (n_epochs - epoch - 1) / epochs_per_sec if epochs_per_sec > 0 else 0

            print(f"\n[Epoch {epoch+1}/{n_epochs}] ({epochs_per_sec:.1f} ep/s, ETA: {eta_sec/60:.1f} min)")
            print(f"  Total loss: {total_loss.item() if hasattr(total_loss, 'item') else total_loss:.3e}")
            print(f"  Components: PDE={losses['pde'].item():.3e}, FluxBC={losses['flux_bc'].item():.3e}, "
                  f"WTHead={losses['wt_head'].item():.3e}")
            print(f"  Data match: {losses['data'].item():.3e}")
            print(f"  Regularization: smooth={losses['zb_smooth'].item():.3e}, curv={losses['zb_curv'].item():.3e}")
            print(f"  z_b range: [{zb_min:.3f}, {zb_max_val:.3f}] m")

        # Sample loss (less frequent for speed)
        if (epoch + 1) % (print_freq * 2) == 0 or epoch == 0:
            sample_losses_dict = compute_full_sample_loss_inverse(
                model_core, qsurf_times_t, theta_times_t, t_min, device,
                t_max=t_max, boundary_ratio=boundary_ratio,
                sample_size=5000, lambda_smooth=lambda_smooth, lambda_curv=lambda_curv
            )
            logger.record_sample_losses(epoch, sample_losses_dict, weights)

            total_sample_loss = sum(weights[key] * sample_losses_dict[key] for key in sample_losses_dict)
            print(f"\n  [Sample Loss at epoch {epoch+1}] Total={total_sample_loss:.3e}")
            print(f"    PDE={sample_losses_dict['pde']:.3e}, FluxBC={sample_losses_dict['flux_bc']:.3e}")
            print(f"    WTHead={sample_losses_dict['wt_head']:.3e}, Data={sample_losses_dict['data']:.3e}")

    # ==================== L-BFGS REFINEMENT PHASE ====================
    if use_lbfgs:
        print("\n" + "="*70)
        print("L-BFGS REFINEMENT PHASE")
        print("="*70)
        print(f"L-BFGS config: max_iter={lbfgs_max_iter}, lr={lbfgs_lr}, history={lbfgs_history_size}")
        print(f"Running {lbfgs_n_epochs} L-BFGS epochs...")

        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change,
            line_search_fn='strong_wolfe'
        )

        # Use larger batch for L-BFGS (full-batch or large mini-batch)
        lbfgs_batch_size = min(batch_size * 4, 20000)
        lbfgs_batch_flux = min(batch_size_flux * 4, 2000)
        lbfgs_batch_theta = min(batch_size_theta * 4, 2000)

        weights = weight_manager.get_weights()

        for lbfgs_epoch in range(lbfgs_n_epochs):
            # Sample PDE points (use larger batch for stability)
            t_col = torch.rand(lbfgs_batch_size, 1, device=device) * t_max
            t_col.requires_grad_(True)

            n_boundary = int(lbfgs_batch_size * boundary_ratio)
            n_interior = lbfgs_batch_size - n_boundary
            n_surface = n_boundary // 2
            n_bottom = n_boundary - n_surface

            u_surface = beta_surface.sample((n_surface,))
            u_bottom = beta_bottom.sample((n_bottom,))
            u_interior = torch.rand(n_interior, device=device)
            u_col = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

            with torch.no_grad():
                zb_col = model_core.predict_water_table(t_col)
            z_col = (-u_col * zb_col).requires_grad_(True)

            # BC points - use all data for L-BFGS (better for full-batch optimization)
            if use_all_data_points:
                t_bc_flux = qsurf_times_t.clone().requires_grad_(True)
                t_bc_theta = theta_times_t.clone().requires_grad_(True)
            else:
                t_bc_flux, _ = gradient_based_sampling(
                    qsurf_times_t, qsurf_values_t, lbfgs_batch_flux,
                    device=device, use_interpolation=True,
                    interp_ratio=interp_ratio, neighbor_ratio=neighbor_ratio,
                    baseline_ratio=baseline_ratio, neighbor_expansion=gradient_neighbor_expansion,
                    gradient_threshold=gradient_threshold, power=gradient_power
                )
                t_bc_theta, _ = gradient_based_sampling(
                    theta_times_t, theta_values_t, lbfgs_batch_theta,
                    device=device, use_interpolation=True,
                    interp_ratio=interp_ratio, neighbor_ratio=neighbor_ratio,
                    baseline_ratio=baseline_ratio, neighbor_expansion=gradient_neighbor_expansion,
                    gradient_threshold=gradient_threshold, power=gradient_power
                )

            z_ic, t_ic = sampling.sample_initial_condition_points(
                model_core, lbfgs_batch_size, t_min, device
            )

            def closure():
                lbfgs_optimizer.zero_grad()
                losses = compute_inverse_losses(
                    model, z_col, t_col, t_bc_flux, t_bc_theta, z_ic, t_ic,
                    lambda_smooth=lambda_smooth, lambda_curv=lambda_curv
                )
                total_loss = sum(weights[key] * losses[key] for key in losses)

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    return torch.tensor(float('inf'), device=device)

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                return total_loss

            try:
                loss_val = lbfgs_optimizer.step(closure)

                if loss_val is None or torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(f"\n  WARNING: L-BFGS produced NaN/Inf at epoch {lbfgs_epoch+1}")
                    print("  Reducing learning rate and retrying...")
                    # Reduce learning rate and recreate optimizer
                    lbfgs_lr = lbfgs_lr * 0.5
                    if lbfgs_lr < 0.01:
                        print("  Learning rate too small, stopping L-BFGS")
                        break
                    lbfgs_optimizer = torch.optim.LBFGS(
                        model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter,
                        history_size=lbfgs_history_size, tolerance_grad=lbfgs_tolerance_grad,
                        tolerance_change=lbfgs_tolerance_change, line_search_fn='strong_wolfe'
                    )
                    continue

            except RuntimeError as e:
                print(f"\n  WARNING: L-BFGS error at epoch {lbfgs_epoch+1}: {e}")
                print("  Stopping L-BFGS phase.")
                break

            if (lbfgs_epoch + 1) % 50 == 0 or lbfgs_epoch == 0:
                with torch.no_grad():
                    t_test = torch.tensor([[t_min], [t_max]], device=device)
                    zb_test = model_core.predict_water_table(t_test).flatten()
                    zb_min_val, zb_max_val = zb_test[0].item(), zb_test[1].item()

                print(f"\n[L-BFGS Epoch {lbfgs_epoch+1}/{lbfgs_n_epochs}]")
                print(f"  Loss: {loss_val.item():.3e}")
                print(f"  z_b range: [{zb_min_val:.3f}, {zb_max_val:.3f}] m")

            logger.record_losses(loss_val, {k: weights[k] * 0 for k in weights})  # Placeholder

        print("\nL-BFGS phase complete.")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    with torch.no_grad():
        t_test = torch.linspace(t_min, t_max, 10, device=device).view(-1, 1)
        zb_test = model_core.predict_water_table(t_test).flatten()
        print(f"Final z_b(t) values:")
        for i in range(len(t_test)):
            print(f"  t={t_test[i].item()/86400:.2f} days: z_b={zb_test[i].item():.3f} m")

    if checkpoint_freq is not None:
        final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
        checkpoint_dict = {
            'epoch': n_epochs - 1,
            'model_state_dict': model_core.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'weight_manager_state': {
                'weights': weight_manager.weights,
                'grad_ema': weight_manager.grad_ema,
            },
            'training_config': {
                'n_epochs': n_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'n_spline_knots': n_spline_knots,
                'lambda_smooth': lambda_smooth,
                'lambda_curv': lambda_curv,
            },
            'normalization_params': {
                'soil_params': soil_params,
                'L': L,
                'S_max': S_max,
                'zr': zr,
            },
            'rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

        torch.save(checkpoint_dict, final_checkpoint_path)
        print(f"\nFinal checkpoint saved: {final_checkpoint_path}")

    losses_np = logger.get_losses_numpy()
    comps_np = logger.get_comps_numpy()

    return model_core, losses_np, comps_np, logger.sample_losses, logger.sample_comps, logger.sample_epochs
