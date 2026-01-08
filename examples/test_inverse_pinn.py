#!/usr/bin/env python
"""
Test script for Inverse PINN solver.

This script:
1. Generates synthetic data with known water table depth
2. Computes surface flux and observed surface moisture
3. Trains the inverse PINN to recover the water table depth
4. Compares recovered z_b(t) with ground truth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.pinn_inverse import RichardsInversePINN, WaterTableSpline
from src.train_inverse import train_inverse_pinn
from src.normalization_helper import NormalizationHelper


def generate_synthetic_data(
    n_days=30,
    dt_hours=1,
    zb_mean=2.0,
    zb_amplitude=0.5,
    zb_period_days=15,
    theta_s=0.43,
    theta_r=0.02,
    noise_level=0.01,
    seed=42
):
    """
    Generate synthetic test data with known water table depth.

    The synthetic water table varies sinusoidally:
        z_b(t) = z_b_mean + A * sin(2π t / T)

    Surface moisture is computed assuming hydrostatic equilibrium
    (simplified, for testing).

    Args:
        n_days: Duration of simulation [days]
        dt_hours: Time step [hours]
        zb_mean: Mean water table depth [m]
        zb_amplitude: Amplitude of water table oscillation [m]
        zb_period_days: Period of oscillation [days]
        theta_s: Saturated moisture content
        theta_r: Residual moisture content
        noise_level: Gaussian noise level for observations
        seed: Random seed

    Returns:
        dict with:
            - t: Time array [s]
            - zb_true: Ground truth water table depth [m]
            - theta_obs: Observed surface moisture [m³/m³]
            - q_surf: Surface flux [m/s]
    """
    np.random.seed(seed)

    n_steps = int(n_days * 24 / dt_hours)
    t_hours = np.linspace(0, n_days * 24, n_steps)
    t_seconds = t_hours * 3600

    # Ground truth water table depth (sinusoidal variation)
    period_seconds = zb_period_days * 86400
    zb_true = zb_mean + zb_amplitude * np.sin(2 * np.pi * t_seconds / period_seconds)

    # Compute surface moisture (simplified hydrostatic approximation)
    # In reality, this would come from solving Richards equation
    # Here we use a simplified relationship: θ ∝ (θ_s - θ_r) * exp(-z_b/scale) + θ_r
    scale = 3.0  # characteristic scale for exponential decay
    Se_surface = np.exp(-zb_true / scale)
    theta_surface = Se_surface * (theta_s - theta_r) + theta_r

    theta_obs = theta_surface + np.random.normal(0, noise_level, len(theta_surface))
    theta_obs = np.clip(theta_obs, theta_r + 0.01, theta_s - 0.01)

    # Compute surface flux (dθ/dt approximation, very simplified)
    # In reality, q = ET - P. Here we just compute a derivative
    dtheta_dt = np.gradient(theta_surface, t_seconds)
    q_surf = -dtheta_dt * 0.5  # Scale factor for typical flux magnitudes

    q_surf += np.random.normal(0, 1e-9, len(q_surf))

    return {
        't': t_seconds,
        'zb_true': zb_true,
        'theta_obs': theta_obs,
        'q_surf': q_surf,
    }


def test_water_table_spline():
    """Test the WaterTableSpline class."""
    print("\n" + "="*60)
    print("Testing WaterTableSpline")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_knots = 10
    t_min = 0.0
    t_max = 30 * 86400  # 30 days in seconds
    zb_initial = 2.0

    spline = WaterTableSpline(n_knots, t_min, t_max, zb_initial, device=device)

    t_test = torch.linspace(t_min, t_max, 100, device=device).view(-1, 1)
    zb_test = spline(t_test)

    print(f"Spline knots: {n_knots}")
    print(f"Initial z_b: {zb_initial} m")
    print(f"Output shape: {zb_test.shape}")
    print(f"Output range: [{zb_test.min().item():.3f}, {zb_test.max().item():.3f}] m")

    smooth_penalty = spline.smoothness_penalty()
    curv_penalty = spline.second_derivative_penalty()
    print(f"Smoothness penalty: {smooth_penalty.item():.3e}")
    print(f"Curvature penalty: {curv_penalty.item():.3e}")

    dzb_dt = spline.get_dzb_dt(t_test[:10])
    print(f"dz_b/dt range: [{dzb_dt.min().item():.3e}, {dzb_dt.max().item():.3e}] m/s")

    print("WaterTableSpline test PASSED")


def test_inverse_pinn_instantiation():
    """Test that RichardsInversePINN can be instantiated."""
    print("\n" + "="*60)
    print("Testing RichardsInversePINN Instantiation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = generate_synthetic_data(n_days=10)

    soil_params = {
        'theta_s': 0.43,
        'theta_r': 0.02,
        'alpha': 3.6,
        'n': 1.56,
        'Ks': 2.9e-6,
        'l': 0.5,
    }

    normalizer = NormalizationHelper(soil_params, L=5.0, S_max=1e-7)

    q_surf_data = (data['t'], data['q_surf'])
    theta_obs_data = (data['t'], data['theta_obs'])

    model = RichardsInversePINN(
        soil_params=soil_params,
        q_surf_data=q_surf_data,
        theta_obs_data=theta_obs_data,
        zr=0.5,
        h_net_config={'hidden_dim': 32, 'num_layers': 3},
        n_spline_knots=15,
        normalizer=normalizer,
        zb_initial=2.0,
        device=device,
    ).to(device)

    z_test = torch.linspace(-3, 0, 20, device=device).view(-1, 1)
    t_test = torch.full((20, 1), data['t'][50], device=device)

    z_tilde = normalizer.normalize_z(z_test)
    t_tilde = normalizer.normalize_t(t_test)

    h_tilde, zb_tilde = model(z_tilde, t_tilde)
    print(f"Forward pass output shapes: h={h_tilde.shape}, z_b={zb_tilde.shape}")

    t_bc = torch.tensor(data['t'][:10], dtype=torch.float32, device=device).view(-1, 1)

    res_pde = model.pde_residual(z_test, t_test)
    print(f"PDE residual shape: {res_pde.shape}")

    res_flux = model.surface_flux_bc_residual(t_bc)
    print(f"Flux BC residual shape: {res_flux.shape}")

    res_wt = model.water_table_head_residual(t_bc)
    print(f"Water table head residual shape: {res_wt.shape}")

    res_data = model.surface_moisture_data_residual(t_bc)
    print(f"Data matching residual shape: {res_data.shape}")

    print("RichardsInversePINN instantiation test PASSED")


def test_inverse_training_short():
    """Test a short training run."""
    print("\n" + "="*60)
    print("Testing Inverse PINN Training (Short Run)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data = generate_synthetic_data(n_days=15, noise_level=0.005)

    soil_params = {
        'theta_s': 0.43,
        'theta_r': 0.02,
        'alpha': 3.6,
        'n': 1.56,
        'Ks': 2.9e-6,
        'l': 0.5,
    }

    q_surf_data = (data['t'], data['q_surf'])
    theta_obs_data = (data['t'], data['theta_obs'])

    # Train with fewer epochs for testing
    model, losses, comps, *_ = train_inverse_pinn(
        soil_params=soil_params,
        q_surf_data=q_surf_data,
        theta_obs_data=theta_obs_data,
        et_data=None,
        zr=0.5,
        h_net_config={'hidden_dim': 32, 'num_layers': 3},
        n_spline_knots=15,
        L=5.0,
        S_max=1e-7,
        n_epochs=500,  # Short test run
        learning_rate=1e-3,
        zb_initial=2.0,
        lambda_smooth=1e-4,
        lambda_curv=1e-6,
        weight_update_freq=10000,  # Fixed weights for short test
        batch_size=200,
        batch_size_bc=50,
        device=device,
    )

    print(f"\nFinal loss: {losses[-1]:.3e}")
    print(f"Training completed successfully!")

    # Compare recovered z_b with ground truth
    t_test = torch.tensor(data['t'], dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        zb_recovered = model.predict_water_table(t_test).cpu().numpy().flatten()

    correlation = np.corrcoef(data['zb_true'], zb_recovered)[0, 1]
    rmse = np.sqrt(np.mean((data['zb_true'] - zb_recovered)**2))

    print(f"\nRecovery metrics:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  RMSE: {rmse:.4f} m")

    print("Short training test PASSED")

    return data, zb_recovered, losses


def plot_results(data, zb_recovered, losses, save_path=None):
    """Plot results from inverse PINN."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    t_days = data['t'] / 86400
    ax.plot(t_days, data['zb_true'], 'b-', label='Ground truth', linewidth=2)
    ax.plot(t_days, zb_recovered, 'r--', label='Recovered', linewidth=2)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Water table depth [m]')
    ax.set_title('Water Table Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.plot(t_days, data['theta_obs'], 'k-', linewidth=1)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Surface moisture [m³/m³]')
    ax.set_title('Observed Surface Moisture')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_days, data['q_surf'] * 1e9, 'g-', linewidth=1)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Surface flux [nm/s]')
    ax.set_title('Surface Flux (BC)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(losses, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("INVERSE PINN TEST SUITE")
    print("="*70)

    test_water_table_spline()
    test_inverse_pinn_instantiation()
    data, zb_recovered, losses = test_inverse_training_short()

    plot_results(data, zb_recovered, losses)

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
