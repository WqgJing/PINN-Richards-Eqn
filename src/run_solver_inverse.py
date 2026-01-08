#!/usr/bin/env python
"""
Run Inverse PINN solver from YAML configuration.

Usage:
    python -m src.run_solver_inverse --config configs/inverse_baseline.yaml
    python -m src.run_solver_inverse --config configs/inverse_baseline.yaml --checkpoint checkpoints_inverse/checkpoint_epoch_5000.pt
    python -m src.run_solver_inverse --config configs/inverse_US-PAS_2021.yaml  # Real data

The config file specifies:
    - Soil parameters (van Genuchten)
    - Data paths (flux, observed moisture)
    - Network architecture
    - Spline configuration
    - Regularization weights
    - Training hyperparameters
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader_inverse import load_inverse_config
from src.train_inverse import train_inverse_pinn_from_config
from src.visualization_inverse import save_inverse_figures


def load_npz_data(path, time_key='times', value_key='values'):
    """Load data from NPZ file."""
    data = np.load(path)
    times = data[time_key]
    values = data[value_key]
    return (times, values)


def load_combined_npz(config):
    """
    Load data from combined NPZ file (e.g., AmeriFlux processed data).

    Returns:
        q_surf_data: tuple (times, q_surf values)
        theta_obs_data: tuple (times, theta_obs values)
        zb_true: ground truth water table depth (if available)
    """
    npz_path = config._config['data'].get('npz_path', None)
    if npz_path is None:
        return None, None, None

    if not os.path.isabs(npz_path):
        # Try relative to config file directory
        config_dir = os.path.dirname(config.config_path)
        npz_path_abs = os.path.join(config_dir, '..', npz_path)
        if not os.path.exists(npz_path_abs):
            npz_path_abs = npz_path
        npz_path = npz_path_abs

    if not os.path.exists(npz_path):
        return None, None, None

    print(f"  Loading from combined NPZ: {npz_path}")
    data = np.load(npz_path)

    time_key = config._config['data'].get('time_key', 'times')
    flux_key = config._config['data'].get('flux_key', 'qsurf')
    theta_key = config._config['data'].get('theta_key', 'smap_sm')
    zb_true_key = config._config['data'].get('zb_true_key', 'zb')

    times = data[time_key]
    q_surf = data[flux_key]
    theta_obs = data[theta_key]

    zb_true = None
    if zb_true_key in data:
        zb_true = data[zb_true_key]

    print(f"  Loaded {len(times)} time points")
    print(f"  Time range: {times[0]:.0f} to {times[-1]:.0f} seconds ({times[-1]/86400:.1f} days)")
    print(f"  q_surf range: [{q_surf.min():.3e}, {q_surf.max():.3e}] m/s")
    print(f"  theta_obs range: [{theta_obs.min():.4f}, {theta_obs.max():.4f}]")
    if zb_true is not None:
        print(f"  z_b true range: [{zb_true.min():.3f}, {zb_true.max():.3f}] m")

    return (times, q_surf), (times, theta_obs), zb_true


def main():
    parser = argparse.ArgumentParser(description='Run Inverse PINN solver')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    args = parser.parse_args()

    print("="*70)
    print("INVERSE PINN SOLVER")
    print("="*70)
    print(f"Config: {args.config}")

    config = load_inverse_config(args.config)
    config.print_summary()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    zb_true = None  # Ground truth (if available, for validation)

    if args.synthetic:
        print("\nUsing SYNTHETIC data for testing...")
        from examples.test_inverse_pinn import generate_synthetic_data
        data = generate_synthetic_data(n_days=30, noise_level=0.005, seed=config.seed)
        q_surf_data = (data['t'], data['q_surf'])
        theta_obs_data = (data['t'], data['theta_obs'])
        zb_true = data['zb_true']
        print(f"  Generated {len(data['t'])} time points over 30 days")
    else:
        print("\nLoading data from files...")
        q_surf_data, theta_obs_data, zb_true = load_combined_npz(config)

        if q_surf_data is None:
            try:
                q_surf_data = load_npz_data(config.flux_data_path)
                print(f"  Loaded flux data: {len(q_surf_data[0])} points from {config.flux_data_path}")
            except Exception as e:
                print(f"  ERROR loading flux data: {e}")
                print("  Use --synthetic flag to run with synthetic test data")
                sys.exit(1)

            try:
                theta_obs_data = load_npz_data(config.theta_obs_data_path)
                print(f"  Loaded theta_obs data: {len(theta_obs_data[0])} points from {config.theta_obs_data_path}")
            except Exception as e:
                print(f"  ERROR loading theta_obs data: {e}")
                print("  Use --synthetic flag to run with synthetic test data")
                sys.exit(1)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nDevice: {device}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    model, losses, comps, sample_losses, sample_comps, sample_epochs = train_inverse_pinn_from_config(
        config=config,
        q_surf_data=q_surf_data,
        theta_obs_data=theta_obs_data,
        et_data=None,
        ic_profile=None,
        device=device,
    )

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    results_path = os.path.join(config.output_dir, 'training_results.npz')
    np.savez(
        results_path,
        losses=np.array(losses),
        sample_losses=np.array(sample_losses),
        sample_epochs=np.array(sample_epochs),
        **{f'comp_{k}': np.array(v) for k, v in comps.items()},
        **{f'sample_comp_{k}': np.array(v) for k, v in sample_comps.items()},
    )
    print(f"Results saved to: {results_path}")

    t_eval = torch.tensor(theta_obs_data[0], dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        zb_pred = model.predict_water_table(t_eval).cpu().numpy().flatten()
        theta_pred = model.predict_surface_moisture(t_eval).cpu().numpy().flatten()

    predictions_path = os.path.join(config.output_dir, 'predictions.npz')
    save_dict = {
        'times': theta_obs_data[0],
        'zb_predicted': zb_pred,
        'theta_predicted': theta_pred,
        'theta_observed': theta_obs_data[1],
        'q_surf': q_surf_data[1],
    }
    if zb_true is not None:
        save_dict['zb_true'] = zb_true
    np.savez(predictions_path, **save_dict)
    print(f"Predictions saved to: {predictions_path}")

    print("\nGenerating visualization figures...")
    save_inverse_figures(
        model=model,
        theta_obs_data=theta_obs_data,
        q_surf_data=q_surf_data,
        soil_params=config.soil_params,
        losses=losses,
        comps=comps,
        output_dir=config.output_dir,
        zb_true=zb_true,
        sample_losses=sample_losses,
        sample_comps=sample_comps,
        sample_epochs=sample_epochs,
        device=device,
    )

    print("\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    print(f"Final loss: {losses[-1]:.3e}")
    print(f"z_b range: [{zb_pred.min():.3f}, {zb_pred.max():.3f}] m")

    theta_rmse = np.sqrt(np.mean((theta_pred - theta_obs_data[1])**2))
    theta_corr = np.corrcoef(theta_pred, theta_obs_data[1])[0, 1]
    print(f"\nSurface moisture matching:")
    print(f"  RMSE: {theta_rmse:.4f}")
    print(f"  Correlation: {theta_corr:.4f}")

    # Compute correlation with ground truth if available
    if zb_true is not None:
        correlation = np.corrcoef(zb_true, zb_pred)[0, 1]
        rmse = np.sqrt(np.mean((zb_true - zb_pred)**2))
        bias = np.mean(zb_pred - zb_true)
        print(f"\nWater table recovery metrics (vs ground truth):")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  RMSE: {rmse:.4f} m")
        print(f"  Bias: {bias:.4f} m")

    print("\nDone!")


if __name__ == '__main__':
    main()
