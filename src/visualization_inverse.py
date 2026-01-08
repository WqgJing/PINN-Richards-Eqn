"""
Visualization functions for Inverse PINN training results.

Matches the style of visualization.py (forward model) with:
- 6-subplot comprehensive results
- 8-subplot loss evolution
- Consistent styling (line plots, no fills, grid)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_inverse_results(model, theta_obs_data, q_surf_data, soil_params,
                         zb_true=None, n_t=200, n_z=100, device='cpu'):
    """
    Comprehensive plotting for inverse PINN results with 6 subplots.

    Args:
        model: Trained RichardsInversePINN model
        theta_obs_data: tuple (times, theta_obs) - observed surface moisture
        q_surf_data: tuple (times, q_surf) - surface flux BC
        soil_params: dict with 'theta_s', 'theta_r', 'alpha', 'n', 'Ks'
        zb_true: optional ground truth water table depth array
        n_t: number of time points for visualization grid
        n_z: number of depth points for visualization grid
        device: computation device
    """
    model.eval()

    t_obs, theta_obs = theta_obs_data
    t_flux, q_surf = q_surf_data

    t_min, t_max = min(t_obs), max(t_obs)
    t_lin = torch.linspace(t_min, t_max, n_t, dtype=torch.float32).to(device)

    # Get water table depth over time
    with torch.no_grad():
        zb_vals = model.predict_water_table(t_lin.view(-1, 1)).cpu().numpy().flatten()

    z_min = -np.max(zb_vals) - 0.5
    z_max = 0.0
    z_lin = torch.linspace(z_min, z_max, n_z, dtype=torch.float32).to(device)

    # Create meshgrid for h(z,t)
    T, Z = torch.meshgrid(t_lin, z_lin, indexing="ij")
    t_flat = T.reshape(-1, 1)
    z_flat = Z.reshape(-1, 1)

    # Compute h(z,t) over the grid
    with torch.no_grad():
        h_flat, _ = model.predict_head(z_flat, t_flat)
    H = h_flat.reshape(n_t, n_z).cpu().numpy()

    # Mask out regions below water table
    T_np = T.cpu().numpy()
    Z_np = Z.cpu().numpy()

    H_masked = H.copy()
    for i, t_val in enumerate(t_lin.cpu().numpy()):
        zb_at_t = zb_vals[i]
        mask = Z_np[i, :] < -zb_at_t
        H_masked[i, mask] = np.nan

    with torch.no_grad():
        t_eval = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)
        theta_pred = model.predict_surface_moisture(t_eval).cpu().numpy().flatten()
        zb_at_obs = model.predict_water_table(t_eval).cpu().numpy().flatten()

    # Predict surface flux at flux observation times
    # NOTE: This is outside no_grad() because predict_surface_flux needs gradients
    # to compute dh/dz via autograd
    t_flux_tensor = torch.tensor(t_flux, dtype=torch.float32, device=device).view(-1, 1)
    q_pred = model.predict_surface_flux(t_flux_tensor).cpu().numpy().flatten()

    with torch.no_grad():
        z_wt = (-torch.tensor(zb_vals, dtype=torch.float32)).to(device).view(-1, 1)
        t_wt = t_lin.view(-1, 1)
        h_at_wt, _ = model.predict_head(z_wt, t_wt)
        h_at_wt = h_at_wt.cpu().numpy().flatten()

    # Create the 6-subplot figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Inverse PINN Results - Water Table Recovery", fontsize=16, fontweight="bold")

    # Convert time to days for plotting
    t_days = t_lin.cpu().numpy() / 86400
    t_obs_days = np.array(t_obs) / 86400
    t_flux_days = np.array(t_flux) / 86400

    # Subplot 1: h(z,t) 2D colormap
    im1 = axs[0, 0].pcolormesh(
        T_np / 86400, Z_np, H_masked, shading="auto", cmap="viridis"
    )
    axs[0, 0].plot(t_days, -zb_vals, "r--", lw=2, label="Water table z_b(t)")
    axs[0, 0].set_title("Pressure Head h(z,t)")
    axs[0, 0].set_xlabel("Time [days]")
    axs[0, 0].set_ylabel("Depth z [m]")
    axs[0, 0].set_ylim(z_min, 0)
    axs[0, 0].legend(loc="lower right")
    plt.colorbar(im1, ax=axs[0, 0], label="h [m]")

    # Subplot 2: z_b(t) - Recovered water table depth
    axs[0, 1].plot(t_days, zb_vals, "b-", lw=2, label="Recovered z_b(t)")
    if zb_true is not None:
        axs[0, 1].plot(t_obs_days, zb_true, "r--", lw=2, alpha=0.7, label="True z_b(t)")
        axs[0, 1].legend()
        axs[0, 1].set_title("Water Table Depth: Recovered vs True")
    else:
        axs[0, 1].set_title("Recovered Water Table Depth z_b(t)")
    axs[0, 1].set_xlabel("Time [days]")
    axs[0, 1].set_ylabel("z_b [m] (positive downward)")
    axs[0, 1].grid(True, alpha=0.3)

    # Subplot 3: h at water table (should be ~0)
    axs[0, 2].plot(t_days, h_at_wt, "m-", lw=2, label="h(-z_b(t), t)")
    axs[0, 2].axhline(0, color="k", lw=1, linestyle="--", alpha=0.7, label="h=0")
    axs[0, 2].set_title("Pressure Head at Water Table")
    axs[0, 2].set_xlabel("Time [days]")
    axs[0, 2].set_ylabel("h(-z_b(t), t) [m]")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend()

    # Subplot 4: Surface moisture - observed vs predicted (INVERSE OBJECTIVE)
    axs[1, 0].plot(t_obs_days, theta_obs, "g-", lw=2, label="Observed (SMAP)")
    axs[1, 0].plot(t_obs_days, theta_pred, "r--", lw=2, label="Predicted")
    axs[1, 0].set_title("Surface Moisture: Obs vs Predicted")
    axs[1, 0].set_xlabel("Time [days]")
    axs[1, 0].set_ylabel("theta [m3/m3]")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    theta_rmse = np.sqrt(np.mean((theta_pred - theta_obs)**2))
    theta_corr = np.corrcoef(theta_pred, theta_obs)[0, 1]
    axs[1, 0].text(0.02, 0.98, f"RMSE={theta_rmse:.4f}\nr={theta_corr:.3f}",
                   transform=axs[1, 0].transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 5: Surface flux - prescribed vs predicted
    q_surf_arr = np.array(q_surf)
    axs[1, 1].plot(t_flux_days, q_surf_arr * 1e6, "g-", lw=2, label="Prescribed (ET-P)")
    axs[1, 1].plot(t_flux_days, q_pred * 1e6, "r--", lw=2, label="PINN predicted")
    axs[1, 1].axhline(0, color="k", lw=1, linestyle="--", alpha=0.5)
    axs[1, 1].set_title("Surface Flux: Prescribed vs Predicted")
    axs[1, 1].set_xlabel("Time [days]")
    axs[1, 1].set_ylabel("q_surf [um/s]")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    q_rmse = np.sqrt(np.mean((q_pred - q_surf_arr)**2))
    q_corr = np.corrcoef(q_pred, q_surf_arr)[0, 1]
    axs[1, 1].text(0.02, 0.98, f"RMSE={q_rmse*1e6:.2f} um/s\nr={q_corr:.3f}",
                   transform=axs[1, 1].transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 6: z_b validation metrics (if ground truth available) or scatter plot
    if zb_true is not None:
        # Scatter plot: predicted vs true
        axs[1, 2].scatter(zb_true, zb_at_obs, alpha=0.6, s=20)
        zb_min = min(zb_true.min(), zb_at_obs.min())
        zb_max = max(zb_true.max(), zb_at_obs.max())
        axs[1, 2].plot([zb_min, zb_max], [zb_min, zb_max], "k--", lw=1.5, label="1:1 line")
        axs[1, 2].set_title("z_b Scatter: Predicted vs True")
        axs[1, 2].set_xlabel("True z_b [m]")
        axs[1, 2].set_ylabel("Predicted z_b [m]")
        axs[1, 2].grid(True, alpha=0.3)
        axs[1, 2].legend()

        zb_corr = np.corrcoef(zb_true, zb_at_obs)[0, 1]
        zb_rmse = np.sqrt(np.mean((zb_at_obs - zb_true)**2))
        zb_bias = np.mean(zb_at_obs - zb_true)
        axs[1, 2].text(0.02, 0.98, f"r={zb_corr:.3f}\nRMSE={zb_rmse:.3f}m\nBias={zb_bias:.3f}m",
                       transform=axs[1, 2].transAxes, va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Show z_b derivative (smoothness check)
        dzb_dt = np.gradient(zb_vals, t_lin.cpu().numpy())
        axs[1, 2].plot(t_days, dzb_dt * 86400, "b-", lw=2, label="dz_b/dt")
        axs[1, 2].axhline(0, color="k", lw=1, linestyle="--", alpha=0.5)
        axs[1, 2].set_title("Water Table Rate of Change")
        axs[1, 2].set_xlabel("Time [days]")
        axs[1, 2].set_ylabel("dz_b/dt [m/day]")
        axs[1, 2].grid(True, alpha=0.3)
        axs[1, 2].legend()

    plt.tight_layout()

    print("\nInverse PINN Summary Statistics:")
    print(f"  Water table depth range: {zb_vals.min():.3f} to {zb_vals.max():.3f} m")
    print(f"  Max |h(-z_b(t), t)|: {np.max(np.abs(h_at_wt)):.6f} m (should be ~0)")
    print(f"  Surface moisture RMSE: {theta_rmse:.4f}")
    print(f"  Surface moisture correlation: {theta_corr:.4f}")

    if zb_true is not None:
        print(f"\nWater Table Recovery Metrics:")
        print(f"  Correlation: {zb_corr:.4f}")
        print(f"  RMSE: {zb_rmse:.4f} m")
        print(f"  Bias: {zb_bias:.4f} m")

    return fig


def plot_inverse_training_losses(losses, comps, sample_losses=None, sample_comps=None, sample_epochs=None):
    """
    Plot training loss evolution with 8 subplots for inverse PINN.

    Args:
        losses: list/array of total weighted losses per epoch
        comps: dict with keys 'pde', 'flux_bc', 'wt_head', 'data', 'ic_h', 'ic_zb', 'zb_smooth', 'zb_curv'
        sample_losses: optional list of losses computed over full dataset
        sample_comps: optional dict of loss components over full dataset
        sample_epochs: optional list of epoch numbers for sample losses
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    title = 'Inverse PINN - Sample Loss Evolution' if sample_losses is not None else 'Inverse PINN - Training Loss Evolution'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plot_sample = sample_losses is not None and sample_comps is not None and sample_epochs is not None
    x_data = np.array(sample_epochs) if plot_sample else np.arange(len(losses))
    total_losses = sample_losses if plot_sample else losses
    loss_comps = sample_comps if plot_sample else comps

    color = 'blue' if plot_sample else 'red'
    style = '-o' if plot_sample else '-'
    ms = 4 if plot_sample else 1

    # Plot 1: Total Weighted Loss
    ax1 = axes[0, 0]
    ax1.semilogy(x_data, total_losses, style, color=color, label='Total Loss', alpha=0.7, markersize=ms)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Weighted Loss')
    ax1.set_title('Total Weighted Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: PDE Loss
    ax2 = axes[0, 1]
    if 'pde' in loss_comps and len(loss_comps['pde']) > 0:
        ax2.semilogy(x_data, loss_comps['pde'], style, color=color, label='PDE Loss', alpha=0.7, markersize=ms)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PDE Loss')
    ax2.set_title('PDE Residual Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Flux BC Loss
    ax3 = axes[0, 2]
    if 'flux_bc' in loss_comps and len(loss_comps['flux_bc']) > 0:
        ax3.semilogy(x_data, loss_comps['flux_bc'], style, color=color, label='Flux BC Loss', alpha=0.7, markersize=ms)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Flux BC Loss')
    ax3.set_title('Surface Flux BC Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Data Matching Loss (INVERSE OBJECTIVE)
    ax4 = axes[0, 3]
    if 'data' in loss_comps and len(loss_comps['data']) > 0:
        ax4.semilogy(x_data, loss_comps['data'], style, color='green', label='Data Loss', alpha=0.7, markersize=ms)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Data Matching Loss')
    ax4.set_title('Data Matching Loss (Inverse Objective)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Water Table Head Loss
    ax5 = axes[1, 0]
    if 'wt_head' in loss_comps and len(loss_comps['wt_head']) > 0:
        ax5.semilogy(x_data, loss_comps['wt_head'], style, color=color, label='WT Head Loss', alpha=0.7, markersize=ms)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('WT Head Loss')
    ax5.set_title('Water Table Head BC Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: IC Losses (h and z_b)
    ax6 = axes[1, 1]
    if 'ic_h' in loss_comps and len(loss_comps['ic_h']) > 0:
        ax6.semilogy(x_data, loss_comps['ic_h'], style, color='purple', label='IC(h)', alpha=0.7, markersize=ms)
    if 'ic_zb' in loss_comps and len(loss_comps['ic_zb']) > 0:
        ax6.semilogy(x_data, loss_comps['ic_zb'], style, color='orange', label='IC(z_b)', alpha=0.7, markersize=ms)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('IC Loss')
    ax6.set_title('Initial Condition Losses')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Regularization Losses (smoothness + curvature)
    ax7 = axes[1, 2]
    if 'zb_smooth' in loss_comps and len(loss_comps['zb_smooth']) > 0:
        ax7.semilogy(x_data, loss_comps['zb_smooth'], style, color='cyan', label='Smoothness', alpha=0.7, markersize=ms)
    if 'zb_curv' in loss_comps and len(loss_comps['zb_curv']) > 0:
        ax7.semilogy(x_data, loss_comps['zb_curv'], style, color='magenta', label='Curvature', alpha=0.7, markersize=ms)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Regularization Loss')
    ax7.set_title('z_b Regularization Losses')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Final Loss Summary (bar plot)
    ax8 = axes[1, 3]
    loss_names = ['Total', 'PDE', 'Flux BC', 'Data', 'WT Head', 'IC(h)', 'IC(zb)', 'Smooth', 'Curv']
    final_vals = [
        total_losses[-1] if len(total_losses) > 0 else 0,
        loss_comps.get('pde', [0])[-1] if 'pde' in loss_comps and len(loss_comps['pde']) > 0 else 0,
        loss_comps.get('flux_bc', [0])[-1] if 'flux_bc' in loss_comps and len(loss_comps['flux_bc']) > 0 else 0,
        loss_comps.get('data', [0])[-1] if 'data' in loss_comps and len(loss_comps['data']) > 0 else 0,
        loss_comps.get('wt_head', [0])[-1] if 'wt_head' in loss_comps and len(loss_comps['wt_head']) > 0 else 0,
        loss_comps.get('ic_h', [0])[-1] if 'ic_h' in loss_comps and len(loss_comps['ic_h']) > 0 else 0,
        loss_comps.get('ic_zb', [0])[-1] if 'ic_zb' in loss_comps and len(loss_comps['ic_zb']) > 0 else 0,
        loss_comps.get('zb_smooth', [0])[-1] if 'zb_smooth' in loss_comps and len(loss_comps['zb_smooth']) > 0 else 0,
        loss_comps.get('zb_curv', [0])[-1] if 'zb_curv' in loss_comps and len(loss_comps['zb_curv']) > 0 else 0,
    ]

    x_bar = np.arange(len(loss_names))
    colors = [color, color, color, 'green', color, 'purple', 'orange', 'cyan', 'magenta']
    bars = ax8.bar(x_bar, final_vals, 0.7, color=colors, alpha=0.7)

    ax8.set_ylabel('Final Loss Value')
    ax8.set_title('Final Loss Summary')
    ax8.set_xticks(x_bar)
    ax8.set_xticklabels(loss_names, rotation=45, ha='right')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_zb_evolution(zb_history, t_values, zb_true=None):
    """
    Plot evolution of recovered water table depth over training.

    Args:
        zb_history: dict with 'epochs' and 'zb_values' lists
        t_values: time array for x-axis
        zb_true: optional ground truth water table
    """
    if not zb_history['epochs']:
        print("No z_b evolution data to plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    t_days = np.array(t_values) / 86400
    n_snapshots = len(zb_history['epochs'])
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    for i, (epoch, zb_vals) in enumerate(zip(zb_history['epochs'], zb_history['zb_values'])):
        alpha = 0.3 + 0.7 * (i / n_snapshots)
        ax.plot(t_days, zb_vals, color=colors[i], alpha=alpha, lw=1,
                label=f'Epoch {epoch}' if i % max(1, n_snapshots // 5) == 0 else None)

    if zb_true is not None:
        ax.plot(t_days, zb_true, 'r--', lw=2.5, label='True z_b')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('z_b [m]')
    ax.set_title('Water Table Recovery Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def save_inverse_figures(model, theta_obs_data, q_surf_data, soil_params,
                         losses, comps, output_dir, zb_true=None,
                         sample_losses=None, sample_comps=None, sample_epochs=None,
                         device='cpu'):
    """
    Save all inverse PINN figures to output directory.

    Args:
        model: Trained RichardsInversePINN model
        theta_obs_data: tuple (times, theta_obs)
        q_surf_data: tuple (times, q_surf)
        soil_params: dict with soil parameters
        losses: training loss history
        comps: loss components dict
        output_dir: directory to save figures
        zb_true: optional ground truth water table
        sample_losses, sample_comps, sample_epochs: optional sample loss data
        device: computation device
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig_results = plot_inverse_results(
        model, theta_obs_data, q_surf_data, soil_params,
        zb_true=zb_true, device=device
    )
    fig_results.savefig(os.path.join(output_dir, 'inverse_results.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_results)
    print(f"  Saved: inverse_results.png")

    fig_losses = plot_inverse_training_losses(
        losses, comps, sample_losses, sample_comps, sample_epochs
    )
    fig_losses.savefig(os.path.join(output_dir, 'training_losses.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_losses)
    print(f"  Saved: training_losses.png")
