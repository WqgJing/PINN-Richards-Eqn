"""Line plots for twin data, identifiability and training diagnostics (no fills)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated default palette); ink for text.
C_TRUTH = '#2a78d6'   # slot 1 blue: truth / reference
C_PRED = '#eb6834'    # slot 2 orange: PINN / prediction
C_3 = '#1baf7a'       # slot 3 aqua
C_4 = '#eda100'       # slot 4 yellow
INK = '#0b0b0b'
INK2 = '#52514e'
SEQ_BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
    'font.size': 9, 'legend.frameon': False,
})


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_twin(tw, path):
    t = tw['t_days']
    fig, ax = plt.subplots(4, 1, figsize=(9, 10))
    ax[0].plot(t, tw['zb_true'], color=C_TRUTH, label='truth (spline fit)')
    ax[0].plot(tw['t_obs_days'], tw['zb_obs_daily'], 'o', ms=3, color=C_PRED, label='prescribed path (daily)')
    ax[0].invert_yaxis(); ax[0].set_ylabel('water-table depth (m)'); ax[0].legend()
    if tw.get('tp_mmd') is None:
        ax[1].plot(t, tw['qsurf'] * 86400e3, color=INK2)
        ax[1].set_ylabel('surface flux, upward positive (mm/d)')
    else:                                                    # twin with a root sink: show the transpiration too
        ax[1].plot(t, tw['qsurf'] * 86400e3, color=C_TRUTH, label='surface flux, upward positive')
        ax[1].plot(t, tw['tp_mmd'], color=C_PRED, label='transpiration (root sink)')
        ax[1].set_ylabel('flux (mm/d)'); ax[1].legend()
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[2].plot(t, tw['theta_obs_hourly'], color=C_TRUTH, label='hourly')
    ax[2].plot(tw['t_obs_days'], tw['theta_obs_daily'], 'o', ms=3, color=C_PRED, label='daily samples')
    ax[2].set_ylabel('soil moisture, top 5 cm (m$^3$/m$^3$)'); ax[2].legend()
    idx = np.linspace(0, t.size - 1, 6).astype(int)
    for c, k in zip(SEQ_BLUE, idx):
        z = tw['xi'] * tw['zb_true'][k]
        ax[3].plot(tw['theta'][k], z, color=c, label=f'day {t[k]:.0f}')
    ax[3].set_xlabel('soil moisture (m$^3$/m$^3$)'); ax[3].set_ylabel('z (m)'); ax[3].legend(ncol=3)
    for a in ax[:3]:
        a.set_xlabel('time (days)')
    _finish(fig, path)


def plot_twin_compare(series, path, ylabel='soil moisture, top 5 cm (m$^3$/m$^3$)'):
    """Soil moisture of several runs of one case: series = [(t_days, theta, label, line style), ...]."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (t, y, label, ls), c in zip(series, (C_TRUTH, C_PRED, C_3, C_4)):
        ax.plot(t, y, ls, color=c, ms=3, label=label)
    ax.set_xlabel('time (days)'); ax.set_ylabel(ylabel); ax.legend()
    _finish(fig, path)


def plot_real_check(t, zb_obs, zb_fit, t_nodes, et_mmd, p_mmd, theta_obs, theta_runs, path):
    """Site window (scripts/make_real.py): the forcing nodes, the soil moisture at the sensor (observed, and forward
    runs theta_runs = [(label, theta), ...] on the hourly times t) and the water table (observed, its spline fit)."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    ax[0].plot(t_nodes, et_mmd, color=C_TRUTH, label='evapotranspiration')
    ax[0].plot(t_nodes, p_mmd, color=C_PRED, label='precipitation')
    ax[0].set_ylabel('flux (mm/d)'); ax[0].legend()
    ax[1].plot(t, theta_obs, color=INK, label='observed')
    for (label, y), c in zip(theta_runs, (C_TRUTH, C_PRED, C_3, C_4)):
        ax[1].plot(t, y, color=c, label=label)
    ax[1].set_ylabel('soil moisture at the sensor (m$^3$/m$^3$)'); ax[1].legend()
    ax[2].plot(t, zb_obs, color=INK, label='observed')
    ax[2].plot(t, zb_fit, color=C_PRED, label='spline fit')
    ax[2].invert_yaxis(); ax[2].set_ylabel('water-table depth (m)'); ax[2].set_xlabel('time (days)'); ax[2].legend()
    _finish(fig, path)


def plot_twin_result(tw, pred, starts_zb, path):
    """Inversion result on a twin: the surface forcing, the top-5 cm soil moisture (data and model at the estimate) and
    the water table (prescribed, and the estimate of every start). tw: twin.npz, pred: predictions.npz of
    scripts/param_invert.py, starts_zb[k]: zb(t) of start k on pred['t_days']."""
    t = pred['t_days']
    fig, ax = plt.subplots(4, 1, figsize=(9, 10.5), sharex=True)
    ax[0].plot(tw['t_days'], tw['qsurf'] * 86400e3, color=C_TRUTH, label='surface flux, upward positive')
    if 'tp_daily_mmd' in tw and np.any(tw['tp_daily_mmd'] > 0):       # twin with a root sink
        ax[0].plot(tw['t_daily'], tw['tp_daily_mmd'], color=C_PRED, label='transpiration (root sink)')
        ax[0].legend(); ax[0].set_ylabel('flux (mm/d)')
    else:
        ax[0].set_ylabel('surface flux, upward positive (mm/d)')
    ax[0].axhline(0, color=INK2, lw=0.6)
    ax[1].plot(pred['t_obs_days'], pred['theta_obs'], color=C_TRUTH, label='FEM twin (given)')
    ax[1].plot(t, pred['theta_obs_pred'], color=C_PRED, lw=1.0, label='PINN at its estimate')
    ax[1].set_ylabel('soil moisture, top 5 cm (m$^3$/m$^3$)'); ax[1].legend()
    ax[2].plot(t, pred['zb_true'], color=C_TRUTH, label='prescribed')
    for k, z in enumerate(starts_zb):
        ax[2].plot(t, z, color=C_PRED, lw=1.0, label=f'PINN estimate ({len(starts_zb)} starts)' if k == 0 else None)
    ax[2].invert_yaxis(); ax[2].set_ylabel('water-table depth (m)'); ax[2].legend()
    for k, z in enumerate(starts_zb):
        ax[3].plot(t, 100 * (z - pred['zb_true']), color=C_PRED, lw=1.0)
    ax[3].axhline(0, color=INK2, lw=0.6)
    ax[3].set_ylabel('water-table depth error (cm)'); ax[3].set_xlabel('time (days)')
    _finish(fig, path)


def plot_identifiability(t_days, zb, sig, path, title=None):       # title unused: figures carry no titles
    """sig: dict label -> sigma_zb(t) [m]."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t_days, zb, color=C_TRUTH)
    ax[0].invert_yaxis(); ax[0].set_ylabel('water-table depth (m)')
    for c, (lab, s) in zip([C_PRED, C_3, C_4, '#e87ba4'], sig.items()):
        ax[1].plot(t_days, s, color=c, label=lab)
    ax[1].set_yscale('log'); ax[1].set_ylabel('lower bound on the depth error, 1 sigma (m)')
    ax[1].set_xlabel('time (days)'); ax[1].legend()
    _finish(fig, path)


def plot_training(hist, pred, path, title):
    """hist: dict of lists (step, loss terms, weights, metrics). pred: final predictions dict."""
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    st = np.asarray(hist['step'])
    for c, k in zip([C_TRUTH, C_PRED, C_3], ['L_pde', 'L_flux', 'L_data']):
        if k in hist:
            ax[0, 0].plot(st, hist[k], color=c, label=k)
    ax[0, 0].set_yscale('log'); ax[0, 0].set_title('normalized loss terms'); ax[0, 0].legend()
    for c, k in zip([C_TRUTH, C_PRED, C_3], ['w_pde', 'w_flux', 'w_data']):
        if k in hist:
            ax[0, 1].plot(st, hist[k], color=c, label=k)
    ax[0, 1].set_yscale('log'); ax[0, 1].set_title('loss weights'); ax[0, 1].legend()
    if 'zb_rmse' in hist and np.isfinite(np.asarray(hist['zb_rmse'], float)).any():
        ax[1, 0].plot(st, hist['zb_rmse'], color=C_PRED, label='zb RMSE [m]')
        ax[1, 0].set_yscale('log'); ax[1, 0].set_title('zb error vs truth'); ax[1, 0].legend()
    ax[1, 1].plot(pred['t_days'], pred['zb_pred'], color=C_PRED, label='PINN')
    if pred.get('zb_true') is not None:
        ax[1, 1].plot(pred['t_days'], pred['zb_true'], color=C_TRUTH, label='truth')
    if pred.get('zb_init') is not None:
        ax[1, 1].plot(pred['t_days'], pred['zb_init'], color=INK2, lw=1.0, label='initial guess')
    ax[1, 1].invert_yaxis(); ax[1, 1].set_ylabel('zb [m]'); ax[1, 1].legend(); ax[1, 1].set_title('water-table depth')
    ax[2, 0].plot(pred['t_days'], pred['theta_obs_pred'], color=C_PRED, label='PINN top-5 cm')
    ax[2, 0].plot(pred['t_obs_days'], pred['theta_obs'], 'o', ms=3, color=C_TRUTH, label='observed')
    ax[2, 0].set_ylabel('theta [-]'); ax[2, 0].legend(); ax[2, 0].set_title('surface moisture fit')
    ax[2, 1].plot(pred['t_days'], pred['q_pred'] * 86400e3, color=C_PRED, label='PINN surface flux')
    ax[2, 1].plot(pred['t_days'], pred['q_surf'] * 86400e3, color=C_TRUTH, lw=1.0, label='prescribed')
    ax[2, 1].set_ylabel('q [mm/d]'); ax[2, 1].legend(); ax[2, 1].set_title('surface flux fit')
    for a in [ax[1, 1], ax[2, 0], ax[2, 1]]:
        a.set_xlabel('days from window start')
    ax[0, 0].set_xlabel('step'); ax[0, 1].set_xlabel('step'); ax[1, 0].set_xlabel('step')
    fig.suptitle(title)
    _finish(fig, path)


def plot_profiles(xi, t_days, theta_ref, theta_pred, zb_ref, zb_pred, path, title):
    """Reference vs PINN theta profiles at a few times (blue = reference, orange = PINN)."""
    idx = np.linspace(0, len(t_days) - 1, 4).astype(int)
    fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for a, k in zip(ax, idx):
        a.plot(theta_ref[k], xi * zb_ref[k], color=C_TRUTH, label='reference')
        a.plot(theta_pred[k], xi * zb_pred[k], color=C_PRED, label='PINN')
        a.set_title(f'day {t_days[k]:.1f}'); a.set_xlabel('theta [-]')
    ax[0].set_ylabel('z [m]'); ax[0].legend()
    fig.suptitle(title)
    _finish(fig, path)


def plot_real_result(t, theta_obs, theta_runs, zb_obs, zb_runs, path):
    """Inversion on site data (scripts/score_real.py): soil moisture at the sensor (observed and model runs
    theta_runs = [(label, theta), ...]) and the water table (observed and estimates zb_runs = [(label, zb or a list of
    zb, one per start), ...]), all on the hourly times t."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(t, theta_obs, color=INK, label='observed')
    for (label, y), c in zip(theta_runs, (C_PRED, C_TRUTH, C_3, C_4)):
        ax[0].plot(t, y, color=c, lw=1.2, label=label)
    ax[0].set_ylabel('soil moisture at the sensor (m$^3$/m$^3$)'); ax[0].legend()
    ax[1].plot(t, zb_obs, color=INK, label='observed')
    for (label, z), c in zip(zb_runs, (C_PRED, C_TRUTH, C_3, C_4)):
        for k, zk in enumerate(np.atleast_2d(z)):
            ax[1].plot(t, zk, color=c, lw=1.2, label=label if k == 0 else None)
    ax[1].invert_yaxis(); ax[1].set_ylabel('water-table depth (m)'); ax[1].set_xlabel('time (days)'); ax[1].legend()
    _finish(fig, path)
