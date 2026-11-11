"""Shared helpers for the scripts: config -> soil, window, truth spline, reference runs."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.data import load_config, load_forcing, select_window, obs_from_profile  # noqa: E402
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy  # noqa: E402
from zbinv.reference import ReferenceSolver, make_grid, hydrostatic_h  # noqa: E402


def load_with_overrides(path, overrides):
    """Load a YAML config and apply `key.sub=value` overrides (values parsed as YAML)."""
    import yaml
    cfg = load_config(path)
    for item in overrides or []:
        key, val = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def through_points(t, points, n_days):
    """Smooth non-periodic curve through control points {'t_days': [...], 'values': [...]}: shape-preserving
    cubic (PCHIP), so the curve does not overshoot between the points."""
    from scipy.interpolate import PchipInterpolator
    tp, vp = np.asarray(points['t_days'], dtype=float), np.asarray(points['values'], dtype=float)
    if tp.shape != vp.shape or tp[0] > 0.0 or tp[-1] < n_days:
        raise ValueError(f'control points need t_days and values of the same length, covering 0..{n_days} d')
    return PchipInterpolator(tp, vp)(t)


def synthetic_window(cfg):
    """Smooth synthetic forcing (no site data): q_surf(t) and zb(t) on 6-hour nodes. q_surf is a sinusoid; zb is a
    sinusoid unless control points are given (synthetic.zb_points, values in m)."""
    s = cfg['synthetic']
    n_days = int(cfg['window']['n_days'])
    t = np.arange(n_days * 4 + 1) / 4.0
    q_mmd = s['q_mean_mmd'] + s['q_amp_mmd'] * np.sin(2 * np.pi * t / s['q_period_days'] + s.get('q_phase', 0.0))
    if 'zb_points' in s:
        zb = through_points(t, s['zb_points'], n_days)
    else:
        zb = s['zb_mean'] + s['zb_amp'] * np.sin(2 * np.pi * t / s['zb_period_days'] + s.get('zb_phase', 0.0))
    return {'t_days': t, 'qsurf': q_mmd / 86400e3, 'zb': zb, 'start_date': 'synthetic', 'n_days': n_days,
            'dates': None, 'smap_sm': None}


def gldas_window(cfg):
    """GLDAS daily forcing of one site with a PRESCRIBED water table (no site water-table data).

    site.gldas_csv: rows of that site from gldas_forcing_global_*.csv (daily means, mm/d: qsurf_mm = E_soil - I,
    Tp_mm = transpiration). Day k of the window sits at t = k d, as in the fixed-WTD sweeps of wtd_pipeline.
    zb(t): smooth curve through prescribed.zb_points (m)."""
    import csv
    n_days = int(cfg['window']['n_days'])
    dates = np.datetime64(cfg['window']['start'], 'D') + np.arange(n_days + 1)
    with open(cfg['site']['gldas_csv']) as f:
        rows = {r['date']: r for r in csv.DictReader(f)}
    q_mmd = np.array([float(rows[str(d)]['qsurf_mm']) for d in dates])
    tp_mmd = np.array([float(rows[str(d)]['Tp_mm']) for d in dates])
    t = np.arange(n_days + 1, dtype=float)
    return {'t_days': t, 'qsurf': q_mmd / 86400e3, 'tp_mmd': tp_mmd,
            'zb': through_points(t, cfg['prescribed']['zb_points'], n_days), 'start_date': str(dates[0]),
            'n_days': n_days, 'dates': dates, 'smap_sm': None}


def setup(cfg):
    soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
    if cfg['site'].get('synthetic', False):
        win = synthetic_window(cfg)
    elif 'gldas_csv' in cfg['site']:
        win = gldas_window(cfg)
    else:
        forcing = load_forcing(cfg['site']['forcing_npz'])
        win = select_window(forcing, cfg['window']['start'], cfg['window']['n_days'])
    sp = CubicBSpline1D(0.0, win['n_days'], cfg['zb_spline']['knot_days'])
    truth = ZbSplineNumpy.fit(sp, win['t_days'], win['zb'], cfg['zb_spline']['zb_min'])
    return soil, win, sp, truth


def hourly_times(n_days, dt_hours=1.0):
    n = int(round(n_days * 24.0 / dt_hours))
    return np.arange(n + 1) * dt_hours / 24.0


def _scalar_or_array(fn):
    def g(td):
        out = fn(np.atleast_1d(td))
        return float(out[0]) if np.ndim(td) == 0 else out
    return g


def run_reference(cfg, soil, win, zb_model, t_eval, n_nodes=None, verbose=True):
    rc = cfg['reference']
    xi = make_grid(n_nodes or rc['n_nodes'], rc.get('grid_gamma', 1.0))
    zb_fn = _scalar_or_array(zb_model.zb)
    zbdot_fn = _scalar_or_array(zb_model.dzb_dt)
    q_fn = lambda td: np.interp(td, win['t_days'], win['qsurf'])
    h0 = hydrostatic_h(xi, zb_fn(0.0), soil.L)
    solver = ReferenceSolver(soil, xi, h_crit_m=rc.get('h_crit_m'), crit_width=rc.get('crit_width', 0.1))
    res = solver.solve(t_eval, zb_fn, zbdot_fn, q_fn, h0, rtol=rc.get('rtol', 1e-6),
                       atol=rc.get('atol', 1e-9), max_step_hours=rc.get('max_step_hours', 1.0),
                       verbose=verbose)
    res['theta_obs'] = obs_from_profile(res['xi'], res['theta'], res['zb'], cfg['obs']['depth_m'])
    return res


def bound_from_jacobian(cfg, J_h, truth, t_eval):
    """Cramer-Rao bound on zb(t) from J_h[k, j] = d theta_obs(t_k) / d c_j on hourly times t_eval:
    Cov(c) >= sigma^2 (J^T J)^-1, mapped to sigma_zb(t) with the spline Jacobian G = d zb / d c.
    Returns (summary, {label: sigma_zb(t)}, G) for daily and hourly observations and the noise levels ident.sigmas."""
    step = int(round(cfg['obs']['interval_hours']))
    k_daily = np.arange(0, t_eval.size, step)
    G = truth.dzb_dcoef(t_eval)                       # metres per coefficient
    summary, sig_curves = {}, {}
    for obs_label, J in [('daily', J_h[k_daily]), ('hourly', J_h)]:
        sv = np.linalg.svd(J, compute_uv=False)
        JTJ_inv = np.linalg.pinv(J.T @ J, rcond=1e-12)
        summary[f'{obs_label}_singular_values'] = sv.tolist()
        summary[f'{obs_label}_condition'] = float(sv[0] / max(sv[-1], 1e-300))
        for sigma in cfg.get('ident', {}).get('sigmas', [0.005, 0.01, 0.02]):
            cov_zb = sigma ** 2 * np.einsum('ij,jk,ik->i', G, JTJ_inv, G)
            s_zb = np.sqrt(np.maximum(cov_zb, 0.0))
            g_mean = G.mean(axis=0)                       # window-mean zb (lowest-frequency component)
            s_mean = float(np.sqrt(max(sigma ** 2 * g_mean @ JTJ_inv @ g_mean, 0.0)))
            summary[f'{obs_label}_sigma{sigma}'] = {
                'median_m': float(np.median(s_zb)), 'max_m': float(s_zb.max()),
                'rms_m': float(np.sqrt(np.mean(s_zb ** 2))), 'window_mean_zb_m': s_mean}
            sig_curves[f'sigma_theta={sigma} ({obs_label} obs)'] = s_zb
            print(f"[ident] {obs_label} obs, sigma_theta={sigma}: sigma_zb median {np.median(s_zb):.3f} m, "
                  f"rms {np.sqrt(np.mean(s_zb ** 2)):.3f} m, max {s_zb.max():.3f} m; "
                  f"window-mean zb {s_mean:.3f} m", flush=True)
        print(f"[ident] {obs_label} condition number {summary[f'{obs_label}_condition']:.2e}", flush=True)
    return summary, sig_curves, G


def save_bound(out, tag, summary, sig_curves, J_h, t_eval, zb, G):
    """identifiability<tag>.npz / .json / .png in the directory out (keys as scripts/identifiability.py)."""
    import json
    from zbinv import plots
    np.savez(os.path.join(out, f'identifiability{tag}.npz'), J_hourly=J_h, t_days=t_eval, zb=zb, G=G,
             **{k.replace(' ', '_').replace('=', ''): v for k, v in sig_curves.items()})
    with open(os.path.join(out, f'identifiability{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    keep = {k: v for k, v in sig_curves.items() if ('0.005' in k or '0.01 ' in k)}
    plots.plot_identifiability(t_eval, zb, keep, os.path.join(out, f'identifiability{tag}.png'))
