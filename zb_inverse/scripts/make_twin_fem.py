#!/usr/bin/env python
"""Twin from the wtd_pipeline FEM (the solver of the fixed-WTD GLDAS sweeps) with a prescribed, time-varying water table.

All runs share the soil, the GLDAS forcing and zb(t), and start from the hydrostatic profile over zb(0):
  FEM, twin time step, with and without the Feddes root sink   -> the twins: <out>/twin.npz, <out>_nosink/twin.npz
  FEM, daily steps, with the sink                              -> the time stepping of the sweeps, kept for comparison
  FEM, finer steps and grid                                    -> convergence of the FEM itself
  this project's reference solver, no sink                     -> independent check of the FEM, and of the air-entry
                                                                  soil option used by the PINN runs so far

Usage: python -u scripts/make_twin_fem.py --config configs/twin_gldas_fl.yaml
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import numpy as np
import torch

from common import load_with_overrides, setup, hourly_times, run_reference
from zbinv import plots
from zbinv.data import obs_from_profile
from zbinv.soil import SoilVG


def load_fem(root):
    """ForwardSensitivitySolverPyTorch and RootWaterUptakeTorch, loaded by file as generate_training_continuous.py does."""
    mods = {}
    for name in ('fem_pytorch', 'root_water_uptake'):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, 'fem', name + '.py'))
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name]
        spec.loader.exec_module(mods[name])
    return mods['fem_pytorch'].ForwardSensitivitySolverPyTorch, mods['root_water_uptake'].RootWaterUptakeTorch


def run_fem(Solver, Sink, cfg, t_days, zb, q, tp_mmd, n_nodes, depth_m, soil_params=None, quiet=False):
    """One batched solve of the sweep solver. The members share the soil and q_surf(t) [n_t] (or have their own:
    q [B, n_t]); each has its own water table and its own transpiration for the root sink (rows of zb and tp_mmd
    [B, n_t]; T_p = 0 means no sink) and starts from the hydrostatic profile over its zb(0).
    soil_params: van Genuchten parameters other than cfg['soil'], scalars or one value per member [B]."""
    fc, n_mem = cfg['fem'], zb.shape[0]
    if soil_params is None:
        soil_params = {k: cfg['soil'][k] for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')}
    solver = Solver(soil_params, L=fc['L'], n_nodes=n_nodes, device='cpu')
    xi = solver.fem.xi
    h0 = -(1.0 + xi).unsqueeze(0) * torch.as_tensor(zb[:, :1], dtype=xi.dtype)   # hydrostatic above the water table [m]
    sink = Sink(tp_mmd, device='cpu', Tp_units='mm/day')
    res = solver.solve(t_days * 86400.0, zb, (q if np.ndim(q) == 2 else np.tile(q, (n_mem, 1))), theta_initial=solver.vg.theta(h0 / fc['L']),
                       sink_func=sink, max_iter=fc['max_iter'], tol=fc['tol'], omega=fc['omega'], return_full=True)
    out = {k: res[k].cpu().numpy() for k in ('xi', 'theta', 'h', 'theta_surf', 'converged', 'n_iters')}
    # top-5 cm mean with this project's observation operator (the one the PINN uses), next to the solver's own
    out['theta_obs'] = np.stack([obs_from_profile(out['xi'], out['theta'][m], zb[m], depth_m) for m in range(n_mem)])
    if not quiet:
        print(f"[fem] {n_mem} members, {t_days.size - 1} steps of {24 * (t_days[1] - t_days[0]):.2f} h, {n_nodes} nodes: "
              f"converged {100 * out['converged'].mean():.1f}% of steps, max Picard iterations "
              f"{int(out['n_iters'].max())}, surface head min {out['h'][:, :, -1].min():.2f} m", flush=True)
    return out


def diff(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {'rmse': float(np.sqrt(np.mean(d ** 2))), 'max': float(np.abs(d).max()), 'mean': float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    fc, depth = cfg['fem'], cfg['obs']['depth_m']
    soil, win, sp, truth = setup(cfg)
    n_days = win['n_days']
    print(f"[twin] {cfg['site']['name']} {win['start_date']} + {n_days} d; soil: {soil.summary()}", flush=True)
    fit_err = truth.zb(win['t_days']) - win['zb']
    print(f"[twin] truth spline ({sp.n_coef} coefficients) vs prescribed daily zb: RMSE "
          f"{np.sqrt(np.mean(fit_err ** 2)):.4f} m, max {np.abs(fit_err).max():.4f} m", flush=True)
    print(f"[twin] q_surf {win['qsurf'].min() * 86400e3:.1f} .. {win['qsurf'].max() * 86400e3:.2f} mm/d "
          f"(sum {np.trapezoid(win['qsurf'], win['t_days']) * 86400e3:.1f} mm), T_p {win['tp_mmd'].min():.2f} .. "
          f"{win['tp_mmd'].max():.2f} mm/d (sum {np.trapezoid(win['tp_mmd'], win['t_days']):.1f} mm)", flush=True)

    Solver, Sink = load_fem(fc['root'])
    on = lambda t: (truth.zb(t), np.interp(t, win['t_days'], win['qsurf']), np.interp(t, win['t_days'], win['tp_mmd']))

    def two(t):                                              # member 0 with the root sink, member 1 without
        zb, q, tp = on(t)
        return np.stack([zb, zb]), q, np.stack([tp, np.zeros_like(tp)])

    t_tw = hourly_times(n_days, fc['dt_hours'])
    t_chk = hourly_times(n_days, fc['check_dt_hours'])
    zb_tw = truth.zb(t_tw)
    A = run_fem(Solver, Sink, cfg, t_tw, *two(t_tw), fc['n_nodes'], depth)                    # the twins
    D = run_fem(Solver, Sink, cfg, win['t_days'], *two(win['t_days']), fc['n_nodes'], depth)  # daily steps, as the sweeps
    C = run_fem(Solver, Sink, cfg, t_chk, *two(t_chk), fc['check_n_nodes'], depth)            # finer steps and grid
    k_chk = np.round(t_tw / (fc['check_dt_hours'] / 24.0)).astype(int)
    k_day = np.round(win['t_days'] / (fc['dt_hours'] / 24.0)).astype(int)

    t_ref = hourly_times(n_days, cfg['reference'].get('out_dt_hours', 1.0))
    k_ref = np.round(t_ref / (fc['dt_hours'] / 24.0)).astype(int)
    R = run_reference(cfg, soil, win, truth, t_ref)
    soil_ae = SoilVG.from_config({**cfg['soil'], 'air_entry_m': 0.02, 'ae_smooth_m': 0.005}, cfg['scaling']['L'])
    R_ae = run_reference(cfg, soil_ae, win, truth, t_ref)

    summary = {
        'window': win['start_date'], 'n_days': n_days, 'n_coef': sp.n_coef,
        'zb_range_m': [float(zb_tw.min()), float(zb_tw.max())],
        'theta_obs_range_sink': [float(A['theta_obs'][0].min()), float(A['theta_obs'][0].max())],
        'theta_obs_range_nosink': [float(A['theta_obs'][1].min()), float(A['theta_obs'][1].max())],
        'h_surf_min_m_sink': float(A['h'][0, :, -1].min()), 'h_surf_min_m_nosink': float(A['h'][1, :, -1].min()),
        'fem_converged_frac': float(A['converged'].mean()),
        'fem_converged_frac_daily_steps': float(D['converged'].mean()),
        # differences in the top-5 cm mean theta (m3/m3)
        'sink_minus_nosink': diff(A['theta_obs'][0], A['theta_obs'][1]),
        'fem_vs_finer_fem_sink': diff(A['theta_obs'][0], C['theta_obs'][0][k_chk]),
        'fem_vs_finer_fem_nosink': diff(A['theta_obs'][1], C['theta_obs'][1][k_chk]),
        'fem_daily_steps_vs_fem_sink': diff(D['theta_obs'][0], A['theta_obs'][0][k_day]),
        'fem_vs_reference_nosink': diff(A['theta_obs'][1][k_ref], R['theta_obs']),
        'finer_fem_vs_reference_nosink': diff(C['theta_obs'][1][k_chk][k_ref], R['theta_obs']),
        'reference_air_entry_vs_plain': diff(R_ae['theta_obs'], R['theta_obs']),
        'fem_own_surface_mean_vs_obs_operator': diff(A['theta_surf'][0], A['theta_obs'][0]),
    }
    h_floor = cfg['reference'].get('h_surf_admissible_m', -10.0)
    summary['admissible_sink'] = bool(summary['h_surf_min_m_sink'] >= h_floor)
    summary['admissible_nosink'] = bool(summary['h_surf_min_m_nosink'] >= h_floor)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"[twin] d theta_5cm, {k}: RMSE {v['rmse']:.2e}, max {v['max']:.2e}, mean {v['mean']:+.2e}", flush=True)
        else:
            print(f"[twin] {k}: {v}", flush=True)

    import yaml
    k_obs = np.arange(0, t_tw.size, int(round(cfg['obs']['interval_hours'] / fc['dt_hours'])))
    for m, (tag, has_sink) in enumerate([('', True), ('_nosink', False)]):
        out = cfg['output']['dir'] + tag
        os.makedirs(out, exist_ok=True)
        c = copy.deepcopy(cfg)
        c['output']['dir'] = out
        c['fem']['root_sink'] = has_sink
        with open(os.path.join(out, 'twin_config.yaml'), 'w') as f:
            yaml.safe_dump(c, f, sort_keys=False)
        th_obs = A['theta_obs'][m]
        tp_tw = on(t_tw)[2] if has_sink else np.zeros_like(t_tw)
        np.savez(os.path.join(out, 'twin.npz'),
                 t_days=t_tw, xi=A['xi'], theta=A['theta'][m], h_t=A['h'][m] / soil.L,
                 zb_true=zb_tw, qsurf=on(t_tw)[1], theta_obs_hourly=th_obs,
                 t_obs_days=t_tw[k_obs], theta_obs_daily=th_obs[k_obs],
                 zb_obs_daily=win['zb'], t_daily=win['t_days'], q_daily=win['qsurf'],
                 tp_daily_mmd=win['tp_mmd'] if has_sink else np.zeros_like(win['tp_mmd']),
                 truth_coef=truth.coef, zb_min=truth.zb_min, knot_days=sp.dt, n_days=n_days,
                 start_date=win['start_date'], L=soil.L,
                 soil=np.array([soil.theta_s, soil.theta_r, soil.alpha, soil.n, soil.Ks]),
                 theta_surf_fem=A['theta_surf'][m], converged=A['converged'][m],
                 theta_obs_fem_daily_steps=D['theta_obs'][m], theta_obs_reference_nosink=R['theta_obs'],
                 t_reference=t_ref)
        with open(os.path.join(out, 'twin_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        plots.plot_twin({'t_days': t_tw, 'zb_true': zb_tw, 't_obs_days': t_tw[k_obs],
                         'zb_obs_daily': np.interp(t_tw[k_obs], win['t_days'], win['zb']), 'qsurf': on(t_tw)[1],
                         'tp_mmd': tp_tw if has_sink else None, 'theta_obs_hourly': th_obs,
                         'theta_obs_daily': th_obs[k_obs], 'xi': A['xi'], 'theta': A['theta'][m]},
                        os.path.join(out, 'twin.png'))
        print(f"[twin] saved {out}/twin.npz, twin.png, twin_summary.json", flush=True)
    plots.plot_twin_compare(
        [(t_tw, A['theta_obs'][0], 'FEM, with root sink', '-'), (t_tw, A['theta_obs'][1], 'FEM, no root sink', '-'),
         (t_ref, R['theta_obs'], 'reference solver, no root sink', '--'),
         (win['t_days'], D['theta_obs'][0], 'FEM, with root sink, daily steps (as the sweeps)', 'o-')],
        os.path.join(cfg['output']['dir'], 'twin_compare.png'))


if __name__ == '__main__':
    main()
