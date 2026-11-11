#!/usr/bin/env python
"""E0a cross-check: reference solver vs the validated FEM (read-only import), same forcing.

Both get the truth-spline zb(t), piecewise-linear q_surf and a hydrostatic initial profile.
The FEM is run (1) with hourly steps (close to continuous time) and (2) with daily steps as in
production, so the comparison separates time-discretisation error from model differences.

Usage: python -u scripts/crosscheck_fem.py --config configs/twin_pas_apr.yaml
"""
import argparse
import json
import os
import sys
import numpy as np

from common import load_config, load_with_overrides, setup, hourly_times, run_reference
from zbinv.data import obs_from_profile

FEM_SRC = '/storage/project/r-rbras6-0/jjason3/fem_validation/uploaded_data/forward_sensitivity_lib/src'


def run_fem(cfg, soil, win, truth, t_days):
    sys.path.insert(0, FEM_SRC)
    from fem_forward_sensitivity import ForwardSensitivitySolver  # noqa: E402
    sp = {k: cfg['soil'][k] for k in ['theta_s', 'theta_r', 'alpha', 'n', 'Ks']}
    fem = ForwardSensitivitySolver(sp, L=5.0, n_nodes=51)
    zb = truth.zb(t_days)
    q = np.interp(t_days, win['t_days'], win['qsurf'])
    xi = fem.fem.xi
    h0 = -zb[0] * (1.0 + xi)                       # hydrostatic, metres
    se0 = (1.0 + (soil.alpha * np.abs(h0)) ** soil.n) ** (-soil.m)
    theta0 = soil.theta_r + soil.theta_star * se0
    r = fem.solve(t_days * 86400.0, zb, q, theta_initial=theta0, method='picard',
                  omega=0.5, max_iter=200, tol=1e-4)
    obs = obs_from_profile(xi, r['theta'], zb, cfg['obs']['depth_m'])
    return obs, r['converged']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    ap.add_argument('--tag', default='', help='suffix for output file names')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    soil, win, sp, truth = setup(cfg)

    t_h = hourly_times(win['n_days'])
    ref = run_reference(cfg, soil, win, truth, t_h)
    summary = {}
    for label, t_days in [('fem_hourly', t_h), ('fem_daily', np.arange(win['n_days'] + 1, dtype=float))]:
        obs, conv = run_fem(cfg, soil, win, truth, t_days)
        ref_on = np.interp(t_days, ref['t_days'], ref['theta_obs'])
        d = obs - ref_on
        dc = d[conv]
        summary[label] = {
            'n_steps': int(t_days.size), 'converged_frac': float(conv.mean()),
            'rmse_all': float(np.sqrt(np.mean(d ** 2))), 'maxabs_all': float(np.abs(d).max()),
            'rmse_converged': float(np.sqrt(np.mean(dc ** 2))) if dc.size else None,
            'bias_converged': float(dc.mean()) if dc.size else None,
        }
        print(f"[crosscheck] {label}: converged {conv.mean():.2%}, theta_obs RMSE vs reference "
              f"{summary[label]['rmse_all']:.4f} (converged-only {summary[label]['rmse_converged']}), "
              f"max {summary[label]['maxabs_all']:.4f}", flush=True)
    with open(os.path.join(out, f'crosscheck_fem{args.tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
