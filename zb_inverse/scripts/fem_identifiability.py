#!/usr/bin/env python
"""Cramer-Rao bound on zb(t) for the FEM twins of scripts/make_twin_fem.py, from the FEM itself.

J[k, j] = d theta_obs(t_k) / d c_j by central differences in the spline coefficients c, with and without the root
sink, from one batched solve of the wtd_pipeline FEM (the reference solver of scripts/identifiability.py has no sink).
The unperturbed path is part of the batch and is compared with the stored twin.
Writes identifiability_fem.{json,npz,png} into both twin directories.

Usage: python -u scripts/fem_identifiability.py --config configs/twin_gldas_fl.yaml
"""
import argparse
import os
import numpy as np

from common import load_with_overrides, setup, hourly_times, bound_from_jacobian, save_bound
from make_twin_fem import load_fem, run_fem, diff
from zbinv.spline import ZbSplineNumpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    ap.add_argument('--delta', type=float, default=0.02, help='FD step in spline-coefficient units')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    fc = cfg['fem']
    soil, win, sp, truth = setup(cfg)
    n_days = win['n_days']
    t = hourly_times(n_days, fc['dt_hours'])
    q = np.interp(t, win['t_days'], win['qsurf'])
    tp = np.interp(t, win['t_days'], win['tp_mmd'])

    coefs = [truth.coef]                                     # member 0: the twin's own path
    for j in range(sp.n_coef):
        for s in (+1.0, -1.0):
            c = truth.coef.copy()
            c[j] += s * args.delta
            coefs.append(c)
    zb = np.stack([ZbSplineNumpy(truth.s, c, truth.zb_min).zb(t) for c in coefs])
    n = zb.shape[0]
    Solver, Sink = load_fem(fc['root'])
    cfg_jac = {**cfg, 'fem': {**fc, **fc.get('jacobian', {})}}
    res = run_fem(Solver, Sink, cfg_jac, t, np.concatenate([zb, zb]), q,
                  np.concatenate([np.tile(tp, (n, 1)), np.zeros_like(zb)]), fc['n_nodes'], cfg['obs']['depth_m'])

    k_h = np.round(hourly_times(n_days) / (fc['dt_hours'] / 24.0)).astype(int)     # hourly observations
    t_h = t[k_h]
    for first, tag in [(0, ''), (n, '_nosink')]:
        out = cfg['output']['dir'] + tag
        th = res['theta_obs'][first:first + n][:, k_h]
        J = np.stack([(th[1 + 2 * j] - th[2 + 2 * j]) / (2 * args.delta) for j in range(sp.n_coef)], axis=1)
        twin = np.load(os.path.join(out, 'twin.npz'))
        summary = {'window': win['start_date'], 'n_days': n_days, 'n_coef': sp.n_coef, 'solver': 'wtd_pipeline FEM',
                   'root_sink': tag == '', 'delta': args.delta,
                   'twin_vs_this_run': diff(twin['theta_obs_hourly'][k_h], th[0])}
        print(f"[ident] {out}: twin vs this run at the true path, d theta_5cm RMSE "
              f"{summary['twin_vs_this_run']['rmse']:.2e}, max {summary['twin_vs_this_run']['max']:.2e}", flush=True)
        ref = os.path.join(out, 'identifiability.npz')
        if os.path.exists(ref):                              # the same Jacobian from this project's reference solver
            J_ref = np.load(ref)['J_hourly']
            norm, norm_ref = np.linalg.norm(J, axis=0), np.linalg.norm(J_ref, axis=0)
            summary['vs_reference_solver'] = {'column_cosine': ((J * J_ref).sum(axis=0) / (norm * norm_ref)).tolist(),
                                              'column_norm_ratio': (norm / norm_ref).tolist()}
            print(f"[ident] Jacobian vs reference solver: column cosine "
                  f"{np.round(summary['vs_reference_solver']['column_cosine'], 4).tolist()}, norm ratio "
                  f"{np.round(summary['vs_reference_solver']['column_norm_ratio'], 3).tolist()}", flush=True)
        s, sig_curves, G = bound_from_jacobian(cfg, J, truth, t_h)
        summary.update(s)
        if tag == '_nosink' and twin['theta_obs_reference_nosink'].shape == t_h.shape:
            # The zb change that explains the FEM-minus-reference difference in theta: what an exact solution of the
            # continuous equation gets wrong on this twin (the twin's discretisation error, seen as a zb error).
            d_theta = twin['theta_obs_hourly'][k_h] - twin['theta_obs_reference_nosink']
            d_zb = G @ np.linalg.lstsq(J, d_theta, rcond=None)[0]
            inner = t_h <= 25.0
            summary['zb_shift_for_twin_discretisation_m'] = {
                'rms_days_0_25': float(np.sqrt(np.mean(d_zb[inner] ** 2))), 'max_days_0_25': float(np.abs(d_zb[inner]).max()),
                'rms_whole': float(np.sqrt(np.mean(d_zb ** 2)))}
            print(f"[ident] zb shift that explains FEM - reference: {summary['zb_shift_for_twin_discretisation_m']}", flush=True)
        save_bound(out, '_fem', summary, sig_curves, J, t_h, truth.zb(t_h), G)


if __name__ == '__main__':
    main()
