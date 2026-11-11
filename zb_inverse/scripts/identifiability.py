#!/usr/bin/env python
"""E0c: how well can surface theta pin down zb? Cramer-Rao bound from the reference solver.

J[k, j] = d theta_obs(t_k) / d c_j by central differences in the spline coefficients c.
Cov(c) >= sigma^2 (J^T J)^-1, mapped to sigma_zb(t) with the spline Jacobian dzb/dc.
This is the best accuracy any unbiased inverse (PINN or FEM) can reach with that noise.

Usage: python -u scripts/identifiability.py --config configs/twin_pas_apr.yaml [--hourly]
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np

from common import load_with_overrides, setup, hourly_times, run_reference, bound_from_jacobian, save_bound
from zbinv.spline import ZbSplineNumpy

_G = {}


def _obs_for(coef):
    cfg, soil, win, truth, t_eval = _G['cfg'], _G['soil'], _G['win'], _G['truth'], _G['t_eval']
    m = ZbSplineNumpy(truth.s, coef, truth.zb_min)
    return run_reference(cfg, soil, win, m, t_eval, verbose=False)['theta_obs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    ap.add_argument('--tag', default='', help='suffix for output file names')
    ap.add_argument('--delta', type=float, default=0.02, help='FD step in spline-coefficient units')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    soil, win, sp, truth = setup(cfg)
    t_eval = hourly_times(win['n_days'])
    _G.update(cfg=cfg, soil=soil, win=win, truth=truth, t_eval=t_eval)

    jobs = []
    for j in range(sp.n_coef):
        for s in (+1.0, -1.0):
            c = truth.coef.copy()
            c[j] += s * args.delta
            jobs.append(c)
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    print(f"[ident] {win['start_date']} +{win['n_days']} d: {sp.n_coef} coefs, {len(jobs)} solves on {nproc} procs",
          flush=True)
    with Pool(nproc) as pool:
        res = pool.map(_obs_for, jobs)
    J_h = np.stack([(res[2 * j] - res[2 * j + 1]) / (2 * args.delta) for j in range(sp.n_coef)], axis=1)

    zb = truth.zb(t_eval)
    summary = {'window': win['start_date'], 'n_days': win['n_days'], 'n_coef': sp.n_coef}
    s, sig_curves, G = bound_from_jacobian(cfg, J_h, truth, t_eval)
    summary.update(s)
    save_bound(out, args.tag, summary, sig_curves, J_h, t_eval, zb, G)


if __name__ == '__main__':
    main()
