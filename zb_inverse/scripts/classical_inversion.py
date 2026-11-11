#!/usr/bin/env python
"""Referee: classical PDE-constrained inversion on exactly the data a PINN run sees.

Every candidate zb(t) (same 5-day cubic B-spline as the PINN) is run through the reference
finite-volume solver, so the physics is exact; Levenberg-Marquardt fits the spline coefficients
to the (noisy) theta_5cm observations, weighted by sigma_theta. Its zb error is what the data
allow; a PINN error above it belongs to the PINN. The local posterior sigma_zb(t) comes from
(J^T J)^-1 at the solution.

Usage: python -u scripts/classical_inversion.py --config configs/e2_inverse_synth.yaml \
           --set twin_npz=... --set obs.noise_sigma=0.005 --set obs.q_noise_frac=0.05 --out output/classical/<name>
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
import numpy as np

from common import load_with_overrides, run_reference
from zbinv.train import load_dataset, initial_coef
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy

_G = {}


def forward(coef):
    g = _G
    m = ZbSplineNumpy(g['spline'], coef, g['zb_min'])
    try:
        return run_reference(g['tcfg'], g['soil'], g['win'], m, g['t_obs'], n_nodes=g['n_nodes'],
                             verbose=False)['theta_obs']
    except Exception:
        return np.full(g['t_obs'].size, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_nodes', type=int, default=401)
    ap.add_argument('--delta', type=float, default=0.01, help='FD step in spline-coefficient units')
    ap.add_argument('--max_iter', type=int, default=40)
    ap.add_argument('--box', action='store_true',
                    help='keep the coefficients in the support of the path prior (the box of scripts/param_invert.py '
                         '--box, same active-set rule): reference for the constrained estimate')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(cfg)
    knot = cfg.get('zb_spline', {}).get('knot_days', ds['twin_cfg']['zb_spline']['knot_days'])
    spline = CubicBSpline1D(0.0, ds['n_days'], knot)
    coef0, _ = initial_coef(cfg, ds, spline)
    tcfg = dict(ds['twin_cfg'])
    tcfg['obs'] = dict(tcfg['obs'], depth_m=cfg['obs']['depth_m'])
    _G.update(spline=spline, zb_min=ds['zb_min'], tcfg=tcfg, soil=ds['soil'],
              win={'t_days': ds['t_q'], 'qsurf': ds['q_obs']}, t_obs=ds['t_obs'], n_nodes=args.n_nodes)
    sig = float(cfg['sigmas']['theta'])
    y = ds['theta_obs']
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    pool = Pool(nproc)

    def cost_of(th):
        r = (th - y) / sig
        return float(np.mean(r ** 2)) if np.all(np.isfinite(r)) else np.inf

    box = None
    if args.box:
        from zbinv.parametric import CPrior
        pc = cfg.get('prior', {})
        box = CPrior(spline, ds['zb_min'], pc.get('mean_range_m', (0.3, 1.2)), pc.get('step_sd_m', 0.15),
                     pc.get('clip_m', (0.2, 1.5))).box()
        print(f"[lm] coefficients kept in the prior box [{box[0]:.3f}, {box[1]:.3f}]", flush=True)
    c = coef0.copy() if box is None else np.clip(coef0, box[0], box[1])
    th = forward(c)
    cost = cost_of(th)
    lam = 1e-2
    tic = time.time()
    t_h = ds['t_hourly']
    inner = t_h <= ds['n_days'] - spline.dt
    hist = []

    def zb_err(cc):
        e = ZbSplineNumpy(spline, cc, ds['zb_min']).zb(t_h) - ds['zb_true']
        return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(e[inner] ** 2)))

    print(f"[lm] start: cost {cost:.4e}, zb RMSE {zb_err(c)[0]:.4f} m, {nproc} procs", flush=True)
    for it in range(args.max_iter):
        jobs = []
        for j in range(spline.n_coef):
            for s in (+1.0, -1.0):
                cj = c.copy(); cj[j] += s * args.delta; jobs.append(cj)
        res = pool.map(forward, jobs)
        J = np.stack([(res[2 * j] - res[2 * j + 1]) / (2 * args.delta) for j in range(spline.n_coef)], axis=1) / sig
        r = (th - y) / sig
        A, g = J.T @ J, J.T @ r
        free = np.ones(spline.n_coef, dtype=bool)
        if box is not None:      # a coefficient on a bound whose descent direction leaves the box is held there
            free = ~(((c <= box[0]) & (g > 0)) | ((c >= box[1]) & (g < 0)))
            if not free.any():
                break
        Af, gf = A[np.ix_(free, free)], g[free]
        accepted = False
        for _ in range(10):
            step = np.zeros_like(c)
            step[free] = -np.linalg.solve(Af + lam * np.diag(np.diag(Af)), gf)
            if box is not None:
                step = np.clip(c + step, box[0], box[1]) - c
            c_new = c + step
            th_new = forward(c_new)
            cost_new = cost_of(th_new)
            if cost_new < cost:
                rel = (cost - cost_new) / cost
                c, th, cost = c_new, th_new, cost_new
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 4.0
        e_all, e_in = zb_err(c)
        hist.append({'iter': it, 'cost': cost, 'lambda': lam, 'zb_rmse': e_all, 'zb_rmse_inner': e_in})
        print(f"[lm {it:2d}] cost {cost:.6e} lambda {lam:.1e} |step| {np.abs(step).max():.2e} "
              f"zb RMSE {e_all:.4f} (days 0-{ds['n_days'] - spline.dt:.0f}: {e_in:.4f}) m  {time.time() - tic:.0f}s",
              flush=True)
        if not accepted or rel < 1e-7 or np.abs(step).max() < 1e-6:
            break
    pool.close()

    m = ZbSplineNumpy(spline, c, ds['zb_min'])
    zb = m.zb(t_h)
    e = zb - ds['zb_true']
    G = m.dzb_dcoef(t_h)
    cov_c = np.linalg.pinv(A, rcond=1e-12)                    # J already divided by sigma
    s_zb = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G, cov_c, G), 0.0))
    summary = {
        'zb_rmse': float(np.sqrt(np.mean(e ** 2))), 'zb_rmse_inner': float(np.sqrt(np.mean(e[inner] ** 2))),
        'zb_bias': float(e.mean()), 'zb_maxabs': float(np.abs(e).max()),
        'theta_rmse_vs_obs': float(np.sqrt(np.mean((th - y) ** 2))),
        'theta_rmse_vs_truth': float(np.sqrt(np.mean((th - ds['theta_obs_true']) ** 2))),
        'posterior_sigma_zb_median_m': float(np.median(s_zb)),
        'posterior_sigma_zb_rms_inner_m': float(np.sqrt(np.mean(s_zb[inner] ** 2))),
        'iterations': len(hist), 'wall_s': round(time.time() - tic, 1),
        'noise_sigma': cfg['obs'].get('noise_sigma', 0.0), 'q_noise_frac': cfg['obs'].get('q_noise_frac', 0.0),
    }
    np.savez(os.path.join(args.out, 'predictions.npz'), t_days=t_h, zb_pred=zb, zb_true=ds['zb_true'],
             zb_init=ZbSplineNumpy(spline, coef0, ds['zb_min']).zb(t_h), zb_sigma=s_zb,
             theta_obs_pred=th, t_obs_days=ds['t_obs'], theta_obs=y, coef=c)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(hist, f, indent=1)
    print(f"[lm] final: {json.dumps(summary)}", flush=True)
    print("[lm] coefficients " + " ".join(f"{v:+.3f}" for v in c), flush=True)


if __name__ == '__main__':
    main()
