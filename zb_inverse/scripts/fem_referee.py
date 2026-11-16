#!/usr/bin/env python
"""Referee for data sets whose model has the root sink (site data, FEM twins). SCORING ONLY: no PINN run reads its output.

The inversion of scripts/classical_inversion.py with the wtd_pipeline FEM as forward model (that script's reference
solver has no root sink): every candidate zb(t), the same cubic B-spline as the PINN's, is run through the FEM with the
data set's soil and forcing; Levenberg-Marquardt fits the spline coefficients to the soil moisture at the sensor
(central differences, one batched FEM solve per Jacobian). Same starts, cost and --box rule as scripts/param_invert.py.
On site data its distance to the observed water table is what the model and the data allow (soil, forcing, sensor);
the distance of a PINN estimate from the referee is the PINN's own error.

Usage: python -u scripts/fem_referee.py --config configs/p1_param_synth_v3.yaml --set twin_npz=.../real.npz \
           --set obs.depth_m=[0.1,0.1] --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 --box \
           --start hydrostatic_flat --out output/referee/<name>
"""
import argparse
import json
import os
import shutil
import time
import numpy as np

from common import load_with_overrides, hourly_times
from make_twin_fem import load_fem, run_fem
from zbinv.train import load_dataset, initial_coef
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', default='hydrostatic_flat',
                    help='kind[:value] as in scripts/param_invert.py --starts, or file:<predictions.npz> to start at '
                         'the coefficients of another estimate (the solver\'s cost of that path is the first line)')
    ap.add_argument('--delta', type=float, default=0.01, help='FD step in spline-coefficient units')
    ap.add_argument('--max_iter', type=int, default=40)
    ap.add_argument('--box', action='store_true', help='keep the coefficients in the support of the path prior')
    ap.add_argument('--dt_hours', type=float, default=None,
                    help='time step of the FEM (default: the hourly grid of the data set); the forcing stays linear '
                         'between its nodes and the observations are read at their own times')
    ap.add_argument('--value_only', action='store_true',
                    help='with --start file:<predictions.npz>: no inversion. Writes <out>/value_at.json: the solver\'s '
                         'value at the sensor on that path against the value stored in the file (theta_obs_pred), by '
                         'day, and the solver\'s first step from that path split into the part of that value '
                         'difference and the rest')
    ap.add_argument('--path_prior', action='store_true',
                    help='add the path term of the prior to the cost (CPrior.increments: the steps of the knot depths '
                         'over prior.step_sd_m), as scripts/param_invert.py --path_prior: the estimate is then the '
                         'most probable path given the data and that prior')
    ap.add_argument('--twin_out', default=None,
                    help='no inversion: write a twin of the data set to this directory. Its water table is the spline '
                         'fit of the record (truth_coef; with --box clipped to the box), its soil moisture at the '
                         'sensor the FEM on that path (the forward model of this script). Noise is added on reading '
                         '(obs.noise_sigma, obs.noise_seed)')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(cfg)
    tcfg = ds['twin_cfg']
    fc = tcfg['fem']
    tcfg['fem'] = {**fc, **fc.get('jacobian', {'tol': 1.0e-8, 'max_iter': 200})}      # differences need a tight solve
    knot = cfg.get('zb_spline', {}).get('knot_days', tcfg['zb_spline']['knot_days'])
    spline = CubicBSpline1D(0.0, ds['n_days'], knot)
    kind, _, val = args.start.partition(':')
    if kind == 'file':      # start at another estimate on the same spline: 'coef' of its predictions.npz
        coef0 = np.array(np.load(val, allow_pickle=False)['coef'], dtype=float)
        if coef0.shape != (spline.n_coef,):
            raise SystemExit(f'{val}: coefficients {coef0.shape}, the spline of this data set has {spline.n_coef}')
    else:
        coef0, _ = initial_coef(dict(cfg, zb_init={'kind': kind, 'zb_max': 3.0,
                                                   **({'value': float(val)} if val else {})}), ds, spline)
    t_h, t_obs, y = ds['t_hourly'], ds['t_obs'], ds['theta_obs']
    t_s = t_h if args.dt_hours is None else hourly_times(ds['n_days'], args.dt_hours)      # time grid of the solver
    k_obs = np.round((t_obs - t_s[0]) / (t_s[1] - t_s[0])).astype(int)
    if not np.allclose(t_s[k_obs], t_obs, rtol=0, atol=1e-9):
        raise SystemExit('observation times are not on the FEM time grid')
    q = np.interp(t_s, ds['t_q'], ds['q_obs'])
    tp = np.zeros_like(t_s) if ds['tp_mmd'] is None else np.interp(t_s, ds['t_tp'], ds['tp_mmd'])
    depth = cfg['obs']['depth_m']
    Solver, Sink = load_fem(fc['root'])
    sig = float(cfg['sigmas']['theta'])
    info = {'solves': 0, 'not_converged': 0.0}

    def forward(C):
        """theta at the sensor [B, n_obs] for coefficient rows C [B, n_coef]; NaN rows where the solve failed."""
        C = np.atleast_2d(C)
        zb = np.stack([ZbSplineNumpy(spline, c, ds['zb_min']).zb(t_s) for c in C])
        info['solves'] += C.shape[0]
        try:
            A = run_fem(Solver, Sink, tcfg, t_s, zb, q, np.tile(tp, (C.shape[0], 1)), fc['n_nodes'], depth, quiet=True)
        except Exception as e:
            print(f"[ref] solver failed: {type(e).__name__}: {e}", flush=True)
            return np.full((C.shape[0], t_obs.size), np.nan)
        info['not_converged'] = max(info['not_converged'], 1.0 - float(A['converged'].mean()))
        return A['theta_obs'][:, k_obs]

    box, prior = None, None
    if args.box or args.path_prior:
        import torch
        from zbinv.parametric import CPrior
        pc = cfg.get('prior', {})
        cp = CPrior(spline, ds['zb_min'], pc.get('mean_range_m', (0.3, 1.2)), pc.get('step_sd_m', 0.15),
                    pc.get('clip_m', (0.2, 1.5)))
    if args.box:
        box = cp.box()
        print(f"[ref] coefficients kept in the prior box [{box[0]:.3f}, {box[1]:.3f}]", flush=True)
    if args.path_prior:      # p (n_coef - 1,) and P = dp/dc: rows added to the residual and the Jacobian
        def prior(cc):
            p, P = cp.increments(torch.as_tensor(np.asarray(cc, dtype=float)))
            return p.numpy(), P.numpy()
        print(f"[ref] path term in the cost: steps of the knot depths over {cp.step_sd} m", flush=True)

    def cost_of(th, cc):
        """Data misfit mean(r^2); with --path_prior plus sum(p^2) / n_obs (minus twice the log posterior / n_obs)."""
        r = (th - y) / sig
        if not np.all(np.isfinite(r)):
            return np.inf
        data = float(np.mean(r ** 2))
        return data if prior is None else data + float(np.sum(prior(cc)[0] ** 2)) / r.size

    if args.twin_out:
        ct = np.array(ds['truth_coef'], dtype=float)
        if ct.shape != (spline.n_coef,):
            raise SystemExit(f'truth_coef {ct.shape}: the spline of this data set has {spline.n_coef} coefficients')
        if t_obs.size != t_h.size or not np.allclose(t_obs, t_h):
            raise SystemExit('--twin_out needs the hourly observations (obs.use=hourly)')
        if box is not None:
            print(f"[twin] {int(np.sum((ct < box[0]) | (ct > box[1])))} coefficient(s) of the fit clipped to the box",
                  flush=True)
            ct = np.clip(ct, box[0], box[1])
        th_t = forward(ct)[0]
        if not np.all(np.isfinite(th_t)):
            raise SystemExit('the solver failed on the path of the twin')
        tw = dict(np.load(cfg['twin_npz'], allow_pickle=False))
        k_d = np.round((tw['t_obs_days'] - t_h[0]) / (t_h[1] - t_h[0])).astype(int)
        if not np.allclose(t_h[k_d], tw['t_obs_days'], rtol=0, atol=1e-9):
            raise SystemExit('t_obs_days is not on the hourly grid')
        zb_t = ZbSplineNumpy(spline, ct, ds['zb_min']).zb(t_h)
        tw.update(theta_obs_hourly=th_t, theta_obs_daily=th_t[k_d], zb_true=zb_t, zb_spline=zb_t, truth_coef=ct,
                  twin_of=np.array(os.path.abspath(cfg['twin_npz'])))
        os.makedirs(args.twin_out, exist_ok=True)
        np.savez(os.path.join(args.twin_out, os.path.basename(cfg['twin_npz'])), **tw)
        shutil.copy(os.path.join(os.path.dirname(cfg['twin_npz']), 'twin_config.yaml'), args.twin_out)
        print(f"[twin] {args.twin_out}: water table {zb_t.min():.3f}-{zb_t.max():.3f} m, rms of the fit minus the "
              f"record {np.sqrt(np.mean((zb_t - ds['zb_true']) ** 2)):.4f} m; theta at the sensor rms of twin minus "
              f"observed {np.sqrt(np.mean((th_t - ds['theta_obs_true']) ** 2)):.4f}; steps not converged "
              f"{info['not_converged']:.3f}", flush=True)
        return

    c = coef0.copy() if box is None else np.clip(coef0, box[0], box[1])
    th = forward(c)[0]
    cost = cost_of(th, c)
    lam = 1e-2
    tic = time.time()
    inner = t_h <= ds['n_days'] - spline.dt
    n_c = spline.n_coef
    hist = []

    def zb_err(cc):
        e = ZbSplineNumpy(spline, cc, ds['zb_min']).zb(t_h) - ds['zb_true']
        return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(e[inner] ** 2)))

    print(f"[ref] start {args.start}: cost {cost:.4e}, zb RMSE {zb_err(c)[0]:.4f} m", flush=True)
    if args.value_only:
        # The file's estimate has its own value at the sensor, th_p (a PINN's: scripts/param_invert.py). With the
        # solver's value th at the same path and e = th_p - th, the solver's first step from the path (the first trial
        # step of the loop below, before clipping) is  -M (th - y) = M e - M (th_p - y),  M = (A + lam diag A)^-1 J^T:
        # 'value_part' is what the value difference alone moves, 'rest' what is left with the file's own value.
        if kind != 'file':
            raise SystemExit('--value_only needs --start file:<predictions.npz>')
        z = np.load(val, allow_pickle=False)
        if z['t_obs_days'].shape != t_obs.shape or not np.allclose(z['t_obs_days'], t_obs):
            raise SystemExit(f'{val}: other observation times than the data set')
        if not np.allclose(c, coef0):
            raise SystemExit(f'{val}: its path is outside the box')
        th_p = np.asarray(z['theta_obs_pred'], dtype=float)
        res = forward(np.concatenate([c + args.delta * np.eye(n_c), c - args.delta * np.eye(n_c)]))
        if not (np.all(np.isfinite(res)) and np.all(np.isfinite(th))):
            raise SystemExit('the solver failed at the path or at a shifted path')
        J = (res[:n_c] - res[n_c:]).T / (2 * args.delta) / sig
        g = J.T @ ((th - y) / sig)
        if prior is not None:      # the path term is the same for both values: it stays in total and in rest
            p, P = prior(c)
            g = g + P.T @ p
        free = np.ones(n_c, dtype=bool) if box is None else ~(((c <= box[0]) & (g > 0)) | ((c >= box[1]) & (g < 0)))
        Jf = J[:, free]
        Af = Jf.T @ Jf
        gp = 0.0
        if prior is not None:
            Af, gp = Af + P[:, free].T @ P[:, free], P[:, free].T @ p
        Ad = Af + lam * np.diag(np.diag(Af)) + 1e-12 * np.eye(int(free.sum()))
        G = ZbSplineNumpy(spline, c, ds['zb_min']).dzb_dcoef(t_h)[:, free]       # zb moves by G @ step (linearised)
        e = th_p - th
        parts = {'total': -np.linalg.solve(Ad, Jf.T @ ((th - y) / sig) + gp),
                 'value_part': np.linalg.solve(Ad, Jf.T @ (e / sig)),
                 'rest': -np.linalg.solve(Ad, Jf.T @ ((th_p - y) / sig) + gp)}
        n_day = int(np.ceil(t_h[-1] - t_h[0]))

        def by_day(t, v, f):
            day = np.minimum(np.floor(t - t_h[0]).astype(int), n_day - 1)
            return [float(f(v[day == k])) for k in range(n_day)]

        def rms(v):
            return float(np.sqrt(np.mean(v ** 2)))
        out = {'start': args.start, 'fem_dt_hours': float((t_s[1] - t_s[0]) * 24.0), 'cost_solver': cost,
               'cost_file_value': cost_of(th_p, c), 'free': [int(j) for j in np.where(free)[0]],
               'fem_not_converged_max': info['not_converged'],
               'theta_file_minus_solver': {'rms': rms(e), 'bias': float(e.mean()), 'maxabs': float(np.abs(e).max()),
                                           'rms_by_day': by_day(t_obs, e, rms), 'mean_by_day': by_day(t_obs, e, np.mean)},
               'first_step_zb_m': {k: {'rms': rms(G @ v), 'rms_by_day': by_day(t_h, G @ v, rms),
                                       'mean_by_day': by_day(t_h, G @ v, np.mean)} for k, v in parts.items()}}
        with open(os.path.join(args.out, 'value_at.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[val] cost: solver {cost:.6f}, with the file's value {out['cost_file_value']:.6f}; value difference "
              f"rms {rms(e):.2e}, by day {' '.join(f'{v:.1e}' for v in out['theta_file_minus_solver']['rms_by_day'])}")
        for k, v in out['first_step_zb_m'].items():
            print(f"[val] first step, {k:10s}: rms {100 * v['rms']:.2f} cm, mean by day (cm) "
                  f"{' '.join(f'{100 * x:+.2f}' for x in v['mean_by_day'])}", flush=True)
        return
    for it in range(args.max_iter):
        res = forward(np.concatenate([c + args.delta * np.eye(n_c), c - args.delta * np.eye(n_c)]))
        J = np.where(np.isfinite(res[:n_c] - res[n_c:]), res[:n_c] - res[n_c:], 0.0).T / (2 * args.delta) / sig
        r = (th - y) / sig
        A, g = J.T @ J, J.T @ r
        if prior is not None:
            p, P = prior(c)
            A, g = A + P.T @ P, g + P.T @ p
        free = np.ones(n_c, dtype=bool)
        if box is not None:      # a coefficient on a bound whose descent direction leaves the box is held there
            free = ~(((c <= box[0]) & (g > 0)) | ((c >= box[1]) & (g < 0)))
            if not free.any():
                break
        Af, gf = A[np.ix_(free, free)], g[free]
        accepted = False
        for _ in range(10):
            step = np.zeros_like(c)
            step[free] = -np.linalg.solve(Af + lam * np.diag(np.diag(Af)) + 1e-12 * np.eye(int(free.sum())), gf)
            if box is not None:
                step = np.clip(c + step, box[0], box[1]) - c
            th_new = forward(c + step)[0]
            cost_new = cost_of(th_new, c + step)
            if cost_new < cost:
                rel = (cost - cost_new) / cost
                c, th, cost = c + step, th_new, cost_new
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 4.0
        e_all, e_in = zb_err(c)
        hist.append({'iter': it, 'cost': cost, 'lambda': lam, 'zb_rmse': e_all, 'zb_rmse_inner': e_in})
        print(f"[ref {it:2d}] cost {cost:.6e} lambda {lam:.1e} |step| {np.abs(step).max():.2e} zb RMSE {e_all:.4f} "
              f"(days 0-{ds['n_days'] - spline.dt:.0f}: {e_in:.4f}) m  {time.time() - tic:.0f}s", flush=True)
        if not accepted or rel < 1e-7 or np.abs(step).max() < 1e-6:
            break

    m = ZbSplineNumpy(spline, c, ds['zb_min'])
    zb = m.zb(t_h)
    e = zb - ds['zb_true']
    G = m.dzb_dcoef(t_h)
    cov_c = np.linalg.pinv(A, rcond=1e-12)                    # J already divided by sigma
    s_zb = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G, cov_c, G), 0.0))
    summary = {
        'start': args.start, 'fem_dt_hours': float((t_s[1] - t_s[0]) * 24.0), 'zb_rmse': float(np.sqrt(np.mean(e ** 2))),
        'zb_rmse_inner': float(np.sqrt(np.mean(e[inner] ** 2))), 'zb_bias': float(e.mean()),
        'zb_maxabs': float(np.abs(e).max()), 'theta_rmse_vs_obs': float(np.sqrt(np.mean((th - y) ** 2))),
        'cost': cost, 'posterior_sigma_zb_median_m': float(np.median(s_zb)),
        'coef_on_bound': [] if box is None else [int(j) for j in np.where((c <= box[0]) | (c >= box[1]))[0]],
        'iterations': len(hist), 'member_solves': info['solves'], 'fem_not_converged_max': info['not_converged'],
        'wall_s': round(time.time() - tic, 1),
        # with --path_prior 'cost' is data + path term; here the two parts at the estimate
        'path_prior': None if prior is None else {
            'step_sd_m': cp.step_sd, 'cost_data': float(np.mean(((th - y) / sig) ** 2)),
            'cost_prior': float(np.sum(prior(c)[0] ** 2)) / y.size,
            'max_abs_increment_sd': float(np.abs(prior(c)[0]).max())},
    }
    np.savez(os.path.join(args.out, 'predictions.npz'), t_days=t_h, zb_pred=zb, zb_true=ds['zb_true'],
             zb_init=ZbSplineNumpy(spline, coef0, ds['zb_min']).zb(t_h), zb_sigma=s_zb,
             theta_obs_pred=th, t_obs_days=t_obs, theta_obs=y, coef=c)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(hist, f, indent=1)
    print(f"[ref] final: {json.dumps(summary)}", flush=True)
    print("[ref] coefficients " + " ".join(f"{v:+.3f}" for v in c), flush=True)


if __name__ == '__main__':
    main()
