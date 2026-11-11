#!/usr/bin/env python
"""Soil parameters of a site from a CALIBRATION window whose water table is known (nothing is inverted for zb).

Forward model: the wtd_pipeline FEM with the observed water table (hourly record) and the forcing of the window, as
written by scripts/make_real.py. Unknowns: van Genuchten theta_s, theta_r, alpha, n and Ks. Least squares on the
hourly soil moisture at the sensor (scipy trust-region, bounded; forward-difference Jacobian from one batched FEM
solve). One start per job (--start): the calibration is accepted only if the starts agree.
The result is used unchanged for ANOTHER window, where the water table is the unknown.

Usage: python -u scripts/calibrate_soil.py --data output/real_ona_nov21 --out output/soil_ona_nov21 --start 0
       several windows after --data: one soil fitted to all of them (every hourly value has the same weight)
"""
import argparse
import json
import os
import time
import numpy as np
import yaml
from scipy.optimize import least_squares

from common import load_with_overrides  # noqa: F401  (puts src on the path)
from make_twin_fem import load_fem, run_fem
from zbinv import plots

NAMES = ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')
# x = (theta_s, theta_r, ln alpha [1/m], ln (n - 1), ln Ks [m/s]); bounds wide enough for sand to clay
LOWER = np.array([0.30, 0.00, np.log(0.2), np.log(0.10), np.log(1.0e-7)])
UPPER = np.array([0.60, 0.20, np.log(20.0), np.log(9.0), np.log(1.0e-3)])
STEP = np.array([1.0e-3, 1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2])          # forward-difference steps in x
STARTS = [                                                        # starting soils (the first: the data set's own)
    None,
    {'theta_s': 0.43, 'theta_r': 0.045, 'alpha': 14.5, 'n': 2.68, 'Ks': 8.25e-5},     # Carsel & Parrish sand
    {'theta_s': 0.41, 'theta_r': 0.065, 'alpha': 7.5, 'n': 1.89, 'Ks': 1.23e-5},      # Carsel & Parrish sandy loam
    'static',                                                     # retention fitted as if the profile were hydrostatic
]


def to_x(p):
    return np.array([p['theta_s'], p['theta_r'], np.log(p['alpha']), np.log(p['n'] - 1.0), np.log(p['Ks'])])


def to_p(x):
    x = np.asarray(x, dtype=float)
    return {'theta_s': x[..., 0], 'theta_r': x[..., 1], 'alpha': np.exp(x[..., 2]), 'n': 1.0 + np.exp(x[..., 3]),
            'Ks': np.exp(x[..., 4])}


def vg_theta(p, h):
    """van Genuchten theta(h), h <= 0 [m]."""
    m = 1.0 - 1.0 / p['n']
    return p['theta_r'] + (p['theta_s'] - p['theta_r']) * (1.0 + (p['alpha'] * np.abs(h)) ** p['n']) ** (-m)


def static_start(zb, depth, theta_obs, ks):
    """Retention curve through the (water-table depth, theta) pairs, read as a hydrostatic profile."""
    h = -(zb - depth)
    x0 = to_x({'theta_s': float(theta_obs.max()) + 0.01, 'theta_r': 0.03, 'alpha': 2.0, 'n': 2.5, 'Ks': ks})
    fit = least_squares(lambda x: vg_theta(to_p(np.append(x, x0[4])), h) - theta_obs, x0[:4],
                        bounds=(LOWER[:4], UPPER[:4]), x_scale='jac')
    return np.append(fit.x, x0[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, nargs='+',
                    help='directory of a make_real.py data set (calibration window); several: one soil for all of them')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-nfev', type=int, default=80)
    args = ap.parse_args()
    wins = []
    for path in args.data:
        with open(os.path.join(path, 'twin_config.yaml')) as f:
            c = yaml.safe_load(f)
        d = dict(np.load(os.path.join(path, 'real.npz'), allow_pickle=False))
        wins.append({'cfg': c, 't': d['t_days'], 'th': d['theta_obs_hourly'], 'zb': d['zb_true'],
                     'q': np.interp(d['t_days'], d['t_daily'], d['q_daily']),
                     'tp': np.interp(d['t_days'], d['t_daily'], d['tp_daily_mmd']),
                     'start': str(d['start_date']), 'n_days': int(d['n_days'])})
    cfg = wins[0]['cfg']                  # FEM set-up, sensor and the soil of start 0: taken from the first window
    for w in wins[1:]:
        if w['cfg']['fem'] != cfg['fem'] or w['cfg']['obs']['depth_m'] != cfg['obs']['depth_m']:
            raise SystemExit('the calibration windows differ in the FEM set-up or in the sensor depth')
    edges = np.cumsum([0] + [w['t'].size for w in wins])        # the windows one after the other on the time axis
    t, th_obs, zb = (np.concatenate([w[k] for w in wins]) for k in ('t', 'th', 'zb'))
    depth = cfg['obs']['depth_m']
    fc = cfg['fem']
    cfg['fem'] = {**fc, **fc.get('jacobian', {'tol': 1.0e-8, 'max_iter': 200})}      # differences need a tight solve
    Solver, Sink = load_fem(fc['root'])
    n_solves = [0]

    def forward(X):
        """theta at the sensor [B, n_t] for soils X [B, 5]; NaN rows where the solve failed."""
        X = np.atleast_2d(X)
        B = X.shape[0]
        n_solves[0] += B
        out = np.full((B, t.size), np.nan)
        try:
            for w, a, b in zip(wins, edges[:-1], edges[1:]):
                A = run_fem(Solver, Sink, cfg, w['t'], np.tile(w['zb'], (B, 1)), np.tile(w['q'], (B, 1)),
                            np.tile(w['tp'], (B, 1)), fc['n_nodes'], depth, soil_params=to_p(X), quiet=True)
                out[:, a:b] = A['theta_obs']
        except Exception as e:                                   # a soil the solver cannot run: reject the step
            print(f"[cal] solver failed: {type(e).__name__}: {e}", flush=True)
            return np.full((B, t.size), np.nan)
        out[~np.isfinite(out).all(axis=1)] = np.nan
        return out

    cache = {}

    def fun(x):
        th = forward(x)[0]
        cache['x'], cache['th'] = x.copy(), th
        r = np.where(np.isfinite(th), th - th_obs, 1.0)
        print(f"[cal] solve {n_solves[0]:4d}: rmse {np.sqrt(np.mean(r ** 2)):.5f}  "
              + '  '.join(f"{k}={float(v):.4g}" for k, v in to_p(x).items()), flush=True)
        return r

    def jac(x):
        th0 = cache['th'] if np.array_equal(cache.get('x'), x) else forward(x)[0]
        sign = np.where(x + STEP > UPPER, -1.0, 1.0)             # step inwards at an upper bound
        th = forward(x[None, :] + np.diag(sign * STEP))
        J = ((th - th0[None, :]) / (sign * STEP)[:, None]).T
        return np.where(np.isfinite(J), J, 0.0)

    start = STARTS[args.start]
    top, bottom = (0.0, float(depth)) if np.ndim(depth) == 0 else depth
    if start is None:
        x0 = to_x(cfg['soil'])
    elif start == 'static':
        x0 = static_start(zb, (top + bottom) / 2.0, th_obs, cfg['soil']['Ks'])
    else:
        x0 = to_x(start)
    x0 = np.clip(x0, LOWER + 1e-6, UPPER - 1e-6)
    print(f"[cal] {cfg['site']['name']} " + ', '.join(f"{w['start']} + {w['n_days']} d" for w in wins)
          + f", start {args.start}: " + ', '.join(f"{k}={float(v):.4g}" for k, v in to_p(x0).items()), flush=True)
    tic = time.time()
    th_start = forward(x0)[0]
    fit = least_squares(fun, x0, jac=jac, bounds=(LOWER, UPPER), x_scale='jac', max_nfev=args.max_nfev,
                        xtol=1e-6, ftol=1e-8, gtol=1e-10)
    th_fit = forward(fit.x)[0]
    r = th_fit - th_obs
    J = fit.jac
    sv = np.linalg.svd(J / np.maximum(np.linalg.norm(J, axis=0), 1e-300), compute_uv=False)
    cov = np.linalg.pinv(J.T @ J)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    p = {k: float(v) for k, v in to_p(fit.x).items()}
    one = len(wins) == 1
    res = {
        'site': cfg['site']['name'], 'window': wins[0]['start'] if one else [w['start'] for w in wins],
        'n_days': wins[0]['n_days'] if one else [w['n_days'] for w in wins], 'start': args.start,
        'by_window': [{'window': w['start'], 'rmse': float(np.sqrt(np.mean(r[a:b] ** 2))), 'bias': float(r[a:b].mean())}
                      for w, a, b in zip(wins, edges[:-1], edges[1:])],
        'soil_start': {k: float(v) for k, v in to_p(x0).items()}, 'soil': p,
        'rmse_start': float(np.sqrt(np.nanmean((th_start - th_obs) ** 2))),
        'rmse': float(np.sqrt(np.mean(r ** 2))), 'bias': float(r.mean()), 'max': float(np.abs(r).max()),
        'r': float(np.corrcoef(th_fit, th_obs)[0, 1]),
        'on_bound': [NAMES[i] for i in range(5) if fit.active_mask[i] != 0],
        'status': int(fit.status), 'message': fit.message, 'n_solves': n_solves[0], 'minutes': (time.time() - tic) / 60,
        'jacobian_singular_values_scaled': sv.tolist(), 'parameter_correlation': corr.tolist(),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'start{args.start}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(args.out, f'start{args.start}.npz'), t_days=t, theta_obs=th_obs, theta_start=th_start,
             theta_fit=th_fit, zb=zb, x=fit.x, x0=x0, window_edges=edges)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):      # one figure per window
        s = slice(a, b)
        plots.plot_twin_compare([(t[s], th_obs[s], 'observed', '-'), (t[s], th_start[s], 'FEM, starting soil', '-'),
                                 (t[s], th_fit[s], 'FEM, calibrated soil', '-')],
                                os.path.join(args.out, f'start{args.start}.png' if one else f'start{args.start}_w{i}.png'),
                                ylabel='soil moisture at the sensor (m$^3$/m$^3$)')
    if not one:
        print('[cal] by window: ' + ', '.join(f"{w['window']} rmse {w['rmse']:.4f} bias {w['bias']:+.4f}"
                                             for w in res['by_window']), flush=True)
    print(f"[cal] start {args.start}: rmse {res['rmse_start']:.4f} -> {res['rmse']:.4f} (bias {res['bias']:+.4f}, "
          f"r {res['r']:.3f}), on bound: {res['on_bound']}, {res['message']}", flush=True)
    print('[cal] soil: ' + ', '.join(f"{k}: {v:.5g}" for k, v in p.items()), flush=True)
    print(f"[cal] scaled Jacobian singular values: {np.array2string(sv, precision=3)}; "
          f"{n_solves[0]} member solves, {res['minutes']:.1f} min", flush=True)


if __name__ == '__main__':
    main()
