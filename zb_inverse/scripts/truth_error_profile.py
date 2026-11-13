#!/usr/bin/env python
"""SCORING ONLY (never part of the method): where a trained parametric PINN differs from the twin at the TRUE path.

For each run directory (one member each): theta_5cm(t) of the network at the true coefficients vs the twin (hourly),
by day; theta(xi, t) vs the twin's stored profiles, by day and by depth band. Tells whether the model error sits at
the forcing events (time resolution), near the surface (depth resolution), or is a slow drift.

Usage: python -u scripts/truth_error_profile.py --members runs/g1gn_p05 [runs/...] [--out <npz>]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.parametric import load_members  # noqa: E402
from zbinv.physics import theta_obs_model, theta_profile, pde_residual, surface_flux  # noqa: E402

torch.set_default_dtype(torch.float64)
BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.0)]   # in -xi


def water_balance(m, tw, c, ep):
    """Daily water balance of the model error at the true path [mm/d]: d(storage error)/dt = column-integrated PDE
    residual - surface-flux error + (difference of the flux through the water table: a response, the remainder)."""
    t, xi = np.asarray(tw['t_days'], float), np.asarray(tw['xi'], float)
    to_mmd = m.soil.Ks * 86400e3
    tt = torch.as_tensor(t).reshape(-1, 1)
    dq = (surface_flux(m, tt, c).detach().numpy().ravel() * m.soil.Ks - tw['qsurf']) * 86400e3        # mm/d
    order = np.argsort(xi)
    col = np.zeros(t.size)
    for i in range(0, t.size, 48):
        X, T = np.meshgrid(xi, t[i:i + 48])
        R = pde_residual(m, torch.as_tensor(X.reshape(-1, 1)), torch.as_tensor(T.reshape(-1, 1)), c)
        R = R.detach().numpy().reshape(X.shape)
        col[i:i + 48] = np.trapezoid(R[:, order], xi[order], axis=1) * tw['zb_true'][i:i + 48] / m.soil.L * to_mmd
    dW = np.trapezoid(ep[:, order], xi[order], axis=1) * tw['zb_true'] * 1e3                              # mm
    print('  day  storage_err[mm]  d/dt[mm/d]  column_residual  -surface_flux_err  remainder(water-table flux)')
    for k in range(int(np.ceil(t[-1]))):
        s = np.nonzero((t >= k) & (t <= k + 1))[0]
        rate = (dW[s[-1]] - dW[s[0]]) / (t[s[-1]] - t[s[0]])
        r, f = np.trapezoid(col[s], t[s]) / (t[s[-1]] - t[s[0]]), -np.trapezoid(dq[s], t[s]) / (t[s[-1]] - t[s[0]])
        print(f"  {k:3d}  {dW[s[-1]]:+8.3f}        {rate:+7.3f}      {r:+7.3f}          {f:+7.3f}          {rate - r - f:+7.3f}")
    return dW, col, dq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sensor', choices=['obs', 'fem'], default='obs',
                    help="reference at the sensor: the data set's observations (twins: the twin) or, for a site "
                         "data set, the solver at the same path (theta_fem[0])")
    a = ap.parse_args()
    dev = torch.device('cpu')
    out = {}
    for d in a.members:
        (m,), cfg = load_members([d], dev)
        tw = np.load(cfg['twin_npz'], allow_pickle=True)
        t, xi = np.asarray(tw['t_days'], float), np.asarray(tw['xi'], float)
        c = torch.as_tensor(np.asarray(tw['truth_coef'], float))
        tt = torch.as_tensor(t).reshape(-1, 1)
        with torch.no_grad():
            th5 = theta_obs_model(m, tt, cfg['obs']['depth_m'], c).numpy().ravel()
        e5 = th5 - (tw['theta_fem'][0] if a.sensor == 'fem' else tw['theta_obs_hourly'])
        ep = theta_profile(m, torch.as_tensor(xi), torch.as_tensor(t), c).numpy() - tw['theta']      # (Nt, Nx)
        if ep.shape != tw['theta'].shape:
            raise ValueError(f'profile shapes {ep.shape} vs {tw["theta"].shape}')
        days = np.arange(0, int(np.ceil(t[-1])))
        print(f"[truth] {d}: theta_5cm RMSE {np.sqrt(np.mean(e5 ** 2)):.2e}, max {np.abs(e5).max():.2e}; "
              f"profile RMSE {np.sqrt(np.mean(ep ** 2)):.2e}, max {np.abs(ep).max():.2e}", flush=True)
        print('  day   q[mm/d]  th5_rms   th5_mean  th5_max   prof_rms')
        for k in days:
            s = (t >= k) & (t < k + 1)
            print(f"  {k:3d}  {86400e3 * tw['qsurf'][s].mean():8.2f}  {np.sqrt(np.mean(e5[s] ** 2)):.2e}  "
                  f"{e5[s].mean():+.2e}  {np.abs(e5[s]).max():.2e}  {np.sqrt(np.mean(ep[s] ** 2)):.2e}")
        print('  depth band (-xi)   prof_rms   prof_mean  prof_max')
        for lo, hi in BANDS:
            s = (-xi >= lo) & (-xi <= hi)
            print(f"  {lo:4.2f}-{hi:4.2f}          {np.sqrt(np.mean(ep[:, s] ** 2)):.2e}  {ep[:, s].mean():+.2e}  "
                  f"{np.abs(ep[:, s]).max():.2e}")
        k = int(np.argmax(np.abs(e5)))
        print(f"  largest theta_5cm error at day {t[k]:.2f}: {e5[k]:+.2e}; the 10 worst hours: "
              + ' '.join(f"{t[i]:.2f}({1e3 * e5[i]:+.1f}e-3)" for i in np.argsort(-np.abs(e5))[:10]), flush=True)
        water_balance(m, tw, c, ep)
        out[os.path.basename(os.path.normpath(d))] = (e5, ep)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez(a.out, t_days=t, xi=xi, **{f'e5_{k}': v[0] for k, v in out.items()},
                 **{f'ep_{k}': v[1] for k, v in out.items()})


if __name__ == '__main__':
    main()
