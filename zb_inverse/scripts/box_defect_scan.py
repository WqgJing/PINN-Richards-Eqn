#!/usr/bin/env python
"""SCORING ONLY (never part of the method): which space-time boxes' water-balance defect tells the model's actual
error at the TRUE path.

For each run directory (one member each): PDE residual and surface-flux error on a dense grid at the true
coefficients -> the defect of the water balance of the layer [-D, 0] over windows of length dur, as a mean water
content of the layer (what physics.box_water_defect estimates), for a table of (D, dur). Next to it the actual change
of the layer's mean-theta error over the same windows (model - twin). Where the two agree the defect stays in the
layer; where the defect is much larger, the layer has passed it on to the soil below within the window.
Also the sampling error of the n_t x n_xi-point estimate used in training.

Usage: python -u scripts/box_defect_scan.py --members runs/g1gn_p05 [runs/...]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.parametric import load_members  # noqa: E402
from zbinv.physics import theta_profile, pde_residual, surface_flux  # noqa: E402

torch.set_default_dtype(torch.float64)
NX, PER_DAY = 80, 96                                    # dense grid: 80 depth cells, 15-minute time cells
DEPTHS = [0.05, 0.1, 0.25, 0.5, 1.0]                    # layer as a fraction of the column
DURS = [1 / 24, 1 / 8, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]   # window [d]


def dense_fields(m, tw, c):
    """R (Nt, NX) at cell midpoints, surface-flux error f (Nt,), z~ (Nt,); t midpoints of 15-minute cells."""
    n_days = float(tw['t_days'][-1])
    nt = int(round(n_days * PER_DAY))
    t = (np.arange(nt) + 0.5) / PER_DAY
    xi = -(np.arange(NX) + 0.5) / NX
    R = np.zeros((nt, NX))
    for i in range(0, nt, 96):
        X, T = np.meshgrid(xi, t[i:i + 96])
        R[i:i + 96] = pde_residual(m, torch.as_tensor(X.reshape(-1, 1)), torch.as_tensor(T.reshape(-1, 1)),
                                   c).detach().numpy().reshape(X.shape)
    tt = torch.as_tensor(t).reshape(-1, 1)
    q_surf = np.interp(t, tw['t_days'], tw['qsurf']) / m.soil.Ks       # piecewise linear between whole days: exact
    f = surface_flux(m, tt, c).detach().numpy().ravel() - q_surf
    with torch.no_grad():
        z = (m.zb_m(tt, c) / m.soil.L).numpy().ravel()
    return t, R, f, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--n-t', type=int, default=8)
    ap.add_argument('--n-xi', type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for d in a.members:
        (m,), cfg = load_members([d], torch.device('cpu'))
        tw = np.load(cfg['twin_npz'], allow_pickle=True)
        c = torch.as_tensor(np.asarray(tw['truth_coef'], float))
        sig = float(cfg['sigmas']['theta'])
        t, R, f, z = dense_fields(m, tw, c)
        nt, dtt = t.size, 1.0 / PER_DAY / m.T_days
        th, xi_tw = np.asarray(tw['t_days'], float), np.asarray(tw['xi'], float)
        ep = theta_profile(m, torch.as_tensor(xi_tw), torch.as_tensor(th), c).numpy() - tw['theta']   # (Nh, Nx)
        order = np.argsort(xi_tw)
        step = PER_DAY // 24                                           # window starts on the twin's hourly times
        zc = np.concatenate([[0.0], np.cumsum(z)])
        print(f"[scan] {d}: theta* {m.soil.theta_star:.3f}, T {m.T_days:.2f} d, mean zb {z.mean() * m.soil.L:.2f} m, "
              f"sigma_theta {sig}; all entries rms over window starts, in units of sigma_theta", flush=True)
        for D in DEPTHS:
            nD = int(round(D * NX))
            b = z * D * R[:, :nD].mean(1) - f                          # defect rate of the layer (flux units)
            Bc = np.concatenate([[0.0], np.cumsum(b)]) * dtt
            s = order[-xi_tw[order] <= D + 1e-12]
            E = np.trapezoid(ep[:, s], xi_tw[s], axis=1) / D           # layer-mean theta error at the hourly times
            print(f"  layer {D:4.2f} of the column (~{100 * D * z.mean() * m.soil.L:5.1f} cm): layer-mean theta error "
                  f"rms {np.sqrt(np.mean(E ** 2)) / sig:.3f}, max {np.abs(E).max() / sig:.3f}; "
                  f"time to drain its storage at Ks {D * z.mean() * m.T_days:.2f} d")
            print('    window[d]   defect   error_change   defect/change   sampling_error_of_estimate')
            for dur in DURS:
                n = int(round(dur * PER_DAY))
                i0 = np.arange(0, nt - n + 1, step)
                zbar = (zc[i0 + n] - zc[i0]) / n
                delta = m.soil.theta_star * (Bc[i0 + n] - Bc[i0]) / (zbar * D)
                h0 = i0 // step
                dE = E[h0 + n // step] - E[h0]
                # the training estimate: one random time per time stratum, one random depth per depth stratum
                k = rng.integers(0, i0.size, 2000)
                jt = i0[k, None] + np.minimum(((np.arange(a.n_t) + rng.random((k.size, a.n_t))) / a.n_t * n).astype(int),
                                              n - 1)
                jx = np.minimum(((np.arange(a.n_xi) + rng.random((k.size, a.n_t, a.n_xi))) / a.n_xi * nD).astype(int),
                                nD - 1)
                Rs = R[jt[:, :, None], jx].mean(2)
                est = m.soil.theta_star * (z[jt] * D * Rs - f[jt]).mean(1) * (n * dtt) / (z[jt].mean(1) * D)
                r = np.sqrt(np.mean(delta ** 2)), np.sqrt(np.mean(dE ** 2)), np.sqrt(np.mean((est - delta[k]) ** 2))
                print(f"    {dur:7.3f}   {r[0] / sig:8.3f}   {r[1] / sig:8.3f}       {r[0] / max(r[1], 1e-30):8.2f}"
                      f"        {r[2] / sig:8.3f}", flush=True)


if __name__ == '__main__':
    main()
