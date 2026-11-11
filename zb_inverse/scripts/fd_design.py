#!/usr/bin/env python
"""Finite-difference design of zb paths around one estimate, for the slope correction of param_invert.py --fwd-fd.

The 2 n_coef paths are c0 +- scale * sd_i * u_i, with (sd_i^2, u_i) the eigenpairs of the posterior covariance
cov_post stored in <center> (a local_center.npz written by scripts/param_invert.py). In posterior units every pair is
the same move, so every pair changes the observed theta_5cm by the same whitened amount (2 x scale noise sigmas),
whether the direction is strongly or weakly identified. scale = 3 is the radius of the local refinement cloud
(param_invert --inflate), where the operator members were trained.

Each file p<i>.npz / m<i>.npz has the keys of a local_center.npz (center, cov), so a forward PINN for that one path is
trained with sampling.mode=point, sampling.local_file=<file> (hpc/fd_set.sh). No solver, no data, no truth.

Usage: python scripts/fd_design.py --center runs/dc_n005_it2/local_center.npz --out runs/fd_n005 [--scale 3.0]
       [--config configs/p1_param_synth_v3.yaml]   also print the range of zb(t) along every design path
       [--dry-run]                                  print the table only, write nothing
"""
import argparse
import os
import sys
import numpy as np


def design(center, cov_post, scale):
    """Displacements (n_coef, n_coef): column i = scale * sd_i * u_i (largest posterior sd first)."""
    w, U = np.linalg.eigh(0.5 * (cov_post + cov_post.T))
    order = np.argsort(w)[::-1]
    w, U = np.maximum(w[order], 0.0), U[:, order]
    return float(scale) * U * np.sqrt(w)[None, :]


def zb_paths(center, D, config):
    """zb(t) [m], hourly, of the paths center + D[:, i] and center - D[:, i]: (t_days, zb_plus (n, T), zb_minus (n, T)).
    Uses only the spline definition of the training config (window length, knots, zb_min)."""
    import yaml
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from zbinv.train import load_dataset
    from zbinv.spline import CubicBSpline1D, ZbSplineNumpy
    with open(config) as f:
        cfg = yaml.safe_load(f)
    ds = load_dataset(cfg)
    sp = CubicBSpline1D(0.0, ds['n_days'], cfg.get('zb_spline', {}).get('knot_days',
                                                                        ds['twin_cfg']['zb_spline']['knot_days']))
    t = np.linspace(0.0, ds['n_days'], int(round(24 * ds['n_days'])) + 1)
    zb = lambda c: ZbSplineNumpy(sp, c, ds['zb_min']).zb(t)
    return (t, np.stack([zb(center + D[:, i]) for i in range(D.shape[1])]),
            np.stack([zb(center - D[:, i]) for i in range(D.shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--scale', type=float, default=3.0, help='half-width of the design in posterior sd')
    ap.add_argument('--config', default=None, help='training config: also print the zb range of every design path')
    ap.add_argument('--dry-run', action='store_true', help='print the table only, write nothing')
    a = ap.parse_args()
    z = np.load(a.center)
    c0, cov = z['center'], z['cov']
    D = design(c0, z['cov_post'], a.scale)
    if not a.dry_run:
        if a.out is None:
            raise SystemExit('--out is needed unless --dry-run')
        os.makedirs(a.out, exist_ok=True)
        for i in range(D.shape[1]):
            for name, s in (('p', 1.0), ('m', -1.0)):
                np.savez(os.path.join(a.out, f'{name}{i}.npz'), center=c0 + s * D[:, i], cov=cov)
        np.savez(os.path.join(a.out, 'design.npz'), center=c0, displacements=D, scale=a.scale)
    print(f"design of {2 * D.shape[1]} paths around {a.center}, +- {a.scale:g} posterior sd")
    print('direction  |dc|      largest coefficient move')
    for i in range(D.shape[1]):
        k = int(np.argmax(np.abs(D[:, i])))
        print(f'{i:9d}  {np.linalg.norm(D[:, i]):.4f}    c[{k}] {D[k, i]:+.4f}')
    if a.config:
        t, zp, zm = zb_paths(c0, D, a.config)
        z0 = zb_paths(c0, 0.0 * D[:, :1], a.config)[1][0]
        print(f'centre path: zb {z0.min():.3f}-{z0.max():.3f} m; coefficients ' + ' '.join(f'{v:+.3f}' for v in c0))
        print('direction  zb range [m] of the + path, the - path   max |zb - centre| [m] (+, -)   day of that max (+, -)')
        for i in range(D.shape[1]):
            dp, dm = np.abs(zp[i] - z0), np.abs(zm[i] - z0)
            print(f'{i:9d}  {zp[i].min():.3f}-{zp[i].max():.3f}  {zm[i].min():.3f}-{zm[i].max():.3f}   '
                  f'{dp.max():.4f} {dm.max():.4f}   {t[dp.argmax()]:.1f} {t[dm.argmax()]:.1f}')
        print('coefficient displacements by direction (columns = directions):')
        for k in range(D.shape[0]):
            print(f'  c[{k}] ' + ' '.join(f'{v:+8.4f}' for v in D[k]))


if __name__ == '__main__':
    main()
