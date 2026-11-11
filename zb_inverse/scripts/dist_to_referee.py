#!/usr/bin/env python
"""SCORING ONLY: distance of each run's zb(t) from the classical referee's estimate for the same data, by window
segment, in units of the referee's local posterior sigma. The referee estimate is the exact least-squares optimum for
that noise realisation, so this measures convergence of a PINN iteration separately from the noise in the data
(error_by_day.py measures the error against the truth).

Usage: python scripts/dist_to_referee.py --referee output/classical/v3_noise005 runs/<a> runs/<b> ...
"""
import argparse
import os
import numpy as np

EDGES = [0, 5, 10, 15, 20, 22.5, 25, 27.5, 30.01]


def seg(t, v):
    return [float(np.sqrt(np.mean(v[(t >= a) & (t < b)] ** 2))) for a, b in zip(EDGES[:-1], EDGES[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--referee', required=True)
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    ref = np.load(os.path.join(a.referee, 'predictions.npz'))
    t = ref['t_days']
    sig = np.array(seg(t, ref['zb_sigma']))
    inner = t <= 25.0
    hdr = ' '.join(f'{f"{x:g}-{y:g}":>8s}' for x, y in zip(EDGES[:-1], [min(v, 30) for v in EDGES[1:]]))
    print(f"{'|zb - zb_referee| / sigma_ref by day':40s} {hdr}   days 0-25 [cm]  vs truth [cm]")
    print(f"{'referee sigma [cm]':40s} " + ' '.join(f'{100 * v:8.2f}' for v in sig)
          + f"   {'':>14s}  {100 * np.sqrt(np.mean((ref['zb_pred'] - ref['zb_true'])[inner] ** 2)):8.2f}")
    for r in a.runs:
        p = np.load(os.path.join(r, 'predictions.npz'))
        if p['t_days'].shape != t.shape or not np.allclose(p['t_days'], t):
            print(f"{os.path.basename(r.rstrip('/')):40s} time grid differs from the referee's: skipped")
            continue
        d = p['zb_pred'] - ref['zb_pred']
        print(f"{os.path.basename(r.rstrip('/')):40s} " + ' '.join(f'{v:8.2f}' for v in np.array(seg(t, d)) / sig)
              + f"   {100 * np.sqrt(np.mean(d[inner] ** 2)):14.2f}"
              + f"  {100 * np.sqrt(np.mean((p['zb_pred'] - p['zb_true'])[inner] ** 2)):8.2f}")


if __name__ == '__main__':
    main()
