#!/usr/bin/env python
"""zb error by window segment for PINN runs vs the classical referee and its local posterior sigma_zb(t).

Answers: where in the window is each run's zb error, and is it inside the information limit
(referee posterior sigma, from the exact-solver Jacobian)?

Usage: python scripts/error_by_day.py --referee output/classical/v3_noise005 runs/<a> runs/<b> ...
"""
import argparse
import os
import numpy as np

EDGES = [0, 5, 10, 15, 20, 22.5, 25, 27.5, 30.01]


def seg_table(t, e):
    return [float(np.sqrt(np.mean(e[(t >= a) & (t < b)] ** 2))) * 100 for a, b in zip(EDGES[:-1], EDGES[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--referee', required=True)
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    ref = np.load(os.path.join(a.referee, 'predictions.npz'))
    t = ref['t_days']
    hdr = ' '.join(f'{f"{x:g}-{y:g}":>8s}' for x, y in zip(EDGES[:-1], [min(v, 30) for v in EDGES[1:]]))
    print(f"{'zb RMSE [cm] by day':40s} {hdr}   at t=30")
    print(f"{'referee posterior sigma':40s} " + ' '.join(f'{v:8.2f}' for v in seg_table(t, ref['zb_sigma']))
          + f"   {100 * ref['zb_sigma'][-1]:6.2f}")
    print(f"{'referee error':40s} " + ' '.join(f'{v:8.2f}' for v in seg_table(t, ref['zb_pred'] - ref['zb_true']))
          + f"   {100 * (ref['zb_pred'][-1] - ref['zb_true'][-1]):6.2f}")
    for r in a.runs:
        f = os.path.join(r, 'predictions.npz')
        if not os.path.exists(f):
            print(f"{os.path.basename(r.rstrip('/')):40s} (missing)")
            continue
        p = np.load(f)
        e = p['zb_pred'] - p['zb_true']
        print(f"{os.path.basename(r.rstrip('/')):40s} " + ' '.join(f'{v:8.2f}' for v in seg_table(p['t_days'], e))
              + f"   {100 * e[-1]:6.2f}")


if __name__ == '__main__':
    main()
