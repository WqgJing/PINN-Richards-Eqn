#!/usr/bin/env python
"""Figure of one inversion on a twin: forcing, top-5 cm soil moisture (given / PINN) and water table (prescribed / PINN).

Usage: python scripts/twin_result_figure.py --twin output/twin_gldas_fl/twin.npz --inv runs/gs_clean_inv0 \
           [--out output/compare/gs_clean_inv0.png]
Prints the water-table errors of the best start (scoring only: the prescribed path is not used by the inversion).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv import plots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--twin', required=True)
    ap.add_argument('--inv', required=True, help='output directory of scripts/param_invert.py')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tw = np.load(a.twin, allow_pickle=True)
    pred = np.load(os.path.join(a.inv, 'predictions.npz'))
    starts = np.load(os.path.join(a.inv, 'starts.npz'), allow_pickle=True)
    out = a.out or os.path.join(a.inv, 'twin_result.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plots.plot_twin_result(tw, pred, starts['zb'], out)
    t, err = pred['t_days'], pred['zb_pred'] - pred['zb_true']
    edges = np.arange(0.0, t[-1] + 1e-9, 5.0)
    seg = [float(np.sqrt(np.mean(err[(t >= lo) & (t <= hi)] ** 2))) for lo, hi in zip(edges[:-1], edges[1:])]
    d_th = pred['theta_obs_pred'] - np.interp(t, pred['t_obs_days'], pred['theta_obs'])   # model at the estimate - data
    seg_th = [float(np.sqrt(np.mean(d_th[(t >= lo) & (t <= hi)] ** 2))) for lo, hi in zip(edges[:-1], edges[1:])]
    print(f"[result] {a.inv}: zb RMSE {100 * np.sqrt(np.mean(err ** 2)):.2f} cm (days 0-25 "
          f"{100 * np.sqrt(np.mean(err[t <= 25.0] ** 2)):.2f} cm), max {100 * np.abs(err).max():.2f} cm, by 5-day "
          f"segment [cm] {' '.join(f'{100 * s:.2f}' for s in seg)}; theta_5cm fit RMSE "
          f"{np.sqrt(np.mean(d_th ** 2)):.2e}, by segment {' '.join(f'{s:.1e}' for s in seg_th)}; saved {out}", flush=True)


if __name__ == '__main__':
    main()
