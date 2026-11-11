#!/usr/bin/env python
"""Compare runs: zb(t) against the truth (line plot) and a summary table (CSV + stdout).

Usage: python scripts/compare_runs.py OUT.png label=runs/<run> [label=runs/<run> ...]
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = ['#eb6834', '#1baf7a', '#eda100', '#c03bc4', '#6b6b6b', '#8a5a2b']
KEYS = ['zb_rmse', 'zb_rmse_inner', 'zb_bias', 'theta_rmse_hourly', 'data_rmse', 'flux_rms_mmd', 'pde_rms',
        'prof_rmse', 'wall_s']


def main():
    out = sys.argv[1]
    runs = [a.split('=', 1) for a in sys.argv[2:]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1]})
    rows = []
    for k, (label, path) in enumerate(runs):
        p = np.load(os.path.join(path, 'predictions.npz'))
        s = json.load(open(os.path.join(path, 'summary.json')))
        if k == 0:
            ax[0].plot(p['t_days'], p['zb_true'], color='#2a78d6', lw=2.2, label='truth')
            ax[0].plot(p['t_days'], p['zb_init'], color='#9a9a9a', lw=1.0, ls='--', label='initial guess')
        ax[0].plot(p['t_days'], p['zb_pred'], color=C[k % len(C)], lw=1.5, label=label)
        ax[1].plot(p['t_days'], p['zb_pred'] - p['zb_true'], color=C[k % len(C)], lw=1.5, label=label)
        rows.append([label] + [s.get(key, float('nan')) for key in KEYS])
    ax[0].invert_yaxis(); ax[0].set_xlabel('days from window start'); ax[0].set_ylabel('zb [m]')
    ax[1].axhline(0, color='k', lw=0.8); ax[1].set_xlabel('days from window start'); ax[1].set_ylabel('zb error [m]')
    for a in ax:
        a.grid(True, alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    hdr = ['run'] + KEYS
    with open(os.path.splitext(out)[0] + '.csv', 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join([r[0]] + [f'{v:.6g}' for v in r[1:]]) + '\n')
    print(' | '.join(hdr))
    for r in rows:
        print(' | '.join([r[0]] + [f'{v:.4g}' for v in r[1:]]))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
