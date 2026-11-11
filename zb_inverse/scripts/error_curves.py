#!/usr/bin/env python
"""zb error vs time for several runs against the referee error and its +-1 posterior sigma (dashed lines).
Line plots only. Usage: python scripts/error_curves.py --referee output/classical/v3_noise005 --out <png> runs/a runs/b ...
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#c03bc4', '#e0a100', '#5c5c5c', '#00a3c4', '#a3542b']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--referee', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    ref = np.load(os.path.join(a.referee, 'predictions.npz'))
    t = ref['t_days']
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for k, axk in enumerate(ax):
        axk.plot(t, 100 * ref['zb_sigma'], color='#9a9a9a', lw=1.0, ls='--', label='referee posterior sigma')
        axk.plot(t, -100 * ref['zb_sigma'], color='#9a9a9a', lw=1.0, ls='--')
        axk.plot(t, 100 * (ref['zb_pred'] - ref['zb_true']), color='k', lw=1.8, label='classical referee')
        axk.axhline(0, color='k', lw=0.5)
        axk.grid(True, alpha=0.3)
        axk.set_ylabel('zb error [cm]')
    for i, r in enumerate(a.runs):
        p = np.load(os.path.join(r, 'predictions.npz'))
        ax[0].plot(p['t_days'], 100 * (p['zb_pred'] - p['zb_true']), color=COLORS[i % len(COLORS)], lw=1.3,
                   label=os.path.basename(r.rstrip('/')))
        ax[1].plot(p['t_days'], 100 * (p['zb_pred'] - p['zb_true']), color=COLORS[i % len(COLORS)], lw=1.3)
    ax[0].set_ylim(-25, 25)
    ax[1].set_xlim(0, 12); ax[1].set_ylim(-5, 5)
    ax[1].set_xlabel('days from window start')
    ax[0].set_title(a.title, loc='left', fontsize=10)
    ax[1].set_title('zoom: days 0-12', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5, ncol=2, loc='lower left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print('saved', a.out)


if __name__ == '__main__':
    main()
