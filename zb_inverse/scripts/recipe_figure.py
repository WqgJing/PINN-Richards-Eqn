#!/usr/bin/env python
"""Figure + table: the inverse PINN recipe vs the old baseline vs the classical referee (synthetic v3 twin).

Rows: noise 0.005 / noise 0.01 / deep start (1.0 m) at noise 0.005.
Columns: zb(t) | zb error vs time | zb error (days 0-25) during training.
Line plots only. Writes output/compare/recipe_vs_baseline.{png,csv}.

Usage: python scripts/recipe_figure.py
"""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RUNS, CLS, OUT = os.path.join(ROOT, 'runs'), os.path.join(ROOT, 'output', 'classical'), os.path.join(ROOT, 'output', 'compare')
COL = {'truth': '#2a78d6', 'init': '#9a9a9a', 'baseline': '#eb6834', 'recipe': '#1baf7a', 'referee': '#c03bc4'}
CASES = [
    ('noise 0.005', 'v3_e3_fixed_noise005', 'recipe_noise005', 'v3_noise005'),
    ('noise 0.01', 'v3_e3_fixed_noise010', 'recipe_noise010', 'v3_noise010'),
    ('deep start 1.0 m, noise 0.005', 'v10_deep_noise005_lrzb3', 'recipe_deep_noise005', 'v3_noise005'),
]
EXTRA = [('clean', 'v3_e2_fixed', 'recipe_clean', 'v3_clean'),
         ('noise 0.005, seed 1', 'v8_autofix_noise005_seed1', 'recipe_noise005_seed1', 'v3_noise005'),
         ('deep start 1.0 m, noise 0.01', None, 'recipe_deep_noise010', 'v3_noise010')]


def load(path):
    p = dict(np.load(os.path.join(path, 'predictions.npz')))
    s = json.load(open(os.path.join(path, 'summary.json')))
    return p, s


def history(path):
    f = os.path.join(path, 'metrics.csv')
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    return (np.array([float(r['step']) for r in rows]), np.array([float(r['zb_rmse_inner']) for r in rows]))


def main():
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15, 3.6 * len(CASES)))
    table = []
    for i, (label, base, rec, ref) in enumerate(CASES + EXTRA):
        runs = {'baseline': os.path.join(RUNS, base) if base else None, 'recipe': os.path.join(RUNS, rec),
                'referee': os.path.join(CLS, ref)}
        res = {k: load(v) for k, v in runs.items() if v and os.path.exists(os.path.join(v, 'summary.json'))}
        for k, (p, s) in res.items():
            table.append([label, k, s['zb_rmse_inner'], s['zb_rmse'], s.get('data_rmse', s.get('theta_rmse_vs_obs')),
                          s.get('pde_rms', float('nan')), s.get('posterior_sigma_zb_rms_inner_m', float('nan'))])
        if i >= len(CASES):
            continue
        a0, a1, a2 = axes[i]
        p0 = res['recipe'][0]
        a0.plot(p0['t_days'], p0['zb_true'], color=COL['truth'], lw=2.4, label='truth')
        a0.plot(p0['t_days'], p0['zb_init'], color=COL['init'], lw=1.0, ls='--', label='initial guess')
        for k in ('baseline', 'referee', 'recipe'):
            if k not in res:
                continue
            p, s = res[k]
            name = {'baseline': 'old baseline PINN', 'recipe': 'PINN recipe', 'referee': 'classical referee'}[k]
            lab = f"{name} ({100 * s['zb_rmse_inner']:.2f} cm)"
            ls = '--' if k == 'referee' else '-'
            a0.plot(p['t_days'], p['zb_pred'], color=COL[k], lw=1.6, ls=ls, label=lab)
            a1.plot(p['t_days'], 100 * (p['zb_pred'] - p['zb_true']), color=COL[k], lw=1.4, ls=ls, label=name)
            if k != 'referee':
                h = history(runs[k])
                if h is not None:
                    a2.semilogy(h[0], 100 * h[1], color=COL[k], lw=1.5, label=name)
        if 'referee' in res:
            a2.axhline(100 * res['referee'][1]['zb_rmse_inner'], color=COL['referee'], lw=1.2, ls='--',
                       label='classical referee')
        a0.set_ylim(1.3, 0.2)
        a0.set_ylabel('zb [m]'); a0.set_title(label, fontsize=10, loc='left')
        a1.axhline(0, color='k', lw=0.7); a1.axvline(25, color='#9a9a9a', lw=0.8, ls=':')
        a1.set_ylim(-15, 15); a1.set_ylabel('zb error [cm]')
        a2.set_ylabel('zb RMSE days 0-25 [cm]'); a2.set_ylim(0.3, 100)
        for a in (a0, a1):
            a.set_xlabel('days from window start')
        a2.set_xlabel('training step')
        for a in (a0, a1, a2):
            a.grid(True, alpha=0.3)
        a0.legend(fontsize=7.5, loc='lower left'); a2.legend(fontsize=7.5)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, 'recipe_vs_baseline.png')
    fig.savefig(png, dpi=130)
    hdr = ['case', 'method', 'zb_rmse_days0_25_m', 'zb_rmse_all_m', 'data_rmse', 'pde_rms', 'posterior_sigma_inner_m']
    with open(os.path.join(OUT, 'recipe_vs_baseline.csv'), 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in table:
            f.write(','.join([r[0], r[1]] + [f'{v:.6g}' for v in r[2:]]) + '\n')
    for r in table:
        print(f"{r[0]:32s} {r[1]:9s} zb days0-25 {100 * r[2]:6.2f} cm  all {100 * r[3]:6.2f} cm  data {r[4]:.4f}")
    print(f'saved {png}')


if __name__ == '__main__':
    main()
