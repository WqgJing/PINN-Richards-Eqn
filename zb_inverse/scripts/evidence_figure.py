#!/usr/bin/env python
"""One-page evidence that the inverse PINN works on the synthetic (sinusoidal) twin.

(a) loss terms vs step (long run)        (b) zb error vs step, all clean-data runs, with the CRLB scale
(c) zb(t): truth, initial guess, runs     (d) zb error vs time
(e) theta_5cm: observed vs PINN           (f) theta profiles PINN vs reference at deep / shallow / deep WT
Usage: python scripts/evidence_figure.py OUT.png main_run label=run [label=run ...]
"""
import csv
import json
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.train import Trainer  # noqa: E402
from zbinv.physics import theta_profile  # noqa: E402

C_TRUTH, C_PRED = '#2a78d6', '#eb6834'
RUN_C = ['#eb6834', '#1baf7a', '#eda100', '#c03bc4', '#6b6b6b']
C_LOSS = {'L_pde': '#2a78d6', 'L_flux': '#eb6834', 'L_data': '#1baf7a'}


def read_metrics(run):
    with open(os.path.join(run, 'metrics.csv')) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    out['stage'] = [r['stage'] for r in rows]
    return out


def main():
    out_png, main_run = sys.argv[1], sys.argv[2]
    runs = [a.split('=', 1) for a in sys.argv[3:]]
    ident = json.load(open('output/twin_synth_v3/identifiability.json'))['hourly_sigma0.005']
    fig, ax = plt.subplots(3, 2, figsize=(14, 13))

    # (a) loss terms, main run
    m = read_metrics(main_run)
    for k in ['L_pde', 'L_flux', 'L_data']:
        ok = np.isfinite(m[k])
        ax[0, 0].semilogy(m['step'][ok], m[k][ok], color=C_LOSS[k], lw=1.3, label=k.replace('L_', '') + ' (sigma-normalised)')
    st = m['stage']
    for name, ls in [('adam2', ':'), ('lbfgs', '--')]:
        if name in st:
            ax[0, 0].axvline(m['step'][st.index(name)], color='k', lw=0.8, ls=ls)
    ax[0, 0].set_title('(a) loss terms, long run (dotted: zb released + data on; dashed: L-BFGS)')
    ax[0, 0].set_xlabel('step'); ax[0, 0].legend(fontsize=8)

    # (b) zb error vs step for every run
    rows = []
    for k, (label, run) in enumerate(runs):
        mm = read_metrics(run)
        ax[0, 1].semilogy(mm['step'], mm['zb_rmse_inner'], color=RUN_C[k], lw=1.4, label=label)
        s = json.load(open(os.path.join(run, 'summary.json')))
        rows.append((label, s))
    ax[0, 1].axhline(ident['median_m'], color='k', lw=0.9, ls='--', label='CRLB median (noise 0.005)')
    ax[0, 1].axhline(ident['rms_m'], color='k', lw=0.9, ls=':', label='CRLB rms (noise 0.005)')
    ax[0, 1].set_title('(b) zb RMSE vs step (days 0-25)'); ax[0, 1].set_xlabel('step'); ax[0, 1].set_ylabel('m')
    ax[0, 1].legend(fontsize=8)

    # (c), (d) zb(t)
    for k, (label, run) in enumerate(runs):
        p = np.load(os.path.join(run, 'predictions.npz'))
        if k == 0:
            ax[1, 0].plot(p['t_days'], p['zb_true'], color=C_TRUTH, lw=2.6, label='truth')
            ax[1, 0].plot(p['t_days'], p['zb_init'], color='#9a9a9a', lw=1.0, ls='--', label='initial guess (VG inversion of theta_obs)')
            tmax = p['t_days'][-1]
        ax[1, 0].plot(p['t_days'], p['zb_pred'], color=RUN_C[k], lw=1.2, label=label)
        ax[1, 1].plot(p['t_days'], 100 * (p['zb_pred'] - p['zb_true']), color=RUN_C[k], lw=1.2, label=label)
    for a in (ax[1, 0], ax[1, 1]):
        a.axvline(tmax - 5, color='k', lw=0.8, ls=':')
        a.set_xlabel('days')
    ax[1, 0].invert_yaxis(); ax[1, 0].set_ylabel('zb [m]'); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title('(c) water-table depth (right of dotted line: last 5 d, weakly identifiable)')
    ax[1, 1].axhline(0, color='k', lw=0.8); ax[1, 1].set_ylabel('zb error [cm]')
    ax[1, 1].set_ylim(-20, 20); ax[1, 1].set_title('(d) zb error vs time (clipped at +-20 cm)')

    # (e) theta fit, main run
    p = np.load(os.path.join(main_run, 'predictions.npz'))
    ax[2, 0].plot(p['t_obs_days'], p['theta_obs'], color=C_TRUTH, lw=2.4, label='observed (twin, hourly)')
    ax[2, 0].plot(p['t_days'], p['theta_obs_pred'], color=C_PRED, lw=1.2, label='PINN, top 5 cm')
    ax[2, 0].set_xlabel('days'); ax[2, 0].set_ylabel('theta [-]'); ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_title('(e) surface moisture (data term)')

    # (f) profiles vs reference at deep / shallow / deep water table
    cfg = yaml.safe_load(open(os.path.join(main_run, 'config.yaml')))
    cfg['output']['dir'] = '/tmp/evidence_scratch'
    tr = Trainer(cfg)
    tr.model.load_state_dict(torch.load(os.path.join(main_run, 'model.pt'), map_location=tr.dev))
    ds = tr.ds
    for k, (day, ls) in enumerate([(5.0, '-'), (15.0, '--'), (25.0, ':')]):
        i = int(np.argmin(np.abs(ds['t_hourly'] - day)))
        zb_t = ds['zb_true'][i]
        xi = ds['xi_ref']
        prof = theta_profile(tr.model, torch.tensor(xi), torch.tensor(ds['t_hourly'][[i]]))[0].numpy()
        zb_p = float(tr.model.zb_m(torch.tensor([[ds['t_hourly'][i]]])))
        ax[2, 1].plot(ds['theta_ref'][i], xi * zb_t, color=C_TRUTH, lw=2.4, ls=ls,
                      label=f'reference day {day:.0f} (zb {zb_t:.2f} m)')
        ax[2, 1].plot(prof, xi * zb_p, color=C_PRED, lw=1.2, ls=ls, label=f'PINN day {day:.0f} (zb {zb_p:.2f} m)')
    ax[2, 1].set_xlabel('theta [-]'); ax[2, 1].set_ylabel('z [m]'); ax[2, 1].legend(fontsize=8)
    ax[2, 1].set_title('(f) whole profile vs reference solver (not used in training)')

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.suptitle('Inverse PINN on the synthetic twin: q = -1 + 2 sin(2 pi t / 6 d) mm/d, '
                 'zb = 0.6 + 0.25 sin(2 pi t / 20 d) m, clean hourly theta_5cm', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    print('run | zb RMSE all | zb RMSE days 0-25 | zb bias | theta RMSE | PDE rms | wall min')
    for label, s in rows:
        print(f"{label} | {s['zb_rmse']:.4f} | {s['zb_rmse_inner']:.4f} | {s['zb_bias']:+.4f} | "
              f"{s['theta_rmse_hourly']:.2e} | {s['pde_rms']:.4f} | {s['wall_s'] / 60:.1f}")
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()
