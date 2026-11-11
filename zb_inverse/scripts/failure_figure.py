#!/usr/bin/env python
"""Where the inverse PINN breaks (synthetic twin unless noted). Line plots only.

(a) noisy theta: zb(t)              (b) noisy theta: what the PINN fits (zoom)
(c) deep initial guess: zb(t)       (d) weighting / soil-model choices: zb error vs time
(e) real US-PAS forcing (d80, head-limited BC), FORWARD with true zb: theta_5cm
(f) physics limit: zb change equivalent to a 0.005 theta error vs WT depth (steady states)
Usage: python scripts/failure_figure.py OUT.png <depth_sensitivity log>
"""
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_TRUTH, C_GOOD = '#2a78d6', '#1baf7a'
C = ['#eb6834', '#eda100', '#c03bc4', '#6b6b6b']
R = 'runs'


def P(run):
    return np.load(os.path.join(R, run, 'predictions.npz'))


def S(run):
    return json.load(open(os.path.join(R, run, 'summary.json')))


def main():
    out, sens_log = sys.argv[1], sys.argv[2]
    fig, ax = plt.subplots(3, 2, figsize=(14, 13))

    # (a) noise
    good = P('v3_e2_fixed_long')
    ax[0, 0].plot(good['t_days'], good['zb_true'], color=C_TRUTH, lw=2.6, label='truth')
    ax[0, 0].plot(good['t_days'], good['zb_pred'], color=C_GOOD, lw=1.3,
                  label=f"clean data ({100 * S('v3_e2_fixed_long')['zb_rmse_inner']:.1f} cm)")
    for k, (run, lab) in enumerate([('v3_e3_fixed_noise005', 'noise 0.005'), ('v3_e3_fixed_noise010', 'noise 0.01'),
                                    ('v3_e3_discrepancy_noise005', 'noise 0.005, discrepancy weights')]):
        p = P(run)
        ax[0, 0].plot(p['t_days'], p['zb_pred'], color=C[k], lw=1.3,
                      label=f"{lab} ({100 * S(run)['zb_rmse_inner']:.1f} cm)")
    ax[0, 0].invert_yaxis(); ax[0, 0].set_ylabel('zb [m]'); ax[0, 0].set_xlabel('days')
    ax[0, 0].set_title('(a) noisy surface moisture: zb (RMSE days 0-25)'); ax[0, 0].legend(fontsize=8)

    # (b) what is fitted under noise 0.01 (zoom)
    tw = np.load('output/twin_synth_v3/twin.npz')
    p = P('v3_e3_fixed_noise010')
    sel = (p['t_obs_days'] >= 10) & (p['t_obs_days'] <= 14)
    selh = (p['t_days'] >= 10) & (p['t_days'] <= 14)
    ax[0, 1].plot(p['t_obs_days'][sel], p['theta_obs'][sel], color='#9a9a9a', lw=0.8, label='observed (truth + noise 0.01)')
    ax[0, 1].plot(tw['t_days'][selh], tw['theta_obs_hourly'][selh], color=C_TRUTH, lw=2.4, label='true theta_5cm')
    ax[0, 1].plot(p['t_days'][selh], p['theta_obs_pred'][selh], color=C[1], lw=1.4, label='PINN')
    s = S('v3_e3_fixed_noise010')
    ax[0, 1].set_title(f"(b) noise 0.01, days 10-14: PINN vs truth RMSE {s['theta_rmse_hourly']:.4f}, "
                       f"PDE rms {s['pde_rms']:.2f} (clean 0.004)", fontsize=10)
    ax[0, 1].set_xlabel('days'); ax[0, 1].set_ylabel('theta [-]'); ax[0, 1].legend(fontsize=8)

    # (c) deep initial guess
    p = P('v3_e2_fixed_init_deep')
    ax[1, 0].semilogy(p['t_days'], p['zb_true'], color=C_TRUTH, lw=2.6, label='truth')
    ax[1, 0].semilogy(p['t_days'], p['zb_init'], color='#9a9a9a', lw=1.0, ls='--', label='initial guess (constant 1.0 m)')
    ax[1, 0].semilogy(p['t_days'], p['zb_pred'], color=C[0], lw=1.4, label=f"PINN (RMSE {S('v3_e2_fixed_init_deep')['zb_rmse']:.1f} m)")
    q = P('v3_e2_fixed_init_shallow')
    ax[1, 0].semilogy(q['t_days'], q['zb_pred'], color=C_GOOD, lw=1.2, label='PINN from 0.4 m guess (for contrast)')
    ax[1, 0].invert_yaxis(); ax[1, 0].set_ylabel('zb [m] (log)'); ax[1, 0].set_xlabel('days')
    ax[1, 0].set_title('(c) deep initial guess: zb runs away'); ax[1, 0].legend(fontsize=8)

    # (d) weighting and soil-model choices (clean data)
    for k, (run, lab) in enumerate([('e2_inverse_synth_v100', 'plain VG (no air entry), gradient-norm weights'),
                                    ('v3_e2_gradnorm', 'air entry, gradient-norm weights'),
                                    ('v3_e2_discrepancy', 'air entry, discrepancy weights')]):
        p = P(run)
        ax[1, 1].plot(p['t_days'], 100 * (p['zb_pred'] - p['zb_true']), color=C[k], lw=1.3,
                      label=f"{lab} ({100 * S(run)['zb_rmse_inner']:.1f} cm)")
    ax[1, 1].plot(good['t_days'], 100 * (good['zb_pred'] - good['zb_true']), color=C_GOOD, lw=1.3,
                  label=f"air entry, fixed weights ({100 * S('v3_e2_fixed_long')['zb_rmse_inner']:.1f} cm)")
    ax[1, 1].axhline(0, color='k', lw=0.8); ax[1, 1].axvline(25, color='k', lw=0.8, ls=':')
    ax[1, 1].set_ylim(-25, 25); ax[1, 1].set_xlabel('days'); ax[1, 1].set_ylabel('zb error [cm]')
    ax[1, 1].set_title('(d) clean data, other design choices: zb error'); ax[1, 1].legend(fontsize=8)

    # (e) real forcing, forward (zb = truth)
    tr = np.load('output/twin_pas_d80_hlim/twin.npz')
    p = P('e1_forward_d80_hlim_log')
    s = S('e1_forward_d80_hlim_log')
    ax[2, 0].plot(tr['t_days'], tr['theta_obs_hourly'], color=C_TRUTH, lw=2.4, label='reference solver')
    ax[2, 0].plot(p['t_days'], p['theta_obs_pred'], color=C[0], lw=1.3, label=f"PINN, true zb given (RMSE {s['theta_rmse_hourly']:.3f})")
    a2 = ax[2, 0].twinx()
    a2.plot(tr['t_days'], tr['q_act_hourly'] * 86400e3, color='#9a9a9a', lw=0.7)
    a2.set_ylabel('surface flux [mm/d] (grey; + = evaporation)', color='#6b6b6b')
    ax[2, 0].set_xlabel('days from 2021-06-20'); ax[2, 0].set_ylabel('theta_5cm [-]')
    ax[2, 0].set_title('(e) real US-PAS forcing: even the FORWARD problem misses'); ax[2, 0].legend(fontsize=8, loc='upper left')

    # (f) physics limit from steady states
    txt = open(sens_log).read()
    blocks = re.split(r'\nq = ', txt)[1:]
    for k, b in enumerate(blocks):
        qv = float(b.split(' mm/d')[0])
        rows = re.findall(r'zb\s+([\d.]+) m\s+theta5\s+(\S+)\s+dtheta/dzb\s+(\S+) /m\s+-> 0.005 error =\s+(\S+) m', b)
        z = np.array([float(r[0]) for r in rows]); e = np.array([float(r[3]) for r in rows])
        ok = np.isfinite(e)
        lab = 'no flux (hydrostatic)' if qv == 0 else (f'infiltration {-qv:.0f} mm/d' if qv < 0 else
                                                    f'evaporation {qv:.1f} mm/d (no steady state beyond 1 m)')
        col = [C_TRUTH, C[0], C[1], C[2], C[3]][k % 5]
        ax[2, 1].semilogy(z[ok], e[ok], color=col, lw=1.5, marker='o', ms=3, label=lab)
    ax[2, 1].axhline(0.05, color='k', lw=0.8, ls=':')
    ax[2, 1].text(4.0, 0.055, '5 cm', fontsize=8)
    ax[2, 1].set_ylim(5e-3, 1e3)
    ax[2, 1].set_xlabel('water-table depth zb [m]'); ax[2, 1].set_ylabel('zb change equal to a 0.005 theta error [m]')
    ax[2, 1].set_title('(f) physics limit: how much the surface still feels the WT (steady states)', fontsize=10)
    ax[2, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.suptitle('Where the inverse PINN breaks (synthetic twin unless noted; US-PAS soil)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
