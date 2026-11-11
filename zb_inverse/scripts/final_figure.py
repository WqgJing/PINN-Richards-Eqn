#!/usr/bin/env python
"""Final comparison figures (line plots only): operator PINN vs joint PINN vs solver referee, one file per data case.

Top: zb(t) of the truth, the referee, the joint PINN and the operator PINN (every LM start overlaid, thin) with its
     +-1 posterior sigma (dashed).  Bottom: zb error [cm] of each, with the referee's +-1 posterior sigma (dashed).
Usage: python scripts/final_figure.py --out output/compare/final \
           --case "n005|output/classical/v3_noise005|runs/box0_n005|runs/v9_long_noise005" ...
       writes output/compare/final_<name>.png per case (joint run may be 'none')
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_REF, C_JOINT, C_PARAM, C_TRUE = '#1b1b1b', '#eb6834', '#2a78d6', '#7a7a7a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output prefix; one file <prefix>_<name>.png per case')
    ap.add_argument('--case', action='append', required=True, help='"name|referee_dir|param_inv_dir|joint_dir|none"')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for spec in a.case:
        name, ref_dir, par_dir, joint_dir = spec.split('|')
        ref = np.load(os.path.join(ref_dir, 'predictions.npz'))
        par = np.load(os.path.join(par_dir, 'predictions.npz'))
        st = np.load(os.path.join(par_dir, 'starts.npz'))
        t, zt = par['t_days'], par['zb_true']
        fig, (top, bot) = plt.subplots(2, 1, figsize=(5.6, 7.2), sharex=True)
        top.plot(t, 100 * zt, color=C_TRUE, lw=3.0, alpha=0.6, label='truth')
        top.plot(ref['t_days'], 100 * ref['zb_pred'], color=C_REF, lw=1.2, label='solver referee')
        bot.plot(ref['t_days'], 100 * (ref['zb_pred'] - ref['zb_true']), color=C_REF, lw=1.2, label='solver referee')
        if joint_dir != 'none':
            jp = np.load(os.path.join(joint_dir, 'predictions.npz'))
            top.plot(jp['t_days'], 100 * jp['zb_pred'], color=C_JOINT, lw=1.2, label='joint PINN')
            bot.plot(jp['t_days'], 100 * (jp['zb_pred'] - jp['zb_true']), color=C_JOINT, lw=1.2, label='joint PINN')
        for k, z in enumerate(st['zb']):
            lab = f"operator PINN ({len(st['zb'])} starts)" if k == 0 else None
            top.plot(t, 100 * z, color=C_PARAM, lw=0.8, alpha=0.7, label=lab)
            bot.plot(t, 100 * (z - zt), color=C_PARAM, lw=0.8, alpha=0.7, label=lab)
        for s in (1, -1):
            top.plot(t, 100 * (par['zb_pred'] + s * par['zb_sigma']), color=C_PARAM, lw=0.8, ls='--',
                     label='operator PINN $\\pm 1\\sigma$' if s == 1 else None)
            bot.plot(ref['t_days'], 100 * s * ref['zb_sigma'], color='#9a9a9a', lw=1.0, ls='--',
                     label='referee $\\pm 1\\sigma$' if s == 1 else None)
        top.set_ylabel('water-table depth (cm)')
        top.invert_yaxis()
        bot.set_ylabel('water-table depth error (cm)')
        bot.set_ylim(-8, 8)
        bot.set_xlabel('time (days)')
        top.legend(fontsize=7, loc='upper left')
        bot.legend(fontsize=7, loc='lower left')
        for axk in (top, bot):
            axk.grid(True, alpha=0.3)
        fig.tight_layout()
        out = f'{a.out}_{name}.png'
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print('saved', out)


if __name__ == '__main__':
    main()
