#!/usr/bin/env python
"""Figure of a PINN inversion on site data against the observations only: soil moisture at the sensor (observed, PINN
at its estimate) and the water table (observed, PINN estimate). No solver line and no hydrostatic reading; the scores
and the full figure are those of scripts/score_real.py.

Usage: python -u scripts/plot_real_pinn.py --data output/real_ona_nov22_caloct22 --pinn runs/<tag> \
           --out output/real_ona_nov22_caloct22/result_<tag>_pinn.png
"""
import argparse
import os
import numpy as np

from common import load_with_overrides  # noqa: F401  (puts src on the path)
from zbinv import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory of a make_real.py data set')
    ap.add_argument('--pinn', required=True, help='directory of a param_invert.py result')
    ap.add_argument('--out', required=True, help='path of the figure (.png)')
    args = ap.parse_args()
    d = dict(np.load(os.path.join(args.data, 'real.npz'), allow_pickle=False))
    pred = dict(np.load(os.path.join(args.pinn, 'predictions.npz'), allow_pickle=False))
    t = d['t_days']
    if not np.allclose(pred['t_days'], t) or not np.allclose(pred['zb_true'], d['zb_true']):
        raise SystemExit('the PINN result belongs to another data set')
    theta_pinn = np.interp(t, pred['t_obs_days'], pred['theta_obs_pred'])
    plots.plot_real_result(t, d['theta_obs_hourly'], [('PINN', theta_pinn)], d['zb_true'], [('PINN', pred['zb_pred'])],
                           args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
