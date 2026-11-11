#!/usr/bin/env python
"""Score a water-table estimate from site data against the OBSERVED water table (scoring only).

Compared on the hourly record of the well: the PINN estimate (scripts/param_invert.py, every start), the FEM referee
(scripts/fem_referee.py, the start with the lowest cost) and the hydrostatic reading of the sensor (zb = sensor depth +
|h(theta)|, no flow, hour by hour). Scores over the whole window and over the hours in which the sensor is below
saturation by more than 3 sigma_theta: while it reads saturated it only bounds the water table from below.

Usage: python -u scripts/score_real.py --data output/real_ona_nov22_cal21 --pinn runs/<tag>_inv2 \
           --referee output/referee/<name>_s0 output/referee/<name>_s1 --out output/real_ona_nov22_cal21/result_<tag>
       --sigma-hind <m>: expected error of the soil transfer (hindcast); reported with the posterior of the inversion
"""
import argparse
import json
import os
import numpy as np
import yaml

from common import load_with_overrides  # noqa: F401  (puts src on the path)
from zbinv import plots
from zbinv.data import obs_layer
from zbinv.soil import SoilVG


def score(est, obs):
    d = np.asarray(est) - np.asarray(obs)
    return {'rmse_m': float(np.sqrt(np.mean(d ** 2))), 'bias_m': float(d.mean()), 'maxabs_m': float(np.abs(d).max()),
            'r': float(np.corrcoef(est, obs)[0, 1]) if np.std(est) > 0 and d.size > 2 else None, 'hours': int(d.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory of a make_real.py data set')
    ap.add_argument('--pinn', required=True, help='directory of a param_invert.py result')
    ap.add_argument('--referee', nargs='*', default=[], help='directories of fem_referee.py results (one per start)')
    ap.add_argument('--sigma', type=float, default=0.005, help='sigma_theta of the inversion')
    ap.add_argument('--sigma-hind', type=float, default=None,
                    help='expected error of the soil transfer (m), from the hindcast of tasks/todo.md finding 73 u: '
                         'adds the uncertainty block and the limits of the estimate to the figure')
    ap.add_argument('--out', required=True, help='path stem of the outputs (.json, .png)')
    args = ap.parse_args()
    with open(os.path.join(args.data, 'twin_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    d = dict(np.load(os.path.join(args.data, 'real.npz'), allow_pickle=False))
    soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
    t, zb_obs, th_obs = d['t_days'], d['zb_true'], d['theta_obs_hourly']
    pred = dict(np.load(os.path.join(args.pinn, 'predictions.npz'), allow_pickle=False))
    starts = dict(np.load(os.path.join(args.pinn, 'starts.npz'), allow_pickle=False))
    if not np.allclose(pred['t_days'], t) or not np.allclose(pred['zb_true'], zb_obs):
        raise SystemExit('the PINN result belongs to another data set')
    top, bottom = obs_layer(cfg['obs']['depth_m'])
    zb_hyd = (top + bottom) / 2.0 - soil.h_from_theta(th_obs)
    dry = th_obs < soil.theta_s - 3.0 * args.sigma                    # the sensor has left saturation
    k_fem = int(np.where(d['fem_water_table'] == 'spline')[0][0])     # FEM over the spline of the observed water table
    res = {
        'data': args.data, 'pinn': args.pinn, 'window': str(d['start_date']), 'n_days': float(d['n_days']),
        'soil': {k: float(cfg['soil'][k]) for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')},
        'hours': int(t.size), 'hours_sensor_below_saturation': int(dry.sum()),
        'zb_observed_range_m': [float(zb_obs.min()), float(zb_obs.max())],
        'spline_of_observed_vs_record': score(d['zb_spline'], zb_obs),
        'pinn': {'all': score(pred['zb_pred'], zb_obs), 'below_saturation': score(pred['zb_pred'][dry], zb_obs[dry]),
                 'start_spread_max_m': float(np.abs(starts['zb'] - pred['zb_pred'][None]).max()),
                 'sigma_median_m': float(np.median(pred['zb_sigma'])),
                 'theta_rmse_vs_obs': float(np.sqrt(np.mean((pred['theta_obs_pred'] - pred['theta_obs']) ** 2)))},
        'hydrostatic_reading': {'all': score(zb_hyd, zb_obs), 'below_saturation': score(zb_hyd[dry], zb_obs[dry])},
        'fem_with_observed_wt_theta_rmse_vs_obs': float(np.sqrt(np.mean((d['theta_fem'][k_fem] - th_obs) ** 2))),
    }
    zb_runs = [(f"PINN estimate ({starts['zb'].shape[0]} starts)", starts['zb'])]
    th_runs = [('PINN at its estimate', np.interp(t, pred['t_obs_days'], pred['theta_obs_pred']))]
    if args.referee:
        refs = []
        for p in args.referee:
            with open(os.path.join(p, 'summary.json')) as f:
                refs.append((json.load(f), dict(np.load(os.path.join(p, 'predictions.npz'), allow_pickle=False)), p))
        s, r, p = min(refs, key=lambda x: x[0]['cost'])
        res['referee'] = {'dir': p, 'all': score(r['zb_pred'], zb_obs),
                          'below_saturation': score(r['zb_pred'][dry], zb_obs[dry]), 'cost': s['cost'],
                          'costs_by_start': {x[0]['start']: x[0]['cost'] for x in refs},
                          'theta_rmse_vs_obs': s['theta_rmse_vs_obs']}
        res['pinn_vs_referee'] = {'all': score(pred['zb_pred'], r['zb_pred']),
                                  'below_saturation': score(pred['zb_pred'][dry], r['zb_pred'][dry])}
        zb_runs.append(('FEM inversion (referee)', r['zb_pred']))
    zb_runs.append(('hydrostatic reading of the sensor', np.clip(zb_hyd, 0.0, 1.5 * zb_obs.max())))
    th_runs.append(('FEM with the observed water table', d['theta_fem'][k_fem]))
    band = None
    if args.sigma_hind is not None:      # posterior of the inversion (holds no soil error) and the soil transfer
        s_tot = np.sqrt(pred['zb_sigma'] ** 2 + args.sigma_hind ** 2)
        e = pred['zb_pred'] - zb_obs
        res['pinn']['uncertainty'] = {
            'sigma_hind_m': args.sigma_hind, 'sigma_total_median_m': float(np.median(s_tot)),
            'below_saturation_within_1_sigma': float(np.mean(np.abs(e[dry]) <= s_tot[dry])),
            'below_saturation_within_2_sigma': float(np.mean(np.abs(e[dry]) <= 2.0 * s_tot[dry])),
            'saturated_hours': int((~dry).sum()),
            # a saturated sensor bounds the water table from below only: it is at or above the estimate
            'saturated_observed_not_below_estimate_plus_2_sigma':
                float(np.mean(e[~dry] >= -2.0 * s_tot[~dry])) if (~dry).any() else None,
        }
        band = ('PINN estimate \u00b1 2 sd (inversion and soil transfer)',
                pred['zb_pred'] - 2.0 * s_tot, pred['zb_pred'] + 2.0 * s_tot)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=2)
    plots.plot_real_result(t, th_obs, th_runs, zb_obs, zb_runs, args.out + '.png', band=band)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
