#!/usr/bin/env python
"""Does the time step of the numerical inversion move its estimate? (scoring only)

Reads results of scripts/fem_referee.py on one data set, run with different --dt_hours (and starts), and compares the
water-table estimates on the hourly record: each against the well, every pair against each other, and a PINN
estimate (scripts/param_invert.py) against each. Nothing here is read by a PINN run.

Usage: python -u scripts/referee_step_check.py --data output/real_nc2_sep23c_own \
           --referee output/referee/nc2_own_sep23c_s1 output/referee/nc2_own_sep23c_dt025_s1 ... \
           --pinn runs/rfnc1dc_it1 --out output/paper/nc2_sep23c/inversion_step
"""
import argparse
import json
import os
import numpy as np
import yaml


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def difference(a, b, day, t_rel):
    """a - b on the hourly record: RMS, bias, largest, and the RMS of every day of the window."""
    d = a - b
    by_day = [rms(d[day == k]) for k in range(int(day.max()) + 1)]
    return {'rms_m': rms(d), 'bias_m': float(d.mean()), 'maxabs_m': float(np.abs(d).max()),
            't_of_max_days': float(t_rel[int(np.abs(d).argmax())]), 'rms_by_day_m': by_day}


def against_well(est, obs):
    d = est - obs
    return {'rmse_m': rms(d), 'bias_m': float(d.mean()), 'maxabs_m': float(np.abs(d).max()),
            'r': float(np.corrcoef(est, obs)[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory of a make_real.py data set')
    ap.add_argument('--referee', nargs='+', required=True, help='directories of fem_referee.py results')
    ap.add_argument('--pinn', default=None, help='directory of a param_invert.py result')
    ap.add_argument('--out', required=True, help='path stem of the output (.json)')
    args = ap.parse_args()
    with open(os.path.join(args.data, 'twin_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    d = dict(np.load(os.path.join(args.data, 'real.npz'), allow_pickle=False))
    t, zb_obs = d['t_days'], d['zb_true']
    t_rel = t - t[0]
    day = np.minimum(np.floor(t_rel).astype(int), int(np.ceil(t_rel[-1])) - 1)
    own_dt = float(cfg['fem']['dt_hours'])

    runs = []
    for p in args.referee:
        with open(os.path.join(p, 'summary.json')) as f:
            s = json.load(f)
        r = dict(np.load(os.path.join(p, 'predictions.npz'), allow_pickle=False))
        if not np.allclose(r['t_days'], t) or not np.allclose(r['zb_true'], zb_obs):
            raise SystemExit(f'{p} belongs to another data set')
        runs.append({'dir': p, 'dt_h': float(s.get('fem_dt_hours', own_dt)), 'start': s['start'], 'zb': r['zb_pred'],
                     'summary': s})
    res = {'data': args.data, 'hours': int(t.size), 'inversions': [], 'pairs': []}
    for a in runs:
        s = a['summary']
        res['inversions'].append({
            'dir': a['dir'], 'fem_dt_hours': a['dt_h'], 'start': a['start'], 'cost': s['cost'],
            'iterations': s.get('iterations'), 'member_solves': s.get('member_solves'), 'wall_s': s.get('wall_s'),
            'fem_not_converged_max': s.get('fem_not_converged_max'), 'coef_on_bound': s.get('coef_on_bound'),
            'theta_rmse_vs_obs': s['theta_rmse_vs_obs'],
            'posterior_sigma_zb_median_m': s.get('posterior_sigma_zb_median_m'),
            'vs_well': against_well(a['zb'], zb_obs)})
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            res['pairs'].append({'a': a['dir'], 'b': b['dir'], 'a_dt_h': a['dt_h'], 'b_dt_h': b['dt_h'],
                                 'a_start': a['start'], 'b_start': b['start'],
                                 'a_minus_b': difference(a['zb'], b['zb'], day, t_rel)})
    if args.pinn:
        pred = dict(np.load(os.path.join(args.pinn, 'predictions.npz'), allow_pickle=False))
        if not np.allclose(pred['t_days'], t) or not np.allclose(pred['zb_true'], zb_obs):
            raise SystemExit('the PINN result belongs to another data set')
        res['pinn'] = {'dir': args.pinn, 'vs_well': against_well(pred['zb_pred'], zb_obs),
                       'sigma_median_m': float(np.median(pred['zb_sigma'])),
                       'minus_inversion': [{'dir': a['dir'], 'fem_dt_hours': a['dt_h'], 'start': a['start'],
                                            **difference(pred['zb_pred'], a['zb'], day, t_rel)} for a in runs]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=2)

    cm = 100.0
    print(f"{'numerical inversion':46s} {'step (h)':>9s} {'cost':>10s} {'it':>3s} {'wall s':>7s} "
          f"{'vs well: rmse':>13s} {'bias':>6s} {'max':>6s} {'r':>7s}")
    for x in res['inversions']:
        w = x['vs_well']
        print(f"{x['dir'][-46:]:46s} {x['fem_dt_hours']:9.4f} {x['cost']:10.6f} {str(x['iterations']):>3s} "
              f"{(x['wall_s'] or 0):7.0f} {cm * w['rmse_m']:13.3f} {cm * w['bias_m']:+6.3f} {cm * w['maxabs_m']:6.3f} "
              f"{w['r']:7.4f}")
    print('\npairs, a - b (cm): rms, max, time of max (days), rms by day')
    for x in res['pairs']:
        q = x['a_minus_b']
        print(f"  {x['a'][-28:]:28s} - {x['b'][-28:]:28s} {cm * q['rms_m']:8.4f} {cm * q['maxabs_m']:8.4f} "
              f"{q['t_of_max_days']:6.2f}  " + ' '.join(f'{cm * v:.3f}' for v in q['rms_by_day_m']))
    if args.pinn:
        print(f"\nPINN {args.pinn} minus numerical inversion (cm): rms, bias, max, time of max (days), rms by day")
        for q in res['pinn']['minus_inversion']:
            print(f"  {q['dir'][-28:]:28s} step {q['fem_dt_hours']:.4f} h {cm * q['rms_m']:8.4f} {cm * q['bias_m']:+8.4f} "
                  f"{cm * q['maxabs_m']:8.4f} {q['t_of_max_days']:6.2f}  "
                  + ' '.join(f'{cm * v:.3f}' for v in q['rms_by_day_m']))


if __name__ == '__main__':
    main()
