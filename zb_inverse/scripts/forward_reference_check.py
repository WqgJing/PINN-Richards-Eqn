#!/usr/bin/env python
"""SCORING ONLY (never part of the method): is the forward error of the networks theirs, or the reference's?

The reference of a site data set (theta, theta_fem[0] of real.npz) is one solve of the sweep solver at fem.dt_hours
and fem.n_nodes. Here that solve is repeated at finer time steps and grids: same water-table path (truth_coef), same
forcing nodes and root sink, linear in time between the nodes, which is how the networks see them. Each forward
network (water table given) is then compared with every level and with the first network of the list.

  sensor   mean theta of the observed layer, hourly: the level's own grid against the network's observation operator
  profile  theta on the STORED grid of the data set, hourly: finer levels are interpolated linearly onto it

Usage: python -u scripts/forward_reference_check.py --data output/real_nc2_sep23c_own \
           --members baseline=runs/fwd_nc2c s1=runs/fs_nc2c_s1 ... \
           --levels 1:101 0.25:101 0.25:201 0.0625:201 0.0625:401 --out output/paper/nc2_sep23c/forward_reference
"""
import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import yaml

from common import hourly_times
from make_twin_fem import load_fem, run_fem
from zbinv.parametric import load_members
from zbinv.physics import theta_obs_model, theta_profile
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy

torch.set_default_dtype(torch.float64)


def stats(d):
    d = np.asarray(d)
    return {'rms': float(np.sqrt(np.mean(d ** 2))), 'max': float(np.abs(d).max()), 'mean': float(d.mean())}


def by_day(d, t):
    """RMS of an hourly series, or of hourly profiles (Nt, Nx), over each day of the window."""
    return [float(np.sqrt(np.mean(d[(t >= k) & (t <= k + 1)] ** 2))) for k in range(int(np.ceil(t[-1])))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='data set directory of scripts/make_real.py')
    ap.add_argument('--members', nargs='+', required=True, help='name=run directory; the first is the baseline')
    ap.add_argument('--levels', nargs='+', required=True, help='<time step in hours>:<nodes>, coarse to fine')
    ap.add_argument('--tol-factor', type=float, default=0.01,
                    help='the finest level is solved once more with the Picard tolerance times this (0: skip)')
    ap.add_argument('--out', required=True, help='output stem: <stem>.json and <stem>.npz')
    a = ap.parse_args()

    with open(os.path.join(a.data, 'twin_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    ds = np.load(os.path.join(a.data, 'real.npz'), allow_pickle=True)
    fc, depth = cfg['fem'], cfg['obs']['depth_m']
    n_days = int(ds['n_days'])
    t_h, t_n = np.asarray(ds['t_days'], float), np.asarray(ds['t_daily'], float)
    q_n, tp_n = np.asarray(ds['q_daily'], float), np.asarray(ds['tp_daily_mmd'], float)
    xi_s = np.asarray(ds['xi'], float)                                   # the stored grid
    order = np.argsort(xi_s)
    truth = ZbSplineNumpy(CubicBSpline1D(0.0, n_days, cfg['zb_spline']['knot_days']), ds['truth_coef'],
                          cfg['zb_spline']['zb_min'])
    checks = {'spline_vs_stored_path_max_m': float(np.abs(truth.zb(t_h) - ds['zb_spline']).max())}
    print(f"[ref] {a.data}: {n_days} d, forcing nodes every {24 * (t_n[1] - t_n[0]):.2f} h; path from truth_coef "
          f"differs from the stored one by {checks['spline_vs_stored_path_max_m']:.1e} m at most", flush=True)

    Solver, Sink = load_fem(fc['root'])

    def solve(dt_h, n_nodes, cfg_run):
        t = hourly_times(n_days, dt_h)
        t0 = time.time()
        r = run_fem(Solver, Sink, cfg_run, t, truth.zb(t)[None], np.interp(t, t_n, q_n)[None],
                    np.interp(t, t_n, tp_n)[None], n_nodes, depth)
        k = np.round(t_h / (dt_h / 24.0)).astype(int)
        if np.abs(t[k] - t_h).max() > 1e-9:
            raise ValueError(f'time step {dt_h} h does not hit the hourly times')
        o = np.argsort(r['xi'])
        prof = np.stack([np.interp(xi_s[order], r['xi'][o], row[o]) for row in r['theta'][0][k]])
        back = np.empty_like(prof)
        back[:, order] = prof                                            # in the stored order of nodes
        return {'dt_h': dt_h, 'n_nodes': n_nodes, 'sensor': r['theta_obs'][0][k], 'profile': back,
                'converged': float(r['converged'].mean()), 'max_iters': int(r['n_iters'].max()),
                'seconds': time.time() - t0}

    levels = []
    for item in a.levels:
        dt_h, n_nodes = item.split(':')
        levels.append(solve(float(dt_h), int(n_nodes), cfg))
        L = levels[-1]
        print(f"[ref] level {item}: {L['seconds']:.0f} s, converged {100 * L['converged']:.1f} %", flush=True)
    key = lambda L: f"{L['dt_h']:g}h/{L['n_nodes']}"
    fine = levels[-1]

    # the first level against the stored reference (same settings when it is fem.dt_hours : fem.n_nodes)
    checks['first_level_vs_stored'] = {'sensor': stats(levels[0]['sensor'] - ds['theta_fem'][0]),
                                       'profile': stats(levels[0]['profile'] - ds['theta'])}
    print(f"[ref] level {key(levels[0])} minus the stored reference: sensor max "
          f"{checks['first_level_vs_stored']['sensor']['max']:.1e}, profile max "
          f"{checks['first_level_vs_stored']['profile']['max']:.1e}", flush=True)
    if a.tol_factor > 0:
        tight = copy.deepcopy(cfg)
        tight['fem']['tol'] = fc['tol'] * a.tol_factor
        tight['fem']['max_iter'] = 4 * fc['max_iter']
        T = solve(fine['dt_h'], fine['n_nodes'], tight)
        checks['finest_vs_tighter_tolerance'] = {'tol': tight['fem']['tol'], 'converged': T['converged'],
                                                 'sensor': stats(fine['sensor'] - T['sensor']),
                                                 'profile': stats(fine['profile'] - T['profile'])}
        print(f"[ref] finest level, Picard tolerance {fc['tol']:g} against {tight['fem']['tol']:g}: sensor max "
              f"{checks['finest_vs_tighter_tolerance']['sensor']['max']:.1e}, profile max "
              f"{checks['finest_vs_tighter_tolerance']['profile']['max']:.1e}", flush=True)

    out = {'data': a.data, 'checks': checks, 'levels': [], 'networks': {}}
    print('\n[ref] each level minus the finest (theta, m3/m3)')
    print('  level            sensor RMS   max        profile RMS  max        sensor RMS by day')
    for L in levels:
        ds_, dp = L['sensor'] - fine['sensor'], L['profile'] - fine['profile']
        row = {'level': key(L), 'dt_h': L['dt_h'], 'n_nodes': L['n_nodes'], 'converged': L['converged'],
               'max_iters': L['max_iters'], 'seconds': L['seconds'], 'vs_finest': {
                   'sensor': stats(ds_), 'profile': stats(dp), 'sensor_rms_by_day': by_day(ds_, t_h)}}
        out['levels'].append(row)
        v = row['vs_finest']
        print(f"  {key(L):15s}  {v['sensor']['rms']:.2e}   {v['sensor']['max']:.2e}   {v['profile']['rms']:.2e}   "
              f"{v['profile']['max']:.2e}   " + ' '.join(f'{x:.1e}' for x in v['sensor_rms_by_day']), flush=True)

    dev = torch.device('cpu')
    net, save = {}, {'t_days': t_h, 'xi': xi_s}
    for item in a.members:
        name, d = item.split('=', 1)
        (m,), mcfg = load_members([d], dev)
        c = torch.as_tensor(np.asarray(ds['truth_coef'], float))
        with torch.no_grad():
            s = theta_obs_model(m, torch.as_tensor(t_h).reshape(-1, 1), mcfg['obs']['depth_m'], c).numpy().ravel()
        net[name] = {'dir': d, 'sensor': s,
                     'profile': theta_profile(m, torch.as_tensor(xi_s), torch.as_tensor(t_h), c).numpy()}
        save[f'sensor_net_{name}'] = s
    base = next(iter(net))
    for L in levels:
        save[f"sensor_ref_{L['dt_h']:g}h_{L['n_nodes']}"] = L['sensor']
    save['profile_ref_finest'] = fine['profile']
    save['profile_ref_first'] = levels[0]['profile']
    save[f'profile_net_{base}'] = net[base]['profile']

    print('\n[ref] each network minus each level: sensor RMS (max) | profile RMS (max)')
    for name, N in net.items():
        res = {'dir': N['dir'], 'levels': {}}
        for L in levels:
            ds_, dp = N['sensor'] - L['sensor'], N['profile'] - L['profile']
            res['levels'][key(L)] = {'sensor': stats(ds_), 'profile': stats(dp),
                                     'sensor_rms_by_day': by_day(ds_, t_h), 'profile_rms_by_day': by_day(dp, t_h)}
        res['vs_baseline'] = {'sensor': stats(N['sensor'] - net[base]['sensor']),
                              'profile': stats(N['profile'] - net[base]['profile']),
                              'sensor_rms_by_day': by_day(N['sensor'] - net[base]['sensor'], t_h)}
        out['networks'][name] = res
        print(f'  {name}')
        for k, v in res['levels'].items():
            print(f"    {k:15s}  {v['sensor']['rms']:.2e} ({v['sensor']['max']:.2e}) | {v['profile']['rms']:.2e} "
                  f"({v['profile']['max']:.2e})   by day: " + ' '.join(f'{x:.1e}' for x in v['sensor_rms_by_day']))
        v = res['vs_baseline']
        print(f"    minus {base:9s}  {v['sensor']['rms']:.2e} ({v['sensor']['max']:.2e}) | {v['profile']['rms']:.2e} "
              f"({v['profile']['max']:.2e})   by day: " + ' '.join(f'{x:.1e}' for x in v['sensor_rms_by_day']),
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    np.savez(a.out + '.npz', **save)
    print(f"\n[ref] saved {a.out}.json and {a.out}.npz", flush=True)


if __name__ == '__main__':
    main()
