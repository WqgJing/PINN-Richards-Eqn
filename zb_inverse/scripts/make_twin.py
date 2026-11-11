#!/usr/bin/env python
"""E0a/E0b: build the twin truth with the reference solver, with grid-refinement and mass checks.

Usage: python -u scripts/make_twin.py --config configs/twin_pas_apr.yaml [--refine]
"""
import argparse
import json
import os
import shutil
import numpy as np

from common import load_config, load_with_overrides, setup, hourly_times, run_reference
from zbinv import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    ap.add_argument('--tag', default='', help='suffix for output file names')
    ap.add_argument('--refine', action='store_true', help='also run half and double grids')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    import yaml
    with open(os.path.join(out, 'twin_config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    soil, win, sp, truth = setup(cfg)
    print(f"[twin] {cfg['site']['name']} {win['start_date']} + {win['n_days']} d", flush=True)
    print(f"[twin] soil: {soil.summary()}", flush=True)
    print(f"[twin] spline: {sp.n_coef} coefficients, knot spacing {sp.dt:.2f} d", flush=True)
    fit_err = truth.zb(win['t_days']) - win['zb']
    print(f"[twin] truth spline vs observed daily zb: RMSE {np.sqrt(np.mean(fit_err ** 2)):.4f} m, "
          f"max {np.abs(fit_err).max():.4f} m", flush=True)

    t_eval = hourly_times(win['n_days'], cfg['reference'].get('out_dt_hours', 1.0))
    res = run_reference(cfg, soil, win, truth, t_eval)
    summary = {
        'window': win['start_date'], 'n_days': win['n_days'], 'n_coef': sp.n_coef,
        'zb_range_m': [float(res['zb'].min()), float(res['zb'].max())],
        'theta_obs_range': [float(res['theta_obs'].min()), float(res['theta_obs'].max())],
        'mass_err_max_rel': float(np.abs(res['mass_err_t']).max() / max(np.abs(res['storage_t']).max(), 1e-12)),
        'h_max_m': float(res['h_t'].max() * soil.L),
    }
    # Admissibility: a prescribed evaporation the soil cannot deliver dries the surface node without
    # bound (h -> -1e7 m in a sub-mm layer). Such a twin is not a valid test of any solver.
    h_m = res['h_t'] * soil.L
    z_m = res['xi'][None, :] * res['zb'][:, None]
    dhdz = np.diff(h_m, axis=1) / np.diff(z_m, axis=1)
    summary['h_surf_min_m'] = float(h_m[:, -1].min())
    summary['dhdz_max'] = float(np.abs(dhdz).max())
    h_floor = cfg['reference'].get('h_surf_admissible_m', -10.0)
    summary['admissible'] = bool(summary['h_surf_min_m'] >= h_floor)
    print(f"[twin] theta_obs range {summary['theta_obs_range']}, max h {summary['h_max_m']:.3f} m, "
          f"relative mass error {summary['mass_err_max_rel']:.2e}", flush=True)
    print(f"[twin] surface head min {summary['h_surf_min_m']:.3f} m, max |dh/dz| {summary['dhdz_max']:.2f}; "
          f"admissible (h_surf >= {h_floor} m): {summary['admissible']}", flush=True)

    if args.refine:
        n0 = cfg['reference']['n_nodes']
        for n in [(n0 + 1) // 2, 2 * n0 - 1]:
            r2 = run_reference(cfg, soil, win, truth, t_eval, n_nodes=n)
            d = np.abs(r2['theta_obs'] - res['theta_obs'])
            summary[f'grid_{n}_theta_obs_maxdiff'] = float(d.max())
            print(f"[twin] grid {n} vs {n0}: max |d theta_obs| = {d.max():.2e}, RMSE {np.sqrt(np.mean(d ** 2)):.2e}",
                  flush=True)

    obs_step = int(round(cfg['obs']['interval_hours'] / cfg['reference'].get('out_dt_hours', 1.0)))
    k_obs = np.arange(0, t_eval.size, obs_step)
    extra = {}
    if cfg['reference'].get('h_crit_m') is not None:
        # head-limited surface BC: the observable flux is the actual (delivered) one
        extra['q_act_hourly'] = res['q_act']
        lim = res['q_act'] < res['qsurf'] - 1e-3 / 86400e3
        summary['frac_hours_supply_limited'] = float(lim.mean())
        summary['evap_deficit_mm'] = float(np.trapezoid(res['qsurf'] - res['q_act'], res['t_days']) * 86400e3)
        print(f"[twin] head-limited BC (h_crit {cfg['reference']['h_crit_m']} m): supply-limited "
              f"{100 * lim.mean():.1f}% of hours, evaporation deficit {summary['evap_deficit_mm']:.1f} mm", flush=True)
    np.savez(os.path.join(out, 'twin.npz'), **extra,
             t_days=res['t_days'], xi=res['xi'], theta=res['theta'], h_t=res['h_t'],
             zb_true=res['zb'], qsurf=res['qsurf'], theta_obs_hourly=res['theta_obs'],
             t_obs_days=res['t_days'][k_obs], theta_obs_daily=res['theta_obs'][k_obs],
             zb_obs_daily=win['zb'], t_daily=win['t_days'], q_daily=win['qsurf'],
             truth_coef=truth.coef, zb_min=truth.zb_min, knot_days=sp.dt, n_days=win['n_days'],
             start_date=win['start_date'], L=soil.L,
             soil=np.array([soil.theta_s, soil.theta_r, soil.alpha, soil.n, soil.Ks]))
    with open(os.path.join(out, 'twin_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    tw = {'t_days': res['t_days'], 'zb_true': res['zb'], 't_obs_days': res['t_days'][k_obs],
          'zb_obs_daily': np.interp(res['t_days'][k_obs], win['t_days'], win['zb']),
          'qsurf': res['qsurf'], 'theta_obs_hourly': res['theta_obs'],
          'theta_obs_daily': res['theta_obs'][k_obs], 'xi': res['xi'], 'theta': res['theta'],
          'title': f"{cfg['site']['name']} {win['start_date']} +{win['n_days']} d"}
    plots.plot_twin(tw, os.path.join(out, 'twin.png'))
    print(f"[twin] saved {out}/twin.npz, twin.png, twin_summary.json", flush=True)


if __name__ == '__main__':
    main()
