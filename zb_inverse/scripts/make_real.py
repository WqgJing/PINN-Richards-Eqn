#!/usr/bin/env python
"""Real-site window for the inversion: AmeriFlux half-hourly records -> forcing nodes, hourly soil moisture at the
sensor and the observed water table, in the format of the twins (train.load_dataset), plus a forward check.

  forcing  ET = LE / (rho_w lambda), clipped at 0 as in the forward validation, gaps filled with the mean diurnal
           cycle; nodes every forcing.node_hours are centred means of ET and of the rain, linear in between
             q_surf = (1 - f_T) ET - P    soil evaporation minus infiltration, + up (no runoff is measured: Q_s = 0)
             T_p    = f_T ET              root sink
           f_T = forcing.transpiration_fraction: a number, or 'gldas' (daily T_p / (E_soil + T_p) of the GLDAS
           pixel in site.gldas_csv: the tower measures the total only)
  data     hourly means of the soil-moisture sensor (obs.depth_m = [top, bottom] below the surface)
  truth    hourly means of the observed water-table depth: for scoring, and for the forward check below
  check    the wtd_pipeline FEM with the OBSERVED water table (its spline fit, and the hourly record) and the forcing
           above, for the f_T of the data set and for forcing.check_fractions: can this soil and this forcing
           explain the soil moisture at the sensor at all? Nothing is inverted here.

Usage: python -u scripts/make_real.py --config configs/real_ona_nov21.yaml
"""
import argparse
import copy
import csv
import json
import os
import numpy as np

from common import load_with_overrides, hourly_times
from make_twin_fem import load_fem, run_fem
from zbinv import plots, sitedata
from zbinv.data import obs_layer
from zbinv.soil import SoilVG
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy


def gldas_fraction(path, dates):
    """T_p / (E_soil + T_p) of the GLDAS daily rows for the given dates."""
    with open(path) as f:
        rows = {r['date']: r for r in csv.DictReader(f)}
    e = np.array([float(rows[str(d)]['E_soil_mm']) for d in dates])
    tp = np.array([float(rows[str(d)]['Tp_mm']) for d in dates])
    if np.any(e + tp <= 0.0):
        raise ValueError('GLDAS E_soil + T_p = 0 on a day of the window: no partition')
    return tp / (e + tp)


def site_series(cfg):
    """Half-hourly records around the window -> forcing nodes and hourly observations (module docstring)."""
    sc, fc = cfg['site'], cfg['forcing']
    n_days = int(cfg['window']['n_days'])
    t0 = np.datetime64(cfg['window']['start'], 'm')
    w = fc['node_hours'] / 24.0
    shift = float(sc.get('p_shift_hours', 0.0))
    half = int(fc.get('fill_half_window_days', 7))
    pad = np.timedelta64(half + 1 + int(np.ceil(abs(shift) / 24.0)), 'D')
    t_rec, rec = sitedata.read_amf_halfhourly(sc['amf_csv'], sc['columns'], t0 - pad,
                                              t0 + np.timedelta64(n_days, 'D') + pad)
    t_c = ((t_rec - t0) / np.timedelta64(1, 'm') + 15.0) / 1440.0     # record centres, days from the window start
    used = (t_c >= -w / 2) & (t_c < n_days + w / 2)                   # records the forcing nodes are made of

    p = sitedata.shift_records(rec['p'], shift) if shift else rec['p']
    p_missing = ~np.isfinite(p)
    p = np.where(p_missing, 0.0, p) / 1000.0 / 1800.0                 # mm per half hour -> m/s; missing: no rain
    et, le_missing = sitedata.fill_diurnal(np.clip(rec['le'] * sitedata.LE_TO_ET, 0.0, None), half_window_days=half)

    t_nodes = np.arange(int(round(n_days / w)) + 1) * w
    et_n = sitedata.centred_means(et, t_c, t_nodes, w)
    p_n = sitedata.centred_means(p, t_c, t_nodes, w)
    dates = (t0 + np.round(t_nodes * 1440.0).astype('timedelta64[m]')).astype('datetime64[D]')

    t_h = hourly_times(n_days, 1.0)
    th, n_th = sitedata.fill_short_gaps(
        t_h, sitedata.centred_means(rec['swc'] / 100.0, t_c, t_h, 1.0 / 24.0, allow_missing=True),
        cfg['obs'].get('max_gap_hours', 3.0) / 24.0)
    zb, n_zb = sitedata.fill_short_gaps(
        t_h, sitedata.centred_means(-rec['wtd'], t_c, t_h, 1.0 / 24.0, allow_missing=True),
        cfg['truth'].get('max_gap_hours', 3.0) / 24.0)
    if zb.min() <= cfg['zb_spline']['zb_min'] + 0.01:
        raise ValueError(f"observed water table within {zb.min():.3f} m of the surface: outside the model (zb_min)")
    info = {'le_filled_frac': float(le_missing[used].mean()), 'p_missing_records': int(p_missing[used].sum()),
            'theta_hours_filled': n_th, 'zb_hours_filled': n_zb, 'p_shift_hours': shift,
            'et_sum_mm': float(np.trapezoid(et_n, t_nodes) * 86400e3),
            'p_sum_mm': float(np.trapezoid(p_n, t_nodes) * 86400e3),
            'et_records_sum_mm': float(et[(t_c >= 0) & (t_c < n_days)].sum() * 1800e3),
            'p_records_sum_mm': float(p[(t_c >= 0) & (t_c < n_days)].sum() * 1800e3)}
    return {'t0': t0, 'n_days': n_days, 't_nodes': t_nodes, 'dates': dates, 'et': et_n, 'p': p_n,
            't_h': t_h, 'theta': th, 'zb': zb, 'info': info}


def fraction_nodes(cfg, s, value):
    if value == 'gldas':
        return gldas_fraction(cfg['site']['gldas_csv'], s['dates'])
    return np.full(s['t_nodes'].size, float(value))


def score(model, obs):
    d = np.asarray(model) - np.asarray(obs)
    return {'rmse': float(np.sqrt(np.mean(d ** 2))), 'bias': float(d.mean()), 'max': float(np.abs(d).max()),
            'r': float(np.corrcoef(model, obs)[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], help='override key.sub=value')
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.set)
    fc, depth = cfg['fem'], cfg['obs']['depth_m']
    soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
    s = site_series(cfg)
    n_days, t_h, t_n = s['n_days'], s['t_h'], s['t_nodes']
    print(f"[real] {cfg['site']['name']} {s['t0']} + {n_days} d; soil: {soil.summary()}", flush=True)
    for k, v in s['info'].items():
        print(f"[real] {k}: {v}", flush=True)

    sp = CubicBSpline1D(0.0, n_days, cfg['zb_spline']['knot_days'])
    truth = ZbSplineNumpy.fit(sp, t_h, s['zb'], cfg['zb_spline']['zb_min'])
    zb_fit = truth.zb(t_h)
    fit = score(zb_fit, s['zb'])
    sigma_theta = sitedata.unresolved_rms(s['theta'])
    print(f"[real] water table {s['zb'].min():.3f} .. {s['zb'].max():.3f} m; spline ({sp.n_coef} coefficients) vs "
          f"hourly record: RMSE {fit['rmse']:.4f} m, max {fit['max']:.4f} m", flush=True)
    print(f"[real] theta at the sensor {s['theta'].min():.3f} .. {s['theta'].max():.3f}; rms about its 24-h running "
          f"mean {sigma_theta:.4f}", flush=True)

    # forward check: members = (f_T of the data set, then the check fractions) x (spline water table, hourly record)
    fractions = [cfg['forcing']['transpiration_fraction']] + list(cfg['forcing'].get('check_fractions', []))
    f_nodes = [fraction_nodes(cfg, s, v) for v in fractions]
    q_nodes = [(1.0 - f) * s['et'] - s['p'] for f in f_nodes]                  # m/s, + up
    tp_nodes = [f * s['et'] * 86400e3 for f in f_nodes]                        # mm/d
    on = lambda y: np.interp(t_h, t_n, y)
    n_f = len(fractions)
    zb_mem = np.stack([zb_fit] * n_f + [s['zb']] * n_f)
    q_mem = np.stack([on(q) for q in q_nodes] * 2)
    tp_mem = np.stack([on(tp) for tp in tp_nodes] * 2)
    Solver, Sink = load_fem(fc['root'])
    A = run_fem(Solver, Sink, cfg, t_h, zb_mem, q_mem, tp_mem, fc['n_nodes'], depth)
    t_chk = hourly_times(n_days, fc['check_dt_hours'])                         # the data set's member, finer
    C = run_fem(Solver, Sink, cfg, t_chk, truth.zb(t_chk)[None], np.interp(t_chk, t_n, q_nodes[0])[None],
                np.interp(t_chk, t_n, tp_nodes[0])[None], fc['check_n_nodes'], depth)
    k_chk = np.round(t_h / (fc['check_dt_hours'] / 24.0)).astype(int)
    top, bottom = obs_layer(depth)
    th_hydro = soil.theta(-(s['zb'] - (top + bottom) / 2.0) / soil.L)          # no flow: hydrostatic over the WT

    h_floor = cfg['reference'].get('h_surf_admissible_m', -10.0)
    summary = {
        'site': cfg['site']['name'], 'window': str(s['t0']), 'n_days': n_days, 'n_coef': sp.n_coef,
        'obs_layer_m': [top, bottom], **s['info'],
        'zb_range_m': [float(s['zb'].min()), float(s['zb'].max())], 'zb_spline_vs_record': fit,
        'theta_obs_range': [float(s['theta'].min()), float(s['theta'].max())], 'sigma_theta_unresolved': sigma_theta,
        'transpiration_fraction': fractions[0], 'transpiration_fraction_mean': float(f_nodes[0].mean()),
        'fem_vs_finer_fem': score(A['theta_obs'][0], C['theta_obs'][0][k_chk]),
        'hydrostatic_vs_obs': score(th_hydro, s['theta']),
        'fem_vs_obs': {}, 'h_surf_min_m': {}, 'fem_converged_frac': float(A['converged'].mean()),
    }
    for j, v in enumerate(fractions):
        for kind, m in (('spline', j), ('record', n_f + j)):
            key = f'fT={v}, {kind} water table'
            summary['fem_vs_obs'][key] = score(A['theta_obs'][m], s['theta'])
            summary['h_surf_min_m'][key] = float(A['h'][m, :, -1].min())
    summary['admissible'] = bool(summary['h_surf_min_m'][f'fT={fractions[0]}, spline water table'] >= h_floor)
    print(f"[real] FEM vs finer FEM: RMSE {summary['fem_vs_finer_fem']['rmse']:.2e}; hydrostatic vs observed: RMSE "
          f"{summary['hydrostatic_vs_obs']['rmse']:.4f}, bias {summary['hydrostatic_vs_obs']['bias']:+.4f}", flush=True)
    for key, v in summary['fem_vs_obs'].items():
        print(f"[real] FEM with the observed water table vs observed theta, {key}: RMSE {v['rmse']:.4f}, bias "
              f"{v['bias']:+.4f}, r {v['r']:.3f}, surface head min {summary['h_surf_min_m'][key]:.2f} m", flush=True)

    import yaml
    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    c = copy.deepcopy(cfg)
    c['fem']['root_sink'] = bool(np.any(tp_nodes[0] > 0.0))
    with open(os.path.join(out, 'twin_config.yaml'), 'w') as f:
        yaml.safe_dump(c, f, sort_keys=False)
    k_obs = np.arange(0, t_h.size, int(round(cfg['obs']['interval_hours'])))
    np.savez(os.path.join(out, 'real.npz'),
             # the keys of a twin (train.load_dataset): data and forcing ...
             t_days=t_h, theta_obs_hourly=s['theta'], t_obs_days=t_h[k_obs], theta_obs_daily=s['theta'][k_obs],
             t_daily=t_n, q_daily=q_nodes[0], tp_daily_mmd=tp_nodes[0], qsurf=on(q_nodes[0]), n_days=n_days,
             # ... and what the result is scored against: the observed water table (zb_true), its spline fit, and
             # the FEM profile over that fit
             zb_true=s['zb'], truth_coef=truth.coef, zb_min=truth.zb_min, knot_days=sp.dt,
             xi=A['xi'], theta=A['theta'][0], start_date=str(s['t0']), L=soil.L,
             soil=np.array([soil.theta_s, soil.theta_r, soil.alpha, soil.n, soil.Ks]),
             # forcing parts and the forward check
             et_nodes=s['et'], p_nodes=s['p'], ft_nodes=f_nodes[0], zb_spline=zb_fit, theta_hydrostatic=th_hydro,
             theta_fem=A['theta_obs'], fem_fraction=np.array([str(v) for v in fractions] * 2),
             fem_water_table=np.array(['spline'] * n_f + ['record'] * n_f), converged=A['converged'])
    with open(os.path.join(out, 'real_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    plots.plot_real_check(t_h, s['zb'], zb_fit, t_n, s['et'] * 86400e3, s['p'] * 86400e3, s['theta'],
                          [(f'FEM with the observed water table, $f_T$ = {v}', A['theta_obs'][j])
                           for j, v in enumerate(fractions)], os.path.join(out, 'real_check.png'))
    print(f"[real] saved {out}/real.npz, twin_config.yaml, real_summary.json, real_check.png", flush=True)


if __name__ == '__main__':
    main()
