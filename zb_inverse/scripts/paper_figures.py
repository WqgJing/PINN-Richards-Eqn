#!/usr/bin/env python
"""Figures and tables of a results section for one PINN inversion on site data (REPORTING ONLY: reads finished runs,
trains nothing, and no run reads its output).

  fig1_setup     the data: surface flux and root uptake, soil moisture at the sensor, observed water table
  fig2_forward   forward problem (water table given): PINN against the numerical solution of the same problem
                 (--forward-reference: against that solution at the finest time step of
                 scripts/forward_reference_check.py, next to the data set's own solve)
  fig3_inverse   inverse problem: water table of the PINN, the numerical inversion, the hydrostatic reading, the well
  fig4_training  held-out physics losses of the members through the stages
  fig5_stages    the estimate stage by stage
  fig6_physics   surface flux and PDE residual of the final members at the estimate
  fig7_seeds     (--repeats) the estimate of repeated runs with other training seeds
  tables.md, metrics.json   every number of the tables

Usage: python -u scripts/paper_figures.py --data output/real_nc2_sep23c_own --tag rfnc1 --final runs/rfnc1dc_it1 \
           --referee output/referee/nc2_own_sep23c_s0 ... --forward runs/fwd_nc2c --out output/paper/nc2_sep23c \
           [--soil-note "calibrated on this window (in-sample)"] [--variants w64=runs/fs_nc2c_w64 ...] \
           [--forward-reference output/paper/nc2_sep23c/forward_reference] \
           [--repeats seed1000=runs/rfnc1s1dc_it1 ...] [--single member0=runs/rfnc1_inv4_m0 ...]
  --tag     name of the hpc/real_full_obs.sh run: stages runs/<tag>_inv0..<rounds>, members runs/<tag>_w<k> and
            runs/<tag>_r<r>_m<k>; --final is the inversion that gives the estimate (last round of hpc/dc_loop.sh)
"""
import argparse
import csv
import json
import os
import numpy as np
import torch
import yaml

from common import load_with_overrides  # noqa: F401  (puts src on the path)
from score_real import score
from zbinv import plots
from zbinv.data import obs_layer
from zbinv.parametric import load_members, physics_scales
from zbinv.physics import theta_obs_model, theta_profile, pde_residual, surface_flux
from zbinv.soil import SoilVG
from zbinv.train import load_dataset

torch.set_default_dtype(torch.float64)
plt = plots.plt
DEV = torch.device('cpu')
MM_D = 86400e3                                   # m/s -> mm/d
MEMBER_COLORS = [plots.C_TRUTH, plots.C_PRED, plots.C_3, plots.C_4]
T_LABEL = 'time (days)'


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_npz(path):
    return dict(np.load(path, allow_pickle=False))


def read_metrics(run):
    """metrics.csv of a training run as {column: array}; empty cells are nan, 'stage' stays text."""
    with open(os.path.join(run, 'metrics.csv')) as f:
        rows = list(csv.DictReader(f))
    out = {'stage': np.array([r['stage'] for r in rows])}
    for k in rows[0]:
        if k != 'stage':
            out[k] = np.array([float(r[k]) if r[k] not in ('', None) else np.nan for r in rows])
    return out


def pairs(items):
    return [tuple(x.split('=', 1)) for x in items]


def load_member(run, data_dir):
    (m,), cfg = load_members([run], DEV)
    if os.path.realpath(cfg['twin_npz']) != os.path.realpath(os.path.join(data_dir, 'real.npz')):
        raise SystemExit(f'{run} was trained on another data set ({cfg["twin_npz"]})')
    return m, cfg


def physics_at(m, cfg, c, n_days, nxi=101, per_hour=2):
    """One member at the path c: prescribed and network surface flux [mm/d] on a half-hourly grid, and the PDE
    residual in units of sigma_pde on (t, xi), xi clustered at the surface as the grid of the numerical solver."""
    ds = load_dataset(cfg)
    sig_q, sig_pde = physics_scales(cfg, ds)
    cc = torch.as_tensor(np.asarray(c, float))
    t = np.linspace(0.0, n_days, int(round(24 * per_hour * n_days)) + 1)
    q_net = surface_flux(m, torch.as_tensor(t).reshape(-1, 1), cc).detach().numpy().ravel() * m.soil.Ks * MM_D
    q_obs = np.interp(t, np.asarray(ds['t_q'], float), np.asarray(ds['q_obs'], float)) * MM_D
    xi = -(1.0 - np.linspace(0.0, 1.0, nxi)) ** 1.5
    X, T = np.meshgrid(xi, t, indexing='xy')                                   # (nt, nxi)
    xs, ts = torch.as_tensor(X.reshape(-1, 1)), torch.as_tensor(T.reshape(-1, 1))
    R = np.concatenate([pde_residual(m, xs[i:i + 8192], ts[i:i + 8192], cc).detach().numpy().ravel()
                        for i in range(0, xs.shape[0], 8192)]).reshape(T.shape) / sig_pde
    return {'t': t, 'xi': xi, 'q_net': q_net, 'q_obs': q_obs, 'R': R, 'sig_pde': sig_pde, 'sig_q': sig_q,
            'flux_rel_rms': rms(q_net - q_obs) / rms(q_obs), 'flux_rms_mmd': rms(q_net - q_obs),
            'residual_rms': rms(R), 'residual_max': float(np.abs(R).max())}


def physics_numbers(ph):
    return {k: ph[k] for k in ('flux_rel_rms', 'flux_rms_mmd', 'residual_rms', 'residual_max', 'sig_pde', 'sig_q')}


def step_label(dt_h):
    return f'{dt_h:g} h' if dt_h >= 1 else f'1/{round(1 / dt_h)} h'


def load_forward_reference(stem, d, data_dir, cfg):
    """Result of scripts/forward_reference_check.py: its tables (json) and the numerical solution at its finest
    level on the (t, xi) nodes of the data set (npz)."""
    FR, z = read_json(stem + '.json'), read_npz(stem + '.npz')
    if os.path.realpath(FR['data']) != os.path.realpath(data_dir):
        raise SystemExit(f'{stem} belongs to another data set ({FR["data"]})')
    if not np.allclose(z['t_days'], d['t_days']) or not np.allclose(z['xi'], d['xi']):
        raise SystemExit(f'{stem}: times or nodes differ from the data set')
    L = FR['levels'][-1]
    return FR, {'theta': z['profile_ref_finest'], 'sensor': z[f"sensor_ref_{L['dt_h']:g}h_{L['n_nodes']}"],
                'label': step_label(L['dt_h']), 'own_label': step_label(cfg['fem']['dt_hours']),
                'dt_h': L['dt_h'], 'n_nodes': L['n_nodes'], 'tol': FR['checks']['picard']['tol']}


def forward_errors(th5, prof, ref5, ref):
    """Soil moisture of a network against a numerical solution on the same (t, xi) nodes."""
    if prof.shape != ref.shape:
        raise SystemExit(f'profile shapes {prof.shape} vs {ref.shape}')
    e5, ep = th5 - ref5, prof - ref
    return {'theta_rms': rms(ep), 'theta_max': float(np.abs(ep).max()),
            'theta_rel_l2': float(np.sqrt(np.sum(ep ** 2) / np.sum(ref ** 2))),
            'theta_rms_over_range': rms(ep) / float(ref.max() - ref.min()),
            'sensor_rms': rms(e5), 'sensor_max': float(np.abs(e5).max())}, e5, ep


def forward_eval(run, d, data_dir, k_fem, fine=None):
    """A forward PINN (trained at the spline of the observed water table) against the numerical solution of the
    same problem: theta on the solver's (t, xi) nodes and at the sensor, surface flux, PDE residual. The solution
    is the data set's own solve; with fine (load_forward_reference) the errors against the finest level are added
    under 'vs_fine'."""
    m, cfg = load_member(run, data_dir)
    t, xi = np.asarray(d['t_days'], float), np.asarray(d['xi'], float)
    c = torch.as_tensor(np.asarray(d['truth_coef'], float))
    with torch.no_grad():
        th5 = theta_obs_model(m, torch.as_tensor(t).reshape(-1, 1), cfg['obs']['depth_m'], c).numpy().ravel()
    prof = theta_profile(m, torch.as_tensor(xi), torch.as_tensor(t), c).numpy()                 # (nt, nxi)
    err, e5, ep = forward_errors(th5, prof, d['theta_fem'][k_fem], np.asarray(d['theta'], float))
    ph = physics_at(m, cfg, d['truth_coef'], float(d['n_days']))
    s = read_json(os.path.join(run, 'summary.json'))
    num = {'run': run, **err, **physics_numbers(ph),
           'val_pde': s.get('val_pde'), 'val_flux': s.get('val_flux'), 'steps': s.get('steps'),
           'wall_s': s.get('wall_s'), 'width': cfg['model']['width'], 'depth': cfg['model']['depth'],
           'n_col': cfg['train']['n_col'], 'seed': cfg['train'].get('seed', 0),
           'n_params': int(sum(p.numel() for p in m.head.parameters())),
           'n_inputs': int(m.head.net[0].in_features)}
    fw = {'th5': th5, 'prof': prof, 'e5': e5, 'ep': ep, 'ph': ph}
    if fine is not None:
        num['vs_fine'], fw['e5_fine'], fw['ep_fine'] = forward_errors(th5, prof, fine['sensor'], fine['theta'])
    return num, fw


def layer_label(depth_m):
    top, bottom = obs_layer(depth_m)
    where = f'{100 * top:.0f}–{100 * bottom:.0f} cm' if bottom > top else f'at {100 * top:.0f} cm'
    return f'soil moisture, {where} (m$^3$/m$^3$)'


# ----------------------------------------------------------------------------- figures
def fig_setup(d, th_label, path):
    t = d['t_days']
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
    ax[0].plot(t, d['qsurf'] * MM_D, color=plots.C_TRUTH, label='surface flux, upward positive')
    ax[0].plot(t, d['tp_daily_mmd'], color=plots.C_PRED, label='root uptake')
    ax[0].axhline(0, color=plots.INK2, lw=0.6)
    ax[0].set_ylabel('flux (mm/d)'); ax[0].legend()
    ax[1].plot(t, d['theta_obs_hourly'], color=plots.INK)
    ax[1].set_ylabel(th_label)
    ax[2].plot(t, d['zb_true'], color=plots.INK)
    ax[2].invert_yaxis(); ax[2].set_ylabel('water-table depth (m)'); ax[2].set_xlabel(T_LABEL)
    plots._finish(fig, path)


def fig_forward(d, fw, k_fem, th_label, path, fine=None):
    """Without fine: the PINN against the data set's own numerical solve. With fine (load_forward_reference): the
    upper panels show the solve at the finest time step, the lower panels the difference of the PINN from that
    solve and from the data set's own, and the difference between the two solves."""
    t, xi = d['t_days'], d['xi']
    ref, ref5 = (fine['theta'], fine['sensor']) if fine else (d['theta'], d['theta_fem'][k_fem])
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.4))
    a = ax[0, 0]
    for col, k in zip(plots.SEQ_BLUE, np.linspace(0, t.size - 1, 6).astype(int)):
        depth = -xi * d['zb_spline'][k]
        a.plot(ref[k], depth, color=col, lw=2.4, label=f'day {t[k]:.1f}')
        a.plot(fw['prof'][k], depth, color=plots.INK, lw=0.9, ls='--')
    a.plot([], [], color=plots.INK2, lw=2.4, label='numerical solution')
    a.plot([], [], color=plots.INK, lw=0.9, ls='--', label='PINN')
    a.invert_yaxis(); a.set_xlabel('soil moisture (m$^3$/m$^3$)'); a.set_ylabel('depth (m)'); a.legend(ncol=2)
    a = ax[0, 1]
    a.plot(t, ref5, color=plots.C_TRUTH, lw=2.4, label='numerical solution')
    a.plot(t, fw['th5'], color=plots.C_PRED, lw=1.1, ls='--', label='PINN')
    a.set_xlabel(T_LABEL); a.set_ylabel(th_label); a.legend()
    order = np.argsort(-xi)
    if fine:
        lines = [(fw['ep_fine'], plots.C_PRED, 1.6, '-', f"PINN − numerical, {fine['label']} step"),
                 (fw['ep'], plots.C_TRUTH, 2.4, '-', f"PINN − numerical, {fine['own_label']} step"),
                 (d['theta'] - fine['theta'], plots.INK, 0.9, '--',
                  f"numerical, {fine['own_label']} step − {fine['label']} step")]
        for e, col, lw, ls, lab in lines:
            v = np.sqrt(np.mean(e ** 2, axis=1))
            ax[1, 0].semilogy(t, np.where(v > 0, v, np.nan), color=col, lw=lw, ls=ls, label=lab)
            v = np.sqrt(np.mean(e ** 2, axis=0))[order]
            ax[1, 1].semilogy(-xi[order], np.where(v > 0, v, np.nan), color=col, lw=lw, ls=ls, label=lab)
        ax[1, 0].set_xlabel(T_LABEL)
        ax[1, 0].set_ylabel('RMS over depth of the soil-moisture difference (m$^3$/m$^3$)'); ax[1, 0].legend()
        ax[1, 1].set_ylabel('RMS over time of the soil-moisture difference (m$^3$/m$^3$)'); ax[1, 1].legend()
    else:
        a = ax[1, 0]
        a.plot(t, fw['e5'], color=plots.C_PRED, label='sensor layer, PINN − numerical')
        a.plot(t, np.sqrt(np.mean(fw['ep'] ** 2, axis=1)), color=plots.C_TRUTH, label='RMS over depth')
        a.axhline(0, color=plots.INK2, lw=0.6)
        a.set_xlabel(T_LABEL); a.set_ylabel('soil-moisture error (m$^3$/m$^3$)'); a.legend()
        ax[1, 1].semilogy(-xi[order], np.sqrt(np.mean(fw['ep'] ** 2, axis=0))[order], color=plots.C_TRUTH)
        ax[1, 1].set_ylabel('RMS over time of the soil-moisture error (m$^3$/m$^3$)')
    ax[1, 1].set_xlabel('relative depth, 0 = surface, 1 = water table (–)')
    plots._finish(fig, path)


def fig_inverse(d, pred, ref_zb, zb_hyd, th_label, path):
    t, zb_obs = d['t_days'], d['zb_true']
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    ax[0].plot(t, d['theta_obs_hourly'], color=plots.INK, lw=2.2, label='observed')
    ax[0].plot(t, np.interp(t, pred['t_obs_days'], pred['theta_obs_pred']), color=plots.C_PRED, lw=1.1, label='PINN')
    ax[0].set_ylabel(th_label); ax[0].legend()
    ax[1].plot(t, zb_obs, color=plots.INK, lw=2.2, label='observed')
    ax[1].plot(t, np.clip(zb_hyd, 0.0, 1.5 * zb_obs.max()), color=plots.C_3, lw=1.1, label='hydrostatic reading')
    if ref_zb is not None:
        ax[1].plot(t, ref_zb, color=plots.C_TRUTH, lw=2.2, label='numerical inversion')
    ax[1].plot(t, pred['zb_pred'], color=plots.C_PRED, lw=1.1, label='PINN')
    ax[1].plot(t, pred['zb_pred'] - pred['zb_sigma'], color=plots.C_PRED, lw=0.7, ls=':', label='PINN ± 1 sd')
    ax[1].plot(t, pred['zb_pred'] + pred['zb_sigma'], color=plots.C_PRED, lw=0.7, ls=':')
    ax[1].invert_yaxis(); ax[1].set_ylabel('water-table depth (m)'); ax[1].legend(ncol=2)
    if ref_zb is not None:
        ax[2].plot(t, 100 * (ref_zb - zb_obs), color=plots.C_TRUTH, lw=2.2, label='numerical inversion − observed')
    ax[2].plot(t, 100 * (pred['zb_pred'] - zb_obs), color=plots.C_PRED, lw=1.1, label='PINN − observed')
    ax[2].axhline(0, color=plots.INK2, lw=0.6)
    ax[2].set_ylabel('water-table depth error (cm)'); ax[2].set_xlabel(T_LABEL); ax[2].legend()
    plots._finish(fig, path)


def fig_training(chains, path):
    """chains[k] = metrics of member k, one entry per stage; the steps run on from stage to stage. A stage starts on
    a new held-out set (its own sampling distribution), so the lines are broken there."""
    cols = [('val_pde', 'held-out PDE loss (–)'), ('val_flux', 'held-out surface-flux loss (–)'),
            ('val_box', 'held-out water-balance loss (–)')]
    cols = [(c, lab) for c, lab in cols if any(c in m and np.isfinite(m[c]).any() for ch in chains for m in ch)]
    fig, ax = plt.subplots(len(cols), 1, figsize=(7.5, 2.9 * len(cols)), sharex=True)
    ax = np.atleast_1d(ax)
    for k, ch in enumerate(chains):
        s0 = 0.0
        for j, m in enumerate(ch):
            for a, (c, _) in zip(ax, cols):
                if c in m and np.isfinite(m[c]).any():
                    a.semilogy(s0 + m['step'], m[c], color=MEMBER_COLORS[k % 4], lw=1.1,
                               label=f'member {k + 1}' if j == 0 else None)
            s0 += m['step'][-1]
    for a, (_, lab) in zip(ax, cols):
        a.set_ylabel(lab)
    ax[0].legend(ncol=4); ax[-1].set_xlabel('training step, all stages in sequence')
    plots._finish(fig, path)


def fig_stages(d, stages, path):
    t = d['t_days']
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.6))
    ax[0].plot(t, d['zb_true'], color=plots.INK, lw=2.2, label='observed')
    for col, s in zip(plots.SEQ_BLUE, stages):
        ax[0].plot(t, s['zb'], color=col, lw=1.1, label=s['stage'])
    ax[0].invert_yaxis(); ax[0].set_xlabel(T_LABEL); ax[0].set_ylabel('water-table depth (m)'); ax[0].legend(ncol=2)
    x = np.arange(len(stages))
    ax[1].semilogy(x, [100 * s['vs_well']['rmse_m'] for s in stages], 'o-', ms=4, color=plots.INK, label='from the well')
    if 'vs_fem' in stages[0]:
        ax[1].semilogy(x, [100 * s['vs_fem']['rmse_m'] for s in stages], 'o-', ms=4, color=plots.C_TRUTH,
                       label='from the numerical inversion')
    ax[1].set_xticks(x); ax[1].set_xticklabels([s['stage'] for s in stages])
    ax[1].set_xlabel('stage'); ax[1].set_ylabel('RMS distance of the estimate (cm)'); ax[1].legend()
    plots._finish(fig, path)


def fig_physics(phys, path):
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 8.8))
    ax[0].plot(phys[0]['t'], phys[0]['q_obs'], color=plots.INK, lw=2.2, label='prescribed')
    for k, ph in enumerate(phys):
        ax[0].plot(ph['t'], ph['q_net'], color=MEMBER_COLORS[k % 4], lw=0.9, label=f'member {k + 1}')
        ax[1].semilogy(ph['t'], np.sqrt(np.mean(ph['R'] ** 2, axis=1)), color=MEMBER_COLORS[k % 4], lw=1.1,
                       label=f'member {k + 1}')
        ax[2].semilogy(-ph['xi'], np.sqrt(np.mean(ph['R'] ** 2, axis=0)), color=MEMBER_COLORS[k % 4], lw=1.1,
                       label=f'member {k + 1}')
    ax[0].set_xlabel(T_LABEL); ax[0].set_ylabel('surface flux, upward positive (mm/d)'); ax[0].legend(ncol=3)
    ax[1].set_xlabel(T_LABEL); ax[1].set_ylabel('PDE residual / $\\sigma_{pde}$,\nRMS over depth (–)')
    ax[2].set_xlabel('relative depth, 0 = surface, 1 = water table (–)')
    ax[2].set_ylabel('PDE residual / $\\sigma_{pde}$,\nRMS over time (–)')
    ax[1].legend(ncol=4); ax[2].legend(ncol=4)
    plots._finish(fig, path)


def fig_seeds(d, base_zb, reps, path):
    t = d['t_days']
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax[0].plot(t, d['zb_true'], color=plots.INK, lw=2.2, label='observed')
    ax[0].plot(t, base_zb, color=MEMBER_COLORS[0], lw=1.1, label='first run')
    for k, (name, zb) in enumerate(reps):
        ax[0].plot(t, zb, color=MEMBER_COLORS[(k + 1) % 4], lw=1.1, label=name)
        ax[1].plot(t, 100 * (zb - base_zb), color=MEMBER_COLORS[(k + 1) % 4], lw=1.1, label=f'{name} − first run')
    ax[1].axhline(0, color=plots.INK2, lw=0.6)
    ax[0].invert_yaxis(); ax[0].set_ylabel('water-table depth (m)'); ax[0].legend(ncol=2)
    ax[1].set_ylabel('difference of the estimates (cm)'); ax[1].set_xlabel(T_LABEL); ax[1].legend()
    plots._finish(fig, path)


# ----------------------------------------------------------------------------- tables
def md_table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]) + '\n'


def cm(x):
    return f'{100.0 * x:.2f}'


def score_row(name, s):
    return [name, cm(s['rmse_m']), f"{100.0 * s['bias_m']:+.2f}", cm(s['maxabs_m']),
            '-' if s['r'] is None else f"{s['r']:.4f}"]


SCORE_HEAD = ['RMSE (cm)', 'bias (cm)', 'max (cm)', 'r']


def rms_max(s):
    return f"{s['rms']:.2e} ({s['max']:.2e})"


def reference_table(FR):
    """The numerical solution against itself (json of scripts/forward_reference_check.py): every level against the
    finest one and against the level before it. Where a refinement of the time step follows another by the same
    factor, the last column says how many times smaller the second change is (the factor itself for a first-order
    scheme in its asymptotic range). Returns the rows of the table and the same numbers for metrics.json."""
    rows, out, last = [], [], None                   # last: (step factor, sensor RMS) of the refinement before
    for i, L in enumerate(FR['levels']):
        prev = FR['consecutive'][i - 1] if i else None
        ratio = None
        if prev is not None:
            factor = FR['levels'][i - 1]['dt_h'] / L['dt_h']
            if factor > 1.0 + 1e-9:
                if last is not None and abs(last[0] - factor) < 1e-9:
                    ratio = last[1] / prev['sensor']['rms']
                last = (factor, prev['sensor']['rms'])
        v, finest = L['vs_finest'], i == len(FR['levels']) - 1
        rows.append([step_label(L['dt_h']), L['n_nodes'], L['max_iters'], f"{L['seconds']:.0f}",
                     '-' if finest else rms_max(v['sensor']), '-' if finest else rms_max(v['profile']),
                     rms_max(prev['sensor']) if prev else '-', rms_max(prev['profile']) if prev else '-',
                     f'{ratio:.2f}' if ratio is not None else '-'])
        out.append({'level': L['level'], 'dt_h': L['dt_h'], 'n_nodes': L['n_nodes'], 'converged': L['converged'],
                    'max_iters': L['max_iters'], 'seconds': L['seconds'],
                    'vs_finest': {'sensor': v['sensor'], 'profile': v['profile']},
                    'vs_level_before': {'sensor': prev['sensor'], 'profile': prev['profile']} if prev else None,
                    'change_before_over_this_change': ratio})
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory of a make_real.py data set')
    ap.add_argument('--tag', required=True, help='name of the hpc/real_full_obs.sh run')
    ap.add_argument('--final', required=True, help='directory of the inversion that gives the estimate')
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--members', type=int, default=4)
    ap.add_argument('--referee', nargs='*', default=[], help='directories of fem_referee.py results (one per start)')
    ap.add_argument('--forward', default=None, help='forward PINN at the spline of the observed water table')
    ap.add_argument('--forward-reference', default=None,
                    help='output stem of scripts/forward_reference_check.py: the forward PINNs are then also compared '
                         'with the numerical solution at its finest time step')
    ap.add_argument('--variants', nargs='*', default=[], help='name=dir of further forward PINNs (sensitivity)')
    ap.add_argument('--repeats', nargs='*', default=[], help='name=dir of the final inversion of repeated runs')
    ap.add_argument('--single', nargs='*', default=[], help='name=dir of inversions with one member each')
    ap.add_argument('--soil-note', default='', help='where the soil parameters come from (goes into the table)')
    ap.add_argument('--out', required=True, help='directory of the figures and tables')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = os.path.dirname(os.path.normpath(args.final))
    with open(os.path.join(args.data, 'twin_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    d = read_npz(os.path.join(args.data, 'real.npz'))
    soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
    t, zb_obs, th_obs = d['t_days'], d['zb_true'], d['theta_obs_hourly']
    k_fem = int(np.where(d['fem_water_table'] == 'spline')[0][0])
    th_label = layer_label(cfg['obs']['depth_m'])
    top, bottom = obs_layer(cfg['obs']['depth_m'])
    M, md = {'data': args.data, 'tag': args.tag, 'final': args.final}, []

    # ---- the data
    fig_setup(d, th_label, os.path.join(args.out, 'fig1_setup.png'))
    q = d['qsurf'] * MM_D
    M['setup'] = {
        'window_start': str(d['start_date']), 'n_days': float(d['n_days']), 'hours': int(t.size),
        'sensor_layer_m': [float(top), float(bottom)],
        'soil': {k: float(cfg['soil'][k]) for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')},
        'soil_note': args.soil_note, 'zb_observed_range_m': [float(zb_obs.min()), float(zb_obs.max())],
        'theta_observed_range': [float(th_obs.min()), float(th_obs.max())],
        'downward_surface_flux_mm': float(np.trapezoid(np.clip(-q, 0, None), t)),
        'upward_surface_flux_mm': float(np.trapezoid(np.clip(q, 0, None), t)),
        'root_uptake_mm': float(np.trapezoid(d['tp_daily_mmd'], t)),
        'n_coef': int(d['truth_coef'].size), 'spline_of_observed_vs_record': score(d['zb_spline'], zb_obs)}
    s = M['setup']
    md.append('## 1. Data\n\n' + md_table(['quantity', 'value'], [
        ['data set', args.data], ['window', f"{s['window_start']}, {s['n_days']:g} days, {s['hours']} hourly values"],
        ['soil-moisture sensor', f'{100 * top:g} to {100 * bottom:g} cm'],
        ['soil (van Genuchten)', ', '.join(f'{k} = {v:.5g}' for k, v in s['soil'].items())
         + (f'; {args.soil_note}' if args.soil_note else '')],
        ['observed water table', f"{s['zb_observed_range_m'][0]:.3f} to {s['zb_observed_range_m'][1]:.3f} m"],
        ['observed soil moisture', f"{s['theta_observed_range'][0]:.3f} to {s['theta_observed_range'][1]:.3f}"],
        ['surface flux over the window', f"{s['downward_surface_flux_mm']:.1f} mm downward, "
                                         f"{s['upward_surface_flux_mm']:.1f} mm upward"],
        ['root uptake over the window', f"{s['root_uptake_mm']:.1f} mm"],
        ['water-table path', f"cubic B-spline, {s['n_coef']} coefficients; the spline fitted to the well is "
                             f"{cm(s['spline_of_observed_vs_record']['rmse_m'])} cm RMS from the hourly record"]]))

    # ---- forward problem
    FR = fine = None
    if args.forward_reference:
        FR, fine = load_forward_reference(args.forward_reference, d, args.data, cfg)
    if args.forward:
        num, fw = forward_eval(args.forward, d, args.data, k_fem, fine)
        M['forward'] = num
        fig_forward(d, fw, k_fem, th_label, os.path.join(args.out, 'fig2_forward.png'), fine)
        head = '## 2. Forward problem: PINN against the numerical solution (water table given)\n\n'
        setup = [['run', args.forward],
                 ['network', f"{num['n_inputs']} inputs, {num['depth']} hidden layers of {num['width']}, tanh, "
                             f"{num['n_params']} parameters"],
                 ['training', f"{num['steps']} steps, {num['wall_s'] / 60:.0f} min on one GPU"]]
        physics = [['surface flux: RMS error / RMS of the prescribed flux', f"{100 * num['flux_rel_rms']:.2f} %"],
                   ['PDE residual / sigma_pde: RMS (max)', f"{num['residual_rms']:.3f} ({num['residual_max']:.2f})"],
                   ['held-out losses at the end: PDE, surface flux', f"{num['val_pde']:.4f}, {num['val_flux']:.4f}"]]
        if fine is None:
            md.append(head + md_table(['quantity', 'value'], setup + [
                ['soil moisture, all depths and times: RMS', f"{num['theta_rms']:.2e}"],
                ['soil moisture, all depths and times: relative L2', f"{100 * num['theta_rel_l2']:.3f} %"],
                ['soil moisture, all depths and times: RMS / range', f"{100 * num['theta_rms_over_range']:.3f} %"],
                ['soil moisture, all depths and times: max', f"{num['theta_max']:.2e}"],
                ['soil moisture of the sensor layer: RMS', f"{num['sensor_rms']:.2e}"],
                ['soil moisture of the sensor layer: max', f"{num['sensor_max']:.2e}"]] + physics))
        else:
            # the data set's own solve against the finest level, in the same measures as the network
            own = forward_errors(d['theta_fem'][k_fem], np.asarray(d['theta'], float), fine['sensor'], fine['theta'])[0]
            rows, levels = reference_table(FR)
            M['forward_reference'] = {
                'stem': args.forward_reference, 'finest': {k: fine[k] for k in ('dt_h', 'n_nodes', 'tol')},
                'picard': FR['checks']['picard'], 'first_level_vs_stored': FR['checks']['first_level_vs_stored'],
                'data_set_solve_vs_finest': own, 'levels': levels}
            cols = [num['vs_fine'], num, own]
            md.append(
                head + md_table(['quantity', 'value'], setup + physics)
                + f"\n\nSoil-moisture difference (m3/m3). The numerical solution of the data set has a "
                  f"{fine['own_label']} time step; the finest level of scripts/forward_reference_check.py has "
                  f"{fine['label']} ({fine['n_nodes']} nodes, Picard tolerance {fine['tol']:g}).\n\n"
                + md_table(['quantity', f"PINN − numerical, {fine['label']} step",
                            f"PINN − numerical, {fine['own_label']} step (the data set's)",
                            f"numerical, {fine['own_label']} step − {fine['label']} step"], [
                    ['all depths and times: RMS'] + [f"{e['theta_rms']:.2e}" for e in cols],
                    ['all depths and times: relative L2'] + [f"{100 * e['theta_rel_l2']:.2g} %" for e in cols],
                    ['all depths and times: RMS / range'] + [f"{100 * e['theta_rms_over_range']:.2g} %" for e in cols],
                    ['all depths and times: max'] + [f"{e['theta_max']:.2e}" for e in cols],
                    ['sensor layer: RMS'] + [f"{e['sensor_rms']:.2e}" for e in cols],
                    ['sensor layer: max'] + [f"{e['sensor_max']:.2e}" for e in cols]])
                + f"\n\nThe numerical solution against itself, soil moisture, RMS (max). Picard tolerance "
                  f"{FR['checks']['picard']['tol']:g} at every level (the data set's: "
                  f"{FR['checks']['picard']['tol_of_the_data_set']:g}); the first level differs from the solve "
                  f"stored in the data set by {FR['checks']['first_level_vs_stored']['sensor']['rms']:.1e} RMS "
                  f"at the sensor.\n\n"
                + md_table(['time step', 'nodes', 'Picard iterations, max', 'seconds',
                            'minus the finest level: sensor layer', 'all depths',
                            'minus the level before: sensor layer', 'all depths',
                            'change of the refinement before / this change (sensor layer)'], rows))

    # ---- inverse problem
    pred = read_npz(os.path.join(args.final, 'predictions.npz'))
    starts = read_npz(os.path.join(args.final, 'starts.npz'))
    fs = read_json(os.path.join(args.final, 'summary.json'))
    if not np.allclose(pred['t_days'], t) or not np.allclose(pred['zb_true'], zb_obs):
        raise SystemExit('the final inversion belongs to another data set')
    zb_hyd = (top + bottom) / 2.0 - soil.h_from_theta(th_obs)
    inv = {'pinn': score(pred['zb_pred'], zb_obs), 'hydrostatic_reading': score(zb_hyd, zb_obs),
           'spline_of_observed': score(d['zb_spline'], zb_obs),
           'start_spread_max_m': float(np.abs(starts['zb'] - pred['zb_pred'][None]).max()),
           'starts': int(starts['zb'].shape[0]), 'sigma_median_m': float(np.median(pred['zb_sigma'])),
           'theta_rmse_vs_obs': rms(pred['theta_obs_pred'] - pred['theta_obs'])}
    ref_zb, ref = None, None
    if args.referee:
        refs = [(read_json(os.path.join(p, 'summary.json')), read_npz(os.path.join(p, 'predictions.npz')), p)
                for p in args.referee]
        rs, ref, rp = min(refs, key=lambda x: x[0]['cost'])
        ref_zb = ref['zb_pred']
        inv.update({'numerical_inversion': score(ref_zb, zb_obs), 'pinn_vs_numerical': score(pred['zb_pred'], ref_zb),
                    'numerical_dir': rp, 'numerical_theta_rmse_vs_obs': rs['theta_rmse_vs_obs'],
                    'numerical_costs_by_start': {x[0]['start']: x[0]['cost'] for x in refs},
                    'numerical_spread_between_starts_max_m':
                        float(max(np.abs(x[1]['zb_pred'] - ref_zb).max() for x in refs))})
    M['inverse'] = inv
    fig_inverse(d, pred, ref_zb, zb_hyd, th_label, os.path.join(args.out, 'fig3_inverse.png'))
    rows = [score_row('PINN', inv['pinn'])]
    if ref is not None:
        rows.append(score_row('numerical inversion (FEM, same data and soil)', inv['numerical_inversion']))
    rows += [score_row('hydrostatic reading of the sensor (no flow)', inv['hydrostatic_reading']),
             score_row('spline fitted to the well (best the path can do)', inv['spline_of_observed'])]
    txt = ('## 3. Inverse problem: water-table depth against the well (hourly)\n\n'
           + md_table(['estimate'] + SCORE_HEAD, rows) + '\n')
    rows = [['misfit of the soil moisture at the estimate (RMS)', f"{inv['theta_rmse_vs_obs']:.5f}"],
            [f"largest difference between the {inv['starts']} starts", f"{100 * inv['start_spread_max_m']:.1e} cm"],
            ['posterior standard deviation (median)', f"{cm(inv['sigma_median_m'])} cm"]]
    if ref is not None:
        p = inv['pinn_vs_numerical']
        rows = [['PINN − numerical inversion', f"{cm(p['rmse_m'])} cm RMS, {cm(p['maxabs_m'])} cm max"]] + rows
        rows.append(['numerical inversion: misfit of the soil moisture (RMS)',
                     f"{inv['numerical_theta_rmse_vs_obs']:.5f}"])
        rows.append(['numerical inversion: largest difference between its starts',
                     f"{100 * inv['numerical_spread_between_starts_max_m']:.1e} cm"])
    md.append(txt + md_table(['quantity', 'value'], rows))

    # ---- stage by stage
    names = ['wide'] + [f'round {r}' for r in range(1, args.rounds + 1)] + ['value correction']
    dirs = [os.path.join(runs, f'{args.tag}_inv{r}') for r in range(args.rounds + 1)] + [args.final]
    stages = []
    for name, p in zip(names, dirs):
        if not os.path.isfile(os.path.join(p, 'predictions.npz')):
            print(f'[paper] no result in {p}: stage left out', flush=True)
            continue
        zb, ss = read_npz(os.path.join(p, 'predictions.npz'))['zb_pred'], read_json(os.path.join(p, 'summary.json'))
        st = {'stage': name, 'dir': p, 'zb': zb, 'vs_well': score(zb, zb_obs),
              'theta_rmse_vs_obs': ss['theta_rmse_vs_obs'], 'start_spread_rms_m': ss['start_spread_rms_m'],
              'sigma_median_m': ss['posterior_sigma_zb_median_m'], 'wall_s': ss.get('wall_s')}
        if ref_zb is not None:
            st['vs_fem'] = score(zb, ref_zb)
        stages.append(st)
    M['stages'] = [{k: v for k, v in s.items() if k != 'zb'} for s in stages]
    fig_stages(d, stages, os.path.join(args.out, 'fig5_stages.png'))
    md.append('## 4. The estimate stage by stage\n\n' + md_table(
        ['stage', 'from the well, RMS (cm)', 'from the numerical inversion, RMS (cm)', 'soil-moisture misfit',
         'between starts, RMS (cm)', 'posterior sd, median (cm)'],
        [[s['stage'], cm(s['vs_well']['rmse_m']), cm(s['vs_fem']['rmse_m']) if 'vs_fem' in s else '-',
          f"{s['theta_rmse_vs_obs']:.5f}", f"{100 * s['start_spread_rms_m']:.1e}", cm(s['sigma_median_m'])]
         for s in stages]))

    # ---- training history and cost
    chains, cost = [], []
    for k in range(args.members):
        ch = [os.path.join(runs, f'{args.tag}_w{k}')] + [os.path.join(runs, f'{args.tag}_r{r}_m{k}')
                                                         for r in range(1, args.rounds + 1)]
        chains.append([read_metrics(p) for p in ch if os.path.isfile(os.path.join(p, 'metrics.csv'))])
    fig_training(chains, os.path.join(args.out, 'fig4_training.png'))
    groups = [('wide', [os.path.join(runs, f'{args.tag}_w{k}') for k in range(args.members)])]
    groups += [(f'round {r}', [os.path.join(runs, f'{args.tag}_r{r}_m{k}') for k in range(args.members)])
               for r in range(1, args.rounds + 1)]
    groups.append(('forward networks of the value correction', list(fs.get('fwd', {}).get('members', []))))
    for name, ps in groups:
        ss = [read_json(os.path.join(p, 'summary.json')) for p in ps if os.path.isfile(os.path.join(p, 'summary.json'))]
        if ss:
            cost.append({'stage': name, 'networks': len(ss), 'steps_each': int(np.median([x['steps'] for x in ss])),
                         'gpu_hours': float(sum(x['wall_s'] for x in ss) / 3600.0),
                         'val_pde_median': float(np.median([x['val_pde'] for x in ss])),
                         'val_flux_median': float(np.median([x['val_flux'] for x in ss]))})
    M['cost'] = {'training': cost, 'training_gpu_hours': float(sum(c['gpu_hours'] for c in cost)),
                 'inversions_hours': float(sum(s['wall_s'] or 0.0 for s in stages) / 3600.0)}
    txt = '## 5. Training and cost\n\n' + md_table(
        ['stage', 'networks', 'steps each', 'GPU hours (all networks)', 'held-out PDE loss (median)',
         'held-out flux loss (median)'],
        [[c['stage'], c['networks'], c['steps_each'], f"{c['gpu_hours']:.2f}", f"{c['val_pde_median']:.4f}",
          f"{c['val_flux_median']:.4f}"] for c in cost]) + '\n'
    rows = [['training, all stages', f"{M['cost']['training_gpu_hours']:.1f} GPU hours"],
            ['inversions of all stages', f"{M['cost']['inversions_hours']:.2f} hours"]]
    if args.referee:
        M['cost']['numerical_inversion'] = {'wall_s_best_start': rs['wall_s'], 'iterations_best_start': rs['iterations'],
                                            'cpu_hours_all_starts': float(sum(x[0]['wall_s'] for x in refs) / 3600.0)}
        rows.append(['numerical inversion', f"{rs['wall_s'] / 60:.0f} min for the start with the lowest cost "
                                            f"({rs['iterations']} iterations); "
                                            f"{M['cost']['numerical_inversion']['cpu_hours_all_starts']:.2f} CPU hours "
                                            f"for all {len(refs)} starts"])
    md.append(txt + md_table(['item', 'cost'], rows))

    # ---- physics of the final members at the estimate
    phys, rows, M['physics'] = [], [], []
    for k, p in enumerate(fs['members']):
        m, mc = load_member(p, args.data)
        ph = physics_at(m, mc, pred['coef'], float(d['n_days']))
        phys.append(ph)
        M['physics'].append({'network': p, 'kind': 'operator member', **physics_numbers(ph)})
        rows.append([f'operator member {k + 1}', f"{100 * ph['flux_rel_rms']:.2f} %", f"{ph['residual_rms']:.3f}",
                     f"{ph['residual_max']:.2f}"])
    for k, p in enumerate(fs.get('fwd', {}).get('members', [])):
        m, mc = load_member(p, args.data)
        ph = physics_at(m, mc, fs['fwd']['c_fwd'], float(d['n_days']))
        M['physics'].append({'network': p, 'kind': 'forward network of the value correction', **physics_numbers(ph)})
        rows.append([f'forward network {k + 1} of the value correction', f"{100 * ph['flux_rel_rms']:.2f} %",
                     f"{ph['residual_rms']:.3f}", f"{ph['residual_max']:.2f}"])
    fig_physics(phys, os.path.join(args.out, 'fig6_physics.png'))
    md.append('## 6. Physics of the trained networks at the estimate\n\n' + md_table(
        ['network', 'surface flux: RMS error / RMS of the prescribed', 'PDE residual / sigma_pde: RMS', 'max'], rows))

    # ---- optional: sensitivity of the forward PINN, repeated runs, single members
    if args.variants:
        rows, M['forward_variants'] = [], {}
        base = [('baseline', args.forward)] if args.forward else []
        for name, p in base + pairs(args.variants):
            if not os.path.isfile(os.path.join(p, 'model.pt')):
                print(f'[paper] no model in {p}: variant {name} left out', flush=True)
                continue
            if p == args.forward:
                num, apart = M['forward'], '-'
            else:
                num, fwv = forward_eval(p, d, args.data, k_fem, fine)
                apart = '-'
                if args.forward:                             # this network minus the baseline network
                    e5, ep = fwv['th5'] - fw['th5'], fwv['prof'] - fw['prof']
                    num['minus_baseline'] = {'sensor_rms': rms(e5), 'sensor_max': float(np.abs(e5).max()),
                                             'theta_rms': rms(ep), 'theta_max': float(np.abs(ep).max())}
                    apart = f"{rms(e5):.2e}"
            M['forward_variants'][name] = num
            rows.append([name, num['width'], num['depth'], num['n_col'], num['seed'], num['n_params']]
                        + ([f"{num['vs_fine']['theta_rms']:.2e}", f"{num['vs_fine']['sensor_rms']:.2e}"]
                           if fine is not None else [])
                        + [f"{num['theta_rms']:.2e}", f"{num['sensor_rms']:.2e}", apart,
                           f"{100 * num['flux_rel_rms']:.2f} %", f"{num['residual_rms']:.3f}",
                           f"{num['wall_s'] / 60:.0f}"])
        own_step = f", {step_label(cfg['fem']['dt_hours'])} step (the data set's)" if fine is not None else ''
        md.append('## 7. Forward PINN: training seed, network size, collocation points\n\n'
                  'Soil moisture RMS against the numerical solution (m3/m3).\n\n' + md_table(
            ['run', 'width', 'hidden layers', 'collocation points', 'seed', 'parameters']
            + ([f"all depths, {fine['label']} step", f"sensor layer, {fine['label']} step"] if fine is not None else [])
            + ['all depths' + own_step, 'sensor layer' + own_step, 'sensor layer, minus the baseline network',
               'surface flux', 'PDE residual RMS', 'minutes'], rows))
    if args.repeats:
        rows, reps, M['repeats'] = [score_row('first run', inv['pinn']) + ['-', '-']], [], {}
        for name, p in pairs(args.repeats):
            if not os.path.isfile(os.path.join(p, 'predictions.npz')):
                print(f'[paper] no result in {p}: repeat {name} left out', flush=True)
                continue
            zb = read_npz(os.path.join(p, 'predictions.npz'))['zb_pred']
            M['repeats'][name] = {'dir': p, 'vs_well': score(zb, zb_obs), 'vs_first_run': score(zb, pred['zb_pred']),
                                  'start_spread_rms_m': read_json(os.path.join(p, 'summary.json'))['start_spread_rms_m']}
            if ref_zb is not None:
                M['repeats'][name]['vs_numerical'] = score(zb, ref_zb)
            r = M['repeats'][name]
            rows.append(score_row(name, r['vs_well']) + [cm(r['vs_numerical']['rmse_m']) if ref_zb is not None else '-',
                                                         f"{cm(r['vs_first_run']['rmse_m'])} "
                                                         f"({cm(r['vs_first_run']['maxabs_m'])})"])
            reps.append((name, zb))
        if reps:
            fig_seeds(d, pred['zb_pred'], reps, os.path.join(args.out, 'fig7_seeds.png'))
        md.append('## 8. Repeated runs with other training seeds (against the well)\n\n' + md_table(
            ['run'] + SCORE_HEAD + ['from the numerical inversion, RMS (cm)', 'from the first run, RMS (max) (cm)'],
            rows))
    if args.single:
        rows, M['single_members'] = [], {}
        base = read_npz(os.path.join(runs, f'{args.tag}_inv{args.rounds}', 'predictions.npz'))['zb_pred']
        for name, p in pairs(args.single):
            if not os.path.isfile(os.path.join(p, 'predictions.npz')):
                print(f'[paper] no result in {p}: {name} left out', flush=True)
                continue
            zb = read_npz(os.path.join(p, 'predictions.npz'))['zb_pred']
            M['single_members'][name] = {'dir': p, 'vs_well': score(zb, zb_obs), 'vs_all_members': score(zb, base)}
            r = M['single_members'][name]
            rows.append(score_row(name, r['vs_well']) + [f"{cm(r['vs_all_members']['rmse_m'])} "
                                                         f"({cm(r['vs_all_members']['maxabs_m'])})"])
        md.append(f'## 9. Inversion with one member of round {args.rounds} at a time (against the well)\n\n' + md_table(
            ['member'] + SCORE_HEAD + [f'from the estimate with all members, round {args.rounds}, RMS (max) (cm)'],
            rows))

    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(M, f, indent=1)
    with open(os.path.join(args.out, 'tables.md'), 'w') as f:
        f.write(f'# Results tables: {args.tag} on {args.data}\n\nWritten by scripts/paper_figures.py; every number '
                f'is in metrics.json.\n\n' + '\n'.join(md))
    print(json.dumps(M, indent=1), flush=True)
    print(f'[paper] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
