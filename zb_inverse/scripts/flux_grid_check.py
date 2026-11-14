#!/usr/bin/env python
"""Does the time grid on which the surface-flux error and the PDE residual are evaluated change the numbers?
(scoring only)

For the networks behind one estimate (operator members at the estimate, forward networks of the value correction
at their path; with --forward, forward networks trained at the path of the record), the surface flux of the network
minus the prescribed flux, as a fraction of the RMS of the prescribed flux, on uniform time grids of several steps,
at the forcing nodes only, half-way between them, on log-uniform times in the first day (the early-time sample of
training, train.frac_early_t), and by day on the finest grid. The prescribed flux is linear between the forcing
nodes, as in training (parametric.ParamTrainer.q_surf) and in scripts/paper_figures.py (physics_at). With --residual
also the PDE residual in units of sigma_pde on (t, xi), xi as in physics_at, for several time grids, split into the
rows at forcing-node times and the others, the row t = 0, the first hour and the time after it. The network has
zero surface flux at t = 0 by construction, so the first minutes are also given row by row (FIRST_HOUR_S) for the
flux error and the residual. Nothing here is read by a PINN run.

Usage: python -u scripts/flux_grid_check.py --data output/real_nc2_sep23c_own --final runs/rfnc1dc_it1 \
           --out output/paper/nc2_sep23c/flux_grid
"""
import argparse
import json
import os
import numpy as np
import torch

from paper_figures import load_member, read_json, read_npz, rms, MM_D
from zbinv.parametric import physics_scales
from zbinv.physics import pde_residual, surface_flux
from zbinv.train import load_dataset

PER_HOUR = (1, 2, 4, 12, 60, 600)
RES_PER_HOUR = (2, 12, 60)
FIRST_HOUR_S = (0, 1, 6, 20, 60, 120, 300, 600, 1200, 1800, 3600)     # times [s] of the start-corner rows


def flux_error(m, c, t, t_q, q_obs):
    """Network minus prescribed surface flux [mm/d] at the times t [d], and the prescribed flux there."""
    q_net = np.concatenate([surface_flux(m, torch.as_tensor(t[i:i + 8192]).reshape(-1, 1), c).detach().numpy().ravel()
                            for i in range(0, t.size, 8192)]) * m.soil.Ks * MM_D
    q = np.interp(t, t_q, q_obs) * MM_D
    return q_net - q, q


def residual_on(m, cfg, c, n_days, nodes, res_per_hour=RES_PER_HOUR, nxi=101):
    """PDE residual / sigma_pde on (t, xi). For the time grids res_per_hour: RMS, max and where it is, the RMS of
    the rows at forcing-node times and of the other rows, of the row t = 0, of the rows of the first hour and of
    the rows after it, and the time mean by the trapezoid rule. By day on the finest grid. The rows of the first
    hour at the times FIRST_HOUR_S (one row each, RMS over xi)."""
    _, sig_pde = physics_scales(cfg, load_dataset(cfg))
    xi = -(1.0 - np.linspace(0.0, 1.0, nxi)) ** 1.5

    def rows(t):
        X, T = np.meshgrid(xi, t, indexing='xy')                               # (nt, nxi)
        xs, ts = torch.as_tensor(X.reshape(-1, 1)), torch.as_tensor(T.reshape(-1, 1))
        return np.concatenate([pde_residual(m, xs[i:i + 8192], ts[i:i + 8192], c).detach().numpy().ravel()
                               for i in range(0, xs.shape[0], 8192)]).reshape(T.shape) / sig_pde

    out = {'grids': {}}
    for per_hour in res_per_hour:
        t = np.linspace(0.0, n_days, int(round(24 * per_hour * n_days)) + 1)
        R = rows(t)
        ms = np.mean(R ** 2, axis=1)                                           # mean square over xi, row by row
        at_node = np.abs(t[:, None] - nodes[None, :]).min(1) < 1e-9
        late = t >= 1.0 / 24.0 - 1e-12
        k, j = np.unravel_index(int(np.argmax(np.abs(R))), R.shape)
        out['grids'][str(per_hour)] = {
            'rms': rms(R), 'max': float(np.abs(R).max()), 'max_at_t_days': float(t[k]), 'max_at_xi': float(xi[j]),
            'rows_at_nodes': int(at_node.sum()), 'rms_at_node_times': rms(R[at_node]),
            'rms_other_times': rms(R[~at_node]), 'rms_row_t0': float(np.sqrt(ms[0])),
            'rms_at_node_times_without_t0': rms(R[at_node & (t > 0.0)]),
            'rms_first_hour': float(np.sqrt(ms[~late].mean())), 'rms_after_first_hour': float(np.sqrt(ms[late].mean())),
            'max_after_first_hour': float(np.abs(R[late]).max()),
            'rms_time_trapezoid': float(np.sqrt(np.trapezoid(ms, t) / n_days))}
    day = np.minimum(np.floor(t).astype(int), int(np.ceil(n_days)) - 1)        # t, ms: the finest grid
    out['by_day_rms'] = [float(np.sqrt(ms[day == k].mean())) for k in range(int(day.max()) + 1)]
    t1 = np.asarray(FIRST_HOUR_S, float) / 86400.0
    out['first_hour'] = {'t_s': list(FIRST_HOUR_S),
                         'row_rms': [float(v) for v in np.sqrt(np.mean(rows(t1) ** 2, axis=1))]}
    return out


def one_network(m, cfg, c, n_days, residual=False, res_per_hour=RES_PER_HOUR):
    ds = load_dataset(cfg)
    t_q, q_obs = np.asarray(ds['t_q'], float), np.asarray(ds['q_obs'], float)
    cc = torch.as_tensor(np.asarray(c, float))
    out = {'forcing_nodes': int(t_q.size), 'grids': {}}
    for per_hour in PER_HOUR:
        t = np.linspace(0.0, n_days, int(round(24 * per_hour * n_days)) + 1)
        e, q = flux_error(m, cc, t, t_q, q_obs)
        out['grids'][str(per_hour)] = {'points': int(t.size), 'err_rms_mmd': rms(e), 'q_rms_mmd': rms(q),
                                       'rel': rms(e) / rms(q)}
    day = np.minimum(np.floor(t).astype(int), int(np.ceil(n_days)) - 1)        # t, e, q: the finest grid
    out['by_day_err_rms_mmd'] = [rms(e[day == k]) for k in range(int(day.max()) + 1)]
    out['by_day_q_rms_mmd'] = [rms(q[day == k]) for k in range(int(day.max()) + 1)]
    q_fine = rms(q)
    nodes = t_q[(t_q >= 0.0) & (t_q <= n_days)]
    e_n, _ = flux_error(m, cc, nodes, t_q, q_obs)
    e_m, _ = flux_error(m, cc, 0.5 * (nodes[1:] + nodes[:-1]), t_q, q_obs)
    out['at_nodes'] = {'points': int(nodes.size), 'err_rms_mmd': rms(e_n), 'rel_to_fine_q_rms': rms(e_n) / q_fine}
    out['between_nodes'] = {'points': int(nodes.size - 1), 'err_rms_mmd': rms(e_m),
                            'rel_to_fine_q_rms': rms(e_m) / q_fine}
    e_e, _ = flux_error(m, cc, np.logspace(-3.0, 0.0, 601), t_q, q_obs)             # [1e-3, 1] d, log-uniform
    out['early'] = {'points': 601, 'err_rms_mmd': rms(e_e), 'rel_to_fine_q_rms': rms(e_e) / q_fine}
    e_1, q_1 = flux_error(m, cc, np.asarray(FIRST_HOUR_S, float) / 86400.0, t_q, q_obs)
    out['first_hour'] = {'t_s': list(FIRST_HOUR_S), 'err_mmd': [float(v) for v in e_1],
                         'q_mmd': [float(v) for v in q_1]}
    late = t >= 1.0 / 24.0 - 1e-12                                              # t, e, q: the finest grid
    out['after_first_hour'] = {'err_rms_mmd': rms(e[late]), 'rel_to_fine_q_rms': rms(e[late]) / q_fine}
    if residual:
        out['residual'] = residual_on(m, cfg, cc, n_days, nodes, res_per_hour)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory of a make_real.py data set')
    ap.add_argument('--final', help='directory of the inversion that gives the estimate')
    ap.add_argument('--forward', nargs='*', default=[], metavar='RUN',
                    help='forward networks trained at the path of the record (truth_coef of the data set)')
    ap.add_argument('--residual', action='store_true', help='also the PDE residual on several time grids')
    ap.add_argument('--res-per-hour', type=int, nargs='+', default=list(RES_PER_HOUR),
                    help='time grids of the PDE residual, rows per hour, coarsest first')
    ap.add_argument('--out', required=True, help='path stem of the output (.json)')
    args = ap.parse_args()
    d = read_npz(os.path.join(args.data, 'real.npz'))
    n_days = float(d['n_days'])
    nets = []
    if args.final:
        pred = read_npz(os.path.join(args.final, 'predictions.npz'))
        fs = read_json(os.path.join(args.final, 'summary.json'))
        nets += [('operator member', p, pred['coef']) for p in fs['members']]
        nets += [('forward network of the value correction', p, fs['fwd']['c_fwd'])
                 for p in fs.get('fwd', {}).get('members', [])]
    nets += [('forward network at the path of the record', p, d['truth_coef']) for p in args.forward]
    res = {'data': args.data, 'final': args.final, 'n_days': n_days, 'networks': []}
    where = ('at_nodes', 'between_nodes', 'early')
    print(f"{'network':34s} " + ' '.join(f'{k:>6d}/h' for k in PER_HOUR) + ' '
          + ' '.join(f"{w.split('_')[-1] if w != 'between_nodes' else 'between':>8s}" for w in where)
          + '   (RMS error / RMS prescribed, %)')
    for kind, p, c in nets:
        m, cfg = load_member(p, args.data)
        r = {'network': p, 'kind': kind, **one_network(m, cfg, c, n_days, args.residual, args.res_per_hour)}
        res['networks'].append(r)
        print(f"{os.path.basename(p):34s} " + ' '.join(f"{100 * r['grids'][str(k)]['rel']:8.2f}" for k in PER_HOUR)
              + ' ' + ' '.join(f"{100 * r[w]['rel_to_fine_q_rms']:8.2f}" for w in where), flush=True)
        print('    by day, error rms (mm/d): ' + ' '.join(f'{v:.3f}' for v in r['by_day_err_rms_mmd']))
        print('    by day, prescribed rms (mm/d): ' + ' '.join(f'{v:.3f}' for v in r['by_day_q_rms_mmd']))
        print('    first hour, flux error (mm/d) at t = ' + ', '.join(f'{s} s' for s in FIRST_HOUR_S) + ': '
              + ' '.join(f'{v:.3f}' for v in r['first_hour']['err_mmd'])
              + '; after the first hour, rms error / rms prescribed '
              + f"{100 * r['after_first_hour']['rel_to_fine_q_rms']:.2f} %")
        if args.residual:
            g = r['residual']['grids']
            print('    PDE residual / sigma_pde, rms (max) [rows at node times | other rows]: ' + '; '.join(
                f"{k}/h {v['rms']:.3f} ({v['max']:.1f}) [{v['rms_at_node_times']:.3f} | {v['rms_other_times']:.3f}]"
                for k, v in g.items()))
            print('    PDE residual: row t = 0 | node rows without it | first hour | after the first hour (max) | '
                  'time mean, trapezoid: ' + '; '.join(
                      f"{k}/h {v['rms_row_t0']:.2f} | {v['rms_at_node_times_without_t0']:.3f} | "
                      f"{v['rms_first_hour']:.3f} | {v['rms_after_first_hour']:.3f} "
                      f"({v['max_after_first_hour']:.1f}) | {v['rms_time_trapezoid']:.3f}"
                      for k, v in g.items()))
            v = g[str(args.res_per_hour[-1])]
            print(f"    PDE residual: max at t = {v['max_at_t_days']:.4f} d, xi = {v['max_at_xi']:.4f}; by day (rms): "
                  + ' '.join(f'{x:.3f}' for x in r['residual']['by_day_rms']))
            print('    PDE residual, rows of the first hour (rms over xi) at t = '
                  + ', '.join(f'{s} s' for s in FIRST_HOUR_S) + ': '
                  + ' '.join(f'{x:.2f}' for x in r['residual']['first_hour']['row_rms']), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
