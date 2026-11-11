#!/usr/bin/env python
"""Does the gradient that reaches zb point the right way?

At a PINN state (network + spline coefficients c) compare
  J_net = d theta_5cm(t_obs) / d c  with the network FROZEN (what the joint optimiser sees for zb),
  J_ref = the same derivative through the reference solver (the true, total sensitivity),
both by central differences (step `delta` in coefficient units), plus the data-misfit gradients and the
Gauss-Newton zb step each implies. A wrong-signed J_net explains zb drifting away from the truth.

States: a trained run (`--run runs/<run>`) or a fresh physics-only stage 1 from a config
(`--stage1 <config.yaml> [--set ...]`), i.e. the state at the start of the joint stage.

Usage: python -u scripts/sensitivity_check.py --run runs/v8_autofix_clean --out output/compare/sens_clean.json
       python -u scripts/sensitivity_check.py --stage1 runs/v3_e2_fixed_init_deep/config.yaml --out ...
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_reference  # noqa: E402
from zbinv.train import Trainer  # noqa: E402
from zbinv.physics import theta_obs_model  # noqa: E402
from zbinv.spline import ZbSplineNumpy  # noqa: E402

_G = {}


def ref_forward(coef):
    g = _G
    m = ZbSplineNumpy(g['spline'], coef, g['zb_min'])
    return run_reference(g['tcfg'], g['soil'], g['win'], m, g['t_obs'], n_nodes=401, verbose=False)['theta_obs']


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--stage1')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--delta', type=float, default=0.01)
    args = ap.parse_args()
    cfg_path = os.path.join(args.run, 'config.yaml') if args.run else args.stage1
    cfg = yaml.safe_load(open(cfg_path))
    for item in args.set:
        key, val = item.split('=', 1)
        node = cfg
        for p in key.split('.')[:-1]:
            node = node.setdefault(p, {})
        node[key.split('.')[-1]] = yaml.safe_load(val)
    cfg['output']['dir'] = '/tmp/sensitivity_scratch'
    tr = Trainer(cfg)
    if args.run:
        tr.model.load_state_dict(torch.load(os.path.join(args.run, 'model.pt'), map_location=tr.dev))
    else:
        tr.adam_stage('adam1', cfg['train']['stage1_steps'], joint=False, use_data=False, decay=False)
    ds, sp = tr.ds, tr.spline
    coef = tr.model.zb_spline.coef.detach().cpu().numpy().copy()
    depth = cfg['obs']['depth_m']
    sig = tr.sig_th
    y = ds['theta_obs']

    def net_theta(c):
        with torch.no_grad():
            tr.model.zb_spline.coef.copy_(torch.tensor(c, device=tr.dev))
            return theta_obs_model(tr.model, tr.t_obs_t, depth).cpu().numpy().ravel()

    th_net = net_theta(coef)
    J_net = np.stack([(net_theta(coef + args.delta * e) - net_theta(coef - args.delta * e)) / (2 * args.delta)
                      for e in np.eye(sp.n_coef)], axis=1)
    net_theta(coef)

    tcfg = dict(ds['twin_cfg'])
    tcfg['obs'] = dict(tcfg['obs'], depth_m=depth)
    _G.update(spline=sp, zb_min=ds['zb_min'], tcfg=tcfg, soil=ds['soil'],
              win={'t_days': ds['t_q'], 'qsurf': ds['q_obs']}, t_obs=ds['t_obs'])
    jobs = [coef] + [coef + s * args.delta * e for e in np.eye(sp.n_coef) for s in (1.0, -1.0)]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))) as pool:
        res = pool.map(ref_forward, jobs)
    th_ref = res[0]
    J_ref = np.stack([(res[1 + 2 * k] - res[2 + 2 * k]) / (2 * args.delta) for k in range(sp.n_coef)], axis=1)

    # zb change per unit coefficient (positive = deeper), at the knot centres
    m = ZbSplineNumpy(sp, coef, ds['zb_min'])
    r_net = (th_net - y) / sig
    g_net = J_net.T @ r_net / sig                     # gradient of 0.5 sum r^2 seen by the joint optimiser
    g_ref = J_ref.T @ r_net / sig                     # same residual, true sensitivity
    G = m.dzb_dcoef(ds['t_hourly'])

    def gn_step(J, r):
        A = J.T @ J / sig ** 2
        return -np.linalg.solve(A + 1e-6 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), J.T @ r / sig)

    st_net, st_ref = gn_step(J_net, r_net), gn_step(J_ref, r_net)
    out = {
        'state': args.run or f'stage1:{args.stage1} {args.set}',
        'zb_mean_m': float(m.zb(ds['t_hourly']).mean()), 'zb_true_mean_m': float(ds['zb_true'].mean()),
        'theta_net_minus_obs_mean': float((th_net - y).mean()),
        'theta_net_minus_ref_rms': float(np.sqrt(np.mean((th_net - th_ref) ** 2))),
        'col_cos_net_vs_ref': [cos(J_net[:, k], J_ref[:, k]) for k in range(sp.n_coef)],
        'col_norm_ratio_net_over_ref': [float(np.linalg.norm(J_net[:, k]) / np.linalg.norm(J_ref[:, k]))
                                        for k in range(sp.n_coef)],
        'col_mean_net': [float(J_net[:, k].mean()) for k in range(sp.n_coef)],
        'col_mean_ref': [float(J_ref[:, k].mean()) for k in range(sp.n_coef)],
        'grad_cos_net_vs_ref': cos(g_net, g_ref),
        'gn_step_mean_dzb_m_net': float((G @ st_net).mean()),
        'gn_step_mean_dzb_m_ref': float((G @ st_ref).mean()),
        'truth_minus_current_mean_m': float((ds['zb_true'] - m.zb(ds['t_hourly'])).mean()),
    }
    print(f"[sens] {json.dumps(out)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
