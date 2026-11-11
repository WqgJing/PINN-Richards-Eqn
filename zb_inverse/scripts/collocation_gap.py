#!/usr/bin/env python
"""Does a trained run satisfy the PDE only at the points it trained on?

For each run: normalised PDE loss mean((R / sigma_pde)^2) on
  - the last logged training batch (from metrics.csv: points the network had already seen),
  - fresh points from the same sampler (RAR-D) and from the base sampler, several draws,
  - the fixed uniform validation set,
  - the top 10 cm at the hourly data times vs half an hour later (is physics bent where data sit?).
A fresh-point loss far above the logged one means the network fits the data by violating the PDE
between collocation points (collocation over-fitting).

Usage: python scripts/collocation_gap.py runs/<run> [runs/<run> ...]
"""
import csv
import json
import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.train import Trainer  # noqa: E402
from zbinv.physics import pde_residual  # noqa: E402


def ms(tr, xi, t, chunk=8192):
    out = []
    for i in range(0, xi.shape[0], chunk):
        with torch.enable_grad():
            out.append(pde_residual(tr.model, xi[i:i + chunk], t[i:i + chunk]).detach() / tr.sig_pde)
    return float((torch.cat(out) ** 2).mean())


def last_logged(run):
    rows = list(csv.DictReader(open(os.path.join(run, 'metrics.csv'))))
    out = {}
    for r in rows:
        if r.get('L_pde') not in (None, '', 'nan'):
            out[r['stage']] = float(r['L_pde'])
    return out


def main():
    torch.manual_seed(7)
    for run in sys.argv[1:]:
        cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
        cfg['output']['dir'] = '/tmp/collocation_gap_scratch'
        tr = Trainer(cfg)
        tr.model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location=tr.dev))
        js = json.load(open(os.path.join(run, 'summary.json')))
        tr.sig_pde = float(js.get('sig_pde', tr.sig_pde))
        n = cfg['train']['n_col']
        fresh_rar = [ms(tr, *tr.collocation(n)) for _ in range(5)]
        fresh_base = [ms(tr, *tr.sampler.collocation(n)) for _ in range(5)]
        # top 10 cm (xi > -0.1 / zb ~ 0.17 for zb ~ 0.6 m) at data times vs half an hour later
        th = torch.tensor(tr.ds['t_obs'], device=tr.dev).reshape(-1, 1)
        th = th[(th > 1.0).ravel()]
        xi_top = -0.17 * torch.rand(th.shape[0] * 8, 1, device=tr.dev)
        t_on = th.repeat(8, 1)
        t_off = t_on + 0.5 / 24.0
        out = {
            'sig_pde': tr.sig_pde,
            'logged_last_by_stage': last_logged(run),
            'fresh_rar_mean': float(np.mean(fresh_rar)), 'fresh_rar_draws': fresh_rar,
            'fresh_base_mean': float(np.mean(fresh_base)),
            'uniform_val': ms(tr, tr.val_xi, tr.val_t),
            'top10cm_at_data_times': ms(tr, xi_top, t_on),
            'top10cm_half_hour_later': ms(tr, xi_top, t_off),
            'zb_rmse_inner': js.get('zb_rmse_inner'), 'data_rmse': js.get('data_rmse'),
        }
        print(f"[gap] {run}: {json.dumps(out)}", flush=True)
        with open(os.path.join(run, 'collocation_gap.json'), 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
