#!/usr/bin/env python
"""Where is the PDE residual of a trained run? Line plots of RMS |R| vs time and vs xi, plus the
share of the total squared residual in the first day, near the surface and near the water table.

Usage: python scripts/residual_map.py runs/<run> [runs/<run> ...]
"""
import os
import sys
import json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.train import Trainer  # noqa: E402
from zbinv.physics import pde_residual  # noqa: E402

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def residual_grid(tr, nxi=201, nt=481):
    s = np.linspace(0.0, 1.0, nxi)
    xi = -(1.0 - s) ** 1.5                      # clustered at the surface, as the reference grid
    t = np.linspace(0.0, tr.ds['n_days'], nt)
    X, Tt = np.meshgrid(xi, t, indexing='xy')
    R = []
    xs = torch.tensor(X.reshape(-1, 1), device=tr.dev)
    ts = torch.tensor(Tt.reshape(-1, 1), device=tr.dev)
    for i in range(0, xs.shape[0], 8192):
        R.append(pde_residual(tr.model, xs[i:i + 8192], ts[i:i + 8192]).detach().cpu().numpy().ravel())
    return xi, t, np.concatenate(R).reshape(Tt.shape) / tr.sig_pde


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, run in enumerate(sys.argv[1:]):
        cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
        cfg['output']['dir'] = '/tmp/residual_map_scratch'
        tr = Trainer(cfg)
        tr.model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location=tr.dev))
        xi, t, Rn = residual_grid(tr)
        r2 = Rn ** 2
        tot = r2.sum()
        out = {
            'rms_norm': float(np.sqrt(r2.mean())),
            'share_t_lt_1d': float(r2[t < 1.0].sum() / tot),
            'share_xi_gt_-0.05': float(r2[:, xi > -0.05].sum() / tot),
            'share_xi_lt_-0.9': float(r2[:, xi < -0.9].sum() / tot),
            'rms_norm_excl_first_day': float(np.sqrt(r2[t >= 1.0].mean())),
        }
        i, j = np.unravel_index(np.argmax(np.abs(Rn)), Rn.shape)
        out['max_abs_norm'] = float(np.abs(Rn[i, j]))
        out['argmax_t_days'], out['argmax_xi'] = float(t[i]), float(xi[j])
        print(f"[resmap] {run}: {json.dumps(out)}", flush=True)
        name = os.path.basename(run.rstrip('/'))
        axes[0].semilogy(t, np.sqrt(r2.mean(axis=1)), color=C[k % 4], lw=1.5, label=name)
        axes[1].semilogy(xi, np.sqrt(r2.mean(axis=0)), color=C[k % 4], lw=1.5, label=name)
        with open(os.path.join(run, 'residual_map.json'), 'w') as f:
            json.dump(out, f, indent=2)
    axes[0].set_xlabel('days from window start'); axes[0].set_ylabel('RMS over xi of R / sigma_pde')
    axes[1].set_xlabel('xi (-1 = water table, 0 = surface)'); axes[1].set_ylabel('RMS over t of R / sigma_pde')
    for a in axes:
        a.grid(True, alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(sys.argv[1], 'residual_map.png')
    fig.savefig(out_png, dpi=130)
    print(f"[resmap] saved {out_png}", flush=True)


if __name__ == '__main__':
    main()
