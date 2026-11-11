"""Print the range of the twin state: surface head, near-surface gradients, K, flux scales.
Usage: python scripts/inspect_twin.py output/<twin>/twin.npz [...]"""
import sys
import os
import numpy as np
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402

for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(path), 'twin_config.yaml')))
    soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
    L = float(d['L'])
    h = d['h_t'] * L                      # (nt, nx) metres
    xi = d['xi']
    zb = d['zb_true']
    print(f'== {path}')
    print('keys', sorted(d.files))
    print(f'h_t shape {d["h_t"].shape}, xi[0..2]={xi[:3]}, xi[-3:]={xi[-3:]}')
    hs = h[:, -1] if xi[-1] > xi[0] else h[:, 0]
    print(f'surface h [m]: min {hs.min():.3f}  p5 {np.percentile(hs, 5):.3f}  median {np.median(hs):.3f}  max {hs.max():.3f}')
    print(f'whole-profile h [m]: min {h.min():.3f}; h/L min {h.min() / L:.3f}')
    for k in np.linspace(0, len(zb) - 1, 7).astype(int):
        z = xi * zb[k]
        dz = np.diff(z)
        dhdz = np.diff(h[k]) / dz
        print(f'  t={d["t_days"][k]:6.2f} d zb={zb[k]:.3f} q={d["qsurf"][k] * 86400e3 if d["qsurf"].shape == zb.shape else float("nan"):7.3f} mm/d '
              f'h_surf={hs[k]:9.3f} m  max|dh/dz|={np.abs(dhdz).max():9.2f}  theta_surf={d["theta"][k, -1]:.3f}')
    Ksurf = soil.K(hs / L) * soil.Ks * 86400e3
    print(f'surface K [mm/d]: min {Ksurf.min():.3e} median {np.median(Ksurf):.3e}')
    print(f'theta_obs hourly range {d["theta_obs_hourly"].min():.4f} {d["theta_obs_hourly"].max():.4f}')
    print(f'q (mm/d) range {d["qsurf"].min() * 86400e3:.2f} {d["qsurf"].max() * 86400e3:.2f}; T={soil.T / 86400:.3f} d')
