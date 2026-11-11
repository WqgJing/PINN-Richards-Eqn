#!/usr/bin/env python
"""How much does the top-5-cm moisture feel the water table? (answers "is deep zb identifiable?")

1) Per-knot Cramer-Rao bounds from the identifiability runs (hourly obs), with the true zb at each knot.
2) Steady-state theta_5cm(zb) for no flux (hydrostatic), steady infiltration and steady evaporation, and
   its sensitivity d theta_5cm / d zb: how many cm of WT change one 0.005 sensor error corresponds to.
Usage: python scripts/depth_sensitivity.py <twin_config.yaml> <identifiability.npz> [...]
"""
import os
import sys
import numpy as np
import yaml
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.data import obs_from_profile  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
soil = SoilVG.from_config(cfg['soil'], cfg['scaling']['L'])
print(f'soil: {soil.summary()}')

for path in sys.argv[2:]:
    d = np.load(path, allow_pickle=True)
    print(f'\n== {path}: keys {sorted(d.files)}')
    for k in d.files:
        v = d[k]
        if v.ndim == 1 and v.size < 40:
            print(f'  {k}: ' + ' '.join(f'{x:.3g}' for x in v))


def steady_theta5(zb, q_mmd, n=400):
    """Steady profile for a constant surface flux q (positive up) above a WT at zb; returns theta_5cm."""
    qt = q_mmd / 86400e3 / soil.Ks
    s = np.linspace(0, 1, n)
    xi = -(1 - s) ** 1.5
    zt = xi * zb / soil.L
    sol = solve_ivp(lambda z, h: -qt / soil.K(h) - 1.0, (zt[0], 0.0), [0.0], t_eval=zt, rtol=1e-10, atol=1e-12,
                    method='LSODA')
    if not sol.success or sol.y.shape[1] != n:
        return np.nan
    th = soil.theta(sol.y[0])
    return float(obs_from_profile(xi, th[None, :], np.array([zb]), 0.05)[0])


zbs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
print('\nsteady theta_5cm(zb) and sensitivity (d zb per 0.005 theta error, metres):')
for q in [0.0, -1.0, -3.0, 0.5, 1.0]:
    th = np.array([steady_theta5(z, q) for z in zbs])
    dz = 1e-3
    sens = np.array([(steady_theta5(z + dz, q) - steady_theta5(z - dz, q)) / (2 * dz) for z in zbs])
    print(f'q = {q:+.1f} mm/d (neg = infiltration)')
    for z, t, s in zip(zbs, th, sens):
        print(f'   zb {z:4.2f} m  theta5 {t:.4f}  dtheta/dzb {s:+.4f} /m  -> 0.005 error = {0.005 / max(abs(s), 1e-12):7.3f} m')
