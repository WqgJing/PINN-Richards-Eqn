"""zb_init 'hydrostatic_flat' and sigma_pde 'morozov' (added 2026-11-10, finding 32)."""
import os
import sys
import tempfile
import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from zbinv.spline import ZbSplineNumpy  # noqa: E402
from zbinv.train import Trainer  # noqa: E402


def _cfg(**sigmas):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'e3_inverse_synth_recipe.yaml')))
    cfg['output']['dir'] = tempfile.mkdtemp(prefix='zbinv_test_')
    cfg['train'].update(n_col=128, n_bc=32, log_every=1000, resample_every=1000)
    cfg['sampling']['rar']['enabled'] = False
    cfg['sigmas'].update(sigmas)
    return cfg


def test_hydrostatic_flat_init_is_the_median_hydrostatic_depth():
    cfg = _cfg()
    cfg['zb_init']['kind'] = 'hydrostatic_flat'
    tr = Trainer(cfg)
    zb = ZbSplineNumpy(tr.spline, tr.coef0, tr.ds['zb_min']).zb(tr.ds['t_hourly'])
    assert np.allclose(zb, np.median(tr.zb_est_obs), atol=1e-9), (zb.min(), zb.max(), np.median(tr.zb_est_obs))


def test_pde_factor_scales_auto():
    a = Trainer(_cfg()).sig_pde
    b = Trainer(_cfg(pde_factor=0.25)).sig_pde
    assert abs(b / a - 0.25) < 1e-12, (a, b)


def test_morozov_sigma_pde_bounds_and_direction():
    # misfit above the target (hydrostatic start: data rmse ~0.008 vs sigma 0.005) -> stays at the auto cap
    cfg = _cfg(pde='morozov', pde_mz_freq=1)
    tr = Trainer(cfg)
    s0 = tr.sig_pde
    tr.adam_stage('t', 3, joint=True, use_data=True, decay=False)
    assert tr.sig_pde == s0, (tr.sig_pde, s0)
    n_obs = len(tr.ds['t_obs'])                     # 721 hourly values (both window ends)
    assert abs(tr.pde_mz['target'] - (1.0 - 9.0 / n_obs)) < 1e-12, (tr.pde_mz['target'], n_obs)
    # misfit far below the target (declared sigma_theta 10x the misfit) -> tightens, never below the floor
    cfg = _cfg(pde='morozov', pde_mz_freq=1, theta=0.1)
    tr = Trainer(cfg)
    s0 = tr.sig_pde
    tr.adam_stage('t', 40, joint=True, use_data=True, decay=False)
    assert tr.pde_mz['min'] - 1e-15 <= tr.sig_pde < s0, (tr.sig_pde, s0, tr.pde_mz['min'])
