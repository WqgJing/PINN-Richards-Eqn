"""Root water uptake in the PINN residual: same sink as the wtd_pipeline FEM, right units and coordinates."""
import importlib.util
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.physics import pde_residual  # noqa: E402
from zbinv.roots import RootSink  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0)
FEM_ROOTS = '/storage/project/r-rbras6-0/jjason3/wtd_pipeline/fem/root_water_uptake.py'
T_TP = np.arange(8.0)                                                  # days
TP_MMD = np.array([0.0, 0.4, 1.0, 2.5, 3.5, 4.32, 6.0, 9.0])           # below, at and above the Feddes threshold


def test_root_sink_matches_fem_class():
    spec = importlib.util.spec_from_file_location('root_water_uptake', FEM_ROOTS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['root_water_uptake'] = mod
    spec.loader.exec_module(mod)
    fem = mod.RootWaterUptakeTorch(TP_MMD[None, :], device='cpu', Tp_units='mm/day')
    sink = RootSink(T_TP, TP_MMD)
    rng = np.random.default_rng(0)
    n = 4000
    depth = torch.tensor(rng.uniform(0.0, 3.5, n))                     # beyond the 3 m root cut-off too
    h = torch.tensor(np.concatenate([rng.uniform(-0.4, 0.05, n // 2), -10.0 ** rng.uniform(-1.0, 2.2, n // 2)]))
    seen = 0.0
    for k in range(T_TP.size):
        S_fem = fem.compute_sink(depth[None, :], k, h[None, :])[0]
        S = sink(depth, torch.full_like(depth, T_TP[k]), h)
        assert torch.allclose(S, S_fem, rtol=1e-12, atol=1e-22), (k, float((S - S_fem).abs().max()))
        seen = max(seen, float(S.max()))
    assert seen > 1e-8                                                 # the comparison is not of zeros
    assert len(sink.state_dict()) == 0                                 # tables only: saved models stay loadable


class FakeModel:
    soil = SOIL
    T_days = SOIL.T / 86400.0
    sink = None

    def zb_m(self, t):
        return 1.0 + 0.3 * torch.sin(2 * np.pi * t / 10.0)

    def h_tilde(self, xi, t):
        zb = self.zb_m(t)
        z = xi * zb
        return (-(z + zb) + 0.15 * torch.sin(2.1 * z + 0.3) * torch.cos(2 * np.pi * t / 7.0) - 0.05 * z * z) / SOIL.L


def sink_np(depth, t, h):
    """S [1/s] written out independently of roots.py (grass defaults of the FEM)."""
    tp = np.interp(t, T_TP, TP_MMD) / 86400e3
    h3 = -6.0 + 3.0 * np.clip(tp / 5e-8, 0.0, 1.0)
    alpha = np.where(h > -0.10, 0.0, np.where(h > -0.25, (h + 0.10) / -0.15,
                                              np.where(h > h3, 1.0, np.clip((h + 80.0) / (h3 + 80.0), 0.0, 1.0))))
    return alpha * tp * (-np.log(0.96) * 100.0 * 0.96 ** (100.0 * depth) / (1.0 - 0.96 ** 300.0))


def test_residual_gains_sink_in_model_units():
    rng = np.random.default_rng(1)
    xi = torch.tensor(-rng.uniform(0.02, 0.98, 500)).reshape(-1, 1)
    t = torch.tensor(rng.uniform(0.2, 6.8, 500)).reshape(-1, 1)
    plain, rooted = FakeModel(), FakeModel()
    rooted.sink = RootSink(T_TP, TP_MMD)
    dR = (pde_residual(rooted, xi, t) - pde_residual(plain, xi, t)).detach().numpy().ravel()
    zb = plain.zb_m(t).numpy().ravel()
    h = plain.h_tilde(xi, t).numpy().ravel() * SOIL.L
    expect = sink_np(-xi.numpy().ravel() * zb, t.numpy().ravel(), h) * SOIL.L / SOIL.Ks
    assert np.abs(expect).max() > 1e-3                                 # the sink is active in this sample
    assert np.allclose(dR, expect, rtol=1e-9, atol=1e-12), float(np.abs(dR - expect).max())
