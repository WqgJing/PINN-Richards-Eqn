"""PINN residuals vs independent finite differences in physical coordinates (checks the Landau signs)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.physics import pde_residual, surface_flux, theta_obs_model  # noqa: E402
from zbinv.model import HeadNet, InversePINN  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineTorch  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0)


def zb_np(t):
    return 1.0 + 0.3 * np.sin(2 * np.pi * t / 10.0)


def h_phys_np(z, t):
    """Arbitrary smooth head [m] in physical coordinates (z <= 0, t in days)."""
    return -(z + zb_np(t)) + 0.15 * np.sin(2.1 * z + 0.3) * np.cos(2 * np.pi * t / 7.0) - 0.05 * z * z


class FakeModel:
    soil = SOIL
    T_days = SOIL.T / 86400.0

    def zb_m(self, t):
        return 1.0 + 0.3 * torch.sin(2 * np.pi * t / 10.0)

    def h_tilde(self, xi, t):
        zb = self.zb_m(t)
        z = xi * zb
        h = -(z + zb) + 0.15 * torch.sin(2.1 * z + 0.3) * torch.cos(2 * np.pi * t / 7.0) - 0.05 * z * z
        return h / SOIL.L


def phys_residual_fd(z, t, dz=1e-5, dt=1e-4):
    """R = dM/dt~|z + dq~/dz~ with central differences; z [m], t [days]."""
    s, L, Td = SOIL, SOIL.L, SOIL.T / 86400.0
    M = lambda zz, tt: s.storage(h_phys_np(zz, tt) / L)

    def q(zz, tt):
        dh = (h_phys_np(zz + dz, tt) - h_phys_np(zz - dz, tt)) / (2 * dz)     # dh/dz (m/m)
        return -s.K(h_phys_np(zz, tt) / L) * (dh + 1.0)
    dM_dtt = (M(z, t + dt) - M(z, t - dt)) / (2 * dt) * Td
    dq_dzt = (q(z + dz, t) - q(z - dz, t)) / (2 * dz) * L
    return dM_dtt + dq_dzt, q(z, t)


def test_landau_residual_matches_physical_fd():
    rng = np.random.default_rng(0)
    xi = -rng.uniform(0.02, 0.98, 200)
    t = rng.uniform(0.5, 20.0, 200)
    R = pde_residual(FakeModel(), torch.tensor(xi).reshape(-1, 1), torch.tensor(t).reshape(-1, 1))
    R = R.detach().numpy().ravel()
    R_fd, _ = phys_residual_fd(xi * zb_np(t), t)
    scale = np.max(np.abs(R_fd))
    assert np.max(np.abs(R - R_fd)) < 1e-4 * scale, (np.max(np.abs(R - R_fd)), scale)


def test_surface_flux_matches_fd():
    t = np.linspace(0.5, 20.0, 50)
    q = surface_flux(FakeModel(), torch.tensor(t).reshape(-1, 1)).detach().numpy().ravel()
    _, q_fd = phys_residual_fd(np.zeros_like(t), t)
    assert np.allclose(q, q_fd, rtol=1e-6, atol=1e-10)


def _pinn(departure='additive', log_depth_eps=None):
    sp = CubicBSpline1D(0.0, 30.0, 5.0)
    zb = ZbSplineTorch(sp, np.linspace(0.5, 1.2, sp.n_coef), 0.05)
    head = HeadNet(32, 3, t_scale_days=30.0, log_depth_eps=log_depth_eps)
    return InversePINN(SOIL, zb, head, tau_ic_days=0.25, departure=departure)


def test_hard_constraints():
    m = _pinn()
    with torch.no_grad():
        for p in m.head.parameters():
            p.normal_(0, 0.5)                     # make N non-trivial
        t = torch.linspace(0, 30, 31).reshape(-1, 1)
        h_wt = m.h_tilde(-torch.ones_like(t), t)
        assert torch.max(torch.abs(h_wt)) < 1e-14
        xi = torch.linspace(-1, 0, 41).reshape(-1, 1)
        h0 = m.h_tilde(xi, torch.zeros_like(xi))
        hydro = -(m.zb_m(torch.zeros((1, 1))) / SOIL.L) * (1 + xi)
        assert torch.max(torch.abs(h0 - hydro)) < 1e-14


def test_obs_operator_hydrostatic():
    m = _pinn()                                   # N = 0 -> hydrostatic at zb(0) for all t
    t = torch.tensor([[0.0], [3.0]])
    th = theta_obs_model(m, t, 0.05).detach().numpy().ravel()
    zb0 = float(m.zb_m(torch.zeros((1, 1))))
    zb_t = m.zb_m(t).detach().numpy().ravel()
    for k in range(2):
        z = np.linspace(-0.05, 0.0, 20001)
        h = -(zb_t[k] + z)                        # equilibrium at the current WT (N = 0)
        ref = np.trapezoid(SOIL.theta(h / SOIL.L), z) / 0.05
        assert abs(th[k] - ref) < 1e-6, (th[k], ref)


def test_hard_constraints_log_departure():
    m = _pinn(departure='log', log_depth_eps=1e-3)
    with torch.no_grad():
        for p in m.head.parameters():
            p.normal_(0, 0.5)
        t = torch.linspace(0, 30, 31).reshape(-1, 1)
        assert torch.max(torch.abs(m.h_tilde(-torch.ones_like(t), t))) < 1e-14
        xi = torch.linspace(-1, 0, 41).reshape(-1, 1)
        h0 = m.h_tilde(xi, torch.zeros_like(xi))
        hydro = -(m.zb_m(torch.zeros((1, 1))) / SOIL.L) * (1 + xi)
        assert torch.max(torch.abs(h0 - hydro)) < 1e-14
        h = m.h_tilde(xi, torch.full_like(xi, 10.0))
        assert torch.all(h[1:] < 0)                              # unsaturated above the WT
    R = pde_residual(m, -torch.rand(256, 1), 30 * torch.rand(256, 1))
    assert torch.all(torch.isfinite(R))


def test_hard_constraints_conductive_departure():
    """departure='conductive' is the additive departure times the gain g = q~ / (K_r(hydrostatic head) + q~):
    the hard constraints are kept, g is in (0, 1) and grows with the suction of the hydrostatic profile."""
    m, ma = _pinn(departure='conductive'), _pinn()
    m.q_scale = 0.05
    with torch.no_grad():
        for p in m.head.parameters():
            p.normal_(0, 0.5)
        ma.load_state_dict(m.state_dict())                       # the same network in the additive form
        t = torch.linspace(0, 30, 31).reshape(-1, 1)
        assert torch.max(torch.abs(m.h_tilde(-torch.ones_like(t), t))) < 1e-14
        xi = torch.linspace(-1, 0, 41).reshape(-1, 1)
        h0 = m.h_tilde(xi, torch.zeros_like(xi))
        hydro = -(m.zb_m(torch.zeros((1, 1))) / SOIL.L) * (1 + xi)
        assert torch.max(torch.abs(h0 - hydro)) < 1e-14
        tt = torch.full_like(xi, 10.0)
        hyd = -(1 + xi) * m.zb_m(tt) / SOIL.L
        g = m.q_scale / (SOIL.K(hyd) + m.q_scale)
        assert float(g.min()) > 0 and float(g.max()) < 1 and torch.all(g[1:] > g[:-1])
        d_c, d_a = m.h_tilde(xi, tt) - hyd, ma.h_tilde(xi, tt) - hyd
        assert float(d_a.abs().max()) > 1e-3 and torch.max(torch.abs(d_c - g * d_a)) < 1e-13
    R = pde_residual(m, -torch.rand(256, 1), 30 * torch.rand(256, 1))
    assert torch.all(torch.isfinite(R))
    try:
        InversePINN(SOIL, m.zb_spline, m.head, base='initial', departure='conductive')
        raise AssertionError("departure='conductive' with base='initial' not refused")
    except ValueError:
        pass
