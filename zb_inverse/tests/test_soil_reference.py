"""Checks on the soil functions and the reference solver (plain asserts; run via tests/run_tests.py)."""
import os
import sys
import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.reference import ReferenceSolver, make_grid, hydrostatic_h  # noqa: E402

SOIL = dict(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6)


def _soil(L=2.0):
    return SoilVG(L=L, **SOIL)


def test_numpy_torch_agree():
    s = _soil()
    h = np.linspace(-1.5, 0.3, 2001)
    ht = torch.tensor(h, dtype=torch.float64)
    assert np.allclose(s.Se(h), s.Se(ht).numpy(), rtol=1e-12, atol=1e-14)
    assert np.allclose(s.K(h), s.K(ht).numpy(), rtol=1e-12, atol=1e-14)


def test_capacity_matches_finite_difference():
    s = _soil()
    h = np.linspace(-1.0, 0.2, 501)
    eps = 1e-7
    fd = (s.Se(h + eps) - s.Se(h - eps)) / (2 * eps)
    assert np.allclose(s.capacity(h), fd, rtol=1e-5, atol=1e-7)


def test_smooth_equals_plain_vg_away_from_zero():
    s = _soil()
    h_m = np.linspace(-3.0, -0.2, 200)
    se_plain = (1 + (s.alpha * np.abs(h_m)) ** s.n) ** (-s.m)
    assert np.max(np.abs(s.Se(h_m / s.L) - se_plain)) < 1e-4
    assert np.allclose(-s.h_from_theta(s.theta_r + s.theta_star * se_plain), np.abs(h_m), rtol=1e-8)


def test_hydrostatic_is_steady():
    s = _soil()
    xi = make_grid(101, 1.5)
    zb = 1.2
    h0 = hydrostatic_h(xi, zb, s.L)
    r = ReferenceSolver(s, xi).solve(np.array([0.0, 5.0, 10.0]), lambda t: zb, lambda t: 0.0,
                                     lambda t: 0.0, h0, verbose=False)
    assert np.max(np.abs(r['h_t'][-1] - h0)) * s.L < 1e-6


def test_steady_infiltration_matches_ode():
    """Constant downward flux, fixed WT: long-time solution = steady profile from the ODE."""
    s = _soil()
    zb, q = 1.0, -2.0e-3 / 86400.0            # 2 mm/d infiltration
    qt = q / s.Ks
    xi = make_grid(201, 1.5)
    h0 = hydrostatic_h(xi, zb, s.L)
    r = ReferenceSolver(s, xi).solve(np.array([0.0, 400.0]), lambda t: zb, lambda t: 0.0,
                                     lambda t: q, h0, verbose=False)
    # steady: -K (dh/dz + 1) = q  ->  dh/dz~ = -q~/K - 1, from h(-zb)=0 upward
    zt = xi * zb / s.L
    sol = solve_ivp(lambda z, h: -qt / s.K(h) - 1.0, (zt[0], 0.0), [0.0], t_eval=zt, rtol=1e-10, atol=1e-12)
    th_ode = s.theta(sol.y[0])
    th_ref = r['theta'][-1]
    assert np.max(np.abs(th_ref - th_ode)) < 2e-3, np.max(np.abs(th_ref - th_ode))


def test_moving_wt_mass_balance():
    s = _soil()
    xi = make_grid(101, 1.5)
    zb_fn = lambda t: 1.0 + 0.3 * np.sin(2 * np.pi * np.asarray(t) / 10.0)
    zd_fn = lambda t: 0.3 * 2 * np.pi / 10.0 * np.cos(2 * np.pi * np.asarray(t) / 10.0)
    q_fn = lambda t: -3e-8 * (np.sin(2 * np.pi * np.asarray(t) / 3.0) > 0.5)
    h0 = hydrostatic_h(xi, zb_fn(0.0), s.L)
    r = ReferenceSolver(s, xi).solve(np.linspace(0, 20, 20 * 96 + 1), zb_fn, zd_fn, q_fn, h0, verbose=False)
    rel = np.abs(r['mass_err_t']).max() / np.abs(r['storage_t']).max()
    assert rel < 5e-3, rel


def test_head_limited_surface_bc():
    """Undeliverable evaporation (5 mm/d, WT at 1.5 m in sand): the pure flux BC dries the surface
    without bound; the head-limited BC keeps h_surf near h_crit, delivers less than the potential
    flux, and conserves mass with the actual flux."""
    s = _soil()
    xi = make_grid(201, 1.5)
    zb_fn = lambda t: np.full_like(np.asarray(t, dtype=float), 1.5)
    zd_fn = lambda t: np.zeros_like(np.asarray(t, dtype=float))
    q_fn = lambda t: np.full_like(np.asarray(t, dtype=float), 5e-3 / 86400.0)
    h0 = hydrostatic_h(xi, 1.5, s.L)
    t_eval = np.linspace(0, 10, 10 * 24 + 1)
    r = ReferenceSolver(s, xi, h_crit_m=-100.0).solve(t_eval, zb_fn, zd_fn, q_fn, h0, verbose=False)
    h_surf = r['h_t'][:, -1] * s.L
    assert h_surf.min() > -150.0, h_surf.min()          # bounded near h_crit (gate width 0.1 in log|h|)
    assert r['q_act'][-1] < 0.5 * r['qsurf'][-1], (r['q_act'][-1], r['qsurf'][-1])
    rel = np.abs(r['mass_err_t']).max() / np.abs(r['storage_t']).max()
    assert rel < 5e-3, rel


# ---------------------------------------------------------------- air-entry model (Ippisch et al. 2006)
def _soil_ae(L=2.0):
    return SoilVG(L=L, air_entry_m=0.02, ae_smooth_m=0.005, **SOIL)


def test_air_entry_numpy_torch_and_capacity():
    s = _soil_ae()
    h = np.linspace(-1.5, 0.3, 2001)
    ht = torch.tensor(h, dtype=torch.float64)
    assert np.allclose(s.Se(h), s.Se(ht).numpy(), rtol=1e-12, atol=1e-14)
    assert np.allclose(s.K(h), s.K(ht).numpy(), rtol=1e-12, atol=1e-14)
    eps = 1e-7
    fd = (s.Se(h + eps) - s.Se(h - eps)) / (2 * eps)
    assert np.allclose(s.capacity(h), fd, rtol=1e-5, atol=1e-7)


def test_air_entry_matches_ippisch_and_saturates():
    s = _soil_ae()
    h_m = np.linspace(-3.0, -0.10, 200)                 # well below h_e = -2 cm (smooth-min inactive)
    Sc = (1 + (s.alpha * 0.02) ** s.n) ** (-s.m)
    se = (1 + (s.alpha * np.abs(h_m)) ** s.n) ** (-s.m) / Sc
    kr = se ** 0.5 * ((1 - (1 - (se * Sc) ** (1 / s.m)) ** s.m) / (1 - (1 - Sc ** (1 / s.m)) ** s.m)) ** 2
    assert np.allclose(s.Se(h_m / s.L), se, rtol=1e-6)
    assert np.allclose(s.K(h_m / s.L), kr, rtol=1e-5)
    # at the water table the 5 mm smooth-min (4 widths above h_e) leaves Se = 1 - 4e-5, K = 0.9987
    assert abs(s.Se(np.array([0.0]))[0] - 1) < 1e-4 and abs(s.K(np.array([0.0]))[0] - 1) < 2e-3
    hp = np.array([0.05, 0.5]) / s.L                      # above the water table: saturated
    assert np.all(np.abs(s.Se(hp) - 1) < 1e-8) and np.all(np.abs(s.K(hp) - 1) < 1e-7)
    # finite conductivity slope near saturation (plain VG: K falls to ~0.92 within 1 mm)
    assert s.K(np.array([-0.001 / s.L]))[0] > 0.998      # 0.9985 (smoothing); plain VG gives 0.92
    assert np.allclose(-s.h_from_theta(s.theta_r + s.theta_star * se), np.abs(h_m), rtol=1e-8)


def test_air_entry_steady_infiltration_matches_ode():
    s = _soil_ae()
    zb, q = 1.0, -2.0e-3 / 86400.0
    qt = q / s.Ks
    xi = make_grid(201, 1.5)
    h0 = hydrostatic_h(xi, zb, s.L)
    r = ReferenceSolver(s, xi).solve(np.array([0.0, 400.0]), lambda t: zb, lambda t: 0.0,
                                     lambda t: q, h0, verbose=False)
    zt = xi * zb / s.L
    sol = solve_ivp(lambda z, h: -qt / s.K(h) - 1.0, (zt[0], 0.0), [0.0], t_eval=zt, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(r['theta'][-1] - s.theta(sol.y[0]))) < 2e-3
