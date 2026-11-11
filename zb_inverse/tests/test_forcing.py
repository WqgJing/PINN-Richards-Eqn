"""Memory features of the known forcing as inputs of the parametric PINN (forcing.ForcingMemoryFeatures)."""
import os
import sys
import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.physics import pde_residual, surface_flux, theta_obs_model  # noqa: E402
from zbinv.model import HeadNet, InversePINN  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineTorch, CausalPathFeatures, _inv_softplus_np  # noqa: E402
from zbinv.forcing import ForcingMemoryFeatures, split_by_sign  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0, air_entry_m=0.02)
SP = CubicBSpline1D(0.0, 30.0, 5.0)
TAUS = (1.0 / 64, 1.0 / 16, 0.25, 1.0, 4.0, 16.0)
GLDAS_TWIN = os.path.join(ROOT, 'output', 'twin_gldas_fl', 'twin.npz')


def _series(seed=0, n=31, t_last=30.0):
    """A pulsed surface flux [m/s] (mostly small and upward, a few large downward pulses) and a transpiration [mm/d]."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_last, n)
    q = 3e-9 * rng.uniform(0.2, 1.0, n)
    q[rng.choice(n, 5, replace=False)] = -2e-7 * rng.uniform(0.2, 1.0, 5)
    tp = 3.5 * rng.uniform(0.0, 1.0, n)
    return t, q, tp


def _coef(shift=0.0):
    k = np.arange(SP.n_coef)
    return torch.tensor(_inv_softplus_np(0.6 + shift + 0.2 * np.sin(1.3 * k) - 0.05))


def _model(forcing, seed=0, **feat_kw):
    torch.manual_seed(seed)
    feats = CausalPathFeatures(SP, **feat_kw)
    head = HeadNet(32, 3, t_scale_days=30.0, n_cond=feats.n_out + (forcing.n_out if forcing is not None else 0),
                   sqrt_t=True)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.3)
    m = InversePINN(SOIL, ZbSplineTorch(SP, np.zeros(SP.n_coef), 0.05), head, 0.25, features=feats)
    m.forcing = forcing
    return m


def test_memory_of_a_step_and_a_ramp():
    """Known solutions of tau g' = u - g, g(0) = 0: u = 1 -> 1 - exp(-t / tau); u = t -> t - tau + tau exp(-t / tau)."""
    t = torch.linspace(0.0, 6.0, 241).reshape(-1, 1)
    taus = torch.tensor(TAUS).reshape(1, -1)
    step = ForcingMemoryFeatures([(np.array([0.0, 6.0]), np.array([1.0, 1.0]))], TAUS)
    assert torch.max(torch.abs(step(t) - (1.0 - torch.exp(-t / taus)))) < 1e-13
    ramp = ForcingMemoryFeatures([(np.array([0.0, 2.0, 6.0]), np.array([0.0, 2.0, 6.0]))], TAUS)   # scaled by max|u| = 6
    assert torch.max(torch.abs(6.0 * ramp(t) - (t - taus + taus * torch.exp(-t / taus)))) < 1e-12


def test_memory_obeys_its_equation_between_and_across_nodes():
    """g(0) = 0; tau dg/dt = u - g at any t (autograd slope: what the PDE residual differentiates); g and dg/dt are
    continuous across the nodes, where u has its kinks."""
    tn, q, tp = _series()
    f = ForcingMemoryFeatures([(tn, q), (tn, tp)], TAUS)
    assert f.n_out == 2 * len(TAUS) and len(f.state_dict()) == 0           # tables only: nothing saved with a model
    assert float(f(torch.zeros(1, 1)).abs().max()) == 0.0
    g = torch.Generator().manual_seed(3)
    t = (30.0 * torch.rand(4000, 1, generator=g)).requires_grad_(True)
    out = f(t)
    u = np.stack([np.interp(t.detach().numpy().ravel(), tn, s / np.abs(s).max()) for s in (q, tp)], axis=1)
    taus = np.asarray(TAUS)
    for j in range(f.n_out):
        dg = torch.autograd.grad(out[:, j].sum(), t, retain_graph=True)[0].numpy().ravel()
        s, k = divmod(j, len(TAUS))
        resid = taus[k] * dg - (u[:, s] - out[:, j].detach().numpy())
        assert np.max(np.abs(resid)) < 1e-11, (j, np.max(np.abs(resid)))
    # the memories really follow the forcing: the fastest one is close to it, the slowest one is not
    assert np.max(np.abs(out[:, 0].detach().numpy() - u[:, 0])) < 0.2 < np.max(np.abs(out[:, 5].detach().numpy() - u[:, 0]))
    eps = 1e-9
    nodes = torch.tensor(tn[1:-1]).reshape(-1, 1)
    lo, hi = (nodes - eps).requires_grad_(True), (nodes + eps).requires_grad_(True)
    g_lo, g_hi = f(lo), f(hi)
    assert float((g_hi - g_lo).abs().max()) < 1e-6
    for j in range(f.n_out):
        d_lo = torch.autograd.grad(g_lo[:, j].sum(), lo, retain_graph=True)[0]
        d_hi = torch.autograd.grad(g_hi[:, j].sum(), hi, retain_graph=True)[0]
        assert float((d_hi - d_lo).abs().max()) < 1e-5


def test_series_shorter_than_the_window_is_held_constant():
    tn, q, _ = _series(n=30, t_last=29.0)
    f = ForcingMemoryFeatures([(tn, q)], TAUS, t_end=30.0)
    t = torch.linspace(29.01, 30.0, 50).reshape(-1, 1).requires_grad_(True)
    out = f(t)
    u_last = q[-1] / np.abs(q).max()
    for k, tau in enumerate(TAUS):
        dg = torch.autograd.grad(out[:, k].sum(), t, retain_graph=True)[0]
        assert float((tau * dg.reshape(-1) - (u_last - out[:, k])).abs().max()) < 1e-11


def test_split_by_sign_is_exact():
    tn, q, _ = _series(seed=4)
    t2, up, down = split_by_sign(tn, q)
    assert t2.size > tn.size and np.all(np.diff(t2) > 0) and np.all(np.isin(tn, t2))
    assert up.min() == 0.0 and down.max() == 0.0 and up.max() > 0.0 and down.min() < 0.0
    assert np.all(up * down == 0.0)                                        # never both at a node: no sign change inside a segment
    t = np.random.default_rng(1).uniform(0.0, 30.0, 5000)
    q_t = np.interp(t, tn, q)
    assert np.max(np.abs(np.interp(t, t2, up) - np.maximum(q_t, 0.0))) < 1e-18
    assert np.max(np.abs(np.interp(t, t2, down) - np.minimum(q_t, 0.0))) < 1e-18
    # the small upward part gets its own scale: its memories are O(1), not O(evaporation / rain)
    f = ForcingMemoryFeatures([(t2, up), (t2, down)], TAUS)
    out = f(torch.tensor(t).reshape(-1, 1))
    assert float(out[:, 0].max()) > 0.5 and float(out[:, len(TAUS)].min()) < -0.5


def test_model_with_forcing_features():
    """The forcing features are extra inputs of the head and leave everything else as it was: per-sample paths equal
    shared ones, d theta / d c is exact, d h / d t includes the features' slope, a saved state reloads, and a model
    with the local zoom can be started from one without (the forcing columns come before the path features)."""
    from zbinv.parametric import load_state_zoom_aware
    tn, q, tp = _series()
    t2, up, down = split_by_sign(tn, q)
    forcing = ForcingMemoryFeatures([(t2, up), (t2, down), (tn, tp)], TAUS, t_end=30.0)
    m = _model(forcing)
    g = torch.Generator().manual_seed(2)
    n = 60
    xi, t = -torch.rand(n, 1, generator=g), 30.0 * torch.rand(n, 1, generator=g)
    c = _coef()
    ce = c.expand(n, -1)
    assert torch.max(torch.abs(m.h_tilde(xi, t, c) - m.h_tilde(xi, t, ce))) < 1e-13
    r = pde_residual(m, xi.clone().requires_grad_(True), t.clone().requires_grad_(True), c)
    assert torch.isfinite(r).all() and torch.isfinite(surface_flux(m, t.clone().requires_grad_(True), c)).all()
    # the network really uses the features (same weights, other forcing -> other head) ...
    tn5, q5, tp5 = _series(seed=5)
    m_other = _model(ForcingMemoryFeatures([(tn5, np.maximum(q5, 0.0)), (tn5, q5), (tn5, tp5)], TAUS, t_end=30.0))
    assert float((m.h_tilde(xi, t, c) - m_other.h_tilde(xi, t, c)).abs().max()) > 1e-3
    # ... and d h / d t by autograd is the total derivative, through the features too
    tg = t.clone().requires_grad_(True)
    dh = torch.autograd.grad(m.h_tilde(xi, tg, c).sum(), tg)[0]
    d = 1e-6
    fd = (m.h_tilde(xi, t + d, c) - m.h_tilde(xi, t - d, c)) / (2 * d)
    assert float((dh - fd).abs().max()) < 1e-6 * (1.0 + float(fd.abs().max()))
    # exact sensitivity to the path coefficients (the inversion's Jacobian)
    to = torch.linspace(0.5, 29.5, 9).reshape(-1, 1)
    fn = lambda cc: theta_obs_model(m, to, 0.05, cc).reshape(-1)
    J = torch.func.jacfwd(fn)(c)
    for k in range(SP.n_coef):
        e = torch.zeros_like(c)
        e[k] = 1e-5
        fdk = (fn(c + e) - fn(c - e)) / 2e-5
        assert torch.max(torch.abs(J[:, k] - fdk)) < 1e-7 * (1 + float(fdk.abs().max()))
    # saved state: no forcing tables in it; reloads into a model built from the same forcing
    sd = m.state_dict()
    assert not any(k.startswith('forcing') for k in sd)
    m2 = _model(forcing, seed=9)
    load_state_zoom_aware(m2, sd)
    assert float((m2.h_tilde(xi, t, ce) - m.h_tilde(xi, t, ce)).abs().max()) == 0.0
    # local zoom appended after the saved inputs: same function at the start of the local stage
    mz = _model(forcing, seed=11, center=c.numpy(), scale=0.05)
    load_state_zoom_aware(mz, sd)
    C = c[None, :] + 0.1 * torch.randn(n, SP.n_coef, generator=g)
    assert float((mz.h_tilde(xi, t, C) - m.h_tilde(xi, t, C)).abs().max()) == 0.0
    # a network trained without the forcing features cannot be loaded into one with them
    try:
        load_state_zoom_aware(mz, _model(None).state_dict())
        raise AssertionError('missing forcing inputs not detected')
    except (ValueError, RuntimeError):
        pass


def test_build_param_model_forcing_switch():
    """Off by default (the model and its inputs are unchanged); model.forcing_memory_taus_days adds one block of
    memories per forcing series: surface flux (two with forcing_split_sign) and, on a twin with roots, transpiration."""
    from zbinv.parametric import build_param_model
    from zbinv.train import load_dataset
    with open(os.path.join(ROOT, 'configs', 'p1_param_synth_v3.yaml')) as f:
        cfg = yaml.safe_load(f)
    ds = load_dataset(cfg)
    m0 = build_param_model(cfg, ds, SP, torch.device('cpu'))
    assert m0.forcing is None and m0.head.n_cond == m0.features.n_out
    cfg['model']['forcing_memory_taus_days'] = list(TAUS)
    m1 = build_param_model(cfg, ds, SP, torch.device('cpu'))
    assert m1.forcing.n_series == 1 and m1.head.n_cond == m0.head.n_cond + len(TAUS)
    cfg['model']['forcing_split_sign'] = True
    m2 = build_param_model(cfg, ds, SP, torch.device('cpu'))
    assert m2.forcing.n_series == 2 and m2.head.n_cond == m0.head.n_cond + 2 * len(TAUS)
    if os.path.exists(GLDAS_TWIN):                                         # the twin with rain pulses and a root sink
        cfg['twin_npz'] = GLDAS_TWIN
        ds = load_dataset(cfg)
        m3 = build_param_model(cfg, ds, SP, torch.device('cpu'))
        assert m3.sink is not None and m3.forcing.n_series == 3
        t = torch.linspace(0.0, float(ds['n_days']), 2001).reshape(-1, 1)
        out = m3.forcing(t)
        assert torch.isfinite(out).all() and float(out.abs().max()) <= 1.0 + 1e-12
        # the fastest memory of each part of the flux tracks that part (in units of its own maximum)
        q = np.interp(t.numpy().ravel(), ds['t_q'], ds['q_obs'])
        nt = len(TAUS)
        # (after the start: the memories begin at zero, the forcing does not)
        late = t.numpy().ravel() > 0.25
        assert np.median(np.abs(out[:, 0].numpy() - np.maximum(q, 0.0) / ds['q_obs'].max())) < 0.05
        d_down = np.abs(out[:, nt].numpy() - np.minimum(q, 0.0) / np.abs(ds['q_obs'].min()))
        k = int(np.argmax(d_down * late))
        assert d_down[k] < 0.1, (float(t[k]), float(d_down[k]), float(q[k]), float(ds['q_obs'].min()), float(d_down[0]))
        c = torch.tensor(ds['truth_coef']) if ds.get('truth_coef') is not None else _coef()
        xi = -torch.rand(200, 1)
        tt = (float(ds['n_days']) * torch.rand(200, 1)).requires_grad_(True)
        assert torch.isfinite(pde_residual(m3, xi.requires_grad_(True), tt, c)).all()


def test_split_by_sign_with_zeros_at_nodes():
    """A sinusoid sampled at its zeros (zero only to rounding there): no duplicate or misplaced crossing nodes."""
    t = np.linspace(0.0, 30.0, 121)
    q = 1e-8 * np.sin(2 * np.pi * t / 5.0)
    t2, up, down = split_by_sign(t, q)
    assert np.all(np.diff(t2) > 1e-6) and t2.size == t.size
    assert np.max(np.abs(up + down - q)) < 1e-12 * 1e-8
    ForcingMemoryFeatures([(t2, up), (t2, down)], TAUS, t_end=30.0)
