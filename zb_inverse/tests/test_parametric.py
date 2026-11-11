"""Parametric (operator) PINN: per-sample zb paths, causal path features, exact sensitivities (gate G0)."""
import json
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
from zbinv.parametric import CPrior  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0, air_entry_m=0.02)
SP = CubicBSpline1D(0.0, 30.0, 5.0)


def _coef(shift=0.0):
    k = np.arange(SP.n_coef)
    return torch.tensor(_inv_softplus_np(0.6 + shift + 0.2 * np.sin(1.3 * k) - 0.05))


def _model(seed=0):
    torch.manual_seed(seed)
    feats = CausalPathFeatures(SP)
    head = HeadNet(32, 3, t_scale_days=30.0, n_cond=feats.n_out, sqrt_t=True)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.3)
    zb = ZbSplineTorch(SP, np.zeros(SP.n_coef), 0.05)
    return InversePINN(SOIL, zb, head, 0.25, features=feats)


def test_per_sample_c_equals_shared_c():
    m = _model()
    g = torch.Generator().manual_seed(1)
    n = 64
    xi, t = -torch.rand(n, 1, generator=g), 30 * torch.rand(n, 1, generator=g)
    c = _coef()
    ce = c.expand(n, -1)
    assert torch.max(torch.abs(m.h_tilde(xi, t, c) - m.h_tilde(xi, t, ce))) < 1e-13
    assert torch.max(torch.abs(pde_residual(m, xi, t, c) - pde_residual(m, xi, t, ce))) < 1e-12
    assert torch.max(torch.abs(surface_flux(m, t, c) - surface_flux(m, t, ce))) < 1e-12
    assert torch.max(torch.abs(theta_obs_model(m, t, 0.05, c) - theta_obs_model(m, t, 0.05, ce))) < 1e-13
    # different paths per sample really differ, and each row matches its own shared evaluation
    c2 = torch.stack([_coef(0.1 * (i % 3)) for i in range(n)])
    h2 = m.h_tilde(xi, t, c2)
    for i in (0, 1, 2):
        hi = m.h_tilde(xi[i:i + 1], t[i:i + 1], c2[i])
        assert abs(float(h2[i] - hi)) < 1e-13


def test_hard_constraints_per_sample():
    m = _model()
    n = 40
    c = torch.stack([_coef(0.05 * i) for i in range(n)])
    t = torch.linspace(0, 30, n).reshape(-1, 1)
    with torch.no_grad():
        assert torch.max(torch.abs(m.h_tilde(-torch.ones_like(t), t, c))) < 1e-14          # h = 0 at the WT
        xi = -torch.rand(n, 1)
        h0 = m.h_tilde(xi, torch.zeros_like(xi), c)
        hydro = -(m.zb_m(torch.zeros_like(xi), c) / SOIL.L) * (1 + xi)                     # IC: hydrostatic
        assert torch.max(torch.abs(h0 - hydro)) < 1e-14


def test_defaults_reproduce_a_saved_joint_run():
    run = os.path.join(ROOT, 'runs', 'v9_long_noise005')
    from zbinv.train import Trainer
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
    cfg['output']['dir'] = '/tmp/zbinv_test_reload'
    tr = Trainer(cfg)
    tr.model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location=tr.dev))
    _, pred = tr.evaluate()
    ref = np.load(os.path.join(run, 'predictions.npz'))
    assert np.max(np.abs(pred['theta_obs_pred'] - ref['theta_obs_pred'])) < 1e-12
    assert np.max(np.abs(pred['zb_pred'] - ref['zb_pred'])) < 1e-12


def test_memory_features_exact_and_smooth():
    F = CausalPathFeatures(SP, taus_days=(0.25, 4.0))
    # partition of unity: a constant path has constant memories
    c1 = torch.full((SP.n_coef,), 0.7)
    t = torch.linspace(0, 30, 97).reshape(-1, 1)
    assert torch.max(torch.abs(F.memories(t, c1) - 0.7)) < 1e-10
    # Hermite table vs the direct exponential integrator, and the ODE dg/dt = (f - g) / tau
    c = _coef()
    tt = torch.tensor(np.random.default_rng(0).uniform(0.05, 29.9, 200)).reshape(-1, 1)
    for j, tau in enumerate(F.taus):
        direct = SP.filtered_basis_np(tt.numpy().ravel(), tau) @ c.numpy()
        g = F.memories(tt, c)[:, j].numpy()
        assert np.max(np.abs(g - direct)) < 1e-6 * (1 + np.max(np.abs(direct))), (tau, np.max(np.abs(g - direct)))
        ttr = tt.clone().requires_grad_(True)
        gj = F.memories(ttr, c)[:, j]
        dg = torch.autograd.grad(gj.sum(), ttr)[0].numpy().ravel()
        f = SP.basis_np(tt.numpy().ravel()) @ c.numpy()
        ode = (f - gj.detach().numpy()) / tau
        assert np.max(np.abs(dg - ode)) < 2e-3 * np.max(np.abs(ode)), (tau, np.max(np.abs(dg - ode)))
    # at t = 0 the memory equals the path (constant past = hydrostatic history)
    t0 = torch.zeros((1, 1))
    assert torch.max(torch.abs(F.memories(t0, c) - SP.eval_torch(t0, c))) < 1e-12


def test_features_are_causal():
    F = CausalPathFeatures(SP)
    c = _coef().clone().requires_grad_(True)
    for k in range(4, SP.n_coef):
        start = (k - 3) * SP.dt                                # support of B_k begins here
        t = torch.linspace(0.0, start - 2 * F.dg, 25).reshape(-1, 1)
        J = torch.autograd.functional.jacobian(lambda cc: F(t, cc), c)            # (25, n_out, n_coef)
        assert float(J[:, :, k].abs().max()) == 0.0, (k, float(J[:, :, k].abs().max()))


def test_parametric_residual_matches_physical_fd():
    """The Landau residual (with path features that depend on t) vs central differences in physical z, t."""
    m = _model(2)
    c = _coef()
    L, Td = SOIL.L, SOIL.T / 86400.0

    def h_phys(z, t):
        z = torch.as_tensor(z).reshape(-1, 1)
        t = torch.as_tensor(t).reshape(-1, 1)
        with torch.no_grad():
            zb = m.zb_m(t, c)
            return (L * m.h_tilde(z / zb, t, c)).numpy().ravel()

    rng = np.random.default_rng(3)
    t = rng.uniform(0.5, 29.0, 150)
    xi = -rng.uniform(0.05, 0.95, 150)
    with torch.no_grad():
        zb = m.zb_m(torch.tensor(t).reshape(-1, 1), c).numpy().ravel()
    z = xi * zb
    dz, dt = 1e-5, 1e-4
    M = lambda zz, tt: SOIL.storage(h_phys(zz, tt) / L)

    def q(zz, tt):
        dh = (h_phys(zz + dz, tt) - h_phys(zz - dz, tt)) / (2 * dz)
        return -SOIL.K(h_phys(zz, tt) / L) * (dh + 1.0)
    R_fd = (M(z, t + dt) - M(z, t - dt)) / (2 * dt) * Td + (q(z + dz, t) - q(z - dz, t)) / (2 * dz) * L
    R = pde_residual(m, torch.tensor(xi).reshape(-1, 1), torch.tensor(t).reshape(-1, 1), c).detach().numpy().ravel()
    scale = np.max(np.abs(R_fd))
    assert np.max(np.abs(R - R_fd)) < 1e-4 * scale, (np.max(np.abs(R - R_fd)), scale)


def test_dtheta_dc_exact():
    m = _model(4)
    t = torch.linspace(0, 30, 61).reshape(-1, 1)
    fn = lambda cc: theta_obs_model(m, t, 0.05, cc).reshape(-1)
    c = _coef()
    J = torch.func.jacfwd(fn)(c)
    d = 1e-5
    J_fd = torch.stack([(fn(c + d * e) - fn(c - d * e)) / (2 * d) for e in torch.eye(SP.n_coef)], dim=1)
    assert float(torch.max(torch.abs(J - J_fd))) < 1e-7 * (1 + float(torch.max(torch.abs(J_fd))))


def test_prior_paths_in_range():
    pr = CPrior(SP, 0.05, (0.3, 1.2), 0.15, (0.2, 1.5))
    z = pr.sample_zb(2000, torch.Generator().manual_seed(0))
    assert float(z.min()) >= 0.2 and float(z.max()) <= 1.5
    c = pr.sample(500, torch.Generator().manual_seed(1))
    zb = 0.05 + torch.nn.functional.softplus(torch.stack([SP.eval_torch(torch.linspace(0, 30, 121).reshape(-1, 1), ci)
                                                          for ci in c]))
    assert float(zb.min()) > 0.12 and float(zb.max()) < 1.6


def _local_trainer(n_tan, **train):
    from zbinv.parametric import ParamTrainer
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'p1_param_synth_v3.yaml')))
    cfg['sampling'] = {'mode': 'local_file', 'frac_local': 0.9,
                       'local_file': os.path.join(ROOT, 'runs', 'p2_clean_inv0', 'local_center.npz')}
    cfg['obs'].update(noise_sigma=0.0, q_noise_frac=0.0)
    cfg['train'].update(n_val=64, n_tan=n_tan, seed=0, **train)
    cfg['output']['dir'] = '/tmp/zbinv_test_centre'
    return ParamTrainer(cfg)


def test_centre_term_equals_per_point_delta_method():
    """centre_losses = mean_i [ r_i(c0)^2 + g_i^T Sigma g_i ] with g_i = d r_i / d c computed point by point."""
    tr = _local_trainer(n_tan=32)
    torch.manual_seed(3)
    xi, t, tb = tr.batch(64, 16, n_tan=32)[5]
    got = tr.centre_losses(xi, t, tb)
    c0, L = tr.c_dist.local.center, tr.c_dist.local.L
    Sig = L @ L.T

    def brute(fun, n):
        vals = []
        for i in range(n):
            cc = c0.clone().reshape(1, -1).requires_grad_(True)
            r = fun(i, cc)
            g = torch.autograd.grad(r.sum(), cc)[0].reshape(-1)
            vals.append(float(r.detach()) ** 2 + float(g @ Sig @ g))
        return np.mean(vals)

    want_pde = brute(lambda i, cc: pde_residual(tr.model, xi[i:i + 1], t[i:i + 1], cc) / tr.sig_pde, xi.shape[0])
    want_flux = brute(lambda i, cc: (surface_flux(tr.model, tb[i:i + 1], cc) - tr.q_surf(tb[i:i + 1])) / tr.sig_q,
                      tb.shape[0])
    assert abs(float(got['pde_c']) - want_pde) < 1e-10 * (1 + want_pde)
    assert abs(float(got['flux_c']) - want_flux) < 1e-10 * (1 + want_flux)


def test_centre_term_off_by_default():
    tr = _local_trainer(n_tan=0)
    B = tr.batch(32, 4)
    assert B[5] is None and tr.val[5] is None
    assert set(tr.losses(*B)) == {'pde', 'flux'}


def test_local_zoom_features():
    """Zoom block = (phi(c) - phi(c0)) / rho, per-sample and shared c; causal; zoom off leaves phi unchanged."""
    c0 = _coef()
    F0 = CausalPathFeatures(SP)
    Fz = CausalPathFeatures(SP, center=c0.numpy(), scale=0.05)
    assert Fz.n_out == 2 * F0.n_out
    g = torch.Generator().manual_seed(3)
    t = 30.0 * torch.rand(40, 1, generator=g)
    C = c0[None, :] + 0.05 * torch.randn(40, SP.n_coef, generator=g)
    phi = F0(t, C)
    out = Fz(t, C)
    ref = (phi - F0(t, c0)) / 0.05
    assert torch.max(torch.abs(out[:, :F0.n_out] - phi)) == 0.0
    assert torch.max(torch.abs(out[:, F0.n_out:] - ref)) < 1e-10
    # shared c path equals the per-sample path
    out_shared = Fz(t, C[7])
    out_ps = Fz(t, C[7].expand(40, -1))
    assert torch.max(torch.abs(out_shared - out_ps)) < 1e-12
    # still causal
    c = C[0].clone().requires_grad_(True)
    for k in range(4, SP.n_coef):
        start = (k - 3) * SP.dt
        tt = torch.linspace(0.0, start - 2 * Fz.dg, 25).reshape(-1, 1)
        J = torch.autograd.functional.jacobian(lambda cc: Fz(tt, cc), c)
        assert float(J[:, :, k].abs().max()) == 0.0


def test_local_zoom_model_roundtrip():
    """build_param_model with model.local_zoom: centre/scale from the local file, saved and reloaded as buffers."""
    from zbinv.parametric import build_param_model
    from zbinv.train import load_dataset
    with open(os.path.join(ROOT, 'configs', 'p1_param_synth_v3.yaml')) as f:
        cfg = yaml.safe_load(f)
    lf = os.path.join(ROOT, 'runs', 'p2_clean_inv0', 'local_center.npz')
    cfg['sampling'] = {'mode': 'local_file', 'local_file': lf, 'frac_local': 0.9}
    cfg['model']['local_zoom'] = True
    ds = load_dataset(cfg)
    m = build_param_model(cfg, ds, SP, torch.device('cpu'))
    z = np.load(lf)
    assert np.allclose(m.features.zoom_center.numpy(), z['center'])
    assert abs(float(m.features.zoom_scale) - np.sqrt(np.mean(np.diag(z['cov'])))) < 1e-12
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "m.pt")
    torch.save(m.state_dict(), p)
    m2 = build_param_model(cfg, ds, SP, torch.device('cpu'))
    m2.load_state_dict(torch.load(p, weights_only=True))
    t = torch.linspace(0.1, 29.0, 7).reshape(-1, 1)
    xi = -0.3 * torch.ones_like(t)
    c = torch.tensor(z['center'])
    assert torch.max(torch.abs(m.h_tilde(xi, t, c) - m2.h_tilde(xi, t, c))) == 0.0


def test_zoom_warm_start_reproduces_function():
    """A zoom model loaded from a network trained without zoom (zero weights on the zoom inputs) = same function;
    a different anchor is refused."""
    from zbinv.parametric import load_state_zoom_aware
    m0 = _model(5)
    c0 = _coef()

    def zoom_model(center, scale):
        feats = CausalPathFeatures(SP, center=center, scale=scale)
        head = HeadNet(32, 3, t_scale_days=30.0, n_cond=feats.n_out, sqrt_t=True)
        zb = ZbSplineTorch(SP, np.zeros(SP.n_coef), 0.05)
        return InversePINN(SOIL, zb, head, 0.25, features=feats)

    mz = zoom_model(c0.numpy(), 0.05)
    load_state_zoom_aware(mz, m0.state_dict())
    g = torch.Generator().manual_seed(7)
    n = 50
    xi = -torch.rand(n, 1, generator=g)
    t = 30.0 * torch.rand(n, 1, generator=g)
    C = c0[None, :] + 0.1 * torch.randn(n, SP.n_coef, generator=g)
    assert torch.max(torch.abs(mz.h_tilde(xi, t, C) - m0.h_tilde(xi, t, C))) == 0.0
    # zoom -> zoom with the same anchor loads; with another anchor it is refused
    mz2 = zoom_model(c0.numpy(), 0.05)
    load_state_zoom_aware(mz2, mz.state_dict())
    assert torch.max(torch.abs(mz2.h_tilde(xi, t, C) - mz.h_tilde(xi, t, C))) == 0.0
    mz3 = zoom_model(c0.numpy() + 0.01, 0.05)
    try:
        load_state_zoom_aware(mz3, mz.state_dict())
        raise AssertionError('anchor mismatch not detected')
    except ValueError:
        pass


def test_first_order_corrected_model_matches_forward_at_point():
    """LM model = operator + value offset + slope correction at c_k: equals the forward network in value and
    Jacobian at c_k (first-order consistency), and one LM solve of a problem that is linear for the forward network
    recovers the Gauss-Newton solution J_fwd^+ (y - theta_fwd(c_k))."""
    from zbinv.parametric import invert_lm, member_thetas, jacobian
    op, fw = [_model(1), _model(2)], [_model(3)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    ck = _coef()
    f_op = lambda cc: member_thetas(op, t, 0.05, cc).mean(0)
    f_fw = lambda cc: member_thetas(fw, t, 0.05, cc).mean(0)
    off = f_fw(ck) - f_op(ck)
    J_f, J_o = jacobian(f_fw, ck), jacobian(f_op, ck)
    r = invert_lm(op, t, 0.05, f_fw(ck), 1.0, ck, max_iter=0, verbose=False, offset=off, lin=(J_f - J_o, ck))
    assert float(torch.max(torch.abs(r['theta'] - f_fw(ck)))) < 1e-13
    assert float(torch.max(torch.abs(r['J'] - J_f))) < 1e-10 * (1 + float(J_f.abs().max()))
    # without the corrections the model is the operator ensemble
    r0 = invert_lm(op, t, 0.05, f_fw(ck), 1.0, ck, max_iter=0, verbose=False)
    assert float(torch.max(torch.abs(r0['J'] - J_o))) < 1e-10 * (1 + float(J_o.abs().max()))


def test_secant_correction_is_least_change():
    """D dC = dO exactly for independent differences, and D is zero across their span (operator Jacobian kept)."""
    from zbinv.parametric import secant_correction
    g = torch.Generator().manual_seed(3)
    dC = torch.randn(SP.n_coef, 3, generator=g)
    D_true = torch.randn(40, SP.n_coef, generator=g)
    D = secant_correction(dC, D_true @ dC)
    assert float(torch.max(torch.abs(D @ dC - D_true @ dC))) < 1e-10
    v = torch.randn(SP.n_coef, generator=g)
    v = v - dC @ torch.linalg.lstsq(dC, v[:, None]).solution[:, 0]        # component orthogonal to the history
    assert float(torch.max(torch.abs(D @ v))) < 1e-10
    # with n_coef independent differences the linear discrepancy is recovered completely
    dC9 = torch.randn(SP.n_coef, SP.n_coef, generator=g)
    assert float(torch.max(torch.abs(secant_correction(dC9, D_true @ dC9) - D_true))) < 1e-8


def test_affine_slope_is_central_difference_on_fd_design():
    """On the symmetric design c0 +- d_i (scripts/fd_design.py) the fitted slope of a quadratic discrepancy is its
    exact gradient at c0 (the even part cancels), with or without the centre point; the design displacements are
    scale x posterior sd along the posterior eigen-directions."""
    import importlib.util
    import os
    from zbinv.parametric import affine_slope
    spec = importlib.util.spec_from_file_location(
        'fd_design', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'fd_design.py'))
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    g = torch.Generator().manual_seed(5)
    n, m = SP.n_coef, 40
    A = torch.randn(n, n, generator=g)
    cov = (A @ A.T / n + 0.1 * torch.eye(n)).numpy()
    c0 = _coef()
    Dm = torch.tensor(fd.design(c0.numpy(), cov, 3.0))
    # displacements: orthogonal in the posterior metric, each of length 3 posterior sd
    W = Dm.T @ torch.linalg.inv(torch.tensor(cov)) @ Dm
    assert float(torch.max(torch.abs(W - 9.0 * torch.eye(n)))) < 1e-8
    a, G = torch.randn(m, generator=g), torch.randn(m, n, generator=g)
    Q = torch.randn(m, n, n, generator=g)
    Q = Q + Q.transpose(1, 2)
    o = lambda c: a + G @ (c - c0) + 0.5 * torch.einsum('i,mij,j->m', c - c0, Q, c - c0)
    pts = [c0 + s * Dm[:, i] for i in range(n) for s in (1.0, -1.0)]
    for P in (pts, [c0] + pts):
        D, resid = affine_slope(torch.stack(P, 1), torch.stack([o(c) for c in P], 1))
        assert float(torch.max(torch.abs(D - G))) < 1e-9 * (1 + float(G.abs().max()))
        assert 0.0 < resid < 1.0
    # an exactly affine discrepancy is fitted without residual, also from an off-centre set of points
    P = pts + [c0 + 0.3 * Dm[:, 0] + 0.2 * Dm[:, 3]]
    D, resid = affine_slope(torch.stack(P, 1), torch.stack([a + G @ (c - c0) for c in P], 1))
    assert float(torch.max(torch.abs(D - G))) < 1e-9 * (1 + float(G.abs().max())) and resid < 1e-9
    # too few directions: refused rather than silently rank-deficient
    try:
        affine_slope(torch.stack(pts[:6], 1), torch.stack([o(c) for c in pts[:6]], 1))
        assert False, 'rank-deficient design was accepted'
    except ValueError:
        pass


def test_percoef_channels():
    """Per-coefficient channels chi_j = E_taumax,j(t) c_j: appended after the shared block (unchanged), sum to the
    longest memory, each depends on its own coefficient only and is causal; off by default."""
    F0 = CausalPathFeatures(SP)
    Fp = CausalPathFeatures(SP, percoef=True)
    assert Fp.n_out == F0.n_out + SP.n_coef and not F0.percoef
    g = torch.Generator().manual_seed(5)
    t = 30.0 * torch.rand(50, 1, generator=g)
    C = _coef()[None, :] + 0.1 * torch.randn(50, SP.n_coef, generator=g)
    phi, out = F0(t, C), Fp(t, C)
    assert torch.max(torch.abs(out[:, :F0.n_out] - phi)) == 0.0
    chi = out[:, F0.n_out:]
    assert torch.max(torch.abs(chi.sum(1) - F0.memories(t, C)[:, Fp.k_long])) < 1e-12
    assert torch.max(torch.abs(Fp(t, C[3]) - Fp(t, C[3].expand(50, -1)))) < 1e-12
    c = C[0].clone()
    J = torch.autograd.functional.jacobian(lambda cc: Fp(t, cc)[:, F0.n_out:], c)        # (N, n_coef, n_coef)
    off = J - torch.diag_embed(torch.diagonal(J, dim1=1, dim2=2))
    assert float(off.abs().max()) == 0.0
    for k in range(4, SP.n_coef):
        before = t[:, 0] < (k - 3) * SP.dt - 2 * Fp.dg
        assert float(J[before][:, k, k].abs().max()) == 0.0
        assert float(J[~before][:, k, k].abs().max()) > 0.0


def test_percoef_model_sensitivity_is_free_per_coefficient():
    """With the per-coefficient channels d theta / d c is exact (jacfwd = finite differences) and the model builds
    from the config switch model.percoef_memory."""
    from zbinv.parametric import build_param_model
    from zbinv.train import load_dataset
    with open(os.path.join(ROOT, 'configs', 'p1_param_synth_v3.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['percoef_memory'] = True
    ds = load_dataset(cfg)
    m = build_param_model(cfg, ds, SP, torch.device('cpu'))
    assert m.head.n_cond == 2 + len(cfg['model']['memory_taus_days']) + SP.n_coef
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    t = torch.linspace(0.5, 29.5, 9).reshape(-1, 1)
    c = _coef()
    fn = lambda cc: theta_obs_model(m, t, 0.05, cc).reshape(-1)
    J = torch.func.jacfwd(fn)(c)
    d = 1e-5
    for k in range(SP.n_coef):
        e = torch.zeros_like(c)
        e[k] = d
        fd = (fn(c + e) - fn(c - e)) / (2 * d)
        assert torch.max(torch.abs(J[:, k] - fd)) < 1e-7 * (1 + float(fd.abs().max()))


def test_percoef_warm_start_reproduces_function():
    """A model with per-coefficient channels loaded from one without them (zero weights on the new inputs) is the
    same function; a zoomed model cannot be started from a per-coefficient one (the blocks would not line up)."""
    from zbinv.parametric import load_state_zoom_aware
    m0 = _model(6)
    c0 = _coef()

    def model(**kw):
        feats = CausalPathFeatures(SP, **kw)
        head = HeadNet(32, 3, t_scale_days=30.0, n_cond=feats.n_out, sqrt_t=True)
        return InversePINN(SOIL, ZbSplineTorch(SP, np.zeros(SP.n_coef), 0.05), head, 0.25, features=feats)

    g = torch.Generator().manual_seed(8)
    n = 50
    xi = -torch.rand(n, 1, generator=g)
    t = 30.0 * torch.rand(n, 1, generator=g)
    C = c0[None, :] + 0.1 * torch.randn(n, SP.n_coef, generator=g)
    mp = model(percoef=True)
    load_state_zoom_aware(mp, m0.state_dict())
    assert torch.max(torch.abs(mp.h_tilde(xi, t, C) - m0.h_tilde(xi, t, C))) == 0.0
    mzp = model(center=c0.numpy(), scale=0.05, percoef=True)
    load_state_zoom_aware(mzp, m0.state_dict())
    assert torch.max(torch.abs(mzp.h_tilde(xi, t, C) - m0.h_tilde(xi, t, C))) == 0.0
    mp2 = model(percoef=True)
    load_state_zoom_aware(mp2, mp.state_dict())
    assert torch.max(torch.abs(mp2.h_tilde(xi, t, C) - mp.h_tilde(xi, t, C))) == 0.0
    try:
        load_state_zoom_aware(model(center=c0.numpy(), scale=0.05, percoef=True), mp.state_dict())
        raise AssertionError('misaligned feature blocks not detected')
    except ValueError:
        pass


def test_lm_box_keeps_coefficients_in_prior_support():
    """CPrior.box is the support of the prior in coefficient space, and invert_lm(box=...) never leaves it: the start
    is projected and every accepted iterate stays inside, while the cost does not increase. Without a box the same
    call is the unconstrained LM."""
    from zbinv.parametric import invert_lm, member_thetas
    pr = CPrior(SP, 0.05, (0.3, 1.2), 0.15, (0.2, 1.5))
    lo, hi = pr.box()
    C = pr.sample(2000, torch.Generator().manual_seed(0))
    assert float(C.min()) >= lo - 1e-12 and float(C.max()) <= hi + 1e-12
    assert abs(float(torch.nn.functional.softplus(torch.tensor(lo))) + 0.05 - 0.2) < 1e-12
    assert abs(float(torch.nn.functional.softplus(torch.tensor(hi))) + 0.05 - 1.5) < 1e-12
    op = [_model(1), _model(2)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    target = _coef()
    target[-1] = hi + 3.0                                   # data generated outside the box in the last coefficient
    y = member_thetas(op, t, 0.05, target).mean(0)
    c0 = torch.full_like(target, lo - 1.0)                  # start outside the box
    r = invert_lm(op, t, 0.05, y, 1e-3, c0, max_iter=8, verbose=False, box=(lo, hi))
    assert float(r['c'].min()) >= lo - 1e-12 and float(r['c'].max()) <= hi + 1e-12
    costs = [h['cost'] for h in r['hist']]
    assert all(b <= a for a, b in zip(costs, costs[1:]))
    start_cost = float(torch.mean(((member_thetas(op, t, 0.05, torch.clamp(c0, lo, hi)).mean(0) - y) / 1e-3) ** 2))
    assert r['cost'] <= start_cost
    r_free = invert_lm(op, t, 0.05, y, 1e-3, target + 0.01, max_iter=3, verbose=False)
    r_none = invert_lm(op, t, 0.05, y, 1e-3, target + 0.01, max_iter=3, verbose=False, box=None)
    assert float(torch.max(torch.abs(r_free['c'] - r_none['c']))) == 0.0


def test_lm_box_converges_to_the_constrained_minimum():
    """With a box, invert_lm ends at a stationary point of the constrained problem (active set, not a bare clamp of
    the unconstrained step): the residual is orthogonal to the Jacobian column of every coefficient inside the box,
    and for a coefficient on a bound the gradient points out of the box."""
    from zbinv.parametric import invert_lm, member_thetas, jacobian
    lo, hi = CPrior(SP, 0.05, (0.3, 1.2), 0.15, (0.2, 1.5)).box()
    op = [_model(1), _model(2)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    target = _coef()
    target[-1] = hi + 3.0                                   # the data ask for a last coefficient above the box
    fn = lambda c: member_thetas(op, t, 0.05, c).mean(0)    # noqa: E731
    y = fn(target)
    r = invert_lm(op, t, 0.05, y, 1e-3, target, max_iter=100, verbose=False, box=(lo, hi))
    c = r['c']
    assert float(c[-1]) == hi and r['cost'] > 0.0
    res = (fn(c) - y) / 1e-3
    J = jacobian(fn, c) / 1e-3
    g = J.T @ res
    on_lo, on_hi = c <= lo, c >= hi
    inside = ~(on_lo | on_hi)
    assert int(inside.sum()) >= 1
    assert bool(torch.all(g[on_hi] < 0)) and bool(torch.all(g[on_lo] > 0))
    cosine = torch.abs(g) / (torch.linalg.norm(J, dim=0) * torch.linalg.norm(res))
    assert float(cosine[inside].max()) < 1e-3, cosine
    # a bare clamp of the full step leaves the free coefficients short of this point: the constrained cost found here
    # is not above the cost of the clamped unconstrained solution
    r_un = invert_lm(op, t, 0.05, y, 1e-3, target, max_iter=100, verbose=False)
    c_cl = torch.clamp(r_un['c'], lo, hi)
    assert r['cost'] <= float(torch.mean(((fn(c_cl) - y) / 1e-3) ** 2)) + 1e-12


def test_box_defect_is_the_water_balance_of_the_layer():
    """box_water_defect (from the residuals) = storage change of the layer minus what the prescribed surface flux and
    the flux through its lower face supply, computed here from storage and fluxes; for an arbitrary (untrained) model."""
    from zbinv.physics import box_water_defect
    m, c = _model(), _coef()
    t0, dur, depth, nt, nx = 6.0, 3.0, 0.4, 240, 120

    def q_surf(t):
        return 0.02 * torch.sin(1.7 * t)

    tm = (t0 + dur * (torch.arange(nt) + 0.5) / nt).reshape(-1, 1)
    xm = -depth * (torch.arange(nx) + 0.5) / nx
    d = box_water_defect(m, q_surf, xm.reshape(1, 1, nx).expand(1, nt, nx), tm.reshape(1, nt), c.reshape(1, -1),
                         torch.tensor([[dur]]), torch.tensor([[depth]]))
    z = (m.zb_m(tm, c) / SOIL.L).detach()
    got = float(d) * float(z.mean()) * depth / SOIL.theta_star

    gx, gw = np.polynomial.legendre.leggauss(48)
    xg = torch.tensor(-depth * (1.0 - gx) / 2.0).reshape(-1, 1)

    def W(t):                                                  # z~ int_xi_a^0 M dxi at one time
        tt = torch.full_like(xg, t)
        with torch.no_grad():
            M = SOIL.storage(m.h_tilde(xg, tt, c)).reshape(-1)
            return float(m.zb_m(tt[:1], c) / SOIL.L) * depth / 2.0 * float((torch.tensor(gw) * M).sum())

    ta = tm.clone().requires_grad_(True)
    xa = torch.full_like(ta, -depth).requires_grad_(True)
    za = m.zb_m(ta, c) / SOIL.L
    zdot = m.T_days * torch.autograd.grad(za.sum(), ta, create_graph=True)[0]
    h = m.h_tilde(xa, ta, c)
    q = -SOIL.K(h) * (torch.autograd.grad(h.sum(), xa, create_graph=True)[0] / za + 1.0)
    G = (q - SOIL.storage(h) * xa * zdot).detach()             # flux through the lower face (which moves with zb)
    supply = float((-q_surf(tm) + G).mean()) * dur / m.T_days
    want = W(t0 + dur) - W(t0) - supply
    scale = abs(W(t0 + dur) - W(t0)) + abs(supply)
    assert scale > 1e-4 and abs(got - want) < 1e-3 * scale, (got, want, scale)


def test_box_term_optional():
    tr = _local_trainer(n_tan=0)
    B = tr.batch(32, 4)
    assert len(B) == 7 and B[6] is None and tr.val[6] is None and 'box' not in tr.heldout_loss()

    tr = _local_trainer(n_tan=0, n_box=3, box_n_t=4, box_n_xi=2)
    xi, t, c, dur, depth = tr.val[6]
    assert xi.shape == (12, 4, 2) and t.shape == (12, 4) and c.shape[0] == 12
    T = float(tr.ds['n_days'])
    assert abs(tr.box_dt_max - tr.soil.theta_star * 1.5 / tr.soil.Ks / 86400.0) < 1e-9      # drain time of a 1.5 m column
    assert float(dur.min()) >= tr.box_dt_min * (1 - 1e-12) and float(dur.max()) <= tr.box_dt_max * (1 + 1e-12)
    assert float(depth.min()) >= tr.box_d_min * (1 - 1e-12) and float(depth.max()) <= 1.0
    assert float(t.min()) >= 0.0 and float(t.max()) <= T * (1 + 1e-12) and bool((t.max(1).values - t.min(1).values <= dur[:, 0]).all())
    assert bool((xi <= 0).all()) and bool((xi >= -depth.unsqueeze(2)).all())
    B = tr.batch(32, 4, n_box=3)
    L = tr.losses(*B)
    assert set(L) == {'pde', 'flux', 'box'} and L['box'].requires_grad and bool(torch.isfinite(L['box']))
    a, b = tr.heldout_loss(chunk=8)['box'], tr.heldout_loss()['box']
    assert np.isfinite(a) and abs(a - b) < 1e-10 * (1 + abs(b))
