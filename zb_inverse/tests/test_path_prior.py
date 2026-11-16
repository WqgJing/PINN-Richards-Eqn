"""Path term of the inversion cost (CPrior.increments) and its use in invert_lm and posterior_cov."""
import os
import sys
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from zbinv.soil import SoilVG  # noqa: E402
from zbinv.model import HeadNet, InversePINN  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineTorch, CausalPathFeatures, _inv_softplus_np  # noqa: E402
from zbinv.parametric import CPrior, inv_softplus_t, invert_lm, posterior_cov, member_thetas, jacobian  # noqa: E402

torch.set_default_dtype(torch.float64)
SOIL = SoilVG(theta_s=0.43, theta_r=0.02, alpha=3.6, n=1.56, Ks=2.9e-6, L=2.0, air_entry_m=0.02)
SP = CubicBSpline1D(0.0, 30.0, 5.0)
SD = 0.15


def _prior(sp=SP):
    return CPrior(sp, 0.05, (0.3, 1.2), SD, (0.1, 1.5))


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


def test_p_is_the_step_of_the_knot_depths_over_the_sd():
    sp = CubicBSpline1D(0.0, 11.0, 1.0)                      # 14 coefficients, as the site windows
    pr = _prior(sp)
    p, P = pr.increments(torch.full((sp.n_coef,), 0.3))
    assert float(p.abs().max()) == 0.0 and P.shape == (sp.n_coef - 1, sp.n_coef)      # a flat path costs nothing
    z = torch.tensor([0.10, 1.50, 0.11, 0.97] + [0.5] * (sp.n_coef - 4))              # knot depths [m]
    p, _ = pr.increments(inv_softplus_t(z - 0.05))
    assert torch.allclose(p, torch.diff(z) / SD, rtol=0, atol=1e-10)
    assert abs(float(p[0]) - 1.4 / SD) < 1e-9                # a 1.4 m step between two knots is 9.3 sd


def test_jacobian_of_the_term_against_central_differences():
    pr = _prior()
    c = _coef() + 0.3 * torch.sin(torch.arange(SP.n_coef) * 2.1)
    _, P = pr.increments(c)
    d = 1e-6
    for k in range(SP.n_coef):
        e = torch.zeros(SP.n_coef)
        e[k] = d
        fd = (pr.increments(c + e)[0] - pr.increments(c - e)[0]) / (2 * d)
        assert float((fd - P[:, k]).abs().max()) < 1e-7, k


def test_training_draws_have_independent_unit_steps():
    """Before clipping the draws of CPrior.sample have independent N(0, 1) entries of p: sum(p^2) is minus twice the
    log density of the training prior over the shape of the path."""
    pr = CPrior(SP, 0.05, (5.0, 6.0), SD, (0.2, 100.0))      # level far from the clip: no draw is clipped
    C = pr.sample(4000, torch.Generator().manual_seed(0))
    p = torch.stack([pr.increments(c)[0] for c in C])        # (4000, n_coef - 1)
    assert abs(float(p.mean())) < 0.02
    assert abs(float((p ** 2).mean()) - 1.0) < 0.03          # se of the mean of p^2 over 32000 values: 0.008
    corr = np.corrcoef(p.numpy().T)
    assert float(np.abs(corr - np.eye(SP.n_coef - 1)).max()) < 0.08      # se 0.016 each


def test_lm_with_the_term_ends_at_the_minimum_of_the_stated_cost():
    pr = _prior()
    op = [_model(1), _model(2)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    fn = lambda c: member_thetas(op, t, 0.05, c).mean(0)     # noqa: E731
    sig = 2e-3
    y = fn(_coef()) + sig * torch.randn(60, generator=torch.Generator().manual_seed(3))
    c0 = _coef(0.2)
    r = invert_lm(op, t, 0.05, y, sig, c0, max_iter=200, verbose=False, prior=pr.increments)
    c = r['c']

    def cost(cc):
        return float(torch.mean(((fn(cc) - y) / sig) ** 2) + torch.sum(pr.increments(cc)[0] ** 2) / 60)
    assert abs(r['cost'] - cost(c)) < 1e-12 * max(1.0, r['cost'])            # the cost reported is the stated one
    # stationary: the gradient J^T r + P^T p vanishes against the size of its two parts
    J = jacobian(fn, c) / sig
    res = (fn(c) - y) / sig
    p, P = pr.increments(c)
    g = J.T @ res + P.T @ p
    assert float(g.abs().max()) < 1e-4 * float((J.T @ res).abs().max() + (P.T @ p).abs().max()), g
    # and a minimum: no nearby path has a lower cost
    gen = torch.Generator().manual_seed(4)
    for _ in range(20):
        assert cost(c + 1e-3 * torch.randn(SP.n_coef, generator=gen)) >= cost(c) - 1e-12
    # the term is active there: the estimate is not a minimum of the data misfit alone, and the fit without the term,
    # started at it, lowers the data misfit. (Each run only lowers its own cost; which of two runs from another start
    # ends lower in the data misfit is not fixed on a model with more than one minimum, so it is not asserted.)
    r0 = invert_lm(op, t, 0.05, y, sig, c, max_iter=200, verbose=False)
    data = lambda cc: float(torch.mean(((fn(cc) - y) / sig) ** 2))      # noqa: E731
    assert data(r0['c']) < data(c)
    assert r0['cost'] == data(r0['c'])


def test_without_data_the_term_alone_gives_a_flat_path():
    pr = _prior()
    op = [_model(1)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    y = member_thetas(op, t, 0.05, _coef()).mean(0)
    r = invert_lm(op, t, 0.05, y, 1e9, _coef(0.1), max_iter=200, verbose=False, prior=pr.increments)      # no data weight
    z = torch.nn.functional.softplus(r['c'])
    assert float(z.max() - z.min()) < 1e-5, z


def test_lm_without_the_term_is_the_call_without_the_argument():
    op = [_model(1), _model(2)]
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    y = member_thetas(op, t, 0.05, _coef()).mean(0)
    a = invert_lm(op, t, 0.05, y, 1e-3, _coef(0.05), max_iter=5, verbose=False)
    b = invert_lm(op, t, 0.05, y, 1e-3, _coef(0.05), max_iter=5, verbose=False, prior=None)
    assert float((a['c'] - b['c']).abs().max()) == 0.0 and a['cost'] == b['cost']
    assert a['cost'] == float(torch.mean(((a['theta'] - y) / 1e-3) ** 2))      # the data misfit alone, as before


def test_posterior_precision_gets_the_term():
    pr = _prior()
    op = [_model(1)]                                         # one member: no ensemble covariance
    t = torch.linspace(0.5, 30, 60).reshape(-1, 1)
    c = _coef()
    sig = 2e-3
    J = jacobian(lambda cc: member_thetas(op, t, 0.05, cc).mean(0), c)
    _, P = pr.increments(c)
    cov, cov_noise, Se = posterior_cov(J, sig, op, t, 0.05, c, P=P)
    H = J.T @ J / sig ** 2 + P.T @ P
    assert float(Se.abs().max()) == 0.0
    assert torch.allclose(cov @ H, torch.eye(SP.n_coef), atol=1e-6) and torch.allclose(cov, cov_noise, rtol=1e-9)
    cov0, cov0_noise, _ = posterior_cov(J, sig, op, t, 0.05, c)
    assert torch.equal(cov0_noise, torch.linalg.pinv(J.T @ J / sig ** 2, rtol=1e-12))      # unchanged without P
    sd, sd0 = torch.sqrt(torch.diag(cov)), torch.sqrt(torch.diag(cov0))
    assert bool(torch.all(sd <= sd0 * (1 + 1e-9)))           # a prior never widens the posterior
