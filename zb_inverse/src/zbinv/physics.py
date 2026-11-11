"""Residuals of the inverse problem (all dimensionless, t~ = t / T).

PDE in the Landau coordinate xi = z / zb(t), z~ = zb / L:
    R = dM/dt~|xi - (xi * zdot / z~) dM/dxi + (1 / z~) dq~/dxi
    M = Se(h) + ss~ h,   q~ = -K(h) ((1 / z~) dh/dxi + 1),   zdot = dz~/dt~
Surface flux:  q~(xi = 0, t) - q_surf~(t)
Data:          mean theta over the top depth_m  -  theta_obs(t)
Inputs are tensors (N, 1); t is in days from the window start.
Root water uptake (optional, model.sink = roots.RootSink): R gains S L / Ks, with S [1/s] the uptake rate.
Water balance of space-time boxes (optional): the same residuals, added up over a layer and a time window.
c (optional, parametric PINN): zb-path spline coefficients, (n_coef,) shared or (N, n_coef) per point.
"""
import numpy as np
import torch

from .data import GL_X, GL_W, obs_layer


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def _zb(model, t, c):
    """zb(t) [m]; c is passed only for the parametric model (keeps the c-free model interface unchanged)."""
    return model.zb_m(t) if c is None else model.zb_m(t, c)


def _h(model, xi, t, c):
    return model.h_tilde(xi, t) if c is None else model.h_tilde(xi, t, c)


def pde_residual(model, xi, t_days, c=None):
    soil = model.soil
    Td = model.T_days
    xi = xi.detach().requires_grad_(True)
    t = t_days.detach().requires_grad_(True)
    z = _zb(model, t, c) / soil.L
    zdot = Td * _grad(z, t)
    h = _h(model, xi, t, c)
    M = soil.storage(h)
    dM_dt = Td * _grad(M, t)
    dM_dxi = _grad(M, xi)
    dh_dxi = _grad(h, xi)
    q = -soil.K(h) * (dh_dxi / z + 1.0)
    dq_dxi = _grad(q, xi)
    R = dM_dt - (xi * zdot / z) * dM_dxi + dq_dxi / z
    sink = getattr(model, 'sink', None)
    if sink is not None:
        R = R + sink(-xi * z * soil.L, t, h * soil.L) * (soil.L / soil.Ks)
    return R


def surface_flux(model, t_days, c=None):
    """Model Darcy flux q~ at xi = 0 (positive upward)."""
    soil = model.soil
    xi0 = torch.zeros_like(t_days).requires_grad_(True)
    z = _zb(model, t_days, c) / soil.L
    h = _h(model, xi0, t_days, c)
    dh_dxi = _grad(h, xi0)
    return -soil.K(h) * (dh_dxi / z + 1.0)


def box_water_defect(model, q_surf, xi, t_days, c, dur_days, depth):
    """Water balance of space-time boxes [-depth, 0] x [t0, t0 + dur] in (xi, t), one zb path per box.

    With the prescribed surface flux q_surf~ the layer between the surface and xi_a = -depth obeys
        W(t1) - W(t0) = int [ -q_surf~ + (q~ - M xi zdot)(xi_a) - z~ int S~ dxi ] dt~,     W = z~ int_xi_a^0 M dxi.
    For the model, the defect of this balance is exactly
        B = int [ z~ int_xi_a^0 R dxi - (q~(0, t) - q_surf~(t)) ] dt~
    (R: the PDE residual, sink included), so it is zero for the exact solution whatever the quadrature. It is
    estimated here by means over the given points: xi (n, nt, nx) in the layer, t_days (n, nt) in the window.
    A residual of one sign adds up in B over the window; one that changes sign cancels (the pointwise mean square
    does not tell these apart, the error of the solution does).
    q_surf: callable t_days (N, 1) -> q_surf~ (N, 1); c (n, n_coef); dur_days, depth (n, 1).
    Returns theta* B / (z~ depth) (n, 1): the defect as a mean water content of the layer."""
    soil = model.soil
    n, nt, nx = xi.shape
    tt = t_days.reshape(-1, 1)
    ct = c.repeat_interleave(nt, dim=0)
    R = pde_residual(model, xi.reshape(-1, 1), t_days.unsqueeze(2).expand(-1, -1, nx).reshape(-1, 1),
                     ct.repeat_interleave(nx, dim=0)).reshape(n, nt, nx).mean(2)
    f = (surface_flux(model, tt, ct) - q_surf(tt)).reshape(n, nt)
    z = (_zb(model, tt, ct) / soil.L).reshape(n, nt)
    B = (z * depth * R - f).mean(1, keepdim=True) * (dur_days / model.T_days)
    return soil.theta_star * B / (z.mean(1, keepdim=True) * depth)


_GLX = torch.tensor(GL_X, dtype=torch.float64)
_GLW = torch.tensor(GL_W / 2.0, dtype=torch.float64)


def theta_obs_model(model, t_days, depth_m, c=None):
    """Mean theta over the observed layer (data.obs_layer: the top depth_m, or [top, bottom] of a buried sensor;
    4-point Gauss-Legendre in xi), shape (N, 1)."""
    soil = model.soil
    glx = _GLX.to(t_days.device)
    glw = _GLW.to(t_days.device)
    top, bottom = obs_layer(depth_m)
    zb = _zb(model, t_days, c)                                # (N, 1) metres
    xi = -(top / zb + ((bottom - top) / zb) * (1.0 - glx[None, :]) / 2.0)   # (N, 4)
    tt = t_days.expand(-1, glx.numel())
    cc = c if (c is None or c.dim() == 1) else c.repeat_interleave(glx.numel(), dim=0)
    h = _h(model, xi.reshape(-1, 1), tt.reshape(-1, 1), cc).reshape(-1, glx.numel())
    return (soil.theta(h) * glw[None, :]).sum(dim=1, keepdim=True)


def theta_profile(model, xi_nodes, t_days, c=None):
    """theta on a grid: xi_nodes (Nx,), t_days (Nt,) -> (Nt, Nx), no grad. c: (n_coef,) shared, if parametric."""
    with torch.no_grad():
        X, Tt = torch.meshgrid(xi_nodes, t_days, indexing='xy')
        h = _h(model, X.reshape(-1, 1), Tt.reshape(-1, 1), c).reshape(Tt.shape)
        return model.soil.theta(h)
