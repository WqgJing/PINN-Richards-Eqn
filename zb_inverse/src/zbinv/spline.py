"""Uniform cubic B-spline for the water-table depth zb(t).

zb(t) = zb_min + softplus(f(t)),   f(t) = sum_k c_k B_k(t)

- Time is in days relative to the window start.
- n_coef = n_intervals + 3. The spline is C2, so dzb/dt is continuous (needed by the
  Landau advection term).
- The same class evaluates in numpy (truth / reference) and torch (learnable).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_np(x):
    return np.logaddexp(0.0, x)


def _inv_softplus_np(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 30.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))


class CubicBSpline1D:
    def __init__(self, t0_days, t1_days, knot_days):
        self.t0 = float(t0_days)
        self.t1 = float(t1_days)
        self.n_int = max(1, int(math.ceil((self.t1 - self.t0) / float(knot_days) - 1e-9)))
        self.dt = (self.t1 - self.t0) / self.n_int
        self.n_coef = self.n_int + 3

    # local coordinates ------------------------------------------------------
    def _local_np(self, t):
        x = (np.asarray(t, dtype=float) - self.t0) / self.dt
        i = np.clip(np.floor(x), 0, self.n_int - 1).astype(int)
        return i, x - i

    def _local_torch(self, t):
        x = (t - self.t0) / self.dt
        i = torch.clamp(torch.floor(x.detach()), 0, self.n_int - 1)
        return i.long(), x - i

    @staticmethod
    def _weights(u):
        u2, u3 = u * u, u * u * u
        w0 = (1.0 - u) ** 3 / 6.0
        w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
        w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
        w3 = u3 / 6.0
        return w0, w1, w2, w3

    @staticmethod
    def _dweights(u):
        return (-(1.0 - u) ** 2 / 2.0, 1.5 * u * u - 2.0 * u, -1.5 * u * u + u + 0.5, u * u / 2.0)

    # numpy -------------------------------------------------------------------
    def basis_np(self, t):
        """Dense basis matrix B (len(t), n_coef)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i, u = self._local_np(t)
        B = np.zeros((t.size, self.n_coef))
        for j, w in enumerate(self._weights(u)):
            B[np.arange(t.size), i + j] += w
        return B

    def dbasis_np(self, t):
        """d/dt of the basis, per day."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i, u = self._local_np(t)
        D = np.zeros((t.size, self.n_coef))
        for j, w in enumerate(self._dweights(u)):
            D[np.arange(t.size), i + j] += w / self.dt
        return D

    # torch -------------------------------------------------------------------
    @staticmethod
    def _pick(coef, idx):
        """coef[idx] for shared coef (n_coef,) or per-sample coef (N, n_coef) (one row per idx entry)."""
        if coef.dim() == 1:
            return coef[idx]
        return coef.gather(1, idx.unsqueeze(1)).squeeze(1)

    def eval_torch(self, t, coef):
        """f(t) for t of shape (N, 1) in days; differentiable in t and coef.
        coef: (n_coef,) shared by all points, or (N, n_coef), one coefficient vector per point."""
        tf = t.reshape(-1)
        i, u = self._local_torch(tf)
        w = self._weights(u)
        out = torch.zeros_like(tf)
        for j in range(4):
            out = out + w[j] * self._pick(coef, i + j)
        return out.reshape(t.shape)

    def deval_torch(self, t, coef):
        """df/dt (per day), same shapes as eval_torch; differentiable in t (C1 spline -> continuous)."""
        tf = t.reshape(-1)
        i, u = self._local_torch(tf)
        w = self._dweights(u)
        out = torch.zeros_like(tf)
        for j in range(4):
            out = out + (w[j] / self.dt) * self._pick(coef, i + j)
        return out.reshape(t.shape)

    # causal memory of the path ----------------------------------------------------
    def filtered_basis_np(self, t, tau_days, h_days=1.0 / 1440.0):
        """E(t) (len(t), n_coef): exponential memory of the basis, dE/dt = (B(t) - E) / tau, E(0) = B(0)
        (constant past = hydrostatic history before the window, consistent with the initial condition).
        g(t) = E(t) . c is the causal moving average of f(t) = B(t) . c with time constant tau: it depends only
        on f over [0, t]. Exact exponential integrator for B linear over each sub-step h (1 min)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        n = int(np.ceil((self.t1 - self.t0) / h_days))
        tg = self.t0 + np.arange(n + 1) * (self.t1 - self.t0) / n
        h = tg[1] - tg[0]
        B = self.basis_np(tg)
        a = math.exp(-h / tau_days)
        b1 = 1.0 - a
        b2 = 1.0 - (tau_days / h) * (1.0 - a)
        E = np.empty_like(B)
        E[0] = B[0]
        for k in range(n):
            E[k + 1] = a * E[k] + b1 * B[k] + b2 * (B[k + 1] - B[k])
        out = np.empty((t.size, self.n_coef))
        for j in range(self.n_coef):
            out[:, j] = np.interp(t, tg, E[:, j])
        return out


class CausalPathFeatures(nn.Module):
    """Smooth causal features of the zb path for a parametric network, all linear in the coefficients c:
        f(t; c), dt_knot * df/dt(t; c), and g_tau(t; c) = E_tau(t) . c for each tau (exponential memories).
    E_tau is tabulated on a grid (default 15 min) and interpolated by cubic Hermite with the exact slope
    (B - E) / tau, so every feature is C1 in t (the PDE needs d/dt of the network; clamped lags would put kinks
    in dh/dt). Feature k at time t depends on c only through the path on [0, t] (causal).

    Optional local zoom (center c0, scale rho): the features are then [phi(t; c), phi(t; c - c0) / rho]. phi is
    linear in c, so the second block is (phi(t; c) - phi(t; c0)) / rho: the same function class (the first layer
    is linear), but the variation of the path inside a local ball of radius ~rho enters the network at O(1)
    instead of O(rho), i.e. standard input normalisation for the local training distribution.

    Optional per-coefficient channels (percoef): chi_j(t; c) = E_taumax,j(t) c_j, one per coefficient (their sum is
    the longest memory g_taumax). The 2 + n_tau shared channels carry d h / d c_j only through n_tau exponential
    lag kernels shared by all coefficients; with chi_j the network's response to each coefficient is a free function
    of (xi, t), still causal (E_j(t) = 0 before the support of B_j starts), so any causal linearised response
    (sensitivity) is representable."""

    def __init__(self, spline, taus_days=(0.25, 1.0, 4.0, 16.0), grid_days=1.0 / 96.0, center=None, scale=None,
                 percoef=False):
        super().__init__()
        self.s = spline
        self.zoom = center is not None
        if self.zoom:
            if scale is None or not float(scale) > 0.0:
                raise ValueError('local zoom needs a positive scale')
            self.register_buffer('zoom_center', torch.tensor(np.asarray(center, dtype=float), dtype=torch.float64))
            self.register_buffer('zoom_scale', torch.tensor(float(scale), dtype=torch.float64))
        self.taus = tuple(float(x) for x in taus_days)
        n = int(np.ceil((spline.t1 - spline.t0) / grid_days))
        tg = spline.t0 + np.arange(n + 1) * (spline.t1 - spline.t0) / n
        self.t0, self.dg, self.n_grid = spline.t0, float(tg[1] - tg[0]), n + 1
        B = spline.basis_np(tg)
        E = np.stack([spline.filtered_basis_np(tg, tau) for tau in self.taus])            # (n_tau, n_grid, n_coef)
        S = np.stack([(B - E[k]) / tau for k, tau in enumerate(self.taus)])
        self.register_buffer('E', torch.tensor(E, dtype=torch.float64))
        self.register_buffer('S', torch.tensor(S, dtype=torch.float64))
        self.n_base = 2 + len(self.taus)
        self.percoef = bool(percoef)
        self.k_long = int(np.argmax(self.taus))
        self.n_out = self.n_base * (2 if self.zoom else 1) + (spline.n_coef if self.percoef else 0)

    def memory_rows(self, t):
        """E_tau(t) for all taus (cubic Hermite in t): (n_tau, N, n_coef)."""
        tf = t.reshape(-1)
        x = (tf - self.t0) / self.dg
        k = torch.clamp(torch.floor(x.detach()), 0, self.n_grid - 2).long()
        u = (x - k)[None, :, None]
        h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
        h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        return (h00 * self.E[:, k] + h10 * self.dg * self.S[:, k]
                + h01 * self.E[:, k + 1] + h11 * self.dg * self.S[:, k + 1])

    @staticmethod
    def _contract(rows, coef):
        if coef.dim() == 1:
            return torch.einsum('anc,c->na', rows, coef)
        return torch.einsum('anc,nc->na', rows, coef)

    def memories(self, t, coef):
        """g_tau(t; c) for all taus: (N, n_tau). coef (N, n_coef) or (n_coef,)."""
        return self._contract(self.memory_rows(t), coef)

    def base(self, t, coef, rows=None):
        f = self.s.eval_torch(t, coef).reshape(-1, 1)
        fdot = self.s.dt * self.s.deval_torch(t, coef).reshape(-1, 1)
        return torch.cat([f, fdot, self._contract(self.memory_rows(t) if rows is None else rows, coef)], dim=1)

    def forward(self, t, coef):
        rows = self.memory_rows(t)
        out = [self.base(t, coef, rows)]
        if self.zoom:
            out.append(self.base(t, coef - self.zoom_center, rows) / self.zoom_scale)
        if self.percoef:
            out.append(rows[self.k_long] * coef)              # (N, n_coef): E_taumax,j(t) c_j
        return out[0] if len(out) == 1 else torch.cat(out, dim=1)


class ZbSplineNumpy:
    """Fixed (truth) zb(t) in metres; t in days from window start."""

    def __init__(self, spline, coef, zb_min):
        self.s = spline
        self.coef = np.asarray(coef, dtype=float)
        self.zb_min = float(zb_min)

    @classmethod
    def fit(cls, spline, t_days, zb_m, zb_min, ridge=1e-8):
        B = spline.basis_np(t_days)
        y = _inv_softplus_np(np.asarray(zb_m) - zb_min)
        A = B.T @ B + ridge * np.eye(spline.n_coef)
        coef = np.linalg.solve(A, B.T @ y)
        return cls(spline, coef, zb_min)

    def zb(self, t_days):
        return self.zb_min + _softplus_np(self.s.basis_np(t_days) @ self.coef)

    def dzb_dt(self, t_days):
        """m/day"""
        f = self.s.basis_np(t_days) @ self.coef
        return (1.0 / (1.0 + np.exp(-f))) * (self.s.dbasis_np(t_days) @ self.coef)

    def dzb_dcoef(self, t_days):
        """Jacobian (len(t), n_coef) of zb with respect to the coefficients, metres."""
        f = self.s.basis_np(t_days) @ self.coef
        return (1.0 / (1.0 + np.exp(-f)))[:, None] * self.s.basis_np(t_days)


class ZbSplineTorch(nn.Module):
    """Learnable zb(t) in metres; t in days from window start (tensor (N,1))."""

    def __init__(self, spline, coef_init, zb_min):
        super().__init__()
        self.s = spline
        self.zb_min = float(zb_min)
        self.coef = nn.Parameter(torch.as_tensor(np.asarray(coef_init), dtype=torch.float64))

    def forward(self, t_days, coef=None):
        """coef=None: the learnable coefficients; else (n_coef,) or per-sample (N, n_coef)."""
        return self.zb_min + F.softplus(self.s.eval_torch(t_days, self.coef if coef is None else coef))
