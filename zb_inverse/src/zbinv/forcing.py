"""Causal memory features of the known forcing (surface flux, transpiration) for the head of the parametric PINN.

The forcing series are piecewise linear in time (daily nodes for site forcing), so the solution has kinks in time
where the forcing has them, while the head's other time inputs (Fourier features, features of the zb path) are smooth.
Each feature is an exponential memory g of one forcing series u(t):

    tau g' = u - g,    g(0) = 0      (no forcing before the window, as the hydrostatic initial condition assumes)

On a segment [t_k, t_k+1], where u = u_k + m_k (t - t_k), the memory is known in closed form,

    g(t) = u(t) - m_k tau + (g_k - u_k + m_k tau) exp(-(t - t_k) / tau),

so it is evaluated exactly at any t (no table). It is C1 in t (g' = (u - g) / tau is continuous, so d h / d t stays
continuous, as it is in the soil below the surface), and its slope has the forcing's kinks at the right times.
A ladder of tau spans the response times: hours at the surface, days at depth (a sum of exponentials approximates
the t^(-1/2) kernel of a diffusive response to a boundary flux). Scaled by max |u| so the inputs are O(1).
No trainable weights, and not a function of the zb path.

A series that changes sign (the surface flux: evaporation upward, infiltration downward, an order of magnitude apart)
can be split into its two parts (split_by_sign), each a piecewise-linear series scaled by its own maximum.
"""
import numpy as np
import torch
import torch.nn as nn


def memory_at_nodes(t, u, tau):
    """g at the nodes t (K,) of the piecewise-linear u (K,), for one tau; g[0] = 0. Exact recursion."""
    g = np.zeros_like(u, dtype=np.float64)
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        m = (u[k + 1] - u[k]) / dt
        g[k + 1] = u[k + 1] - m * tau + (g[k] - u[k] + m * tau) * np.exp(-dt / tau)
    return g


def split_by_sign(t, u):
    """Piecewise-linear u on nodes t -> (t2, max(u, 0), min(u, 0)), exactly: t2 has the zero crossings as extra nodes,
    so both parts are piecewise linear on t2 and add up to u. Node values that are zero to rounding (below 1e-12 of
    the largest) count as zero, so that no crossing falls on a node."""
    t, u = np.asarray(t, dtype=np.float64).reshape(-1), np.asarray(u, dtype=np.float64).reshape(-1)
    u = np.where(np.abs(u) < 1e-12 * np.abs(u).max(), 0.0, u)
    k = np.nonzero(u[:-1] * u[1:] < 0)[0]
    tz = t[k] - u[k] * (t[k + 1] - t[k]) / (u[k + 1] - u[k])
    t2 = np.unique(np.concatenate([t, tz]))
    u2 = np.interp(t2, t, u)
    u2[np.isin(t2, tz)] = 0.0
    return t2, np.maximum(u2, 0.0), np.minimum(u2, 0.0)


class ForcingMemoryFeatures(nn.Module):
    def __init__(self, series, taus_days, t_end=None):
        """series: list of (t_nodes_days, values), each 1-D with increasing t starting at the window start (0);
        taus_days: memory times [d]; t_end: end of the window [d], a series that stops before it is held constant
        from its last node (as the clamped interpolation of the forcing in the residuals does).
        Output: (N, n_series * n_tau), series by series."""
        super().__init__()
        taus = np.asarray(taus_days, dtype=np.float64).reshape(-1)
        if taus.size == 0 or np.any(taus <= 0):
            raise ValueError('forcing memory times must be positive')
        self.n_series, self.n_tau = len(series), taus.size
        self.n_out = self.n_series * self.n_tau
        # tables, not weights: rebuilt from the dataset when a model is built, kept out of the state dict
        self.register_buffer('taus', torch.as_tensor(taus).reshape(1, -1), persistent=False)
        for s, (t, u) in enumerate(series):
            t = np.asarray(t, dtype=np.float64).reshape(-1)
            u = np.asarray(u, dtype=np.float64).reshape(-1)
            if t.size != u.size or t.size < 2 or np.any(np.diff(t) <= 0) or abs(t[0]) > 1e-9:
                raise ValueError('forcing series: nodes must increase from t = 0, one value per node')
            if t_end is not None and t[-1] < float(t_end) - 1e-9:
                t, u = np.append(t, float(t_end)), np.append(u, u[-1])
            scale = float(np.abs(u).max())
            u = u / scale if scale > 0 else u
            g = np.stack([memory_at_nodes(t, u, tau) for tau in taus], axis=1)          # (K, n_tau)
            self.register_buffer(f't{s}', torch.as_tensor(t), persistent=False)
            self.register_buffer(f'u{s}', torch.as_tensor(u).reshape(-1, 1), persistent=False)
            self.register_buffer(f'm{s}', torch.as_tensor(np.diff(u) / np.diff(t)).reshape(-1, 1), persistent=False)
            self.register_buffer(f'g{s}', torch.as_tensor(g), persistent=False)

    def forward(self, t_days):
        """t_days: (N, 1) in [0, T]. Differentiable in t (the segment index is piecewise constant)."""
        tf = t_days.reshape(-1, 1)
        out = []
        for s in range(self.n_series):
            t, u, m, g = (getattr(self, f'{name}{s}') for name in 'tumg')
            k = torch.clamp(torch.searchsorted(t, tf.detach().reshape(-1).contiguous(), right=True) - 1, 0, t.numel() - 2)
            dt = tf - t[k].reshape(-1, 1)
            uk, mk = u[k], m[k]                                                         # (N, 1)
            out.append(uk + mk * dt - mk * self.taus + (g[k] - uk + mk * self.taus) * torch.exp(-dt / self.taus))
        return torch.cat(out, dim=1)
