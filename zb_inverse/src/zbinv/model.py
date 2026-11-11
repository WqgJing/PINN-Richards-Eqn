"""PINN for h(xi, t) with hard water-table and initial conditions, plus a spline zb(t).

Equilibrium-plus-departure form (base='current', default). Two departure forms:
  departure='additive' (default):
    h~(xi, t) = (1 + xi) * [ -zb~(t) + s(t) * N(xi, t) ]
  departure='log':
    h~(xi, t) = -(1 + xi) * zb~(t) * exp( s(t) * N(xi, t) )
  with s(t) = 1 - exp(-t / tau).
- -(1 + xi) zb~(t) = -(z + zb)/L is the zero-flux (hydrostatic) profile for the CURRENT water
  table, so N only has to learn the departure from equilibrium caused by surface forcing.
- h~(-1, t) = 0 exactly (water table); h~(xi, 0) = hydrostatic at zb(0) (exact IC).
- N's last layer starts at zero, so training starts from equilibrium (zero flux everywhere).
- 'log' makes the departure multiplicative in |h|: a dry surface layer where |h| grows by orders of
  magnitude over a few mm (supply-limited evaporation; |h| ~ depth^-0.4 for this VG-Mualem soil) is
  then a gentle function of N. It keeps h < 0 above the water table (no perched saturation).
  With log_depth_eps the network also gets the input ln(eps - xi), in which that power law is linear.
base='initial' keeps the old form (hydrostatic at zb(0) for all t), which implies large spurious
fluxes when the WT moves and trained poorly.
zb(t) is NOT a network input: it reaches the profile only through the equilibrium baseline and
the physics residuals.
"""
import math
import torch
import torch.nn as nn


class HeadNet(nn.Module):
    """MLP for the departure N. Optional extra inputs (all off by default = the original network):
    n_cond conditioning features (parametric PINN: features of the zb path, see spline.CausalPathFeatures) and
    sqrt_t, an input sqrt(t / t_scale) for the sqrt(t) surface layer where the hydrostatic initial condition
    meets a non-zero surface flux."""

    SQRT_T_EPS_DAYS = 1e-3       # keeps d/dt sqrt(t + eps) finite at t = 0

    def __init__(self, width=64, depth=5, t_scale_days=30.0, fourier_periods_days=(2, 4, 8, 16, 32, 64),
                 log_depth_eps=None, n_cond=0, sqrt_t=False):
        super().__init__()
        self.t_scale = float(t_scale_days)
        self.register_buffer('omegas', torch.tensor([2 * math.pi / p for p in fourier_periods_days],
                                                    dtype=torch.float64))
        self.log_eps = None if log_depth_eps is None else float(log_depth_eps)
        self.n_cond = int(n_cond)
        self.sqrt_t = bool(sqrt_t)
        in_dim = 2 + 2 * len(fourier_periods_days) + (1 if self.log_eps else 0) + (1 if self.sqrt_t else 0) + self.n_cond
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        last = nn.Linear(d, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        for mod in self.net:
            if isinstance(mod, nn.Linear) and mod is not last:
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, xi, t_days, cond=None):
        feats = [2.0 * xi + 1.0, 2.0 * t_days / self.t_scale - 1.0]
        if self.log_eps:
            # ln(eps - xi) scaled to about [-1, 0]: resolves a thin layer at the surface (xi -> 0)
            feats.append(torch.log(self.log_eps - xi) / abs(math.log(self.log_eps)))
        if self.omegas.numel():
            wt = t_days * self.omegas[None, :]
            feats += [torch.sin(wt), torch.cos(wt)]
        if self.sqrt_t:
            feats.append(2.0 * torch.sqrt((t_days + self.SQRT_T_EPS_DAYS) / self.t_scale) - 1.0)
        if self.n_cond:
            if cond is None or cond.shape[1] != self.n_cond:
                raise ValueError(f'HeadNet expects {self.n_cond} conditioning features')
            feats.append(cond)
        return self.net(torch.cat(feats, dim=1))


class InversePINN(nn.Module):
    V_MAX = 15.0     # numerical guard for departure='log': |h| <= zb * e^15 (far beyond any h_crit)

    def __init__(self, soil, zb_spline, head, tau_ic_days=0.25, base='current', departure='additive',
                 features=None):
        super().__init__()
        self.soil = soil
        self.zb_spline = zb_spline
        self.head = head
        # parametric PINN: features(t, c) -> conditioning inputs of the head (spline.CausalPathFeatures);
        # then every method takes the zb-path coefficients c, (n_coef,) or per-sample (N, n_coef)
        self.features = features
        self.sink = None            # optional root water uptake (roots.RootSink): part of the physics, as the soil is
        self.forcing = None         # optional memory features of the known forcing (forcing.ForcingMemoryFeatures)
        self.tau = float(tau_ic_days)
        self.T_days = soil.T / 86400.0
        if base not in ('current', 'initial'):
            raise ValueError(base)
        if departure not in ('additive', 'log'):
            raise ValueError(departure)
        if departure == 'log' and base != 'current':
            raise ValueError("departure='log' needs base='current'")
        self.base = base
        self.departure = departure

    def zb_m(self, t_days, c=None):
        return self.zb_spline(t_days) if c is None else self.zb_spline(t_days, c)

    def zb0_tilde(self, like, c=None):
        if c is None:
            t0 = torch.zeros((1, 1), dtype=like.dtype, device=like.device)
            return self.zb_spline(t0) / self.soil.L
        t0 = torch.zeros((like.shape[0], 1), dtype=like.dtype, device=like.device)
        return self.zb_spline(t0, c) / self.soil.L

    def _head(self, xi, t_days, c):
        if self.features is None:
            return self.head(xi, t_days)
        if c is None:
            raise ValueError('parametric model: pass the zb-path coefficients c')
        cond = self.features(t_days, c)
        if self.forcing is not None:                 # before the path features, which may grow (zoom) in later stages
            cond = torch.cat([self.forcing(t_days), cond], dim=1)
        return self.head(xi, t_days, cond)

    def h_tilde(self, xi, t_days, c=None):
        s = 1.0 - torch.exp(-t_days / self.tau)
        if self.departure == 'log':
            v = s * self._head(xi, t_days, c)
            v = self.V_MAX * torch.tanh(v / self.V_MAX)           # smooth bound, exact at v = 0 (IC)
            return -(1.0 + xi) * (self.zb_m(t_days, c) / self.soil.L) * torch.exp(v)
        zb_base = self.zb_m(t_days, c) / self.soil.L if self.base == 'current' else self.zb0_tilde(xi, c)
        return (1.0 + xi) * (-zb_base + s * self._head(xi, t_days, c))
