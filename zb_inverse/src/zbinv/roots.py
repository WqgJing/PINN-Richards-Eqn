"""Root water uptake of the wtd_pipeline FEM (fem/root_water_uptake.py, RootWaterUptakeTorch) for the PINN residual.

S(depth, t, h) = alpha(h, T_p) * T_p(t) * b(depth)   [1/s], a sink in the water balance:  d theta/dt + dq/dz + S = 0
  alpha  Feddes stress: 0 above h1 (too wet), rising to 1 at h2, 1 down to h3, falling to 0 at h4;
         h3 moves from h3_low to h3_high as T_p rises to tp_threshold
  b      Gale & Grigal root density, -ln(beta) 100 beta^(100 depth) [1/m], cut at z_root_max and renormalised
T_p(t) is linear between its daily values, as the surface flux is. Defaults: the FEM's (grass / pasture).
"""
import numpy as np
import torch
import torch.nn as nn

from .data import torch_interp


class RootSink(nn.Module):
    def __init__(self, t_days, tp_mmd, h1=-0.10, h2=-0.25, h3_low=-6.0, h3_high=-3.0, h4=-80.0, tp_threshold=5e-8,
                 beta=0.96, z_root_max=3.0):
        super().__init__()
        # tables, not weights: kept out of the state dict so saved models load with or without a sink
        self.register_buffer('t_tp', torch.as_tensor(np.asarray(t_days, dtype=np.float64)), persistent=False)
        self.register_buffer('tp', torch.as_tensor(np.asarray(tp_mmd, dtype=np.float64)) / 86400e3, persistent=False)
        self.h1, self.h2, self.h3_low, self.h3_high, self.h4 = h1, h2, h3_low, h3_high, h4
        self.tp_threshold, self.beta, self.z_root_max = tp_threshold, beta, z_root_max

    def transpiration(self, t_days):
        """T_p(t) [m/s]."""
        return torch_interp(t_days, self.t_tp, self.tp)

    def stress(self, h_m, tp):
        h3 = self.h3_low + (self.h3_high - self.h3_low) * torch.clamp(tp / self.tp_threshold, 0.0, 1.0)
        wet = (self.h1 - h_m) / (self.h1 - self.h2)
        dry = (h_m - self.h4) / (h3 - self.h4)
        return torch.clamp(torch.minimum(wet, dry), 0.0, 1.0)

    def root_density(self, depth_m):
        d = torch.clamp(depth_m, min=0.0)
        b = -np.log(self.beta) * 100.0 * torch.exp(np.log(self.beta) * 100.0 * d)
        b = torch.where(d > self.z_root_max, torch.zeros_like(b), b)
        return b / (1.0 - self.beta ** (100.0 * self.z_root_max))

    def forward(self, depth_m, t_days, h_m):
        """Uptake rate S [1/s] at depth depth_m >= 0 [m], time t_days and head h_m [m]; tensors of one shape."""
        tp = self.transpiration(t_days)
        return self.stress(h_m, tp) * tp * self.root_density(depth_m)
