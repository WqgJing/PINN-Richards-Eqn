"""Smooth van Genuchten-Mualem soil functions in dimensionless form.

One definition serves both the reference solver (numpy) and the PINN (torch), so the
twin experiment solves exactly the same constitutive model in both.

Scaling (same as the FEM and the legacy PINN code):
    h~ = h / L,  K~ = K / Ks,  q~ = q / Ks,  t~ = t / T,  T = theta_star * L / Ks.

Smoothing near h = 0 (needed for second derivatives in the PINN):
    |h| -> sqrt(h^2 + eps)                (eps = abs_smooth_m^2, in metres)
    Se  = b + (1 - b) * Se_vg(|h|)       b = sigmoid(h / blend_width)
    K   = b + (1 - b) * kr(Se_vg)
Away from |h| ~ blend_width this is the standard VG-Mualem model.

Air-entry option (air_entry_m = |h_e|, recommended for n < 2):
    For n < 2 the Mualem conductivity of plain VG drops with infinite slope at saturation (here K
    falls to 0.92 Ks at |h| = 1 mm and 0.44 Ks at 4 cm), a near-singular layer just above the water
    table. Vogel et al. (2001, AWR 24) and Ippisch et al. (2006, AWR 29) fix it with an air-entry
    head h_e < 0 (HYDRUS: "VG-Mualem with air-entry value of -2 cm"):
        Se = Se_vg(hc) / S_c,   S_c = Se_vg(h_e)
        kr = Se^l * [ (1 - (1 - (Se S_c)^(1/m))^m) / (1 - (1 - S_c^(1/m))^m) ]^2
    with hc = min(h, h_e), here a smooth minimum of width ae_smooth_m so that the PINN has second
    derivatives: hc = h_e - w softplus((h_e - h) / w). Se = K = 1 for h >> h_e. The blend and |h|
    smoothing above are then not used.
"""
import numpy as np
import torch
from scipy.special import expit


def _is_torch(x):
    return isinstance(x, torch.Tensor)


class SoilVG:
    def __init__(self, theta_s, theta_r, alpha, n, Ks, L, l=0.5, Ss=1.0e-5,
                 blend_width_m=0.02, abs_smooth_m=1.0e-3, air_entry_m=None, ae_smooth_m=0.005):
        self.theta_s = float(theta_s)
        self.theta_r = float(theta_r)
        self.alpha = float(alpha)          # 1/m
        self.n = float(n)
        self.m = 1.0 - 1.0 / self.n
        self.Ks = float(Ks)                # m/s
        self.l = float(l)
        self.L = float(L)                  # m
        self.theta_star = self.theta_s - self.theta_r
        self.T = self.theta_star * self.L / self.Ks          # s
        self.alpha_t = self.alpha * self.L                   # alpha~
        self.ss_t = float(Ss) * self.L / self.theta_star     # storage coefficient in tilde form
        self.w_t = float(blend_width_m) / self.L             # blend width in h~ units
        self.eps_t = (float(abs_smooth_m) / self.L) ** 2
        self.air_entry = air_entry_m is not None
        if self.air_entry:
            self.he_t = -abs(float(air_entry_m)) / self.L         # air-entry head (h~ units, < 0)
            self.wae_t = float(ae_smooth_m) / self.L               # smooth-min width
            self.S_c = float((1.0 + (self.alpha_t * -self.he_t) ** self.n) ** (-self.m))
            self._kr_den = float((1.0 - (1.0 - self.S_c ** (1.0 / self.m)) ** self.m) ** 2)

    @classmethod
    def from_config(cls, soil_cfg, L):
        return cls(theta_s=soil_cfg['theta_s'], theta_r=soil_cfg['theta_r'],
                   alpha=soil_cfg['alpha'], n=soil_cfg['n'], Ks=soil_cfg['Ks'], L=L,
                   l=soil_cfg.get('l', 0.5), Ss=soil_cfg.get('Ss', 1.0e-5),
                   blend_width_m=soil_cfg.get('blend_width_m', 0.02),
                   abs_smooth_m=soil_cfg.get('abs_smooth_m', 1.0e-3),
                   air_entry_m=soil_cfg.get('air_entry_m'), ae_smooth_m=soil_cfg.get('ae_smooth_m', 0.005))

    # ------------------------------------------------------------------ helpers
    def _sigmoid(self, x):
        return torch.sigmoid(x) if _is_torch(x) else expit(x)

    def _sabs(self, h):
        return (h * h + self.eps_t) ** 0.5

    def _blend(self, h):
        return self._sigmoid(h / self.w_t)

    # ------------------------------------------------------------------ VG core
    def Se_vg(self, h):
        """Plain VG effective saturation evaluated at the smoothed |h| (always < 1)."""
        return (1.0 + (self.alpha_t * self._sabs(h)) ** self.n) ** (-self.m)

    def kr_from_Se(self, se):
        """Mualem relative conductivity for 0 < Se < 1."""
        x = 1.0 - se ** (1.0 / self.m)
        if _is_torch(x):
            x = torch.clamp(x, min=1e-30)
        else:
            x = np.maximum(x, 1e-30)
        return se ** self.l * (1.0 - x ** self.m) ** 2

    # ------------------------------------------------------------------ air-entry model
    def _softplus(self, x):
        """Stable softplus, the same formula in numpy and torch: max(x, 0) + log1p(exp(-|x|))."""
        if _is_torch(x):
            return torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    def _hc(self, h):
        """Smooth min(h, h_e), always < h_e < 0."""
        return self.he_t - self.wae_t * self._softplus((self.he_t - h) / self.wae_t)

    def _Se_ae(self, h):
        return (1.0 + (self.alpha_t * -self._hc(h)) ** self.n) ** (-self.m) / self.S_c

    def _kr_ae(self, se):
        x = 1.0 - (se * self.S_c) ** (1.0 / self.m)
        x = torch.clamp(x, min=1e-30) if _is_torch(x) else np.maximum(x, 1e-30)
        return se ** self.l * (1.0 - x ** self.m) ** 2 / self._kr_den

    # ------------------------------------------------------------------ public
    def Se(self, h):
        if self.air_entry:
            return self._Se_ae(h)
        b = self._blend(h)
        return b + (1.0 - b) * self.Se_vg(h)

    def theta(self, h):
        return self.theta_r + self.theta_star * self.Se(h)

    def K(self, h):
        if self.air_entry:
            return self._kr_ae(self._Se_ae(h))
        b = self._blend(h)
        return b + (1.0 - b) * self.kr_from_Se(self.Se_vg(h))

    def storage(self, h):
        """Water storage variable M = Se + ss~ * h~ used in the mass balance."""
        return self.Se(h) + self.ss_t * h

    def capacity(self, h):
        """dSe/dh~ (analytic, numpy). Add ss_t for the total capacity."""
        h = np.asarray(h, dtype=float)
        if self.air_entry:
            hc = self._hc(h)
            a_s = self.alpha_t * -hc
            u = a_s ** self.n
            dse_dhc = self.m * self.n * self.alpha_t * a_s ** (self.n - 1.0) * (1.0 + u) ** (-self.m - 1.0)
            dhc_dh = expit((self.he_t - h) / self.wae_t)
            return dse_dhc * dhc_dh / self.S_c
        s = self._sabs(h)
        a_s = self.alpha_t * s
        u = a_s ** self.n
        se_vg = (1.0 + u) ** (-self.m)
        dse_vg_ds = -self.m * self.n * self.alpha_t * a_s ** (self.n - 1.0) * (1.0 + u) ** (-self.m - 1.0)
        dse_vg_dh = dse_vg_ds * (h / s)
        b = expit(h / self.w_t)
        db_dh = b * (1.0 - b) / self.w_t
        return db_dh * (1.0 - se_vg) + (1.0 - b) * dse_vg_dh

    # ------------------------------------------------------------------ inverse (plain VG)
    def h_from_theta(self, theta):
        """VG inverse in metres (dimensional; ignores the smoothing), for initial guesses only."""
        se = np.clip((np.asarray(theta, dtype=float) - self.theta_r) / self.theta_star, 1e-9, 1 - 1e-12)
        if self.air_entry:
            se = se * self.S_c
        return -((se ** (-1.0 / self.m) - 1.0) ** (1.0 / self.n)) / self.alpha

    def summary(self):
        return (f"theta_s={self.theta_s} theta_r={self.theta_r} alpha={self.alpha} n={self.n} "
                f"Ks={self.Ks:.3e} L={self.L} T={self.T / 86400:.3f} d alpha~={self.alpha_t:.3f} "
                f"ss~={self.ss_t:.2e} blend_w~={self.w_t:.4f}"
                + (f" air-entry h_e={self.he_t * self.L:.3f} m (S_c={self.S_c:.5f}, smooth {self.wae_t * self.L:.4f} m)"
                   if self.air_entry else ""))
