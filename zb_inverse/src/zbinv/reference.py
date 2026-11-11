"""Method-of-lines reference solver for Richards' equation in the Landau coordinate.

Problem (dimensionless, z up, xi = z / zb(t) in [-1, 0]):
    d(z M)/dt|xi + d/dxi [ q - xi * zdot * M ] = 0          (conservative form)
    M = Se(h) + ss~ * h,   q = -K(h) * ((1/z) dh/dxi + 1)
    h(xi = -1) = 0                       water table
    q(xi = 0)  = q_surf(t)               surface flux, positive upward

Vertex-centred finite volumes on a (possibly stretched) xi grid; time integration with
scipy BDF. Uses the same SoilVG functions as the PINN, so it is the exact twin truth.
"""
import time as _time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def make_grid(n_nodes, gamma=1.0):
    """xi nodes from -1 (WT) to 0 (surface); gamma > 1 clusters nodes at the surface."""
    s = np.linspace(0.0, 1.0, n_nodes)
    return -(1.0 - s) ** gamma


class ReferenceSolver:
    def __init__(self, soil, xi, h_crit_m=None, crit_width=0.1):
        """h_crit_m: optional minimum surface head (m, < 0). With it the surface condition is the
        system-dependent BC of HYDRUS-1D (Simunek et al.): the prescribed flux is the potential flux;
        under evaporation it is reduced once the surface head approaches h_crit_m, so the soil only
        evaporates what it can deliver (smooth gate in log|h|, width crit_width). Without it the
        flux BC is exact and an undeliverable evaporation dries the surface without bound."""
        self.soil = soil
        self.h_crit_t = None if h_crit_m is None else float(h_crit_m) / soil.L
        self.crit_width = float(crit_width)
        self.xi = np.asarray(xi, dtype=float)
        n = self.xi.size
        self.dxi = np.diff(self.xi)
        self.xi_f = 0.5 * (self.xi[:-1] + self.xi[1:])
        V = np.empty(n)
        V[1:-1] = 0.5 * (self.xi[2:] - self.xi[:-2])
        V[-1] = 0.5 * self.dxi[-1]
        V[0] = 0.5 * self.dxi[0]
        self.V = V

    def fluxes(self, h, z, zdot, qs):
        """Face fluxes F (n-1,) and the top boundary flux, all dimensionless."""
        s = self.soil
        K = s.K(h)
        M = s.storage(h)
        Kf = 0.5 * (K[:-1] + K[1:])
        q_f = -Kf * (np.diff(h) / (z * self.dxi) + 1.0)
        M_f = 0.5 * (M[:-1] + M[1:])
        F = q_f - self.xi_f * zdot * M_f
        return F, self.top_flux(h[-1], qs), M

    def top_flux(self, h_top, qs):
        """Actual surface flux: qs (potential) unless evaporation is supply-limited (see __init__)."""
        if self.h_crit_t is None or qs <= 0.0:
            return qs
        x = np.log(max(-h_top, 1e-300) / -self.h_crit_t) / self.crit_width
        return qs * np.exp(-np.logaddexp(0.0, x))

    def rhs(self, h_int, z, zdot, qs):
        h = np.concatenate(([0.0], h_int))
        F, F_top, M = self.fluxes(h, z, zdot, qs)
        dzM = np.empty_like(h)
        dzM[1:-1] = -(F[1:] - F[:-1]) / self.V[1:-1]
        dzM[-1] = -(F_top - F[-1]) / self.V[-1]
        cap = self.soil.capacity(h) + self.soil.ss_t
        dh = (dzM - zdot * M) / (z * cap)
        return dh[1:]

    def solve(self, t_days_eval, zb_fn, zbdot_fn, qsurf_fn, h0, rtol=1e-6, atol=1e-9,
              max_step_hours=1.0, verbose=True):
        """zb_fn(t_days)->m, zbdot_fn(t_days)->m/day, qsurf_fn(t_days)->m/s. Returns dict."""
        s = self.soil
        T_days = s.T / 86400.0
        L = s.L

        def fun(tt, y):
            td = tt * T_days
            z = zb_fn(td) / L
            zdot = zbdot_fn(td) * T_days / L          # dz~/dt~
            qs = qsurf_fn(td) / s.Ks
            return self.rhs(y, z, zdot, qs)

        n_int = self.xi.size - 1
        jac_sp = diags([np.ones(n_int - 1), np.ones(n_int), np.ones(n_int - 1)], [-1, 0, 1])
        t_eval = np.asarray(t_days_eval, dtype=float) / T_days
        tic = _time.time()
        sol = solve_ivp(fun, (t_eval[0], t_eval[-1]), np.asarray(h0, dtype=float)[1:],
                        method='BDF', t_eval=t_eval, rtol=rtol, atol=atol,
                        jac_sparsity=jac_sp, max_step=max_step_hours / 24.0 / T_days)
        if verbose:
            print(f"  reference: {sol.message} nfev={sol.nfev} njev={sol.njev} "
                  f"time={_time.time() - tic:.1f}s", flush=True)
        if not sol.success:
            raise RuntimeError(f"reference solver failed: {sol.message}")
        h = np.concatenate((np.zeros((1, sol.t.size)), sol.y), axis=0).T   # (Nt, Nnodes)
        t_days = sol.t * T_days
        return self._diagnostics(t_days, h, zb_fn, zbdot_fn, qsurf_fn)

    def _diagnostics(self, t_days, h, zb_fn, zbdot_fn, qsurf_fn):
        s = self.soil
        T_days = s.T / 86400.0
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=float), t_days.shape).copy()
        zb = bc(zb_fn(t_days))
        z = zb / s.L
        zdot = bc(zbdot_fn(t_days)) * T_days / s.L
        qs = bc(qsurf_fn(t_days)) / s.Ks
        storage = np.empty(t_days.size)
        F_bot = np.empty(t_days.size)
        q_top_model = np.empty(t_days.size)
        q_act = np.empty(t_days.size)
        for k in range(t_days.size):
            F, q_act[k], M = self.fluxes(h[k], z[k], zdot[k], qs[k])
            storage[k] = z[k] * np.dot(self.V[1:], M[1:])
            F_bot[k] = F[0]
            # surface Darcy flux from the top two nodes (check of the flux BC, first order)
            Kt = 0.5 * (s.K(h[k, -1]) + s.K(h[k, -2]))
            q_top_model[k] = -Kt * ((h[k, -1] - h[k, -2]) / (z[k] * self.xi[-1] - z[k] * self.xi[-2]) + 1.0)
        # mass balance: d(storage)/dt~ = F_bot - q_act (actual surface flux)  (integrated with trapezoid on output times)
        tt = t_days / T_days
        flux_in = np.concatenate(([0.0], np.cumsum(0.5 * np.diff(tt) * ((F_bot - q_act)[1:] + (F_bot - q_act)[:-1]))))
        mass_err = (storage - storage[0]) - flux_in
        return {
            't_days': t_days, 'xi': self.xi, 'h_t': h, 'theta': s.theta(h), 'zb': zb,
            'qsurf': qs * s.Ks, 'q_act': q_act * s.Ks, 'storage_t': storage, 'mass_err_t': mass_err,
            'q_top_model': q_top_model * s.Ks,
        }


def hydrostatic_h(xi, zb_m, L):
    """Equilibrium (zero-flux) head h~ = -(z + zb)/L on the xi grid."""
    return -(zb_m / L) * (1.0 + np.asarray(xi, dtype=float))
