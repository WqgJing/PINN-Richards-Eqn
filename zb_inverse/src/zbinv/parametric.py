"""Parametric (operator) PINN for the water-table inverse problem.

Why: in the joint PINN one network both solves the PDE and fits the noisy theta data, so any slack in the soft
physics penalty is spent on absorbing noise or bending weakly identified zb directions; every weighting only
moves that error around (tasks/todo.md findings 17-33). Here the roles are separated:

1. Physics-only training. The network learns the solution OPERATOR of this window's Richards problem,
       c -> h~(xi, t; c) = (1 + xi) [ -zb~(t; c) + s(t) N(xi, t, phi(t; c)) ],
   for zb paths zb(t; c) = zb_min + softplus(B(t) . c) drawn from a prior (or a local distribution).
   phi = causal path features (spline.CausalPathFeatures). Loss = PDE + surface-flux residuals only, each scaled
   by its physical sigma. The theta data never enter the training (no data-vs-physics weight anywhere).
2. Inversion. zb is fitted to theta_5cm through the trained network: Levenberg-Marquardt over the 9 spline
   coefficients with the exact autograd Jacobian d theta / d c (c is a network input, so this is the total
   derivative including the learned lag of the surface behind the water table).
3. Error model without the numerical solver: an ensemble of independently trained networks; their spread is
   the surrogate covariance added to the noise covariance in the posterior.
"""
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from .spline import CubicBSpline1D, ZbSplineNumpy, ZbSplineTorch, CausalPathFeatures
from .model import HeadNet, InversePINN
from .physics import pde_residual, surface_flux, theta_obs_model, box_water_defect
from .data import torch_interp, obs_layer
from .roots import RootSink
from .forcing import ForcingMemoryFeatures, split_by_sign
from .train import load_dataset, signal_rate_scale, initial_coef, Sampler

torch.set_default_dtype(torch.float64)


def inv_softplus_t(y):
    return y + torch.log(-torch.expm1(-y))


# ----------------------------------------------------------------------------- zb-path distributions
class CPrior:
    """Smooth random zb paths -> spline coefficients. Knot depths = mean + centred Gaussian random walk:
    mean ~ U(mean_range), increments ~ N(0, step_sd) per knot interval, clipped to `clip` [m]."""

    def __init__(self, spline, zb_min, mean_range=(0.3, 1.2), step_sd=0.15, clip=(0.2, 1.5), device='cpu'):
        self.k, self.zb_min = spline.n_coef, float(zb_min)
        self.lo, self.hi = float(mean_range[0]), float(mean_range[1])
        self.step_sd, self.clip = float(step_sd), (float(clip[0]), float(clip[1]))
        self.dev = device

    def sample_zb(self, n, gen=None):
        m = self.lo + (self.hi - self.lo) * torch.rand(n, 1, device=self.dev, generator=gen)
        walk = torch.cumsum(self.step_sd * torch.randn(n, self.k, device=self.dev, generator=gen), dim=1)
        z = m + walk - walk.mean(dim=1, keepdim=True)
        return torch.clamp(z, *self.clip)

    def sample(self, n, gen=None):
        return inv_softplus_t(self.sample_zb(n, gen) - self.zb_min)

    def box(self):
        """Support of the prior in coefficient space: every coefficient lies in [lo, hi] (knot depths in `clip`)."""
        lo, hi = inv_softplus_t(torch.tensor(self.clip, dtype=torch.float64) - self.zb_min).tolist()
        return lo, hi


class LocalGaussian:
    """c ~ N(center, cov)."""

    def __init__(self, center, cov, device='cpu'):
        self.center = torch.as_tensor(np.asarray(center), dtype=torch.float64, device=device)
        cov = torch.as_tensor(np.asarray(cov), dtype=torch.float64, device=device)
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-12 * float(torch.trace(cov)) / cov.shape[0]
        self.L = torch.linalg.cholesky(cov + jitter * torch.eye(cov.shape[0], device=device))
        self.dev = device

    def sample(self, n, gen=None):
        z = torch.randn(n, self.center.numel(), device=self.dev, generator=gen)
        return self.center[None, :] + z @ self.L.T


class MixtureC:
    """frac_local from the local Gaussian, the rest from the prior (keeps the whole support trained)."""

    def __init__(self, prior, local=None, frac_local=0.0):
        self.prior, self.local = prior, local
        self.frac = float(frac_local) if local is not None else 0.0

    def sample(self, n, gen=None):
        n_l = int(round(self.frac * n))
        parts = []
        if n_l:
            parts.append(self.local.sample(n_l, gen))
        if n - n_l:
            parts.append(self.prior.sample(n - n_l, gen))
        return torch.cat(parts, 0)


class PointC:
    """Every sample is the same path c0: the network becomes an ordinary forward PINN for that one zb path
    (accurate value at c0; d theta / d c only through the optional centre / sensitivity term, train.n_tan > 0,
    which needs the covariance cov of the delta-method objective: local = N(c0, cov), never sampled)."""

    def __init__(self, center, cov=None, device='cpu'):
        self.center = torch.as_tensor(np.asarray(center), dtype=torch.float64, device=device)
        self.local = None if cov is None else LocalGaussian(center, cov, device)

    def sample(self, n, gen=None):
        return self.center[None, :].expand(n, -1).clone()


def local_cov_from_width(coef, spline, width_m):
    """Diagonal coefficient covariance giving ~width_m [m] of zb spread per knot (softplus slope at c)."""
    slope = 1.0 / (1.0 + np.exp(-np.asarray(coef, dtype=float)))
    return np.diag((float(width_m) / slope) ** 2)


# ----------------------------------------------------------------------------- model
def build_param_model(cfg, ds, spline, device):
    mc = cfg['model']
    center = scale = None
    zoom_file = mc.get('zoom_file')
    if zoom_file is None and mc.get('local_zoom', False):
        sc = cfg.get('sampling', {})
        if sc.get('mode') != 'local_file':
            raise ValueError('model.local_zoom needs sampling.mode=local_file (its centre and covariance)')
        zoom_file = sc['local_file']
    if zoom_file is not None:
        # input normalisation for the local stage: anchor path c0 and radius rho (rms sd) of a local distribution
        z = np.load(zoom_file)
        center, scale = z['center'], float(np.sqrt(np.mean(np.diag(z['cov']))))
    feats = CausalPathFeatures(spline, tuple(mc.get('memory_taus_days', (0.25, 1.0, 4.0, 16.0))),
                               grid_days=float(mc.get('memory_grid_days', 1.0 / 96.0)), center=center, scale=scale,
                               percoef=bool(mc.get('percoef_memory', False)))
    forcing = None
    if mc.get('forcing_memory_taus_days'):          # memories of the known forcing as inputs of the head (default: none)
        if mc.get('forcing_split_sign', False):      # evaporation and infiltration as two series, each on its own scale
            t2, q_up, q_down = split_by_sign(ds['t_q'], ds['q_obs'])
            series = [(t2, q_up), (t2, q_down)]
        else:
            series = [(ds['t_q'], ds['q_obs'])]
        if ds.get('tp_mmd') is not None:
            series.append((ds['t_tp'], ds['tp_mmd']))
        forcing = ForcingMemoryFeatures(series, mc['forcing_memory_taus_days'], t_end=ds['n_days'])
    head = HeadNet(mc.get('width', 128), mc.get('depth', 6), t_scale_days=ds['n_days'],
                   fourier_periods_days=tuple(mc.get('fourier_periods_days', (2, 4, 8, 16, 32, 64))),
                   n_cond=feats.n_out + (forcing.n_out if forcing is not None else 0),
                   sqrt_t=bool(mc.get('sqrt_t', False)))
    # the module's own coefficients are unused in parametric mode (every call passes c); kept frozen
    zb_mod = ZbSplineTorch(spline, np.zeros(spline.n_coef), ds['zb_min'])
    zb_mod.coef.requires_grad_(False)
    model = InversePINN(ds['soil'], zb_mod, head, mc.get('tau_ic_days', 0.25), base='current',
                        departure=mc.get('departure', 'additive'), features=feats)
    if model.departure == 'conductive':          # the flux scale of physics_scales (sigma_q = q_frac * this)
        model.q_scale = float(np.sqrt(np.mean((ds['q_obs'] / ds['soil'].Ks) ** 2)))
    if ds.get('tp_mmd') is not None:                 # the twin has root water uptake: the same sink in the residual
        model.sink = RootSink(ds['t_tp'], ds['tp_mmd'])
    model.forcing = forcing
    return model.to(device)


def load_state_zoom_aware(model, sd):
    """Load a saved state into a parametric model. A model with the local zoom (spline.CausalPathFeatures) can be
    started from a network trained without it: the zoom inputs get zero first-layer weights, so the loaded function
    is reproduced exactly and training then learns the zoomed sensitivity. Two zoom models must share the anchor.
    The same holds for any path-feature block appended after the saved ones (zoom, per-coefficient channels)."""
    feats = model.features
    if feats is not None:
        msd = model.state_dict()
        k = 'head.net.0.weight'
        sd_zoom = 'features.zoom_center' in sd
        if sd_zoom and feats.zoom and not (torch.equal(sd['features.zoom_center'], msd['features.zoom_center'])
                                           and torch.equal(sd['features.zoom_scale'], msd['features.zoom_scale'])):
            raise ValueError('init_from has a different zoom anchor')
        n_old, n_new = sd[k].shape[1], msd[k].shape[1]
        if n_old != n_new:
            # the saved path features must be a leading block of the model's: [shared] or [shared, zoom]
            if n_old - (n_new - feats.n_out) != feats.n_base * (2 if sd_zoom else 1):
                raise ValueError(f'cannot extend first layer {tuple(sd[k].shape)} to {tuple(msd[k].shape)}')
            w_new = torch.zeros_like(msd[k])
            w_new[:, :n_old] = sd[k]
            sd = {**sd, k: w_new}
        if feats.zoom and not sd_zoom:
            sd = {**sd, 'features.zoom_center': msd['features.zoom_center'],
                  'features.zoom_scale': msd['features.zoom_scale']}
    model.load_state_dict(sd)


def physics_scales(cfg, ds):
    """sigma_q~ (flux) and sigma_pde (rate of change of Se), the same physical scales as the joint PINN."""
    sg = cfg['sigmas']
    soil = ds['soil']
    sig_q = float(sg.get('q_frac', 0.1)) * float(np.sqrt(np.mean((ds['q_obs'] / soil.Ks) ** 2)))
    sig_pde = float(sg.get('pde_factor', 1.0)) * signal_rate_scale(ds['t_obs'], ds['theta_obs'], soil,
                                                                    float(sg['theta']))
    return sig_q, sig_pde


# ----------------------------------------------------------------------------- physics-only trainer
class ParamTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        tc = cfg['train']
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(int(tc.get('seed', 0)))
        np.random.seed(int(tc.get('seed', 0)))
        self.ds = ds = load_dataset(cfg)
        self.soil = ds['soil']
        knot = cfg.get('zb_spline', {}).get('knot_days', ds['twin_cfg']['zb_spline']['knot_days'])
        self.spline = CubicBSpline1D(0.0, ds['n_days'], knot)
        self.model = build_param_model(cfg, ds, self.spline, self.dev)
        if tc.get('init_from'):
            load_state_zoom_aware(self.model, torch.load(tc['init_from'], map_location=self.dev, weights_only=True))
            print(f"[param] initialised from {tc['init_from']}", flush=True)
        self.sampler = Sampler(ds, cfg, self.dev)
        self.frac_early = float(tc.get('frac_early_t', 0.0))
        self.t_q = torch.tensor(ds['t_q'], device=self.dev)
        self.q_obs = torch.tensor(ds['q_obs'] / self.soil.Ks, device=self.dev)
        self.sig_q, self.sig_pde = physics_scales(cfg, ds)
        pc = cfg.get('prior', {})
        self.prior = CPrior(self.spline, ds['zb_min'], pc.get('mean_range_m', (0.3, 1.2)), pc.get('step_sd_m', 0.15),
                            pc.get('clip_m', (0.2, 1.5)), self.dev)
        self.c_dist = self.make_c_dist(cfg.get('sampling', {}))
        # centre (sensitivity) term, optional: n_tan > 0 needs a local distribution (its centre and covariance)
        self.n_tan = int(tc.get('n_tan', 0))
        if self.n_tan and self.c_dist.local is None:
            raise ValueError('train.n_tan > 0 needs a local sampling distribution (centre + covariance)')
        # water balance of space-time boxes, optional (default: none). Box sizes run from the smallest scale of the
        # data (its time step; its depth as a fraction of the deepest admissible water table) to the whole column
        # and the time gravity flow at Ks takes to drain the storage of the deepest admissible column (theta* zb / Ks):
        # a defect stays in a layer's balance no longer than the layer keeps its water (over longer windows it has
        # been passed on below and the defect overstates the error; scripts/box_defect_scan.py). The defect is
        # measured against the precision of the water-content data (sigmas.theta).
        self.n_box = int(tc.get('n_box', 0))
        self.box_nt, self.box_nxi = int(tc.get('box_n_t', 8)), int(tc.get('box_n_xi', 4))
        self.sig_th = float(cfg['sigmas']['theta'])
        self.box_dt_min = float(np.min(np.diff(ds['t_obs'])))
        self.box_d_min = obs_layer(cfg['obs']['depth_m'])[1] / float(pc.get('clip_m', (0.2, 1.5))[1])
        self.box_dt_max = min(float(ds['n_days']),
                              self.model.T_days * float(pc.get('clip_m', (0.2, 1.5))[1]) / self.soil.L)
        self.truth_c = torch.tensor(np.asarray(ds['truth_coef']), device=self.dev)
        self.t_obs = torch.tensor(ds['t_hourly'], device=self.dev).reshape(-1, 1)
        self.diag_truth = float(cfg['obs'].get('q_noise_frac', 0.0)) == 0.0   # twin theta valid as truth only for clean q
        self.out = cfg['output']['dir']
        os.makedirs(self.out, exist_ok=True)
        self.rows, self.step, self.t0 = [], 0, time.time()
        self.make_heldout()
        print(f"[param] device {self.dev}; sigmas: q~={self.sig_q:.3e} pde={self.sig_pde:.4e}; "
              f"params {sum(p.numel() for p in self.model.head.parameters())}", flush=True)

    # -- sampling
    def make_c_dist(self, sc):
        mode = sc.get('mode', 'prior')
        if mode == 'prior':
            return MixtureC(self.prior)
        if mode == 'local_truth':           # DIAGNOSTIC ONLY (gate G1): centred on the true coefficients
            c0 = np.asarray(self.ds['truth_coef'], dtype=float)
            loc = LocalGaussian(c0, local_cov_from_width(c0, self.spline, sc.get('local_width_m', 0.05)), self.dev)
            return MixtureC(self.prior, loc, sc.get('frac_local', 1.0))
        if mode == 'point':                 # forward PINN at one path (the current estimate)
            z = np.load(sc['local_file'])
            return PointC(z['center'], z['cov'], self.dev)
        if mode == 'local_file':            # refinement: centre and covariance from the last inversion
            z = np.load(sc['local_file'])
            loc = LocalGaussian(z['center'], z['cov'], self.dev)
            return MixtureC(self.prior, loc, sc.get('frac_local', 0.7))
        raise ValueError(mode)

    def times(self, n):
        n_e = int(round(self.frac_early * n))
        t = self.sampler.t(n - n_e)
        if n_e:      # log-uniform on [1e-3, 1] d: the surface layer right after the start
            te = 10.0 ** (-3.0 + 3.0 * torch.rand(n_e, 1, device=self.dev))
            t = torch.cat([t, te], 0)
        return t

    def boxes(self, n, gen=None):
        """n space-time boxes [-depth, 0] x [t0, t0 + dur], each at its own path, with stratified points:
        xi (n, nt, nx), t (n, nt), c (n, n_coef), dur [d] (n, 1), depth (fraction of the column) (n, 1).
        Duration and depth are log-uniform over [box_dt_min, box_dt_max] and [box_d_min, 1]."""
        nt, nx, T = self.box_nt, self.box_nxi, float(self.ds['n_days'])
        dur = self.box_dt_min * (self.box_dt_max / self.box_dt_min) ** torch.rand(n, 1, device=self.dev)
        t0 = torch.rand(n, 1, device=self.dev) * (T - dur)
        depth = self.box_d_min ** torch.rand(n, 1, device=self.dev)
        j = torch.arange(nt, device=self.dev).reshape(1, nt)
        t = t0 + dur * (j + torch.rand(n, nt, device=self.dev)) / nt
        k = torch.arange(nx, device=self.dev).reshape(1, 1, nx)
        xi = -depth.unsqueeze(2) * (k + torch.rand(n, nt, nx, device=self.dev)) / nx
        return xi, t, self.c_dist.sample(n, gen), dur, depth

    def batch(self, n_col, n_bc, gen=None, n_tan=0, n_box=0):
        """Fresh points: (xi, t, c) for the PDE, (tb, cb) for the surface flux, and, if n_tan > 0, centre points
        tan = (xi, t, tb) where the sensitivity term is evaluated (c = centre there); if n_box > 0, boxes for the
        water balance (see boxes)."""
        xi = self.sampler.xi(n_col)
        t = self.times(n_col)
        c = self.c_dist.sample(n_col, gen)
        tb = self.times(n_bc)
        cb = self.c_dist.sample(n_bc, gen)
        tan = None
        if n_tan:
            n_tb = max(1, (n_tan * n_bc) // n_col)          # same PDE : flux point ratio as the main set
            tan = (self.sampler.xi(n_tan), self.times(n_tan), self.times(n_tb))
        return xi, t, c, tb, cb, tan, (self.boxes(n_box, gen) if n_box else None)

    def make_heldout(self):
        """Fixed held-out physics set from the current sampling distribution (checkpoint choice, solver-free)."""
        g = torch.Generator(device=self.dev).manual_seed(4321)
        n = int(self.cfg['train'].get('n_val', 8192))
        self.val = self.batch(n, n // 4, gen=g, n_tan=self.n_tan, n_box=4 * self.n_box)

    # -- losses
    def q_surf(self, t):
        return torch_interp(t, self.t_q, self.q_obs)

    def losses(self, xi, t, c, tb, cb, tan=None, box=None):
        R = pde_residual(self.model, xi, t, c) / self.sig_pde
        f = (surface_flux(self.model, tb, cb) - self.q_surf(tb)) / self.sig_q
        out = {'pde': (R ** 2).mean(), 'flux': (f ** 2).mean()}
        if tan is not None:
            out.update(self.centre_losses(*tan))
        if box is not None:
            out['box'] = ((box_water_defect(self.model, self.q_surf, *box) / self.sig_th) ** 2).mean()
        return out

    def centre_losses(self, xi, t, tb):
        """Delta-method form of the local part of the objective at the centre c0 of the local distribution
        N(c0, Sigma):  E_c[r(c)^2] ~= r(c0)^2 + grad_c r(c0)^T Sigma grad_c r(c0)  (r = scaled PDE or flux residual).
        The second term is zero only if the network's c-derivative obeys the linearised (sensitivity) equations at
        c0, i.e. the Jacobian d theta / d c that the inversion uses is the physical one there. Same sigma scales and
        the same Sigma as the sampled term: no new weight."""
        Lc = self.c_dist.local.L
        c0 = self.c_dist.local.center

        def value_and_tangent(fun, n):
            cc = c0.expand(n, -1).clone().requires_grad_(True)
            r = fun(cc)
            # every residual depends only on its own row of cc, so d(sum r)/d cc = per-point gradients
            g = torch.autograd.grad(r.sum(), cc, create_graph=True)[0]
            return (r ** 2).mean() + ((g @ Lc) ** 2).sum(1).mean()

        return {'pde_c': value_and_tangent(lambda cc: pde_residual(self.model, xi, t, cc) / self.sig_pde, xi.shape[0]),
                'flux_c': value_and_tangent(lambda cc: (surface_flux(self.model, tb, cc) - self.q_surf(tb)) / self.sig_q,
                                            tb.shape[0])}

    @staticmethod
    def total(L):
        return sum(L.values())

    def heldout_loss(self, chunk=4096):
        xi, t, c, tb, cb, tan, box = self.val
        tot = {'pde': 0.0, 'flux': 0.0}
        for i in range(0, xi.shape[0], chunk):
            R = pde_residual(self.model, xi[i:i + chunk], t[i:i + chunk], c[i:i + chunk]).detach() / self.sig_pde
            tot['pde'] += float((R ** 2).sum())
        f = (surface_flux(self.model, tb, cb).detach() - self.q_surf(tb)) / self.sig_q
        tot['flux'] = float((f ** 2).sum())
        out = {'pde': tot['pde'] / xi.shape[0], 'flux': tot['flux'] / tb.shape[0]}
        if tan is not None:
            out.update({k: float(v.detach()) for k, v in self.centre_losses(*tan).items()})
        if box is not None:
            nb, tot_b = max(1, chunk // (self.box_nt * self.box_nxi)), 0.0
            for i in range(0, box[0].shape[0], nb):
                d = box_water_defect(self.model, self.q_surf, *(b[i:i + nb] for b in box)).detach() / self.sig_th
                tot_b += float((d ** 2).sum())
            out['box'] = tot_b / box[0].shape[0]
        return out

    def truth_theta_err(self):
        """Diagnostic only (never used for any choice): theta_5cm at the TRUE path vs the twin; meaningful only
        when the training forcing is the twin's (clean q)."""
        with torch.no_grad():
            th = theta_obs_model(self.model, self.t_obs, self.cfg['obs']['depth_m'], self.truth_c).cpu().numpy().ravel()
        e = th - self.ds['theta_obs_hourly']
        return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())

    def log(self, stage, L, extra=None):
        hv = self.heldout_loss()
        row = {'step': self.step, 'stage': stage, 'time_s': round(time.time() - self.t0, 1),
               **{f'L_{k}': float(v) for k, v in L.items()}, **{f'val_{k}': v for k, v in hv.items()},
               'val_total': sum(hv.values())}
        if self.diag_truth:
            row['theta_rmse_truth'], row['theta_max_truth'] = self.truth_theta_err()
        if extra:
            row.update(extra)
        self.rows.append(row)
        msg = ' '.join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"[{stage}] {msg}", flush=True)
        return row

    # -- stages
    def adam(self, stage, n_steps, lr, lr_final):
        tc = self.cfg['train']
        params = list(self.model.head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        gamma = (lr_final / lr) ** (1.0 / max(n_steps, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        good = copy.deepcopy(self.model.state_dict())
        n_col, n_bc, log_every = tc['n_col'], tc['n_bc'], tc.get('log_every', 500)
        for i in range(n_steps):
            L = self.losses(*self.batch(n_col, n_bc, n_tan=self.n_tan, n_box=self.n_box))   # fresh points, paths
            loss = self.total(L)
            if not torch.isfinite(loss):
                print(f"[{stage}] non-finite loss at step {self.step}; restoring last good state", flush=True)
                self.model.load_state_dict(good)
                for g in opt.param_groups:
                    g['lr'] *= 0.5
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc.get('clip', 1e3))
            opt.step()
            sched.step()
            self.step += 1
            if self.step % log_every == 0 or i == n_steps - 1:
                self.log(stage, {k: v.detach() for k, v in L.items()}, {'lr': opt.param_groups[0]['lr']})
                good = copy.deepcopy(self.model.state_dict())

    def lbfgs(self, stage, n_blocks, iters):
        """L-BFGS in blocks, each on a FRESH fixed set (no over-fitting of one set in 11-D); the state with the
        lowest held-out physics loss is kept."""
        tc = self.cfg['train']
        params = list(self.model.head.parameters())
        best = self.heldout_loss()
        best_v, best_state = sum(best.values()), copy.deepcopy(self.model.state_dict())
        for b in range(n_blocks):
            B = self.batch(tc.get('lbfgs_n_col', 32768), tc.get('lbfgs_n_bc', 4096),
                           n_tan=tc.get('lbfgs_n_tan', 2 * self.n_tan),
                           n_box=tc.get('lbfgs_n_box', 4 * self.n_box))
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=100, tolerance_grad=1e-12,
                                    tolerance_change=1e-16, line_search_fn='strong_wolfe')
            state = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                L = self.losses(*B)
                loss = self.total(L)
                loss.backward()
                state['L'] = {k: v.detach() for k, v in L.items()}
                return loss

            opt.step(closure)
            self.step += iters
            if not all(math.isfinite(float(v)) for v in state['L'].values()):
                print(f"[{stage}] non-finite; restoring best and stopping", flush=True)
                break
            row = self.log(stage, state['L'], {'block': b})
            if row['val_total'] < best_v:
                best_v, best_state = row['val_total'], copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        print(f"[{stage}] kept held-out physics loss {best_v:.4e}", flush=True)

    def run(self):
        tc = self.cfg['train']
        with open(os.path.join(self.out, 'config.yaml'), 'w') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        self.log('init', {'pde': float('nan'), 'flux': float('nan')})
        if tc.get('adam_steps', 0):
            self.adam('adam', tc['adam_steps'], tc['lr'], tc.get('lr_final', tc['lr']))
        if tc.get('lbfgs_blocks', 0):
            self.lbfgs('lbfgs', tc['lbfgs_blocks'], tc.get('lbfgs_iters', 200))
        self.save()

    def save(self):
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(self.out, 'metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, restval='')
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        torch.save(self.model.state_dict(), os.path.join(self.out, 'model.pt'))
        final = self.heldout_loss()
        summ = {**{f'val_{k}': v for k, v in final.items()}, 'steps': self.step,
                'sig_pde': self.sig_pde, 'sig_q': self.sig_q, 'wall_s': round(time.time() - self.t0, 1)}
        if self.diag_truth:
            summ['theta_rmse_truth'], summ['theta_max_truth'] = self.truth_theta_err()
        if self.dev.type == 'cuda':
            summ['peak_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(summ, f, indent=2)
        print(f"[param] saved {self.out}: {summ}", flush=True)


# ----------------------------------------------------------------------------- inversion through the network
def load_members(run_dirs, device):
    """Trained parametric networks (an ensemble), all for the same window and forcing."""
    members, cfg0 = [], None
    for d in run_dirs:
        with open(os.path.join(d, 'config.yaml')) as f:
            cfg = yaml.safe_load(f)
        ds = load_dataset(cfg)
        sp = CubicBSpline1D(0.0, ds['n_days'], cfg.get('zb_spline', {}).get('knot_days',
                                                                           ds['twin_cfg']['zb_spline']['knot_days']))
        m = build_param_model(cfg, ds, sp, device)
        m.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location=device, weights_only=True))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        members.append(m)
        cfg0 = cfg0 or cfg
    return members, cfg0


def member_thetas(members, t_obs, depth, c):
    """(K, n) theta_5cm of every member at the shared coefficients c (n_coef,)."""
    return torch.stack([theta_obs_model(m, t_obs, depth, c).reshape(-1) for m in members])


def jacobian(fn, c):
    """d fn / d c (n, n_coef), exact (forward-mode autograd through the network)."""
    try:
        return torch.func.jacfwd(fn)(c)
    except Exception as exc:                       # pragma: no cover - fallback if an op lacks forward AD
        print(f"[invert] jacfwd failed ({exc}); using reverse mode", flush=True)
        return torch.autograd.functional.jacobian(fn, c, vectorize=True)


def secant_correction(dC, dO):
    """Least-change (minimum Frobenius norm) matrix D (n, n_coef) with D dC = dO (multi-secant / Broyden condition).
    dC (n_coef, m): differences of earlier paths from the current one; dO (n, m): the matching differences of the
    value discrepancy theta_fwd - theta_op. D v = 0 for v orthogonal to the earlier differences, so the operator
    Jacobian is kept there. For m > n_coef (or dependent columns) the conditions hold in the least-squares sense."""
    return dO @ torch.linalg.pinv(dC)


def affine_slope(C, O):
    """Slope D (n, n_coef) of the least-squares affine fit O_j ~ a + D C_j over the points C (n_coef, m), O (n, m);
    the points must span every coefficient direction (m > n_coef). For a symmetric design c0 +- d_i it is the central
    difference (O(c0 + d_i) - O(c0 - d_i)) / 2 along every d_i (the even part of O cancels), with or without c0.
    Also returns the rms of the fit residual over the rms of O about its mean (curvature + noise of the points)."""
    Cc = C - C.mean(1, keepdim=True)
    Oc = O - O.mean(1, keepdim=True)
    if int(torch.linalg.matrix_rank(Cc)) < C.shape[0]:
        raise ValueError(f'the {C.shape[1]} points do not span the {C.shape[0]} coefficient directions')
    D = Oc @ torch.linalg.pinv(Cc)
    resid = float(torch.sqrt(torch.mean((Oc - D @ Cc) ** 2)) / torch.sqrt(torch.mean(Oc ** 2)))
    return D, resid


def invert_lm(members, t_obs, depth, y, sigma, c0, max_iter=60, lam0=1e-2, verbose=True, offset=None, lin=None,
              box=None):
    """Levenberg-Marquardt on mean((theta_model(c) - y)^2) / sigma^2 with the exact (autograd) Jacobian of the model.
    Same loop and stopping rules as scripts/classical_inversion.py.
    theta_model(c) = theta_bar(c)            ensemble mean of the operator members
                     + offset                value correction at one path c_k (zeroth order), optional
                     + dJ (c - c_k)          sensitivity correction at c_k (first order), optional, lin = (dJ, c_k)
    offset = theta_fwd(c_k) - theta_bar(c_k) and dJ = J_fwd(c_k) - J_bar(c_k) come from forward PINN(s) trained at
    the one path c_k (scripts/param_invert.py --fwd [--fwd-jac]). With both, the model equals the forward PINN in
    value and slope at c_k; the operator members only supply the curvature away from c_k.
    dJ can also be the fitted slope of theta_fwd - theta_bar over forward PINNs at a finite-difference design around
    c_k (--fwd-fd, affine_slope) or a multi-secant estimate (--fwd-prev, secant_correction).
    box = (lo, hi): the coefficients stay in [lo, hi], the support of the prior the members were trained on (CPrior.box;
    outside it they extrapolate). The start and every trial step are projected onto the box, and a coefficient that
    sits on a bound while the descent direction points out of the box is held there (active set): the others solve
    the reduced system, so the iteration converges to the constrained minimum (free gradient 0, bound gradient
    pointing out). Without a box every coefficient is free and this is the plain LM."""
    def fn(c):
        th = member_thetas(members, t_obs, depth, c).mean(0)
        if offset is not None:
            th = th + offset
        if lin is not None:
            th = th + lin[0] @ (c - lin[1])
        return th

    def cost_of(th):
        r = (th - y) / sigma
        return float(torch.mean(r ** 2)) if bool(torch.all(torch.isfinite(r))) else math.inf

    proj = (lambda v: v) if box is None else (lambda v: torch.clamp(v, box[0], box[1]))
    c = proj(c0.clone())
    th = fn(c)
    cost, lam = cost_of(th), lam0
    hist, step = [], torch.zeros_like(c)
    rel = 1.0
    for it in range(max_iter):
        J = jacobian(fn, c) / sigma
        r = (th - y) / sigma
        A, g = J.T @ J, J.T @ r
        free = torch.ones_like(c, dtype=torch.bool)
        if box is not None:
            free = ~(((c <= box[0]) & (g > 0)) | ((c >= box[1]) & (g < 0)))
            if not bool(free.any()):
                break
        Af, gf = A[free][:, free], g[free]
        accepted = False
        for _ in range(10):
            step = torch.zeros_like(c)
            step[free] = -torch.linalg.solve(Af + lam * torch.diag(torch.diag(Af)), gf)
            step = proj(c + step) - c
            th_new = fn(c + step)
            cost_new = cost_of(th_new)
            if cost_new < cost:
                rel = (cost - cost_new) / cost
                c, th, cost = c + step, th_new, cost_new
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 4.0
        hist.append({'iter': it, 'cost': cost, 'lambda': lam, 'step_max': float(step.abs().max())})
        if verbose:
            print(f"[lm {it:2d}] cost {cost:.6e} lambda {lam:.1e} |step| {float(step.abs().max()):.2e}", flush=True)
        if not accepted or rel < 1e-9 or float(step.abs().max()) < 1e-7:
            break
    J = jacobian(fn, c)
    return {'c': c, 'theta': th, 'cost': cost, 'J': J, 'hist': hist, 'last_step': step}


def posterior_cov(J, sigma, members, t_obs, depth, c):
    """Coefficient covariance with noise sigma^2 I plus the ensemble (surrogate) covariance Sigma_e:
    (J^T W J)^-1, W = (sigma^2 I + Sigma_e)^-1. Also returns the noise-only covariance."""
    n = J.shape[0]
    th = member_thetas(members, t_obs, depth, c)
    K = th.shape[0]
    Se = torch.zeros(n, n, dtype=J.dtype, device=J.device)
    if K > 1:
        D = th - th.mean(0, keepdim=True)
        Se = D.T @ D / (K - 1)
    W = torch.linalg.inv(sigma ** 2 * torch.eye(n, dtype=J.dtype, device=J.device) + Se)
    cov = torch.linalg.pinv(J.T @ W @ J, rtol=1e-12)
    cov_noise = torch.linalg.pinv(J.T @ J / sigma ** 2, rtol=1e-12)
    return cov, cov_noise, Se
