"""Three-stage training of the zb-inverse PINN on a twin (or real) dataset.

Stage 1  Adam, zb frozen at its initial guess, loss = PDE + flux    (consistent forward state)
Stage 2  Adam, joint (network + zb spline), loss = PDE + flux + data (forward mode: no data, zb frozen)
Stage 3  L-BFGS (strong Wolfe), fixed points, weights frozen, no causal weights

Logs every `log_every` steps: loss terms, weights, and errors against the twin truth.
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

from .soil import SoilVG
from .spline import CubicBSpline1D, ZbSplineNumpy, ZbSplineTorch, _inv_softplus_np
from .model import HeadNet, InversePINN
from .physics import pde_residual, surface_flux, theta_obs_model, theta_profile
from .weighting import GradNormBalancer, CausalWeights, DiscrepancyBalancer
from .data import torch_interp, obs_layer
from . import plots

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- data
def load_dataset(cfg):
    tw = dict(np.load(cfg['twin_npz'], allow_pickle=False))
    twin_dir = os.path.dirname(cfg['twin_npz'])
    with open(os.path.join(twin_dir, 'twin_config.yaml')) as f:
        tcfg = yaml.safe_load(f)
    soil = SoilVG.from_config(tcfg['soil'], tcfg['scaling']['L'])
    n_days = float(tw['n_days'])
    rng = np.random.default_rng(cfg['obs'].get('noise_seed', 0))
    if cfg['obs'].get('use', 'daily') == 'hourly':
        t_obs, th_true = tw['t_days'], tw['theta_obs_hourly']
    else:
        t_obs, th_true = tw['t_obs_days'], tw['theta_obs_daily']
    th_obs = th_true + cfg['obs'].get('noise_sigma', 0.0) * rng.standard_normal(th_true.shape)
    # Observed surface flux. A head-limited twin (reference.h_crit_m) stores the actual, delivered
    # flux hourly: that is what eddy covariance measures, so it is the observable. Otherwise the
    # prescribed flux on its own nodes (daily for site forcing, 6-hourly for the synthetic case).
    if 'q_act_hourly' in tw:
        t_q, q_true = tw['t_days'], tw['q_act_hourly']
    else:
        t_q, q_true = tw['t_daily'], tw['q_daily']
    q_obs = q_true * (1.0 + cfg['obs'].get('q_noise_frac', 0.0) * rng.standard_normal(q_true.shape))
    return {
        'soil': soil, 'n_days': n_days, 'twin_cfg': tcfg,
        't_hourly': tw['t_days'], 'zb_true': tw['zb_true'], 'theta_obs_hourly': tw['theta_obs_hourly'],
        'theta_ref': tw['theta'], 'xi_ref': tw['xi'], 'truth_coef': tw['truth_coef'],
        'zb_min': float(tw['zb_min']), 't_obs': t_obs, 'theta_obs': th_obs, 'theta_obs_true': th_true,
        't_q': t_q, 'q_obs': q_obs, 'q_true': q_true,
        # transpiration taken up by roots [mm/d on t_tp], for twins made with the root sink (make_twin_fem.py)
        't_tp': tw.get('t_daily'), 'tp_mmd': tw['tp_daily_mmd'] if tcfg.get('fem', {}).get('root_sink', False) else None,
    }


def signal_rate_scale(t_obs, theta_obs, soil, sigma_theta):
    """sigma_pde 'auto': RMS of d Se / d t~ of the noise-free signal, estimated from day means.

    Differencing noisy hourly data measures the noise, sqrt(2) sigma / (theta* dt~): sigma_theta 0.005 gives
    1.35 vs 0.084 for clean data, i.e. a PDE term 256x weaker than intended (1000x at 0.01), so the network
    fits the noise by bending the physics. Day means cut the noise by sqrt(n per day); the remaining known
    noise variance of each difference, sigma^2 (1/n_i + 1/n_j) / (theta* dt~)^2, is subtracted.
    Incomplete days (fewer than half the usual count) are dropped. Daily data: n = 1 per day."""
    t_obs = np.asarray(t_obs, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    day = np.floor(t_obs + 1e-9).astype(int)
    keys, counts = np.unique(day, return_counts=True)
    keys = keys[counts >= 0.5 * counts.max()]
    tm = np.array([t_obs[day == k].mean() for k in keys])
    se = np.array([(theta_obs[day == k].mean() - soil.theta_r) / soil.theta_star for k in keys])
    nk = np.array([(day == k).sum() for k in keys], dtype=float)
    dtt = np.diff(tm) / (soil.T / 86400.0)
    rate2 = (np.diff(se) / dtt) ** 2
    noise2 = (float(sigma_theta) / soil.theta_star) ** 2 * (1.0 / nk[1:] + 1.0 / nk[:-1]) / dtt ** 2
    var = float(np.mean(rate2 - noise2))
    if var < 0.25 * float(np.mean(rate2)):
        print(f"[train] WARNING sigma_pde auto: noise removes {100 * (1 - var / np.mean(rate2)):.0f}% of the "
              f"day-mean rate variance; signal barely resolved, using 0.5 x raw", flush=True)
        var = 0.25 * float(np.mean(rate2))
    return float(np.sqrt(var))


def initial_coef(cfg, ds, spline):
    kind = cfg['zb_init']['kind']
    zmin = ds['zb_min']
    if kind == 'truth':
        if len(ds['truth_coef']) != spline.n_coef:
            raise ValueError('truth init needs the same knots as the twin')
        return np.array(ds['truth_coef'], dtype=float), None
    if kind == 'constant':
        c = float(_inv_softplus_np(cfg['zb_init']['value'] - zmin))
        return np.full(spline.n_coef, c), None
    if kind in ('hydrostatic', 'hydrostatic_flat'):
        soil = ds['soil']
        top, bottom = obs_layer(cfg['obs']['depth_m'])
        est = (top + bottom) / 2.0 - soil.h_from_theta(ds['theta_obs'])
        est = np.clip(est, zmin + 0.05, cfg['zb_init'].get('zb_max', 3.0))
        if kind == 'hydrostatic_flat':
            # constant at the median hydrostatic depth: the pointwise inversion ignores the lag of the surface
            # behind the WT, and its shape at the window end (no later data) survives the inverse as a
            # +11 / -20 cm end-span error (recipe runs, finding 31); a flat start carries no such shape
            c = float(_inv_softplus_np(float(np.median(est)) - zmin))
            return np.full(spline.n_coef, c), est
        fit = ZbSplineNumpy.fit(spline, ds['t_obs'], est, zmin, ridge=1e-6)
        return fit.coef, est
    raise ValueError(kind)


# ----------------------------------------------------------------------------- sampling
class Sampler:
    def __init__(self, ds, cfg, device):
        self.n_days = ds['n_days']
        self.dev = device
        self.frac_rain = cfg['train'].get('frac_rain_t', 0.3)
        q = ds['q_obs']
        jump = np.abs(np.diff(q)) + np.abs(np.minimum(q[1:], 0.0))
        p = jump + 0.1 * jump.mean() + 1e-30
        self.p_int = torch.tensor(p / p.sum(), device=device)
        self.t_q = torch.tensor(ds['t_q'], device=device)

    def t(self, n):
        n_r = int(self.frac_rain * n)
        tu = torch.rand(n - n_r, 1, device=self.dev) * self.n_days
        k = torch.multinomial(self.p_int, n_r, replacement=True)
        t0, t1 = self.t_q[k], self.t_q[k + 1]
        tr = (t0 + torch.rand(n_r, device=self.dev) * (t1 - t0)).reshape(-1, 1)
        return torch.cat([tu, tr], 0)

    def xi(self, n):
        n1, n2 = int(0.5 * n), int(0.35 * n)
        n3 = n - n1 - n2
        beta = torch.distributions.Beta(torch.tensor(1.0, device=self.dev), torch.tensor(4.0, device=self.dev))
        a = -torch.rand(n1, device=self.dev)
        b = -beta.sample((n2,))
        c = -1.0 + beta.sample((n3,))
        return torch.cat([a, b, c]).reshape(-1, 1)

    def collocation(self, n):
        return self.xi(n), self.t(n)


# ----------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg['train'].get('seed', 0))
        np.random.seed(cfg['train'].get('seed', 0))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ds = ds = load_dataset(cfg)
        self.soil = ds['soil']
        self.mode = cfg.get('mode', 'inverse')
        knot = cfg.get('zb_spline', {}).get('knot_days', ds['twin_cfg']['zb_spline']['knot_days'])
        self.spline = CubicBSpline1D(0.0, ds['n_days'], knot)
        coef0, self.zb_est_obs = initial_coef(cfg, ds, self.spline)
        self.coef0 = coef0
        zb_mod = ZbSplineTorch(self.spline, coef0, ds['zb_min'])
        mc = cfg['model']
        head = HeadNet(mc.get('width', 64), mc.get('depth', 5), t_scale_days=ds['n_days'],
                       fourier_periods_days=tuple(mc.get('fourier_periods_days', (2, 4, 8, 16, 32, 64))),
                       log_depth_eps=mc.get('log_depth_eps'))
        self.model = InversePINN(self.soil, zb_mod, head, mc.get('tau_ic_days', 0.25),
                                 base=mc.get('base', 'current'), departure=mc.get('departure', 'additive')).to(self.dev)
        self.sampler = Sampler(ds, cfg, self.dev)
        T = lambda a: torch.tensor(np.asarray(a, dtype=float), device=self.dev)
        self.t_q_t, self.q_obs_t = T(ds['t_q']), T(ds['q_obs'] / self.soil.Ks)
        self.t_obs_t = T(ds['t_obs']).reshape(-1, 1)
        self.th_obs_t = T(ds['theta_obs']).reshape(-1, 1)
        sg = cfg['sigmas']
        self.sig_th = float(sg['theta'])
        self.sig_q = float(sg.get('q_frac', 0.1)) * float(np.sqrt(np.mean((ds['q_obs'] / self.soil.Ks) ** 2)))
        # PDE scale (model-error sigma). 'auto': RMS rate of change of the saturation signal, from day means of
        #   the observations with the known noise variance removed (signal_rate_scale; 'auto_hourly' = old,
        #   noise-dominated estimator).
        # 'stage1': start from auto, then after the forward-only stage use the residual RMS the network
        #   actually reaches there (an empirical model-error estimate). 'ml': start from auto, then in the
        #   joint stage keep sigma_pde^2 = EMA of mean R^2 on a uniform validation set, i.e. the maximum-
        #   likelihood estimate of the model-error variance given the current fit. A number: fixed.
        #   Why: with sigma_pde ~20x above the residual the network reaches, violating the PDE is cheap, so
        #   with noisy theta the network bends the physics to fit noise (E3: PDE rms 0.44 vs 0.004 clean).
        # 'morozov': start from auto; in the joint Adam stage scale sigma_pde so that the data misfit sits at the
        #   noise level expected for n_coef free parameters, E[L_data] = 1 - n_coef / n_obs (discrepancy
        #   principle): below it the network is absorbing noise by bending the physics (tighten), above it the
        #   physics is too stiff for zb to move (loosen). sigma_pde stays in [auto * pde_mz_min_frac, auto].
        # pde_factor (auto modes only): sigma_pde = factor * auto.
        pde_mode = sg.get('pde', 'auto')
        self.pde_mode = pde_mode if pde_mode in ('stage1', 'ml', 'morozov') else 'fixed'
        self.pde_ml = {'freq': int(sg.get('pde_ml_freq', 100)), 'beta': float(sg.get('pde_ml_beta', 0.9)), 'ema': None}
        if pde_mode in ('auto', 'stage1', 'ml', 'morozov'):
            self.sig_pde = float(sg.get('pde_factor', 1.0)) * signal_rate_scale(
                ds['t_obs'], ds['theta_obs'], self.soil, self.sig_th)
            self.pde_mz = {'freq': int(sg.get('pde_mz_freq', 100)), 'eta': float(sg.get('pde_mz_eta', 0.25)),
                           'beta': float(sg.get('pde_mz_beta', 0.9)), 'ema': None, 'max': self.sig_pde,
                           'min': self.sig_pde * float(sg.get('pde_mz_min_frac', 0.1)),
                           'target': 1.0 - self.spline.n_coef / float(len(ds['t_obs']))}
        elif pde_mode == 'auto_hourly':
            # pre-2026-11-09 estimator, kept only to reproduce old runs: noise-dominated for hourly noisy data
            se = (ds['theta_obs'] - self.soil.theta_r) / self.soil.theta_star
            rate = np.diff(se) / (np.diff(ds['t_obs']) / (self.soil.T / 86400.0))
            self.sig_pde = float(np.sqrt(np.mean(rate ** 2)))
        else:
            self.sig_pde = float(sg['pde'])
        rc = cfg.get('sampling', {}).get('rar', {})
        self.rar = {'enabled': rc.get('enabled', True), 'k': rc.get('k', 1.0), 'c': rc.get('c', 1.0),
                    'pool_factor': rc.get('pool_factor', 8)}
        wc = cfg['weighting']
        self.names = ['pde', 'flux', 'data'] if self.mode == 'inverse' else ['pde', 'flux']
        if wc.get('mode', 'gradnorm') == 'discrepancy':
            # PDE = objective; theta data and flux held at their noise level (Morozov)
            self.bal = DiscrepancyBalancer(self.names, freq=wc.get('freq', 100), eta=wc.get('eta', 0.5),
                                           targets=wc.get('targets'))
        else:
            self.bal = GradNormBalancer(self.names, freq=wc.get('freq', 100), alpha=wc.get('alpha', 0.9),
                                        fixed=(wc.get('mode', 'gradnorm') == 'fixed'))
        self.causal = CausalWeights(ds['n_days'], wc.get('n_chunks', 16), wc.get('eps', 0.05),
                                    wc.get('eps_max', 5.0), enabled=wc.get('causal', False))
        g = torch.Generator(device='cpu').manual_seed(12345)
        self.val_xi = (-torch.rand(4096, 1, generator=g)).to(self.dev)
        self.val_t = (torch.rand(4096, 1, generator=g) * ds['n_days']).to(self.dev)
        self.split = bool(cfg['train'].get('split', False)) and self.mode == 'inverse'
        self.out = cfg['output']['dir']
        os.makedirs(self.out, exist_ok=True)
        self.hist = {}
        self.rows = []
        self.step = 0
        self.t0 = time.time()
        print(f"[train] device={self.dev} mode={self.mode} spline coefs={self.spline.n_coef} "
              f"sigmas: theta={self.sig_th} q~={self.sig_q:.3e} pde={self.sig_pde}", flush=True)
        print(f"[train] soil: {self.soil.summary()}", flush=True)

    # ------------------------------------------------------------------ sampling
    def collocation(self, n):
        """RAR-D (Wu et al. 2023): draw a pool from the base sampler, keep n points with
        probability proportional to |R|^k / mean(|R|^k) + c. Defaults k = c = 1 (paper defaults)."""
        if not self.rar['enabled']:
            return self.sampler.collocation(n)
        pool_xi, pool_t = self.sampler.collocation(n * int(self.rar['pool_factor']))
        res = []
        for i in range(0, pool_xi.shape[0], 8192):
            with torch.enable_grad():
                r = pde_residual(self.model, pool_xi[i:i + 8192], pool_t[i:i + 8192])
            res.append(r.detach().abs().reshape(-1))
        e = torch.cat(res) ** self.rar['k']
        e = torch.nan_to_num(e, nan=0.0, posinf=0.0)
        p = e / e.mean().clamp(min=1e-300) + self.rar['c']
        idx = torch.multinomial(p / p.sum(), n, replacement=False)
        return pool_xi[idx], pool_t[idx]

    # ------------------------------------------------------------------ losses
    def q_surf(self, t):
        return torch_interp(t, self.t_q_t, self.q_obs_t)

    def losses(self, xi_c, t_c, t_b, use_data, causal=True):
        R = pde_residual(self.model, xi_c, t_c) / self.sig_pde
        L = {'pde': self.causal.weighted_mean(R ** 2, t_c) if causal else (R ** 2).mean()}
        rq = (surface_flux(self.model, t_b) - self.q_surf(t_b)) / self.sig_q
        L['flux'] = (rq ** 2).mean()
        if use_data:
            th = theta_obs_model(self.model, self.t_obs_t, self.cfg['obs']['depth_m'])
            L['data'] = (((th - self.th_obs_t) / self.sig_th) ** 2).mean()
        return L

    def total(self, L):
        return sum(self.bal.w[k] * v for k, v in L.items())

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        m, ds, soil = self.model, self.ds, self.soil
        th_h = torch.tensor(ds['t_hourly'], device=self.dev).reshape(-1, 1)
        with torch.no_grad():
            zb = m.zb_m(th_h).cpu().numpy().ravel()
            th_obs_h = theta_obs_model(m, th_h, self.cfg['obs']['depth_m']).cpu().numpy().ravel()
            th_at_obs = theta_obs_model(m, self.t_obs_t, self.cfg['obs']['depth_m']).cpu().numpy().ravel()
        with torch.enable_grad():
            q = surface_flux(m, th_h.clone()).detach().cpu().numpy().ravel() * soil.Ks
            R = pde_residual(m, self.val_xi, self.val_t).detach().cpu().numpy().ravel()
        q_pres = np.interp(ds['t_hourly'], ds['t_q'], ds['q_true'])
        ks = slice(None, None, 6)
        xs = slice(None, None, 4)
        prof = theta_profile(m, torch.tensor(ds['xi_ref'][xs], device=self.dev),
                             torch.tensor(ds['t_hourly'][ks], device=self.dev)).cpu().numpy()
        e = zb - ds['zb_true']
        # the last knot span is weakly identifiable (the surface lags the WT; see the Cramer-Rao bound)
        inner = ds['t_hourly'] <= ds['n_days'] - self.spline.dt
        met = {
            'zb_rmse': float(np.sqrt(np.mean(e ** 2))), 'zb_bias': float(e.mean()), 'zb_maxabs': float(np.abs(e).max()),
            'zb_rmse_inner': float(np.sqrt(np.mean(e[inner] ** 2))),
            'theta_rmse_hourly': float(np.sqrt(np.mean((th_obs_h - ds['theta_obs_hourly']) ** 2))),
            'data_rmse': float(np.sqrt(np.mean((th_at_obs - ds['theta_obs']) ** 2))),
            'flux_rms_mmd': float(np.sqrt(np.mean((q - q_pres) ** 2)) * 86400e3),
            'pde_rms': float(np.sqrt(np.mean(R ** 2))),
            'prof_rmse': float(np.sqrt(np.mean((prof - ds['theta_ref'][ks][:, xs]) ** 2))),
        }
        pred = {'t_days': ds['t_hourly'], 'zb_pred': zb, 'zb_true': ds['zb_true'],
                'zb_init': ZbSplineNumpy(self.spline, self.coef0, ds['zb_min']).zb(ds['t_hourly']),
                'theta_obs_pred': th_obs_h, 't_obs_days': ds['t_obs'], 'theta_obs': ds['theta_obs'],
                'q_pred': q, 'q_surf': q_pres}
        return met, pred

    def log(self, stage, L, extra=None):
        met, _ = self.evaluate()
        row = {'step': self.step, 'stage': stage, 'time_s': round(time.time() - self.t0, 1)}
        for k in self.names:
            row[f'L_{k}'] = float(L[k]) if k in L else float('nan')
            row[f'w_{k}'] = self.bal.w[k]
        row['eps'] = self.causal.eps
        row['sig_pde'] = self.sig_pde
        row.update(met)
        if extra:
            row.update(extra)
        self.rows.append(row)
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        self.hist = {k: [r.get(k, float('nan')) for r in self.rows] for k in keys}
        print(f"[{stage} {self.step:6d}] " + " ".join(
            f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            if k not in ('step', 'stage')), flush=True)
        return met

    # ------------------------------------------------------------------ stages
    def _set_zb_trainable(self, flag):
        self.model.zb_spline.coef.requires_grad_(flag)

    def val_pde_rms(self):
        """RMS of the raw PDE residual on the fixed uniform validation set."""
        with torch.enable_grad():
            R = pde_residual(self.model, self.val_xi, self.val_t).detach()
        return float(torch.sqrt(torch.mean(R ** 2)))

    def adam_stage(self, stage, n_steps, joint, use_data, decay):
        tc = self.cfg['train']
        self._set_zb_trainable(joint)
        groups = [{'params': list(self.model.head.parameters()), 'lr': tc['lr']}]
        # stage 1 keeps lr constant; the final Adam stage decays it (forward and inverse alike: a constant
        # lr left the forward run on a noise floor, flux loss jumping 1e-3 <-> 3e-1)
        lr_end = tc.get('lr_final', tc['lr']) if decay else tc['lr']
        if joint:
            groups.append({'params': [self.model.zb_spline.coef], 'lr': tc['lr_zb']})
        opt = torch.optim.Adam(groups)
        finals = [lr_end] + ([tc.get('lr_zb_final', tc['lr_zb'])] if joint else [])
        gammas = [(f / g['lr']) ** (1.0 / max(n_steps, 1)) for f, g in zip(finals, groups)]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s, gm=gm: gm ** s for gm in gammas])
        net_params = list(self.model.head.parameters())
        good = copy.deepcopy(self.model.state_dict())
        n_col, n_bc, every = tc['n_col'], tc['n_bc'], tc.get('resample_every', 100)
        log_every = tc.get('log_every', 250)
        for i in range(n_steps):
            if i % every == 0:
                xi_c, t_c = self.collocation(n_col)
                t_b = self.sampler.t(n_bc)
            L = self.losses(xi_c, t_c, t_b, use_data)
            self.bal.maybe_update(self.step, L, net_params)
            loss = self.total(L)
            if not torch.isfinite(loss):
                print(f"[{stage}] non-finite loss at step {self.step}; restoring last good state", flush=True)
                self.model.load_state_dict(good)
                for g in opt.param_groups:
                    g['lr'] *= 0.5
                continue
            opt.zero_grad(set_to_none=True)
            if self.split and joint and use_data:
                # Bilevel, first-order (cf. Hao et al. 2023, Bi-PINN): the network solves the FORWARD problem
                # (PDE + flux) for the current zb and never sees the data, so it cannot bend the physics to fit
                # noise; zb follows the data misfit only, so the PDE penalty cannot lock it (moving zb with the
                # network momentarily fixed raises the PDE residual, which in the joint loss resists every zb
                # update: zb stalled at ~6 cm once sigma_pde was tightened, clean data included).
                L_fwd = sum(self.bal.w[k] * v for k, v in L.items() if k != 'data')
                g_net = torch.autograd.grad(L_fwd, net_params, retain_graph=True, allow_unused=True)
                g_zb = torch.autograd.grad(self.bal.w['data'] * L['data'], [self.model.zb_spline.coef])[0]
                for p, g in zip(net_params, g_net):
                    p.grad = g
                self.model.zb_spline.coef.grad = g_zb
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(net_params, tc.get('clip', 1e3))
            opt.step()
            sched.step()
            self.step += 1
            if self.pde_mode == 'ml' and use_data and self.step % self.pde_ml['freq'] == 0:
                r2 = self.val_pde_rms() ** 2
                e = self.pde_ml['ema']
                self.pde_ml['ema'] = r2 if e is None else self.pde_ml['beta'] * e + (1 - self.pde_ml['beta']) * r2
                self.sig_pde = math.sqrt(self.pde_ml['ema'])
            if self.pde_mode == 'morozov' and use_data and self.step % self.pde_mz['freq'] == 0:
                mz, ld = self.pde_mz, float(L['data'].detach())
                mz['ema'] = ld if mz['ema'] is None else mz['beta'] * mz['ema'] + (1 - mz['beta']) * ld
                s = self.sig_pde * (mz['ema'] / mz['target']) ** mz['eta']
                self.sig_pde = float(min(max(s, mz['min']), mz['max']))
            if self.causal.maybe_anneal():
                print(f"[{stage}] causal eps -> {self.causal.eps}", flush=True)
            if self.step % log_every == 0 or i == n_steps - 1:
                self.log(stage, {k: v.detach() for k, v in L.items()},
                         {'lr': opt.param_groups[0]['lr'], 'causal_min_w': self.causal.last_min_w})
                good = copy.deepcopy(self.model.state_dict())

    def lbfgs_stage(self, n_iters, joint, use_data):
        tc = self.cfg['train']
        self._set_zb_trainable(joint)
        params = list(self.model.head.parameters()) + ([self.model.zb_spline.coef] if joint else [])
        xi_c, t_c = self.collocation(tc.get('lbfgs_n_col', 16384))
        t_b = self.sampler.t(tc.get('lbfgs_n_bc', 2048))
        chunk = tc.get('lbfgs_chunk', 100)
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, history_size=100, tolerance_grad=1e-12,
                                tolerance_change=1e-16, line_search_fn='strong_wolfe')
        state = {}

        def closure():
            opt.zero_grad(set_to_none=True)
            L = self.losses(xi_c, t_c, t_b, use_data, causal=False)
            loss = self.total(L)
            loss.backward()
            state['L'] = {k: v.detach() for k, v in L.items()}
            return loss

        best_loss = float(closure())
        best_state = copy.deepcopy(self.model.state_dict())
        print(f"[lbfgs] start loss {best_loss:.4e}", flush=True)
        for _ in range(max(1, n_iters // chunk)):
            opt.step(closure)
            cur = float(closure())                 # loss at the current parameters
            self.step += chunk
            if not math.isfinite(cur):
                print("[lbfgs] non-finite loss; restoring best state and stopping", flush=True)
                break
            self.log('lbfgs', state['L'], {'lbfgs_loss': cur})
            if cur < best_loss:
                best_loss, best_state = cur, copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        print(f"[lbfgs] kept best loss {best_loss:.4e}", flush=True)

    # ------------------------------------------------------------------ zb pre-fit (between stage 1 and the joint stage)
    # Why: from a poor initial zb the theta misfit dominates the joint loss (deep start: L_data ~96 vs L_pde 0.02),
    # so the network bends the physics before zb moves and the run settles in a physics-violating state (zb -> 9 m).
    # The frozen-network sensitivity d theta_5 / d c has the right sign (scripts/sensitivity_check.py: GN step -0.33 m
    # where -0.35 m was needed). So: damped Gauss-Newton steps on the 9 coefficients with that Jacobian, each followed
    # by a physics-only re-solve of the network (warm start); a step is kept only if the data misfit of the RE-SOLVED
    # state falls (Levenberg-Marquardt acceptance on the bilevel objective). The network never fits data here.
    def zb_prefit(self):
        pc = self.cfg['train'].get('zb_prefit', {})
        n_outer, n_resolve = int(pc.get('outer', 5)), int(pc.get('resolve_steps', 1000))
        delta, lam, max_step = float(pc.get('delta', 0.01)), float(pc.get('lambda0', 1e-2)), float(pc.get('max_step', 1.0))
        depth = self.cfg['obs']['depth_m']
        coef = self.model.zb_spline.coef
        y = self.th_obs_t.reshape(-1)

        def theta_at(c):
            with torch.no_grad():
                coef.copy_(c)
                return theta_obs_model(self.model, self.t_obs_t, depth).reshape(-1)

        def cost(th):
            return float((((th - y) / self.sig_th) ** 2).mean())

        for it in range(n_outer):
            c0 = coef.detach().clone()
            th0 = theta_at(c0)
            cost0 = cost(th0)
            eye = torch.eye(c0.numel(), dtype=c0.dtype, device=c0.device)
            J = torch.stack([(theta_at(c0 + delta * e) - theta_at(c0 - delta * e)) / (2 * delta) for e in eye], dim=1)
            theta_at(c0)
            Js, r = J / self.sig_th, (th0 - y) / self.sig_th
            A, g = Js.T @ Js, Js.T @ r
            saved = copy.deepcopy(self.model.state_dict())
            accepted = False
            for _ in range(6):
                step = -torch.linalg.solve(A + lam * torch.diag(torch.diag(A)), g)
                step = step * min(1.0, max_step / float(step.abs().max()))
                theta_at(c0 + step)
                self.adam_stage('prefit', n_resolve, joint=False, use_data=False, decay=False)
                cost1 = cost(theta_at(coef.detach().clone()))
                if cost1 < cost0:
                    lam = max(lam / 3.0, 1e-6)
                    accepted = True
                    break
                self.model.load_state_dict(saved)
                lam *= 4.0
            met = self.log('prefit-gn', {}, {'prefit_cost_before': cost0, 'prefit_cost_after': cost1 if accepted else cost0,
                                             'prefit_lambda': lam, 'prefit_step_max': float(step.abs().max())})
            print(f"[prefit {it}] data cost {cost0:.4e} -> {cost1 if accepted else cost0:.4e} accepted={accepted} "
                  f"lambda {lam:.1e} zb_rmse {met['zb_rmse']:.4f}", flush=True)
            # stop_ratio < 1: keep stepping only while each step cuts the misfit by at least this factor, i.e. coarse
            # moves into the basin; the slow tail fits the half-trained network's forward error along the weakly
            # identified (deep-WT) direction, which the joint stage then cannot undo (finding 33)
            if not accepted or cost1 > float(pc.get('stop_ratio', 0.9)) * cost0:
                break

    # ------------------------------------------------------------------ augmented Lagrangian (joint stage)
    # min J(theta_data) s.t. r_i = R(x_i)/sigma_pde = 0 at fixed collocation points and f_j = 0 at fixed flux times:
    #   L_A = J + mean_i(lam_i r_i) + mu/2 mean_i(r_i^2) + mean_j(nu_j f_j) + mu/2 mean_j(f_j^2),
    #   every `inner_steps`: lam <- lam + mu r, nu <- nu + mu f  (method of multipliers; Lu et al. 2021 hPINN,
    #   Basir & Senocak 2022 PECANN). Why: with a pure penalty the only "multiplier" is mu * r, so carrying the data
    #   pressure needs a non-zero residual (physics bent to fit noise), and zb's gradient is the frozen-network one
    #   (PDE term locks zb). The accumulated lam converges to the adjoint field, so lam^T dR/dzb restores the
    #   total derivative for zb while r -> 0. Data = objective (sigma_theta scaled); PDE and flux = constraints.
    def _alm_lagrangian(self, xi, t, tb, lam, nu, mu):
        r = pde_residual(self.model, xi, t) / self.sig_pde
        f = (surface_flux(self.model, tb) - self.q_surf(tb)) / self.sig_q
        th = theta_obs_model(self.model, self.t_obs_t, self.cfg['obs']['depth_m'])
        J = (((th - self.th_obs_t) / self.sig_th) ** 2).mean()
        LA = J + (lam * r).mean() + 0.5 * mu * (r ** 2).mean() + (nu * f).mean() + 0.5 * mu * (f ** 2).mean()
        return LA, {'pde': (r ** 2).mean().detach(), 'flux': (f ** 2).mean().detach(), 'data': J.detach()}

    def _alm_residuals(self, chunk=8192):
        a = self.alm
        r = []
        for i in range(0, a['xi'].shape[0], chunk):
            with torch.enable_grad():
                r.append((pde_residual(self.model, a['xi'][i:i + chunk], a['t'][i:i + chunk]) / self.sig_pde).detach())
        with torch.enable_grad():
            f = ((surface_flux(self.model, a['tb']) - self.q_surf(a['tb'])) / self.sig_q).detach()
        return torch.cat(r), f

    def alm_stage(self, n_steps):
        tc, ac = self.cfg['train'], self.cfg.get('alm', {})
        K = int(ac.get('inner_steps', 500))
        mu, mu_max = float(ac.get('mu0', 1.0)), float(ac.get('mu_max', 100.0))
        growth, ratio = float(ac.get('mu_growth', 2.0)), float(ac.get('violation_ratio', 0.5))
        Nc, Nb = int(ac.get('n_col', 16384)), int(ac.get('n_bc', 2048))
        xi, t = self.sampler.collocation(Nc)                 # FIXED points: each carries its own multiplier
        tb = self.sampler.t(Nb)
        self.alm = {'xi': xi, 't': t, 'tb': tb, 'lam': torch.zeros_like(xi), 'nu': torch.zeros_like(tb), 'mu': mu}
        self._set_zb_trainable(True)
        groups = [{'params': list(self.model.head.parameters()), 'lr': tc['lr']},
                  {'params': [self.model.zb_spline.coef], 'lr': tc['lr_zb']}]
        opt = torch.optim.Adam(groups)
        finals = [tc.get('lr_final', tc['lr']), tc.get('lr_zb_final', tc['lr_zb'])]
        gammas = [(f / g['lr']) ** (1.0 / max(n_steps, 1)) for f, g in zip(finals, groups)]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s, gm=gm: gm ** s for gm in gammas])
        net_params = list(self.model.head.parameters())
        good = copy.deepcopy(self.model.state_dict())
        bs, bsb, log_every = tc['n_col'], tc['n_bc'], tc.get('log_every', 250)
        prev_v, v = float('inf'), float('nan')
        print(f"[alm] {Nc} fixed collocation points, {Nb} flux times, inner {K} steps, mu0 {mu}, mu_max {mu_max}",
              flush=True)
        for i in range(n_steps):
            idx = torch.randint(0, Nc, (bs,), device=self.dev)
            jdx = torch.randint(0, Nb, (bsb,), device=self.dev)
            LA, L = self._alm_lagrangian(xi[idx], t[idx], tb[jdx], self.alm['lam'][idx], self.alm['nu'][jdx], mu)
            if not torch.isfinite(LA):
                print(f"[alm] non-finite at step {self.step}; restoring last good state", flush=True)
                self.model.load_state_dict(good)
                for g in opt.param_groups:
                    g['lr'] *= 0.5
                continue
            opt.zero_grad(set_to_none=True)
            LA.backward()
            torch.nn.utils.clip_grad_norm_(net_params, tc.get('clip', 1e3))
            opt.step()
            sched.step()
            self.step += 1
            if (i + 1) % K == 0:
                r_all, f_all = self._alm_residuals()
                self.alm['lam'] += mu * r_all
                self.alm['nu'] += mu * f_all
                v = float(torch.sqrt((r_all ** 2).mean() + (f_all ** 2).mean()))
                if v > ratio * prev_v:
                    mu = min(mu * growth, mu_max)
                prev_v = v
                self.alm['mu'] = mu
            if self.step % log_every == 0 or i == n_steps - 1:
                self.log('alm', L, {'lr': opt.param_groups[0]['lr'], 'mu': mu, 'viol': v,
                                    'lam_rms': float(self.alm['lam'].pow(2).mean().sqrt()),
                                    'viol_val': self.val_pde_rms() / self.sig_pde})
                good = copy.deepcopy(self.model.state_dict())

    def alm_lbfgs(self, n_iters):
        """Method of multipliers with L-BFGS subproblems over the full fixed sets: each block of `lbfgs_chunk`
        iterations minimises L_A at fixed (lam, nu, mu), then lam <- lam + mu r, nu <- nu + mu f and the L-BFGS
        memory is reset (the objective changed). Freezing the multipliers instead would drive r to -lam/mu, not 0.
        L_A values from different blocks are not comparable, so the last finite state is kept."""
        a, tc = self.alm, self.cfg['train']
        self._set_zb_trainable(True)
        params = list(self.model.head.parameters()) + [self.model.zb_spline.coef]
        chunk = tc.get('lbfgs_chunk', 100)
        state = {}
        print(f"[alm-lbfgs] {n_iters // chunk} blocks of {chunk} iterations, mu {a['mu']}", flush=True)
        for _ in range(max(1, n_iters // chunk)):
            good = copy.deepcopy(self.model.state_dict())
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, history_size=100, tolerance_grad=1e-12,
                                    tolerance_change=1e-16, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad(set_to_none=True)
                LA, L = self._alm_lagrangian(a['xi'], a['t'], a['tb'], a['lam'], a['nu'], a['mu'])
                LA.backward()
                state['L'] = L
                return LA

            opt.step(closure)
            cur = float(closure())
            self.step += chunk
            if not math.isfinite(cur):
                print("[alm-lbfgs] non-finite; restoring the previous block and stopping", flush=True)
                self.model.load_state_dict(good)
                break
            r_all, f_all = self._alm_residuals()
            a['lam'] += a['mu'] * r_all
            a['nu'] += a['mu'] * f_all
            v = float(torch.sqrt((r_all ** 2).mean() + (f_all ** 2).mean()))
            self.log('alm-lbfgs', state['L'], {'lbfgs_loss': cur, 'mu': a['mu'], 'viol': v,
                                                'lam_rms': float(a['lam'].pow(2).mean().sqrt()),
                                                'viol_val': self.val_pde_rms() / self.sig_pde})

    # ------------------------------------------------------------------ run
    def run(self):
        tc = self.cfg['train']
        inverse = self.mode == 'inverse'
        with open(os.path.join(self.out, 'config.yaml'), 'w') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        self.log('init', {})
        self.adam_stage('adam1', tc['stage1_steps'], joint=False, use_data=False, decay=False)
        if inverse and tc.get('zb_prefit', {}).get('enabled', False):
            self.zb_prefit()
        if self.pde_mode == 'stage1':
            old, self.sig_pde = self.sig_pde, self.val_pde_rms()
            print(f"[train] sigma_pde: {old:.4e} -> {self.sig_pde:.4e} (residual RMS after the forward stage)", flush=True)
        use_alm = inverse and self.cfg.get('alm', {}).get('enabled', False)
        if use_alm:
            self.alm_stage(tc['stage2_steps'])
        else:
            self.adam_stage('adam2', tc['stage2_steps'], joint=inverse, use_data=inverse, decay=True)
        if tc.get('lbfgs_iters', 0) > 0:
            if use_alm:
                self.alm_lbfgs(tc['lbfgs_iters'])
            elif self.split:
                # bilevel: L-BFGS only polishes the forward solution (network, PDE + flux) at the final zb
                self.lbfgs_stage(tc['lbfgs_iters'], joint=False, use_data=False)
            else:
                self.lbfgs_stage(tc['lbfgs_iters'], joint=inverse, use_data=inverse)
        met, pred = self.evaluate()
        self.save(met, pred)
        return met

    def save(self, met, pred):
        keys = list(self.hist.keys())
        with open(os.path.join(self.out, 'metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, restval='')
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        np.savez(os.path.join(self.out, 'predictions.npz'), **{k: np.asarray(v) for k, v in pred.items()
                                                               if v is not None})
        torch.save(self.model.state_dict(), os.path.join(self.out, 'model.pt'))
        final = dict(met)
        final.update({f'w_{k}': self.bal.w[k] for k in self.names})
        final['steps'] = self.step
        final['sig_pde'] = self.sig_pde
        final['wall_s'] = round(time.time() - self.t0, 1)
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(final, f, indent=2)
        title = f"{os.path.basename(self.out)}: zb RMSE {met['zb_rmse']:.3f} m, theta RMSE {met['theta_rmse_hourly']:.4f}"
        plots.plot_training(self.hist, pred, os.path.join(self.out, 'training.png'), title)
        ds = self.ds
        ks = np.linspace(0, len(ds['t_hourly']) - 1, 4).astype(int)
        prof = theta_profile(self.model, torch.tensor(ds['xi_ref'], device=self.dev),
                             torch.tensor(ds['t_hourly'][ks], device=self.dev)).cpu().numpy()
        plots.plot_profiles(ds['xi_ref'], ds['t_hourly'][ks], ds['theta_ref'][ks], prof, ds['zb_true'][ks],
                            pred['zb_pred'][ks], os.path.join(self.out, 'profiles.png'), os.path.basename(self.out))
        print(f"[train] final: {json.dumps(final)}", flush=True)
