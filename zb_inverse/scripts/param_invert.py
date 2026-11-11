#!/usr/bin/env python
"""Invert theta_5cm for zb(t) through trained parametric PINN members (ensemble), multi-start LM.

Usage:
  python -u scripts/param_invert.py --members runs/p1_a runs/p1_b ... --out runs/p1_inv_noise005 \
      [--set obs.noise_sigma=0.01 --set sigmas.theta=0.01] [--prev-center runs/.../local_center.npz]
      [--coarse runs/p1_wide_a ...]                      far starts go through the wide-stage members first
      [--fwd runs/f_a runs/f_b [--fwd-jac | --fwd-fd runs/fd_p0 runs/fd_m0 ... | --fwd-prev runs/f_old ...]]
--fwd: forward PINN(s) trained at the one path c_k of --prev-center (sampling.mode=point). The inverted model is then
theta_op(c) + [theta_fwd(c_k) - theta_op(c_k)] (value-corrected); with --fwd-jac also + (J_fwd - J_op)(c_k) (c - c_k)
(the forward members carry the sensitivities, train.n_tan > 0); with --fwd-fd the slope correction is the finite
difference of theta_fwd - theta_op over forward PINNs at a design of paths around the estimate (scripts/fd_design.py);
with --fwd-prev a multi-secant slope correction from the forward PINNs of earlier rounds. hpc/dc_loop.sh runs the
train / invert rounds, hpc/fd_set.sh trains a design.
The data (theta_obs, and q_obs which must equal the members' training forcing) come from the members' config plus
the overrides. Writes predictions.npz (same keys as the classical referee), starts.npz, summary.json and
local_center.npz (centre + covariance for the next physics-only refinement of the members).
The posterior covariance adds the spread between the networks whose value is fitted to the noise: the operator
members, or with --fwd the forward members on their own path (the operator value cancels at c_k).
The twin truth is used only to SCORE the result (zb errors in summary.json), never in the inversion.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from zbinv.parametric import (load_members, invert_lm, posterior_cov, member_thetas, local_cov_from_width,  # noqa: E402
                              jacobian, secant_correction, affine_slope, CPrior)
from zbinv.train import load_dataset, initial_coef  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy  # noqa: E402
from train import apply_override  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--starts', default='hydrostatic,hydrostatic_flat,constant:1.0,constant:0.3')
    ap.add_argument('--inflate', type=float, default=3.0, help='local sampling sd = inflate x posterior sd')
    ap.add_argument('--floor-width', type=float, default=0.05,
                    help='local sampling: add (floor_width m of zb per knot)^2 = the radius over which G1 showed a '
                         'physics-only network can be made accurate; covers the surrogate bias of the current estimate, '
                         'which the data posterior (and the ensemble, whose members share it) does not see')
    ap.add_argument('--prev-center', default=None)
    ap.add_argument('--fwd', nargs='+', default=None,
                    help='forward PINN(s) trained at one path c_k (sampling.mode=point): the operator ensemble is corrected'
                         ' by the constant offset theta_fwd(c_k) - theta_op(c_k); its Jacobian is used for the step')
    ap.add_argument('--fwd-jac', action='store_true',
                    help='with --fwd: also correct the slope, by J_fwd(c_k) - J_op(c_k) (forward members trained with the '
                         'sensitivity term, train.n_tan > 0): every round is then a Gauss-Newton step with the forward '
                         'PINN\'s own sensitivities')
    ap.add_argument('--fwd-prev', nargs='*', default=[],
                    help='with --fwd (and without --fwd-jac): forward PINNs of earlier rounds, trained at other paths; '
                         'their values give a multi-secant (Broyden) correction of the operator Jacobian')
    ap.add_argument('--fwd-fd', nargs='*', default=[],
                    help='with --fwd (and without --fwd-jac): forward PINNs trained at the paths of a finite-difference '
                         'design around the estimate (scripts/fd_design.py); the slope of theta_fwd - theta_op fitted '
                         'to them corrects the operator Jacobian in every direction (written to fwd_slope.npz)')
    ap.add_argument('--coarse', nargs='+', default=None,
                    help='wide-stage members: every start except "previous" is first inverted through them, then '
                         'through --members (coarse-to-fine continuation; the refined members are only accurate near '
                         'their centre)')
    ap.add_argument('--box', action='store_true',
                    help='keep the estimate itself in the support of the path prior (CPrior.box, the coefficient '
                         'range of the wide-stage training paths): the fine stage is then the constrained fit, as '
                         'the coarse stage always is. A coefficient the data do not identify stops at the bound '
                         'instead of running away')
    args = ap.parse_args()
    tic = time.time()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members, cfg = load_members(args.members, dev)
    q_train = load_dataset(cfg)['q_obs']
    for item in args.set:
        apply_override(cfg, item)
    ds = load_dataset(cfg)
    if not np.allclose(ds['q_obs'], q_train, rtol=0, atol=1e-15):
        raise SystemExit('surface forcing of the data differs from the members\' training forcing')
    sp = CubicBSpline1D(0.0, ds['n_days'], cfg.get('zb_spline', {}).get('knot_days',
                                                                        ds['twin_cfg']['zb_spline']['knot_days']))
    depth, sig = cfg['obs']['depth_m'], float(cfg['sigmas']['theta'])
    t_obs = torch.tensor(ds['t_obs'], device=dev).reshape(-1, 1)
    y = torch.tensor(ds['theta_obs'], device=dev)
    t_h = ds['t_hourly']
    inner = t_h <= ds['n_days'] - sp.dt

    def prior_box(c):
        """Support of the path prior of config c in coefficient space (lo, hi)."""
        pc = c.get('prior', {})
        return CPrior(sp, ds['zb_min'], pc.get('mean_range_m', (0.3, 1.2)), pc.get('step_sd_m', 0.15),
                      pc.get('clip_m', (0.2, 1.5))).box()

    fine_box = prior_box(cfg) if args.box else None
    if fine_box is not None:
        print(f"[box] estimate kept in the prior box [{fine_box[0]:.3f}, {fine_box[1]:.3f}]", flush=True)

    offset, lin, fwd_info = None, None, None
    if args.fwd:
        fwd, fcfg = load_members(args.fwd, dev)
        if not np.allclose(load_dataset(fcfg)['q_obs'], q_train, rtol=0, atol=1e-15):
            raise SystemExit('forward PINN forcing differs from the operator members\' forcing')
        c_fwd = np.load(fcfg['sampling']['local_file'])['center']
        ct = torch.tensor(c_fwd, device=dev)
        th_f = member_thetas(fwd, t_obs, depth, ct)
        th_o = member_thetas(members, t_obs, depth, ct).mean(0)
        offset = th_f.mean(0) - th_o
        fwd_info = {'c_fwd': c_fwd.tolist(), 'offset_rms': float(torch.sqrt(torch.mean(offset ** 2))),
                    'offset_max': float(offset.abs().max()),
                    'fwd_member_sd_rms': float(torch.sqrt(torch.mean(th_f.var(0, unbiased=False)))) if len(fwd) > 1 else None,
                    'theta_fwd_vs_obs_rms': float(torch.sqrt(torch.mean((th_f.mean(0) - y) ** 2))), 'members': args.fwd}
        print(f"[fwd] offset rms {fwd_info['offset_rms']:.2e} max {fwd_info['offset_max']:.2e}; "
              f"forward members sd {fwd_info['fwd_member_sd_rms']}", flush=True)
        if args.fwd_jac:
            # first-order correction: the forward members were trained with the sensitivity term (train.n_tan > 0),
            # so their d theta / d c at c_k is the physical one; the model then matches them in value and slope at c_k
            J_fm = torch.stack([jacobian(lambda cc, m=m: member_thetas([m], t_obs, depth, cc)[0], ct) for m in fwd])
            J_f = J_fm.mean(0)
            J_o = jacobian(lambda cc: member_thetas(members, t_obs, depth, cc).mean(0), ct)
            lin = (J_f - J_o, ct)
            nrm = lambda A: torch.linalg.norm(A, dim=0)
            fwd_info['jac_norm_ratio_fwd_over_op'] = (nrm(J_f) / nrm(J_o)).tolist()
            fwd_info['jac_rel_diff_fwd_vs_op'] = (nrm(J_f - J_o) / nrm(J_o)).tolist()
            fwd_info['jac_member_rel_sd'] = ((torch.sqrt((J_fm - J_f).pow(2).sum(1).mean(0)) / nrm(J_f)).tolist()
                                             if len(fwd) > 1 else None)
            print(f"[fwd] J norm ratio fwd/op by coefficient {' '.join(f'{v:.2f}' for v in fwd_info['jac_norm_ratio_fwd_over_op'])}",
                  flush=True)
            print(f"[fwd] J rel diff fwd vs op             {' '.join(f'{v:.2f}' for v in fwd_info['jac_rel_diff_fwd_vs_op'])}",
                  flush=True)
            if fwd_info['jac_member_rel_sd']:
                print(f"[fwd] J rel sd between fwd members     {' '.join(f'{v:.2f}' for v in fwd_info['jac_member_rel_sd'])}",
                      flush=True)
        def discrepancies(dirs):
            """Value discrepancy o_j = theta_fwd(c_j) - theta_op(c_j) at the path c_j of every forward PINN in dirs
            (those trained at the same path are averaged; the current path c_k is skipped): [(c_j, o_j), ...]."""
            groups = {}
            for d in dirs:
                mj, cj_cfg = load_members([d], dev)
                if not np.allclose(load_dataset(cj_cfg)['q_obs'], q_train, rtol=0, atol=1e-15):
                    raise SystemExit(f'forcing of {d} differs from the operator members\' forcing')
                cj = np.load(cj_cfg['sampling']['local_file'])['center']
                groups.setdefault(cj.tobytes(), (cj, []))[1].extend(mj)
            out = []
            for cj, mj in groups.values():
                if np.allclose(cj, c_fwd, rtol=0, atol=1e-12):
                    continue
                cjt = torch.tensor(cj, device=dev)
                out.append((cjt, member_thetas(mj, t_obs, depth, cjt).mean(0)
                            - member_thetas(members, t_obs, depth, cjt).mean(0)))
            return out

        if args.fwd_fd and not args.fwd_jac:
            # finite-difference slope correction: forward PINNs at the paths of a design around the estimate
            # (scripts/fd_design.py: c0 +- 3 posterior sd along every posterior eigen-direction). The slope of the
            # discrepancy o(c) = theta_fwd(c) - theta_op(c) is the least-squares affine fit over the design and the
            # current path (central differences for the symmetric design); the operator keeps the curvature.
            pts = [(ct, offset)] + discrepancies(args.fwd_fd)
            C_fd, O_fd = torch.stack([p[0] for p in pts], 1), torch.stack([p[1] for p in pts], 1)
            dJ, resid = affine_slope(C_fd, O_fd)
            lin = (dJ, ct)
            J_o = jacobian(lambda cc: member_thetas(members, t_obs, depth, cc).mean(0), ct)
            nrm = lambda A: torch.linalg.norm(A, dim=0)
            rms = lambda A: torch.sqrt(torch.mean(A ** 2, 0)).tolist()
            # per design path: the fitted (linear) part of the discrepancy and what the fit leaves (curvature of the
            # discrepancy along that direction + training noise of that forward PINN)
            lin_part = dJ @ (C_fd - C_fd.mean(1, keepdim=True))
            fwd_info['fd_points'] = len(pts) - 1
            fwd_info['fd_fit_resid_rel'] = resid
            fwd_info['fd_offset_rms'] = rms(O_fd)[1:]
            fwd_info['fd_linear_rms'] = rms(lin_part)[1:]
            fwd_info['fd_resid_rms'] = rms(O_fd - O_fd.mean(1, keepdim=True) - lin_part)[1:]
            fwd_info['jac_rel_diff_fd_vs_op'] = (nrm(dJ) / nrm(J_o)).tolist()
            os.makedirs(args.out, exist_ok=True)
            np.savez(os.path.join(args.out, 'fwd_slope.npz'), dJ=dJ.cpu().numpy(), center=c_fwd)
            print(f"[fwd] finite-difference slope from {len(pts) - 1} design path(s): affine-fit residual / signal "
                  f"{resid:.2f}; |dJ| / |J_op| by coefficient "
                  f"{' '.join(f'{v:.2f}' for v in fwd_info['jac_rel_diff_fd_vs_op'])}", flush=True)
            print('[fwd] by design path, linear part / residual of the discrepancy [1e-4]: '
                  + ' '.join(f'{a * 1e4:.1f}/{b * 1e4:.1f}'
                             for a, b in zip(fwd_info['fd_linear_rms'], fwd_info['fd_resid_rms'])), flush=True)
        elif args.fwd_prev and not args.fwd_jac:
            # multi-secant slope correction (no extra training): forward PINNs of earlier rounds give the value
            # discrepancy o_j = theta_fwd(c_j) - theta_op(c_j) at other paths c_j; D is the least-change matrix with
            # D (c_j - c_k) = o_j - o_k for every j (Broyden's condition), zero across the span of the history
            prev = discrepancies(args.fwd_prev)
            dC, dO = [cj - ct for cj, _ in prev], [oj - offset for _, oj in prev]
            if dC:
                dC, dO = torch.stack(dC, 1), torch.stack(dO, 1)
                lin = (secant_correction(dC, dO), ct)
                fwd_info['secant_points'] = dC.shape[1]
                fwd_info['secant_dc_norm'] = torch.linalg.norm(dC, dim=0).tolist()
                fwd_info['secant_do_rms'] = torch.sqrt(torch.mean(dO ** 2, dim=0)).tolist()
                print(f"[fwd] secant correction from {dC.shape[1]} earlier path(s): |dc| "
                      f"{' '.join(f'{v:.3f}' for v in fwd_info['secant_dc_norm'])}, d offset rms "
                      f"{' '.join(f'{v:.1e}' for v in fwd_info['secant_do_rms'])}", flush=True)

    coarse, coarse_box = None, None
    if args.coarse:
        coarse, ccfg = load_members(args.coarse, dev)
        if not np.allclose(load_dataset(ccfg)['q_obs'], q_train, rtol=0, atol=1e-15):
            raise SystemExit('coarse members\' forcing differs from the operator members\' forcing')
        # the wide members were trained on the prior only: its support is a box in coefficient space, and the coarse
        # stage stays inside it (outside, a weakly identified coefficient can run away into extrapolation)
        coarse_box = prior_box(ccfg)
        print(f"[coarse] wide-stage coefficients kept in the prior box [{coarse_box[0]:.3f}, {coarse_box[1]:.3f}]",
              flush=True)

    def score(c):
        e = ZbSplineNumpy(sp, c, ds['zb_min']).zb(t_h) - ds['zb_true']
        return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(e[inner] ** 2)))

    results = []
    starts = args.starts.split(',')
    if args.prev_center:
        # refinement round: the members were refined around the previous estimate, so start there too
        starts = ['previous'] + starts
    for s in starts:
        if s == 'previous':
            c0 = np.load(args.prev_center)['center']
        else:
            kind, _, val = s.partition(':')
            c_cfg = dict(cfg, zb_init={'kind': kind, 'zb_max': 3.0, **({'value': float(val)} if val else {})})
            c0, _ = initial_coef(c_cfg, ds, sp)
        print(f"== start {s}: zb RMSE {score(c0)[0]:.4f} m", flush=True)
        c_start = torch.tensor(c0, device=dev)
        if coarse is not None and s != 'previous':
            rc = invert_lm(coarse, t_obs, depth, y, sig, c_start, verbose=False, box=coarse_box)
            c_start = rc['c'].detach()
            print(f"== start {s}: after the coarse (wide) stage cost {rc['cost']:.5f}, zb RMSE "
                  f"{score(c_start.cpu().numpy())[0]:.4f} m, coefficients "
                  + " ".join(f"{v:+.3f}" for v in c_start.cpu().numpy()), flush=True)
        r = invert_lm(members, t_obs, depth, y, sig, c_start, offset=offset, lin=lin, box=fine_box)
        c = r['c'].detach().cpu().numpy()
        e_all, e_in = score(c)
        zb_s = ZbSplineNumpy(sp, c, ds['zb_min']).zb(t_h)
        print(f"== start {s}: cost {r['cost']:.5f}, {len(r['hist'])} iters, zb RMSE {e_all:.4f} m "
              f"(days 0-25 {e_in:.4f}); zb {zb_s.min():.3f}-{zb_s.max():.3f} m, coefficients "
              + ' '.join(f'{v:+.3f}' for v in c), flush=True)
        results.append({'start': s, 'c0': c0, 'c': c, 'cost': r['cost'], 'iters': len(r['hist']),
                        'zb_rmse': e_all, 'zb_rmse_inner': e_in, 'theta': r['theta'].detach().cpu().numpy(),
                        'J': r['J'], 'last_step': r['last_step'].detach().cpu().numpy()})
    best = min(results, key=lambda r: r['cost'])            # chosen by the data misfit only
    c_hat = torch.tensor(best['c'], device=dev)
    # surrogate covariance Sigma_e = spread of the networks whose VALUE is fitted. With --fwd the fitted value at c_k
    # is the forward members' mean (the operator's value cancels there), so it is their spread on the one path they
    # were trained on; the operator members' spread would charge the estimate for an error that was removed.
    if args.fwd:
        cov, cov_noise, Se = posterior_cov(best['J'], sig, fwd, t_obs, depth, ct)
    else:
        cov, cov_noise, Se = posterior_cov(best['J'], sig, members, t_obs, depth, c_hat)
    cov, cov_noise = cov.cpu().numpy(), cov_noise.cpu().numpy()
    m = ZbSplineNumpy(sp, best['c'], ds['zb_min'])
    zb = m.zb(t_h)
    G = m.dzb_dcoef(t_h)
    s_zb = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G, cov, G), 0.0))
    s_zb_noise = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G, cov_noise, G), 0.0))
    zbs = np.stack([ZbSplineNumpy(sp, r['c'], ds['zb_min']).zb(t_h) for r in results])
    spread_all = float(np.max(np.sqrt(np.mean((zbs - zb[None]) ** 2, axis=1))))
    spread_in = float(np.max(np.sqrt(np.mean((zbs[:, inner] - zb[None, inner]) ** 2, axis=1))))
    e = zb - ds['zb_true']
    th_members = member_thetas(members, t_obs, depth, c_hat).cpu().numpy()
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, 'predictions.npz'), t_days=t_h, zb_pred=zb, zb_true=ds['zb_true'],
             zb_init=ZbSplineNumpy(sp, best['c0'], ds['zb_min']).zb(t_h), zb_sigma=s_zb, zb_sigma_noise=s_zb_noise,
             theta_obs_pred=best['theta'], t_obs_days=ds['t_obs'], theta_obs=ds['theta_obs'], coef=best['c'],
             theta_members=th_members)
    np.savez(os.path.join(args.out, 'starts.npz'), t_days=t_h, zb=zbs, starts=np.array([r['start'] for r in results]),
             cost=np.array([r['cost'] for r in results]), coef=np.stack([r['c'] for r in results]))
    # next refinement: centre = estimate; covariance = (inflate x posterior sd)^2 + displacement since last centre
    #   + floor (trust radius of the network, finding 37: a posterior-sized region excludes the truth when the
    #   estimate still carries surrogate bias, and the refinement stalls)
    cov_local = args.inflate ** 2 * cov + local_cov_from_width(best['c'], sp, args.floor_width)
    if args.prev_center:
        d = best['c'] - np.load(args.prev_center)['center']
        cov_local = cov_local + np.outer(d, d)
    np.savez(os.path.join(args.out, 'local_center.npz'), center=best['c'], cov=cov_local, cov_post=cov)
    step_over_sigma = None
    if args.prev_center:
        # stopping rule for iterated refinement: move of zb since the previous centre, per day segment, in units of
        # the posterior sd (days 0 .. n_days - knot)
        dz = zb - ZbSplineNumpy(sp, np.load(args.prev_center)['center'], ds['zb_min']).zb(t_h)
        edges = np.arange(0.0, ds['n_days'] - sp.dt + 1e-9, 2.5)
        step_over_sigma = [float(np.sqrt(np.mean(dz[(t_h >= a) & (t_h < b)] ** 2)) /
                                 np.sqrt(np.mean(s_zb[(t_h >= a) & (t_h < b)] ** 2)))
                           for a, b in zip(edges[:-1], edges[1:])]
    summary = {
        'best_start': best['start'], 'zb_rmse': float(np.sqrt(np.mean(e ** 2))),
        'zb_rmse_inner': float(np.sqrt(np.mean(e[inner] ** 2))), 'zb_bias': float(e.mean()),
        'zb_maxabs': float(np.abs(e).max()),
        'theta_rmse_vs_obs': float(np.sqrt(np.mean((best['theta'] - ds['theta_obs']) ** 2))),
        'theta_rmse_vs_truth': float(np.sqrt(np.mean((best['theta'] - ds['theta_obs_true']) ** 2))),
        'start_spread_rms_m': spread_all, 'start_spread_rms_inner_m': spread_in,
        'posterior_sigma_zb_median_m': float(np.median(s_zb)),
        'posterior_sigma_zb_rms_inner_m': float(np.sqrt(np.mean(s_zb[inner] ** 2))),
        'posterior_sigma_noise_only_rms_inner_m': float(np.sqrt(np.mean(s_zb_noise[inner] ** 2))),
        'ensemble_theta_sd_rms': float(np.sqrt(np.mean(np.diag(Se.cpu().numpy())))),
        'n_members': len(members), 'members': args.members,
        'starts': [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in r.items()
                    if k in ('start', 'cost', 'iters', 'zb_rmse', 'zb_rmse_inner')} for r in results],
        'noise_sigma': cfg['obs'].get('noise_sigma', 0.0), 'q_noise_frac': cfg['obs'].get('q_noise_frac', 0.0),
        'wall_s': round(time.time() - tic, 1),
        'fwd': fwd_info,
        'step_over_sigma_by_segment': step_over_sigma,
        'step_over_sigma_max': None if step_over_sigma is None else max(step_over_sigma),
        'box': None if fine_box is None else list(fine_box),
        'coef_on_bound': None if fine_box is None else
        [int(i) for i in np.where((best['c'] <= fine_box[0]) | (best['c'] >= fine_box[1]))[0]],
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('starts', 'members')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
