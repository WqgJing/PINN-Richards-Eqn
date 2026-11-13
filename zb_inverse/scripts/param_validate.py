#!/usr/bin/env python
"""SCORING ONLY (never part of the method): accuracy of trained parametric PINN members against the reference solver.

- theta_5cm(t) of the network (ensemble mean) vs the reference solver on fixed test paths: the true path, draws
  around it, prior draws, and optional estimates (npz files with 'coef' or 'center');
- sensitivity at the true path: autograd J_net = d theta / d c vs reference central differences J_ref;
- projected zb bias of the surrogate error at the truth, b = (J_ref^T J_ref)^-1 J_ref^T e, by day segment, against
  the referee's local posterior sigma.

Usage: python -u scripts/param_validate.py --members runs/p1_a ... --out runs/p1_a/validation.json
           [--referee output/classical/v3_noise005] [--coef runs/p1_inv/predictions.npz ...]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_reference  # noqa: E402
from zbinv.parametric import load_members, member_thetas, CPrior, LocalGaussian, local_cov_from_width  # noqa: E402
from zbinv.train import load_dataset  # noqa: E402
from zbinv.spline import CubicBSpline1D, ZbSplineNumpy  # noqa: E402

torch.set_default_dtype(torch.float64)
EDGES = [0, 5, 10, 15, 20, 22.5, 25, 27.5, 30.01]
_G = {}


def ref_theta(coef):
    g = _G
    m = ZbSplineNumpy(g['spline'], coef, g['zb_min'])
    return run_reference(g['tcfg'], g['soil'], g['win'], m, g['t_obs'], n_nodes=401, verbose=False)['theta_obs']


def seg_rms(t, v):
    return [float(np.sqrt(np.mean(v[(t >= a) & (t < b)] ** 2))) for a, b in zip(EDGES[:-1], EDGES[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--referee', default=None)
    ap.add_argument('--coef', nargs='*', default=[])
    ap.add_argument('--n-local', type=int, default=3)
    ap.add_argument('--n-prior', type=int, default=4)
    ap.add_argument('--local-width', type=float, default=0.05)
    ap.add_argument('--delta', type=float, default=0.01)
    ap.add_argument('--jac-at', default=None, help='npz (coef or center): compare J_net with J_ref at this path '
                    'instead of the truth (e.g. the centre a member was refined around)')
    ap.add_argument('--jac-corr', default=None, help='fwd_slope.npz written by param_invert.py --fwd-fd: score '
                    'J_net + dJ (the slope-corrected model of the inversion) instead of J_net')
    ap.add_argument('--fixed', type=int, nargs='*', default=[], help='coefficients held on a bound of the prior box '
                    '(coef_on_bound of the summary.json): also report the Jacobian error and the contraction over '
                    'the free coefficients, the only ones the boxed inversion moves')
    args = ap.parse_args()
    dev = torch.device('cpu')
    members, cfg = load_members(args.members, dev)
    ds = load_dataset(cfg)
    if ds.get('tp_mmd') is not None:   # site data, FEM twins: the PINN and the data set's solver have the root sink
        raise SystemExit("this data set has a root sink and the reference solver of this script has none: its "
                         "scores would not be PINN-against-solver errors. Use scripts/truth_error_profile.py "
                         "--sensor fem (value at the well's path) or scripts/fem_referee.py (inversion).")
    sp = CubicBSpline1D(0.0, ds['n_days'], cfg.get('zb_spline', {}).get('knot_days',
                                                                        ds['twin_cfg']['zb_spline']['knot_days']))
    depth = cfg['obs']['depth_m']
    c_true = np.asarray(ds['truth_coef'], dtype=float)
    g = torch.Generator().manual_seed(7)
    paths = {'truth': c_true}
    loc = LocalGaussian(c_true, local_cov_from_width(c_true, sp, args.local_width))
    for i, c in enumerate(loc.sample(args.n_local, g).numpy()):
        paths[f'local{i}'] = c
    pc = cfg.get('prior', {})
    pr = CPrior(sp, ds['zb_min'], pc.get('mean_range_m', (0.3, 1.2)), pc.get('step_sd_m', 0.15),
                pc.get('clip_m', (0.2, 1.5)))
    for i, c in enumerate(pr.sample(args.n_prior, g).numpy()):
        paths[f'prior{i}'] = c
    for f in args.coef:
        z = np.load(f)
        paths[os.path.basename(os.path.dirname(f)) or f] = z['coef'] if 'coef' in z else z['center']
    c_jac, jac_name = c_true, 'truth'
    if args.jac_at:
        z = np.load(args.jac_at)
        c_jac, jac_name = (z['coef'] if 'coef' in z else z['center']), 'jac_at'
        paths['jac_at'] = c_jac

    tcfg = dict(ds['twin_cfg'])
    tcfg['obs'] = dict(tcfg['obs'], depth_m=depth)
    _G.update(spline=sp, zb_min=ds['zb_min'], tcfg=tcfg, soil=ds['soil'],
              win={'t_days': ds['t_q'], 'qsurf': ds['q_obs']}, t_obs=ds['t_hourly'])
    names = list(paths)
    fd_jobs = [c_jac + s * args.delta * e for e in np.eye(sp.n_coef) for s in (1.0, -1.0)]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))) as pool:
        res = pool.map(ref_theta, [paths[n] for n in names] + fd_jobs)
    th_ref = dict(zip(names, res[:len(names)]))
    fd = res[len(names):]
    J_ref = np.stack([(fd[2 * k] - fd[2 * k + 1]) / (2 * args.delta) for k in range(sp.n_coef)], axis=1)

    t_obs = torch.tensor(ds['t_hourly']).reshape(-1, 1)
    t_h = ds['t_hourly']
    out = {'members': args.members, 'paths': {}}
    for n in names:
        thm = member_thetas(members, t_obs, depth, torch.tensor(paths[n])).numpy()
        e = thm.mean(0) - th_ref[n]
        zb = ZbSplineNumpy(sp, paths[n], ds['zb_min']).zb(t_h)
        out['paths'][n] = {'theta_rmse': float(np.sqrt(np.mean(e ** 2))), 'theta_max': float(np.abs(e).max()),
                           'theta_rmse_days0_1': float(np.sqrt(np.mean(e[t_h <= 1.0] ** 2))),
                           'theta_rmse_by_segment': seg_rms(t_h, e),
                           'member_rmse': [float(np.sqrt(np.mean((x - th_ref[n]) ** 2))) for x in thm],
                           'ensemble_sd_rms': float(np.sqrt(np.mean(thm.std(0, ddof=1) ** 2))) if len(thm) > 1 else 0.0,
                           'zb_range_m': [float(zb.min()), float(zb.max())]}
        print(f"[val] {n:28s} theta RMSE {out['paths'][n]['theta_rmse']:.2e} max {out['paths'][n]['theta_max']:.2e} "
              f"zb {zb.min():.2f}-{zb.max():.2f} m | by segment "
              f"{' '.join(f'{v:.1e}' for v in out['paths'][n]['theta_rmse_by_segment'])}", flush=True)

    fn = lambda cc: member_thetas(members, t_obs, depth, cc).mean(0)
    J_net = torch.func.jacfwd(fn)(torch.tensor(c_jac)).numpy()
    if args.jac_corr:
        dJ = np.load(args.jac_corr)['dJ']
        if dJ.shape != J_net.shape:
            raise SystemExit(f'--jac-corr has shape {dJ.shape}, the Jacobian {J_net.shape}')
        out['jac_corr'] = args.jac_corr
        _, S0, Vt0 = np.linalg.svd(J_ref, full_matrices=False)
        rel0 = [float(np.linalg.norm((J_net - J_ref) @ v) / s) for v, s in zip(Vt0, S0)]
        out['rel_jac_err_by_direction_uncorrected'] = rel0
        print(f"[val] rel J error by direction, uncorrected   {' '.join(f'{v:.2f}' for v in rel0)}", flush=True)
        J_net = J_net + dJ
    cos = [float(J_net[:, k] @ J_ref[:, k] / (np.linalg.norm(J_net[:, k]) * np.linalg.norm(J_ref[:, k]) + 1e-300))
           for k in range(sp.n_coef)]
    ratio = [float(np.linalg.norm(J_net[:, k]) / (np.linalg.norm(J_ref[:, k]) + 1e-300)) for k in range(sp.n_coef)]
    e_truth = fn(torch.tensor(c_jac)).numpy() - th_ref[jac_name]
    b_c = np.linalg.lstsq(J_ref, e_truth, rcond=None)[0]
    G = ZbSplineNumpy(sp, c_jac, ds['zb_min']).dzb_dcoef(t_h)
    # weak directions: relative Jacobian error along the right singular vectors of J_ref (smallest last), and the
    # predicted per-round contraction of the refinement loop, M = pinv(J_net) (J_net - J_ref) (to first order the
    # next estimate's error is M times the current centre's error)
    _, S, Vt = np.linalg.svd(J_ref, full_matrices=False)
    rel_dir = [float(np.linalg.norm((J_net - J_ref) @ v) / s) for v, s in zip(Vt, S)]
    M = np.linalg.pinv(J_net) @ (J_net - J_ref)
    out['jac_at'] = jac_name
    out['singular_values_ref'] = S.tolist()
    out['rel_jac_err_by_direction'] = rel_dir
    out['contraction_spectral_radius'] = float(np.max(np.abs(np.linalg.eigvals(M))))
    out['contraction_norm_2'] = float(np.linalg.norm(M, 2))
    print(f"[val] J at {jac_name}: singular values {' '.join(f'{v:.2g}' for v in S)}", flush=True)
    print(f"[val] rel J error by direction (strong->weak) {' '.join(f'{v:.2f}' for v in rel_dir)}", flush=True)
    print(f"[val] predicted contraction per round: spectral radius {out['contraction_spectral_radius']:.2f}, "
          f"2-norm {out['contraction_norm_2']:.2f}", flush=True)
    if args.fixed:
        free = [k for k in range(sp.n_coef) if k not in args.fixed]
        Jn, Jr = J_net[:, free], J_ref[:, free]
        _, Sf, Vtf = np.linalg.svd(Jr, full_matrices=False)
        Mf = np.linalg.pinv(Jn) @ (Jn - Jr)
        out['fixed'] = list(args.fixed)
        out['singular_values_ref_free'] = Sf.tolist()
        out['rel_jac_err_by_direction_free'] = [float(np.linalg.norm((Jn - Jr) @ v) / s) for v, s in zip(Vtf, Sf)]
        out['contraction_spectral_radius_free'] = float(np.max(np.abs(np.linalg.eigvals(Mf))))
        out['contraction_norm_2_free'] = float(np.linalg.norm(Mf, 2))
        print(f"[val] coefficients {args.fixed} fixed: singular values {' '.join(f'{v:.2g}' for v in Sf)}", flush=True)
        print(f"[val] coefficients {args.fixed} fixed: rel J error by direction (strong->weak) "
              f"{' '.join(f'{v:.2f}' for v in out['rel_jac_err_by_direction_free'])}", flush=True)
        print(f"[val] coefficients {args.fixed} fixed: predicted contraction per round: spectral radius "
              f"{out['contraction_spectral_radius_free']:.2f}, 2-norm {out['contraction_norm_2_free']:.2f}", flush=True)
    bias_zb = G @ b_c
    out['sensitivity_truth'] = {'col_cos': cos, 'col_norm_ratio': ratio}
    out['projected_bias_cm_by_segment'] = [100 * v for v in seg_rms(t_h, bias_zb)]
    inner = t_h <= ds['n_days'] - sp.dt
    out['projected_bias_rms_inner_cm'] = float(100 * np.sqrt(np.mean(bias_zb[inner] ** 2)))
    print(f"[val] J cos   {' '.join(f'{v:.3f}' for v in cos)}", flush=True)
    print(f"[val] J ratio {' '.join(f'{v:.3f}' for v in ratio)}", flush=True)
    print(f"[val] projected zb bias by segment [cm] {' '.join(f'{v:.2f}' for v in out['projected_bias_cm_by_segment'])}"
          f"  (days 0-25 rms {out['projected_bias_rms_inner_cm']:.2f})", flush=True)
    # value-corrected inversion (accurate theta at the estimate, network Jacobian for the step): its fixed point solves
    # J_net^T (y - theta(c)) = 0, so the noise maps to c by A = (J_net^T J_ref)^-1 J_net^T instead of the exact
    # (J_ref^T J_ref)^-1 J_ref^T. Column scaling of J_net cancels; only direction errors cost accuracy.
    # Reported: zb sd of that estimator / zb sd of the exact one, by segment (sigma cancels).
    A = np.linalg.solve(J_net.T @ J_ref, J_net.T)
    sd_net = np.sqrt(np.einsum('ij,jk,ik->i', G, A @ A.T, G))
    sd_ref = np.sqrt(np.einsum('ij,jk,ik->i', G, np.linalg.inv(J_ref.T @ J_ref), G))
    out['corrected_sd_ratio_by_segment'] = [a / b for a, b in zip(seg_rms(t_h, sd_net), seg_rms(t_h, sd_ref))]
    print(f"[val] value-corrected estimator sd / exact sd {' '.join(f'{v:.2f}' for v in out['corrected_sd_ratio_by_segment'])}",
          flush=True)
    if args.referee:
        r = np.load(os.path.join(args.referee, 'predictions.npz'))
        sig = seg_rms(r['t_days'], r['zb_sigma'])
        out['referee_sigma_cm_by_segment'] = [100 * v for v in sig]
        out['bias_over_sigma_by_segment'] = [b / (100 * s) for b, s in zip(out['projected_bias_cm_by_segment'], sig)]
        print(f"[val] referee sigma by segment [cm]    {' '.join(f'{100 * v:.2f}' for v in sig)}", flush=True)
        print(f"[val] bias / sigma                      {' '.join(f'{v:.2f}' for v in out['bias_over_sigma_by_segment'])}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.splitext(args.out)[0] + '_jac.npz', c=c_jac, J_net=J_net, J_ref=J_ref)


if __name__ == '__main__':
    main()
