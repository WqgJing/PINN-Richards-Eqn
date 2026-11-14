# zb_inverse: PINN inverse for zb(t) from surface flux + surface soil moisture

Plan: `/storage/home/hcoda1/9/jjason3/.claude/plans/piped-roaming-cocoa.md`

## Acceptance criteria (E2, clean twin, one 30-day window)
- zb RMSE <= 0.05 m where the Cramer-Rao bound allows it (report the bound next to the result)
- data RMSE <= 0.002; flux residual <= 2% of RMS(q); PDE residual RMS <= 1e-3 (normalized)
- < 1 h on one H200; holds for >= 1 weighting scheme across 3 seeds

## Checklist
- [x] soil.py (smooth VG, numpy+torch), spline.py, data.py, reference.py (MOL, BDF)
- [x] tests: numpy/torch agree, capacity FD, steady infiltration vs ODE, hydrostatic steady, mass balance
- [x] E0: twins (apr, d80, season), FEM cross-check, identifiability (5-day knots)
- [x] E0b: 801-node twins + refinement, matched-resolution FEM check, identifiability 10/15-day knots
- [x] Twin admissibility check (surface head, max |dh/dz|); synthetic v2 (admissible); head-limited surface BC (HYDRUS)
      in the reference + test; d80 head-limited twins (h_crit -100 / -1000 m)
- [x] model: departure='log' + log-depth input feature (for dry surface layers), tests
- [x] model.py, physics.py, weighting.py, train.py, scripts/train.py
- [x] tests: Landau residual vs physical FD, surface flux vs FD, hard constraints (both forms), obs operator, head-limited BC
- [ ] E1 forward (zb = truth): theta RMSE vs reference < 0.005
- [ ] E2 inverse clean twin (d80 first, hourly obs), seeds x3, weighting ablation (gradnorm / fixed)
- [x] Operator-PINN inverse on synthetic v3: clean, noise 0.005 / 0.01, noise seeds 0 / 1 (finding 62)
- [ ] E4 full season; E5 real US-PAS data

## Working notes
- CONSTRAINT (user, 2026-11-09): the method must stay a PINN. No hybrid FEM/FV + PINN inversion. The classical
  inversion (scripts/classical_inversion.py) is a referee for scoring only, never part of the method.
- Login node: never run python; everything via sbatch (`hpc/*.slurm`, env python by absolute path:
  `module` is not available in batch shells).
- VERSION CONTROL (user, 2026-11-11: "every try, commit and push"): after every try (a code change and/or a
  scored result) commit and push to github.com/WqgJing/PINN-Richards-Eqn (PUBLIC), branch master.
  The repository root is the parent folder pinn/ (.git there), sparse checkout: only zb_inverse/ is checked out;
  the January joint-PINN files at the repository root stay tracked and untouched (skip-worktree), and
  pinn/README.md, MANIFEST.md, docs/, legacy/ are local only (.git/info/exclude). Commit with
  `cd /storage/project/r-rbras6-0/jjason3/pinn && git add zb_inverse && git commit && git push origin master`
  (never `git sparse-checkout disable`, never force-push). Tracked: code, configs, tests, this file and, under
  output/, only result_*.json/png, soil_*/start*.json/png and compare/ (zb_inverse/.gitignore). Not tracked:
  runs/, logs/, data/, *.npz, *.pt.
- US-PAS FEM: L=5, 51 nodes, no sink, daily Picard steps; only 83% converged; theta_surf uses L_dim=8.0 bug.
- Reference grid: 201 vs 401 nodes differ by 3.8e-3 in theta_obs (about first order) -> twins at 801 nodes.
- FEM (51 nodes) vs reference (201): theta_obs RMSE 0.034 (apr) / 0.015 (d80) on converged steps;
  matched-resolution check pending.
- Identifiability (5-day knots, CRLB): daily obs sigma 0.005 -> median sigma_zb 0.70 m (apr), 0.24 m (d80);
  hourly obs -> 0.13 m (apr), 0.035 m (d80). End-of-window knots poorly constrained (lag).
  => only the slow part of zb is recoverable; hourly in-situ obs needed; shallow WT best.

- Full season (227 d, 5-day knots) is far more identifiable than 30-day windows: hourly obs sigma 0.01 ->
  median sigma_zb 0.023 m (max 0.37 m); daily obs sigma 0.005 -> median 0.057 m.
- Reference grid 401 vs 801: max 1.1e-3 in theta_obs (twins now at 801 nodes).

## Debug findings (2026-11-09)
1. Causal weights on sigma-normalised losses: eps = 1 with chunk losses ~1e4 gave weight 0 to all but the
   first chunk -> PDE solved only on day 0-2. Causal weighting now off by default (it needs a hand-set eps).
2. Gradient-norm balancing that equalises all terms pushed w_pde to 0.03 -> data fitted (theta RMSE 0.005)
   with PDE residual ~8e3, zb meaningless. Fixed: PDE is the reference (w = 1), others = g_pde / g_k.
3. Baseline "hydrostatic at zb(0) for all t" implies huge spurious fluxes when the WT moves (d80: 1.55 ->
   0.23 m). Replaced by hydrostatic at the CURRENT zb(t) + learned departure (zero flux at init).
4. sigma_pde = 0.01 was ~1e4 below the natural residual scale at rain fronts; now data-driven
   (RMS d Se_obs / d t~).
5. L-BFGS optimised a different objective (no causal) and worsened the fit; now same objective + keep best.
6. Added RAR-D residual-adaptive sampling (Wu et al. 2023, k = c = 1).

## Results
(2026-11-10) Operator-PINN inverse on the synthetic v3 twin: all five cases meet the acceptance of the plan
(finding 62 has the table; figures output/compare/final_*.png).
- What changed: the joint PINN (one network fits data and physics) was replaced by networks trained on physics
  only: an operator PINN ensemble h(xi, t; c) (4 members, wide stage on the path prior, then local refinement), a
  finite-difference design of 18 forward PINNs per forcing for the slope correction, and 2 forward PINNs at the
  current estimate per loop round for the value correction. zb is fitted to theta_5cm by LM over the 9 spline
  coefficients with exact autograd Jacobians, 5 starts, and the estimate is confined to the support of the path
  prior the networks were trained on (zb in [0.2, 1.5] m, param_invert.py --box).
- Where: src/zbinv/parametric.py (invert_lm, box), scripts/param_train.py, scripts/param_invert.py (--fwd-fd,
  --coarse, --box), scripts/fd_design.py, hpc/dc_loop.sh, hpc/fd_set.sh, scripts/param_validate.py (scoring only),
  scripts/classical_inversion.py (solver referee, scoring only), scripts/final_figure.py.
- How verified: tests/test_parametric.py (21 pass, job 15448340); zb RMSE against the truth and against the solver
  referee by segment (scripts/error_by_day.py); distance to the boxed solver estimate (scripts/dist_to_referee.py,
  <= 0.13 sigma everywhere); Jacobian of the corrected model against the solver at every final estimate
  (scripts/param_validate.py); five starts per case agree to < 1e-4 cm.
- Limits: twin experiment only (synthetic v3, soil parameters known, one 30-day window, two noise seeds). Where the
  data push c[8] out of the prior range (3 of 5 cases) days 25-30 are set by the prior bound, not by the data;
  the reported sigma is the unconstrained Laplace value. The acceptance targets come from the unconstrained
  solver estimate. Cost: 4 + 4 x (refinement rounds) operator trainings, 18 design PINNs and 2 PINNs per loop
  round, per forcing; wall time was not measured cleanly (queue congestion). E4 (full season) and E5 (real
  US-PAS data) are not done.

## Findings 2026-11-09 (afternoon)
7. ROOT CAUSE of the forward failure: the twins were inadmissible. Synthetic v1 (q up to 5.5 mm/d evaporation, WT to 0.85 m) and
   US-PAS d80 (13 mm/d) ask the sand to evaporate more than it can deliver; with a pure flux BC the surface node dried to
   h = -2e7 m / -5e8 m, |dh/dz| ~ 1e11. No smooth network can represent that. make_twin now reports h_surf_min, max |dh/dz|.
8. Synthetic v2: q = -1 + 2 sin(2 pi t / 6 d) mm/d, zb = 0.6 + 0.25 sin(2 pi t / 20 d). h_surf >= -0.73 m, max |dh/dz| 1.98,
   theta_obs 0.248-0.333; 401/801/1601-node theta_obs agree to 1.2e-6 (v1: 1e-3). Reference solve 3 s (v1: 400 s).
9. Head-limited surface BC (HYDRUS system-dependent; gate in log|h| at h_crit): d80 twin with h_crit -100 m and -1000 m give the
   same supply-limited fraction (14.4% of hours) and evaporation deficit (29.2 mm / 30 d); theta_obs min 0.1323 vs 0.1321.
   The observables do not depend on h_crit. But the dry crust is sub-mm (|h| ~ depth^-0.42 from h_crit to ~-2 m within ~2 mm),
   below the continuum scale; the 51-node FEM (dz = 10 cm) never sees it. A PINN must not try to resolve it -> open design
   question for real forcing (options: Kirchhoff / matric-flux-potential output; log departure + log-depth feature, done).
   Twin's observable flux = actual delivered flux (q_act_hourly), which is what eddy covariance measures.
10. Synthetic v2 E1 forward (V100, 15 min): theta RMSE 0.0027 (< 0.005, PASS), profile RMSE 0.0025, flux error 0.038 mm/d.
    Stage-2 lr stayed constant in forward mode (bug) -> noisy plateau; fixed (final Adam stage always decays).
11. Synthetic v2 E2 inverse (gradnorm, V100, 22 min): zb RMSE 0.085 m (init 0.166; CRLB median 0.014 m), theta RMSE 1.5e-5,
    w_data at the cap 100. Data over-fitted 200x below the noise level while the PDE stayed at the forward floor -> zb absorbs
    the physics error. Two causes:
    a) residual map (scripts/residual_map.py): 85% of the squared PDE residual within ~2 cm of the WT, peaking with |dzb/dt|.
       Plain VG-Mualem with n = 1.56 < 2: K drops to 0.92 Ks at |h| = 1 mm, 0.44 Ks at 4 cm (Ippisch et al. 2006).
       Fix: air-entry model (h_e = -2 cm, HYDRUS option), smooth-min 5 mm; soil.air_entry_m. Twin v3.
    b) gradient-norm balancing raises w_data as the data get fitted (positive feedback -> over-fitting). New option
       weighting.mode = discrepancy: PDE = objective, data and flux held at their sigma-normalised noise level (L_k = 1) by
       dual ascent on log w_k (Morozov discrepancy principle). Ablation v3: gradnorm / discrepancy / fixed + noisy (0.005).
12. Twin v3 (air-entry): 401/801/1601 theta_obs agree to 1e-6; CRLB hourly sigma 0.005 -> median 0.008 m, rms 0.021 m (v2: 0.014 / 0.075).
13. v3 runs (synthetic, clean data unless noted; output/compare/v3_weighting_ablation.{png,csv}):
    | run                  | zb RMSE | zb RMSE excl. last 5 d | zb bias | theta RMSE | flux err mm/d | pde_rms |
    | E1 forward           |   -     |   -                    |   -     | 0.0018     | 0.038         | 0.025   |
    | E2 gradnorm          | 0.055   | 0.054                  | -0.029  | 1.7e-5     | 0.038         | 0.022   |
    | E2 discrepancy       | 0.073   | 0.073                  | -0.040  | 0.0047     | 0.204         | 0.0019  |
    | E2 fixed (sigma)     | 0.026   | 0.014                  | -0.003  | 1.1e-4     | 0.038         | 0.0069  |
    | E3 discrepancy, noise 0.005 | 0.051 | 0.048            | -0.025  | 0.0008 (vs truth) | 0.092  | 0.017   |
    Air-entry alone: gradnorm 0.085 -> 0.055. All schemes recover the shallow phase (days 10-20) to ~1 cm; the deep phase
    (zb 0.85 m) is pulled toward the shallow initial guess by 0.10 m (adaptive) vs 0.02-0.035 m (fixed); the last ~3 days
    diverge in every run (CRLB). gradnorm: over-fits data (w_data at cap). discrepancy: meets L_data = L_flux = 1 with a
    systematic (not noise-like) misfit, flattens zb in the deep phase (static WT is easier for the network) and shifts the
    flux phase. Fixed sigma-normalised weights (Gaussian likelihood, sigma_pde = RMS dSe_obs/dt~) best.
    Running: fixed x seeds 1, 2; fixed + noise 0.005 / 0.01; fixed long (60k Adam + 8k L-BFGS).
14. Robustness (fixed weights, synthetic v3): seeds 0/1/2 -> zb RMSE days 0-25 1.4/1.0/1.7 cm; 2x longer 0.8 cm; shallow start
    0.4 m 1.7 cm; deep start 1.0 m DIVERGES (zb -> ~14 m); noise 0.005 -> 5.9 cm, noise 0.01 -> 9.1 cm (PDE rms 0.23/0.44 vs
    0.004 clean: the PINN bends the physics to follow noise spikes). Figures: output/compare/evidence_synthetic_inverse.png,
    output/compare/failure_modes.png.
15. Depth limit (scripts/depth_sensitivity.py, steady states): zb change equal to a 0.005 theta error: hydrostatic 4 cm at 0.85 m,
    14 cm at 2 m; infiltration 1 mm/d: 7 cm at 0.85 m, 64 cm at 1.5 m, 3.8 m at 2 m, none beyond (theta_5cm pinned at 0.230 by
    gravity drainage); evaporation 0.5 mm/d: no steady state beyond ~1 m.
16. REFEREE (scripts/classical_inversion.py: reference solver + Levenberg-Marquardt over the same 9 spline coefficients, same
    noisy data, 2-3 min on 8 CPUs): clean 0.0 cm; noise 0.005 + 5% q -> 0.82 cm (posterior sigma 0.79 cm); noise 0.01 -> 1.5 cm
    (posterior 1.6 cm); theta vs truth 0.0004 / 0.0008 (noise filtered). => NOT ill-posed at this depth: the PINN's 5.9 / 9.1 cm
    is the PINN. Hypothesis: sigma_pde (auto 0.084) ~20x looser than the residual the network reaches makes physics violation
    cheap. Testing: sigma_pde = 0.0042 (diagnostic), 'stage1' (residual after forward stage), 'ml' (max-likelihood EMA).
17. sigma_pde hypothesis REFUTED (v4, noise 0.005 unless clean; zb RMSE days 0-25):
    tight 0.0042 -> 6.2 cm (data under-fit 0.0064); 'ml' -> 5.9 cm; clean 'ml' -> 6.5 cm (clean data under-fit 0.0047!);
    'stage1' (sigma rose to 0.123) -> 2.8 cm, still improving at the end; clean 'stage1' -> 1.1 cm.
    Mechanism seen in the histories: with strong physics weight zb LOCKS - moving zb with the network momentarily fixed
    raises the PDE residual, so the PDE term resists every zb update; the network fits physics at the wrong zb and the
    data misfit grows (clean ml: 0.001 -> 0.0047 while zb error 5.6 -> 6.5 cm). Weak physics: zb moves but slowly, and the
    network can fit noise. => joint soft-penalty optimisation is the problem, not the data.
    Fix under test: train.split (bilevel, first-order): network <- PDE + flux only; zb <- data only. Batch v5.
18. train.split (first-order bilevel) FAILED: zb drifts deeper without bound (clean: zb error days 0-25 7.7 cm at step 6000,
    then 93 cm, bias +2.4 m; noise 0.005: 67 cm / 13 cm (seed 1); noise 0.01: 91 cm) while the theta misfit stays ~0.005.
    Cause: zb is pushed by the data gradient through the FROZEN network, i.e. the rigid hydrostatic-shift sensitivity;
    the network then re-solves PDE + flux and the surface returns to the flux-controlled state, the misfit returns, zb is
    pushed again -> drift along the weakly identifiable direction. A bilevel scheme needs the implicit (total) hypergradient.
    Keep train.split off (default).
19. Joint PINN, noise 0.005, 2x training (v5_joint_noise005_long), at step 50k: zb 6.3 cm, theta misfit 0.0034 (< noise 0.005:
    fitting noise), PDE rms 0.21 -> not slow convergence; it settles on a noise-fitting, physics-bending solution.
CONCLUSION (2026-11-09): soft-penalty PINN works on clean data (0.8-1.7 cm) but cannot reach the information limit with
    realistic noise: weak penalty -> fits noise (6 cm); strong -> zb locks (6 cm); one intermediate setting 2.8 cm (not robust);
    first-order bilevel diverges. Classical inversion (FV solver + LM, same 9 coefficients): 0.8 cm / 1.5 cm in 2-3 min.
    A full bilevel (implicit hypergradient) or hard-constraint PINN may close the gap but turns the PINN into a slower
    PDE-constrained solver. Recommendation: classical inversion for zb (with posterior sigma); PINN only where it adds
    something (e.g. unknown soil curves).
20. Augmented-Lagrangian PINN (alm.enabled): data = objective; PDE + flux = equality constraints at FIXED points (16384 / 2048)
    with per-point multipliers (lam -> adjoint field, restores the total zb gradient). Adam minibatches + multiplier update
    every 500 steps, mu 1 -> <=100 (x2 unless violation halves); L-BFGS blocks of 100 with multiplier update between blocks
    (freezing lam drives r to -lam/mu: smoke test showed L_A -> -152). Batch v6: ALM noise 0.005 (2 seeds), noise 0.01,
    clean; feature ablation (Fourier periods >= 8 d) with joint and ALM. Target: noise 0.005 <= 1.6 cm, 0.01 <= 3 cm.
21. Joint PINN, noise 0.005, 2x training, FINAL: zb 17 cm (days 0-25), bias +15 cm. During L-BFGS the noise over-fit grew:
    theta misfit 0.0033 -> 0.0027 (noise 0.005), PDE rms 0.24 -> 0.31, zb error 4.3 -> 17 cm. Longer training makes the
    soft-penalty PINN worse on noisy data (classic over-fitting); confirms the joint penalty formulation, not convergence speed.
22. v6 FAILED: ALM noise 0.005 -> 8.4 / 8.8 cm (2 seeds), noise 0.01 -> 9.4 cm, clean -> 6.6 cm (worse than joint 0.8-1.7);
    no-fast-Fourier: joint 6.5 cm, ALM 7.0 cm (no gain -> fast features are not the cause).
    ALM history: constraint violation on the fixed set stalls at 0.2-0.5 (normalised; same on held-out points, so no point
    over-fitting); multipliers grow linearly without converging (lam rms 8 -> 280); mu hits its cap 100 at step ~12k; clean data
    under-fit (0.005) -> zb locked, as with the strong penalty. The method of multipliers needs r -> 0, which the network
    cannot reach (residual floor), so it degenerates into an ever-stronger penalty.
    REFINED ROOT CAUSE: the network's own forward error floor (PDE residual ~0.2 sigma_pde; theta ~0.002) is comparable to the
    zb signal under noise. Every joint formulation trades that slack: strong physics -> zb locks; weak physics -> slack absorbs
    noise; constraints -> multipliers diverge. Options left (all PINN-only): (a) parametric PINN h(xi, t; c) trained on physics
    only over the zb-coefficient space, then invert with exact autograd sensitivities (network never sees data; amortised;
    cheap posterior); (b) bilevel Gauss-Newton with warm-started PINN re-solves; (c) lower the floor (second-order optimiser).
23. USER (2026-11-09): "keep in mind with the baseline. only move forward until you think current one won't help" -> before
    any new formulation (parametric PINN etc.), exhaust fixes inside the baseline (fixed sigma weights, same model/loss).
    NEW EVIDENCE in the baseline logs: noisy runs show a TRAIN/HELD-OUT PDE gap that the clean run does not.
      clean:      logged L_pde 0.009 -> raw rms 0.008; held-out uniform pde_rms 0.007 (consistent)
      noise 0.005: logged L_pde 0.06 -> raw rms 0.020; held-out pde_rms 0.23 (11x larger)
    Points are reused for 100 Adam steps (resample_every 100) and L-BFGS uses ONE fixed set (16384) -> hypothesis H23:
    with noisy data the network fits noise by violating the PDE BETWEEN collocation points (collocation over-fitting).
    On honest (fresh) points, fitting 0.005 noise would cost L_pde ~ 5-10 vs a data gain <= 1, so the honest objective
    already penalises it; the sampled one does not. Smallest fix = config only: resample every step, no fixed-point L-BFGS.
    Check: scripts/collocation_gap.py (fresh-point loss vs logged loss on 6 saved runs; CPU job 15405536).
    Runs v7 (baseline configs + train.resample_every=1, train.lbfgs_iters=0): v7_rs1_noise005, v7_rs1_noise010,
    v7_rs1_clean (regression), v7_rs1_norar_noise005 (RAR off). Go: noise 0.005 <= 2 cm, 0.01 <= 3 cm, clean <= 2 cm.
24. H23 (collocation over-fitting) mostly REFUTED by scripts/collocation_gap.py: normalised by each run's OWN sigma_pde,
    fresh-point L_pde / logged L_pde = 1.5x (noise 0.005), 1.7x (0.01), 1.6x (clean) -> no noise-specific gap. The 11x in
    finding 23 came from dividing the noisy runs' residual by the CLEAN sigma_pde. v7 runs cancelled.
    ROOT CAUSE FOUND (baseline bug): sigma_pde 'auto' = RMS of HOURLY differences of the NOISY theta_obs, i.e. it measured
    the noise: sqrt(2) sigma / (theta* dt~) -> 1.353 (sigma 0.005), 2.702 (0.01) vs 0.0843 clean (startup log lines of
    v3_e3_fixed_noise005/010). The PDE term in every noisy baseline run was 256x / 1000x weaker than on clean data, so the
    network fitted the noise by bending the physics (pde_rms 0.23 / 0.44). Finding 16's "auto 0.084" for the noisy runs was
    an unchecked assumption; 'tight' 0.0042 was 320x (not 20x) tighter than what those runs used; 'stage1' (0.123, the
    only noisy setting near the clean scale) was the best noisy run (2.8 cm, still improving).
    FIX: train.signal_rate_scale: day means of theta_obs (noise / sqrt(24)) minus the known noise variance of each
    difference; incomplete days dropped; old estimator kept as sigmas.pde = auto_hourly. tests/test_weights.py: within
    15% of the clean value at sigma 0.005 / 0.01 (5 seeds each), daily data within 0.6-1.6x; old estimator > 10x off.
    17/17 tests pass (job 15405661). Runs v8 = exact baseline configs with the fixed estimator: v8_autofix_noise005,
    _noise010, _clean (regression; sigma_pde changes slightly), _noise005_seed1.
25. v8 RESULTS (exact baseline configs, fixed sigma_pde estimator; zb RMSE days 0-25; referee = classical LM):
    | case        | old baseline | v8 fixed estimator        | referee (posterior sigma) |
    | noise 0.005 | 5.9 cm       | 1.69 cm (seed 1: 1.85 cm) | 0.78 cm (0.76)            |
    | noise 0.01  | 9.1 cm       | 2.26 cm                   | 1.21 cm (1.54)            |
    | clean       | 1.43 cm      | ~1.7 cm (sigma 0.077)     | 0.00 cm                   |
    noise 0.005: data misfit 0.0048 (= noise, no over-fit), theta vs truth 0.0007 (noise filtered), pde_rms 0.013 (was 0.23);
    zb still improving at the end (3.0 -> 1.8 cm in L-BFGS). Remaining gap ~ sqrt(clean floor^2 + statistical^2):
    sqrt(1.5^2 + 0.78^2) = 1.7. The clean floor fell to 0.8 cm with 2x training (v3_e2_fixed_long) -> v9_long_noise005/010
    (stage2 60k, L-BFGS 8k). Target: within ~1.5x of the referee (noise 0.005 <= 1.2 cm, 0.01 <= 1.8 cm).
26. Deep-init failure (v3_e2_fixed_init_deep, clean, constant 1.0 m): at the start of the joint stage L_pde jumps 0.02 -> 4.4
    (residual ~2 sigma_pde) and zb moves AWAY from the truth (bias +0.35 -> +6 m) while the data misfit falls 0.0067 -> 0.0018:
    the network fits the data by bending physics and zb follows the frozen-network gradient (in xi coordinates a deeper zb
    moves the 5 cm nodes toward xi = 0, where the departure N is steep, so the partial derivative can point the wrong way).
    Suspect: lr_zb 1e-2 vs network lr 1e-3 -> zb moves up to ~0.7 cm per step, faster than the network can re-solve the
    physics (two-time-scale rule: inner solver faster than outer variable). Runs v10 (lr_zb 1e-3 -> 1e-5):
    v10_deep_clean_lrzb3, v10_deep_noise005_lrzb3, v10_hydro_noise005_lrzb3 (does slower zb hurt the normal start?).
27. v10 (lr_zb 1e-3 -> 1e-5, otherwise v8): normal start noise 0.005 -> 1.11 cm (vs 1.69 at lr_zb 1e-2); deep start (1.0 m)
    still diverges (clean 9.8 m, noise 0.005 9.7 m RMSE days 0-25). Joint stage, deep start: L_pde 0.02 -> 5 within 500 steps
    while zb barely moves (bias 0.347 -> 0.333): stage 1 at 1.0 m leaves theta misfit 0.049 (L_data ~96 vs L_pde 0.02), so
    the network bends physics first; zb then drifts.
    scripts/sensitivity_check.py (frozen-network d theta_5/d c vs reference-solver FD, output/compare/sens_{A,B,C}.json):
    sign hypothesis REFUTED - all column means negative (deeper = drier) in every state; column cosines 0.43-0.99;
    deep stage-1 state: GN step on J_net moves zb -0.33 m (truth is -0.35 m away; J_ref linearisation -0.81 m).
    J_net over-states the last knots 6-250x (frozen N = instant hydrostatic re-equilibration; the real surface lags).
    => the deep-start failure is the ORDER of moves (network bends physics before zb moves), not a wrong gradient.
28. v9 LONG (fixed estimator, stage2 60k + L-BFGS 8k, lr_zb 1e-2): noise 0.005 -> 0.86 cm days 0-25 (0.96 all; referee 0.78 /
    0.82; posterior sigma 0.76); noise 0.01 -> 1.31 cm (2.2 all; referee 1.21 / 1.48). PINN within ~10% of the referee on
    days 0-25 at both noise levels. Data misfit = noise (0.0048 / 0.0098); theta vs truth 0.0007 / 0.0010.
29. zb pre-fit (train.zb_prefit, default off): after stage 1, damped GN steps on the 9 coefficients with the frozen-network
    Jacobian, each followed by a physics-only re-solve (1000 Adam steps); kept only if the re-solved data misfit falls.
    Smoke (deep start, tiny budgets): zb RMSE 0.41 -> 0.28 -> 0.19 in two accepted steps. Runs v12 (lr_zb 1e-3):
    deep 1.0 m clean / noise 0.005, deep 1.5 m noise 0.005, hydrostatic start noise 0.005 (must not hurt: v10 1.11 cm).
30. v10 (lr_zb 1e-3, standard schedule): noise 0.01 1.67 cm (lr_zb 1e-2: 2.26), clean 1.19 (1.37), noise 0.005 seed 1 1.68
    (seed 0 1.11) -> lr_zb 1e-3 no worse anywhere. v11 long + lr_zb 1e-3, noise 0.005: 0.89 cm (v9 0.86).
    v12 PRE-FIT (lr_zb 1e-3, standard schedule): deep 1.0 m clean 1.64 cm (was 9.8 m), deep 1.0 m noise 0.005 0.97 cm
    (was 9.7 m), deep 1.5 m noise 0.005 1.37 cm, hydrostatic start + pre-fit 1.66 cm (no pre-fit 1.11 / 1.68 over seeds:
    within seed spread). Days 26-30 (last knot span, weakly identifiable) still 2-5 cm in some runs (referee 1-3 cm).
    RECIPE = configs/e3_inverse_synth_recipe.yaml (fixed estimator + lr_zb 1e-3 + pre-fit + long schedule).
    Confirmation batch: recipe_noise005 (seeds 0, 1), recipe_noise010, recipe_clean, recipe_deep_noise005, recipe_deep_noise010.
    Acceptance: days 0-25 zb RMSE <= 1.5x referee (noise 0.005 <= 1.2 cm, 0.01 <= 1.8 cm), clean <= 1 cm, deep start same.
31. RECIPE BATCH (configs/e3_inverse_synth_recipe.yaml; output/compare/recipe_vs_baseline.{png,csv}); zb RMSE days 0-25 / all:
    | case                  | recipe        | referee       | acceptance (days 0-25) |
    | noise 0.005           | 1.26 / 4.16   | 0.78 / 0.82   | <= 1.2  FAIL (marginal) |
    | noise 0.005, seed 1   | 0.98 / 4.00   | 0.78 / 0.82   | PASS                    |
    | noise 0.01            | 1.51 / 3.85   | 1.21 / 1.48   | <= 1.8  PASS            |
    | clean                 | 0.95 / 3.51   | 0.00 / 0.00   | <= 1.0  PASS            |
    | deep 1.0 m, 0.005     | 1.62 / 1.65   | 0.78 / 0.82   | FAIL                    |
    | deep 1.0 m, 0.01      | 1.41 / 1.65   | 1.21 / 1.48   | PASS                    |
    Days 0-22 match the referee curve almost exactly. TWO DEFECTS:
    a) END SPAN (days 25-30), hydrostatic start only: +11 cm at day 27, -20 cm at day 30 (end-span RMSE ~10 cm from the
       pre-fit onwards: 7 -> 11 cm during the pre-fit, then flat through 60k Adam). Deep starts end at ~2 cm.
       Hydro and deep runs reach the SAME joint loss (noise 0.005: 0.947 vs 0.948) with very different end spans ->
       at sigma_pde = auto the joint objective is flat along the end-span direction (network slack absorbs it).
       The PDE term alone does prefer the correct end (L_pde 0.0289 vs 0.0263), the data term the wrong one (0.916 vs
       0.919): a tighter sigma_pde (16-64x weight) would make the PDE preference decisive.
    b) L-BFGS: days 0-25 error first DROPS to the referee level (deep 0.78 cm at it 3.4k, noise 0.005 1.11, noise 0.01
       1.25, clean 0.94) then drifts back up while L_flux falls 3x and L_data falls slightly (noise absorbed; the network
       takes ~60 of the 720 data dof: misfit 0.00479 vs 0.005). Clean: held-out pde_rms 0.0055 -> 0.0099 while the
       fixed-set L_pde falls (collocation over-fit). The joint minimum itself is biased; more optimisation goes there.
    Both point at the same thing: sigma_pde = signal scale (0.08) is ~8x looser than the residual the network reaches
    (pde_rms 0.009-0.010) -> physics violation is cheap. Batch v13 (recipe + one change each):
      sigmas.pde = ml (ML-II: sigma_pde^2 = EMA of held-out mean R^2): v13_ml_noise005, _deep_noise005, _noise010, _clean
      sigmas.pde = 0.02 (fixed, ~4x tighter; dose response): v13_sig02_noise005
      lr_zb 1e-2 (v9 kept the end at ~1.4 cm without pre-fit): v13_lrzb2_noise005, v13_lrzb2_deep_noise005
    Watch: all-days RMSE (end span), zb lock (finding 17: clean 'ml' under-fit data 0.0047 at lr_zb 1e-2, no pre-fit).
32. v13 RESULTS (recipe + one change; zb RMSE days 0-25 / all, cm):
    sigmas.pde = ml: noise 0.005 5.5 / 6.0, deep 5.1 / 5.4, noise 0.01 7.4 / 7.2, clean 6.8 / 6.9 -> LOCK (sigma_pde
      collapses to 0.002-0.003, data under-fit 0.0059 / 0.0125 / 0.0050 clean). ML-II feedback (tighter sigma -> smaller
      residual -> tighter) runs to the network floor. Same as finding 17, pre-fit or not. Do not use.
    sigmas.pde = 0.02 (~auto/4): 1.19 / 3.97. No L-BFGS drift (1.59 -> 1.19), misfit 0.00495 (no noise absorption),
      pde_rms 0.0024; BUT the end span is still wrong (max 22 cm): it is created by the pre-fit from the hydrostatic
      start and the tighter physics locks it in.
    lr_zb 1e-2: hydro 1.20 / 1.94, deep 1.04 / 1.90 (end span max ~10 cm).
    => The end-span defect comes from the hydrostatic INITIAL SHAPE (pointwise inversion ignores the lag; deep constant
       starts always end well). New options (tests/test_init_morozov.py, 3 tests PASS, job after 15415848):
       zb_init.kind = hydrostatic_flat (constant at the median hydrostatic depth);
       sigmas.pde_factor (sigma_pde = factor * auto);
       sigmas.pde = morozov (discrepancy principle on the PDE scale: sigma_pde x (L_data_ema / (1 - n_coef/n_obs))^0.25
       every 100 joint steps, within [0.1 auto, auto]; the network may not absorb more than the 9 zb dof of noise).
    Batch v14 (recipe + flat init): factor 1 / 0.5 / 0.25 / 0.125 at noise 0.005; morozov at 0.005 / 0.01 / clean / deep;
      factor 0.25 at 0.01 / clean / deep.
33. v14 RESULTS (recipe + flat start; zb RMSE days 0-25 / all, cm; referee 0.78 / 0.82 at 0.005, 1.21 / 1.48 at 0.01):
    noise 0.005: factor 1 -> 1.09 / 2.16; 0.5 -> 0.88 / 2.12; 0.25 -> 1.30 / 1.87; 0.125 -> 1.78 / 2.29;
                 morozov (settles at sigma 0.015 = auto/5.3) -> 1.32 / 1.62; deep + morozov 1.51 / 3.52.
    noise 0.01:  morozov (sigma 0.029) 1.52 / 2.01; factor 0.25 1.93 / 1.94.   clean: factor 0.25 0.62 / 1.28 (best clean);
                 morozov clean -> floor 0.1 auto (no noise to discriminate) 1.89 / 3.25.
    ERROR BY DAY vs the referee's local posterior sigma (scripts/error_by_day.py; output/compare/error_curves_noise005_v14.png):
      referee posterior sigma, noise 0.005: days 0-5 0.92, 5-10 0.82, 10-20 0.23-0.29, 22.5-25 1.34, 25-27.5 2.08,
      27.5-30 5.46, AT DAY 30 11.7 cm (0.01: 18.6 cm). => the window end is NOT identifiable even for the exact solver;
      the referee's 1.8 cm at day 30 is luck. PINN end errors of 4-12 cm there are within ~1-2 sigma. The real end defect
      was only the hydrostatic-SHAPE start + pre-fit (days 25-27.5: 9-10 cm = 4.5 sigma); flat starts: 1.2-3.5 cm.
    DAYS 0-12: every PINN error curve has the SAME shape as the referee's (-1.8 cm day 2, +1.2 cm day 7: the noise
      projected on the weakest, deep-WT direction), AMPLIFIED: v9 (no pre-fit) 1.0x; pre-fit + auto 1.4x, x0.5 1.25x,
      x0.25 1.9x, morozov 2.0x, x0.125 2.6x. v9 (hydro start, auto, lr_zb 1e-2, NO pre-fit) is the only run within
      ~1.3 posterior sigma in every segment at both noise levels (0.005: 1.22 1.12 0.27 0.14 0.63 1.18 0.98 1.61;
      0.01: 2.20 1.83 0.58 0.17 0.12 0.29 1.34 6.29). Same joint settings WITH pre-fit (v13_lrzb2): days 0-5 2.26.
      Hypothesis: the pre-fit's slow tail (GN on a half-trained network) sets the weak direction to fit the network's
      forward error; the joint stage cannot undo it (tighter physics -> stiffer). Pre-fit is needed only for poor starts.
    Batch v15 (v9 joint settings: auto sigma, lr_zb 1e-2 -> 1e-4, long): hydro no pre-fit seed 1 (0.005, 0.01); flat no
      pre-fit (0.005, 0.01); coarse pre-fit train.zb_prefit.stop_ratio = 0.5 (continue only while each GN step halves the
      misfit): flat, hydro, deep (0.005), deep (0.01).
34. USER (2026-11-10): "make the PINN work ... legit, not ad hoc modifications ... you can change the structure, the
    weighting, the network, how terms are handled, but it must stay a neural network (not inverting the numerical
    solver)." -> stop knob sweeps on the joint PINN. Plan approved: PARAMETRIC (OPERATOR) PINN
    (~/.claude/plans/piped-roaming-cocoa.md; design reviewed by an independent Plan agent).
    Root cause addressed: in the joint PINN the data reach the physics representation. New design:
      - network h~(xi, t; c) learns the solution operator c -> h for this window's forcing from PDE + flux ONLY
        (no theta data in training, no data-vs-physics weight);
      - path features linear in c and causal: f, dt*fdot, exponential memories g_tau = E_tau(t).c (tau 0.25/1/4/16 d,
        dE/dt = (B - E)/tau, E(0) = B(0)); Hermite table -> C1 in t (clamped lags would kink dh/dt);
      - inversion: LM over the 9 coefficients through the network with the exact jacfwd Jacobian (total derivative,
        includes the learned lag; the frozen-network Jacobian of the joint PINN was the instant hydrostatic one);
      - ensemble of members -> surrogate covariance in the posterior (no solver in the method; solver = scoring only);
      - accuracy stage = physics-only refinement around the current estimate (local_center.npz from the inversion).
    Code: spline.py (per-sample coef, deval_torch, filtered_basis_np, CausalPathFeatures), model.py (HeadNet n_cond /
    sqrt_t, InversePINN features + c args; defaults unchanged), physics.py (optional c), src/zbinv/parametric.py,
    scripts/param_train.py / param_invert.py / param_validate.py, configs/p1_param_synth_v3.yaml.
    G0 tests (tests/test_parametric.py, 8 new): per-sample == shared c to 1e-13; hard constraints per sample; a saved
    joint run (v9) reloads with identical theta / zb (1e-12); memory table vs exponential integrator 1e-6 and its
    t-slope vs the ODE; causality (d phi / d c_k = 0 before B_k's support); Landau residual vs physical finite
    differences (1e-4 rel); jacfwd d theta / d c vs FD (1e-7). Full suite 28/28 PASS. CPU smoke of train -> invert ->
    refine -> validate OK (two starts converge to the same c, spread 4e-9 m).
    G1 (feasibility, local distribution around the TRUE path, diagnostic only): p1_g1_noise005 (q_obs with 5% noise,
    for the 0.005 / 0.01 data), p1_g1_clean (q_true). Go if the projected zb bias <= 0.5 posterior sigma on days 0-25
    and LM zb RMSE days 0-25 <= 1.1x referee.
35. v15 = FINAL JOINT-PINN BASELINE (v9 joint settings: auto sigma_pde, lr_zb 1e-2 -> 1e-4, 60k + 8k; zb RMSE days 0-25 / all, cm):
    noise 0.005: hydro no-prefit seed 1 1.07 / 1.07 (v9 seed 0: 0.86 / 0.96); flat no-prefit 1.15 / 1.49;
                 coarse pre-fit (stop_ratio 0.5): flat 0.95 / 2.02, hydro 0.90 / 1.81, deep 1.0 m 1.02 / 2.09.
    noise 0.01:  hydro no-prefit seed 1 1.59 / 1.58 (v9 1.31 / 2.22); flat no-prefit 1.43 / 1.69; deep + p50 1.65 / 3.33.
    -> all 8 meet the days 0-25 acceptance (<= 1.2 / <= 1.8); remaining weaknesses: early window 1.5-2x the referee
       error (weak deep-WT direction), end span up to 10 cm (within the 2-sigma information limit there).
       This is the bar the parametric PINN has to beat (and the joint PINN still lets data reach the physics).
36. G1 PASS (parametric PINN, ONE member trained physics-only on a local distribution around the TRUE path, +-5 cm
    per knot; diagnostic only; 30k Adam + 15 L-BFGS blocks, ~40 min V100):
    - network theta_5cm vs reference solver: truth 1.3e-4 RMSE (single-path forward PINN was 1.8e-3), local draws
      0.7-1.5e-3, prior draws ~1e-2 (not trained there, as intended).
    - sensitivity at the truth: J_net vs J_ref column cosines 0.89-0.998, norm ratios 0.70-1.67 (joint PINN frozen
      Jacobian: end knots 6-250x).
    - projected zb bias of the surrogate error: 0.13 cm rms days 0-25; bias/sigma <= 0.43 in every segment.
    - inversion (LM through the network), zb RMSE days 0-25: noise 0.005 0.69-0.72 cm (referee 0.78), noise 0.01
      1.11 cm (referee 1.21), clean 0.17 cm (all days 0.20). By segment (0.005): 1.06 0.55 0.42 0.15 0.60 1.24 | 1.45
      11.3 vs referee 1.21 0.96 0.31 0.09 0.49 0.91 | 0.45 1.37 -> NO amplification on the weak deep-WT direction
      (joint PINN 1.4-2.6x). Last 2.5 days 11 cm, day 30 21 cm: ~2 posterior sigma (unidentifiable end; different
      starts reach costs within 1e-3 with different end values).
    - starts outside the trained region fail (clean, 1.0 m: cost 37, not converged) -> expected for a local-only
      network; the wide stage exists for this.
    G2/G3 launched (hpc/param_pipeline.sh, no truth anywhere): wide 4 members (prior, 40k Adam + 10 L-BFGS blocks) ->
    inversion -> 2 rounds of local physics-only refinement (frac_local 0.7, 15k Adam lr 3e-4 + 15 blocks) + inversion.
    Pipelines: p1_n005 (wide p1_wide_q005), p1_n010 (shares p1_wide_q005), p1_clean (wide p1_wide_clean).
37. G2/G3 first pass (p1_*: wide 4 members -> LM -> local refinement, cov_local = 9 Sigma_post + dc dc^T, 70% local,
    15k Adam lr 3e-4 + 15 L-BFGS blocks per round). zb RMSE days 0-25 (cm):
                  wide inv0   round 1   round 2
      noise 0.005   5.80        2.97      2.68
      noise 0.01    5.87        3.25      4.01
      clean         5.42        2.42      2.03   (theta misfit 2.7e-4 while zb is 2 cm off -> surrogate bias)
    WIDE STAGE = G2 PASS: all 4 starts (hydrostatic, flat, 1.0 m, 0.3 m) land in the same basin (spread 0.4-0.5 cm
    days 0-25); wide-member theta error at the truth 6.7e-3 -> the wide network only finds the basin (as planned).
    REFINEMENT STALLED: the local region was sized by the DATA posterior (3 sigma_post ~ 0.6 cm mid-window) while the
    estimate still carried 2-3 cm of surrogate bias -> the truth lay outside the region being made accurate. The
    ensemble does not see that bias (members share it). G1 worked with a +-5 cm per knot region.
    FIX (p2_*): cov_local = 9 Sigma_post + dc dc^T + (5 cm per knot)^2 floor (= the radius G1 showed a physics-only
    network resolves accurately; trust radius, not a loss weight), frac_local 0.9, G1-strength rounds (25k Adam
    lr 1e-3 + 15 blocks, warm start), LM also started from the previous estimate. Reuses the wide members and
    re-inverts them (p2_*_inv0). Runs: p2_n005, p2_n010, p2_clean, rounds 1-3 (hpc/param_restart.sh).
    Driver hiccup: login node did not see a just-written file -> param_rounds.sh retries file checks for 2 min.
38. p2 (floored trust region, 5 cm/knot) rounds 1-2. zb RMSE days 0-25 (cm): n005 5.80 -> 2.44 -> 2.09; n010 5.87 -> 3.29;
    clean 5.42 -> 1.96. ALL starts reach the same zb (spread 0.00 cm) = stability achieved; accuracy stalls:
    n005 by segment (0-5, 5-10, 10-15, 15-20, 20-22.5, 22.5-25): inv1 3.83 3.74 0.93 0.22 0.35 0.73; inv2 2.43 3.50 0.24
    0.09 0.70 2.54 (referee sigma 0.92 0.82 0.23 0.29 0.77 1.34). Mid-window (10-22.5) is at the referee level already.
    DIAGNOSIS (scripts/param_validate.py --jac-at <centre>): the network VALUE at its own centre is right (1.1e-4 r1,
    3.2e-4 r2; projected zb bias 0.13 / 0.64 cm) but its JACOBIAN is wrong along the weak directions of J_ref:
    relative error by right singular vector (s = 0.74 ... 0.0023): r1 0.03 0.14 0.13 0.16 0.32 0.53 0.89 1.33 2.54;
    r2 0.09 0.20 0.14 0.18 0.18 0.33 1.01 1.26 2.00. The refinement loop's error map is M = pinv(J_net)(J_net - J_ref)
    (next error = M x current centre error): 2-norm 3.5 (r1), 6.5 (r2) -> the weak directions are not pulled in.
    Cause: the absolute J error (~0.01) = value error over the training ball (1-2e-3) / ball radius (~0.1 coef units):
    a sampled ball alone cannot give a better Jacobian.
    FIX (p3t): sensitivity (centre) term = delta-method form of the local objective at the centre c0,
    E_c[r^2] ~= r(c0)^2 + grad_c r^T Sigma grad_c r (PDE and flux; same sigmas, same Sigma as the trust region; no new
    weight; c-direction gradient-enhanced PINN). Zero only if d h / d c obeys the linearised Richards (sensitivity)
    equations at c0 -> J_net = J_ref there -> each round becomes a true Gauss-Newton step. train.n_tan (default 0 = off),
    tests: centre term = per-point brute force to 1e-10; off by default; all 10 parametric tests pass.
    Smoke (A100): 3.8 GB peak, ~2.5x cost per Adam step. Launched p3t_n005 / p3t_clean rounds 2-4 from the SAME
    round-1 state as p2 (members p2_*_r1, centre p2_*_inv1); p2 round 3 continues as the control.
    G4 prep: seed-1 referees output/classical/v3s1_noise00{5,10}; seed-1 wide members p1_wide_q005s1_w* (p2s1_n005 inv0).
39. p3t (centre / sensitivity term) round 2 vs p2 round 2: n005 2.00 vs 2.09 cm, clean 1.06 vs 1.16 cm; weak-direction J
    error UNCHANGED (0.99 1.25 2.12 vs 1.01 1.26 2.00); centre value error 2.6e-4 vs 3.2e-4; ~3x GPU cost per round.
    -> p3t stopped (code kept, off by default). p2 control continued: n005 days 0-25 inv3 1.31, inv4 1.35, inv5 1.02 cm:
    no convergence, the estimate WANDERS by ~1 cm per round along weak directions (inv3->inv4: days 0-10 better, 20-25
    worse). Not a usable stopping state.
    Shared-bias diagnosis (p2 r4 at centre inv3): value error at centre 2.2e-4 (days 25-27.5: 5.9e-4), ensemble theta
    sd only 9e-5 -> the error is SYSTEMATIC (same in all seeds); J ratio (net/ref) for knots 0-2 = 0.84 0.86 0.75 in
    every round (cos 0.99): the network under-responds to the path by 15-25% (also explains the 2.5e-3 theta error at
    the truth 2 cm from the centre). Train ~ val loss ~0.01 floor; Adam lr 1e-3 from a warm start blows the val loss up
    (0.03 -> 3.1) before it recovers (the warm start buys nothing).
    Twin reference is grid-converged (401/1601 nodes: <1e-6 theta), so the floor is the network's, not the solver's.
    Hypothesis: input scaling. In a local ball the path features vary by ~5% of their range; the network learns the
    sensitivity magnitude slowly. FIX candidate (dxF): model.local_zoom -> features [phi(c), phi(c - c0)/rho], rho =
    rms local sd (same function class, input normalisation; trains from scratch). Tests: 12/12 parametric pass.
    Diagnostic batch, all from scratch at the SAME centre (p2_n005_inv3), 2 members each, scored by param_validate
    --jac-at: dxA control, dxB 10 memory taus, dxC width 256, dxD frac_local 1.0, dxE single path (attainable accuracy
    at one c), dxF local zoom. Baseline (p2 r4, warm start): value 2.0-2.2e-4, J rel err 0.05 0.18 0.14 0.26 0.29 0.30
    0.40 0.78 0.30, projected bias 0.40-0.44 cm days 0-25.
40. Diagnostic batch results (all at centre p2_n005_inv3; theta error at centre / J ratio knots 0-2 / projected bias):
    FROM SCRATCH (28k steps): dxA control 5.9e-4 / .67 .76 .69 / 1.09 cm; dxB 10 taus 3.9e-4 / .72 .78 .68 / 0.57;
    dxC width 256 3.0e-4 / .76 .80 .61 / 0.47; dxD frac_local 1 4.6e-4 / .71 .76 .59 / 0.80; dxF zoom 5.3e-4 / .70 .78
    .73 / 1.04. WARM (A/B vs p2 r4): dxG zoom 2.1e-4 / .85 .88 .77 / 0.42 vs baseline 2.2e-4 / .84 .86 .75 / 0.44.
    -> warm start DOES matter (from scratch val loss 0.12 vs 0.012); zoom hypothesis REJECTED; width and memory
    taus help from scratch but none removes the operator's ~2e-4 floor or the J under-response.
    dxE = ordinary forward PINN at ONE path (sampling point, from scratch, same budget): theta error 3.6e-5 at that
    path (6x better than any operator net), projected bias 0.03 cm.
    Value-corrected estimator analysis (param_validate, new line): with an accurate theta at the estimate and the
    operator's Jacobian for the step, the fixed point solves J_op^T (y - theta(c)) = 0; noise maps through
    (J_op^T J_ref)^-1 J_op^T; column scaling cancels. Its zb sd / exact sd = 1.01-1.07 by segment (p2 r4 and r5).
    NEW DESIGN (value-corrected refinement, two physics-only PINNs, no solver): forward PINN at c_k (sampling.mode=
    point) + operator ensemble (Jacobian, model between iterates); param_invert --fwd inverts theta_op(c) +
    [theta_fwd(c_k) - theta_op(c_k)]; stop when the zb move per 2.5-day segment < 0.2 posterior sd (hpc/dc_loop.sh).
    First iteration (from p2_n005_inv3, operator p2_n005_r4, forward dxE): days 0-25 1.31 -> 0.96 cm, all 5 starts
    identical (cost 0.98590). Launched dc_n005 (it 2..), dc_n010 and dc_clean (from p2 inv2, operator p2 r2).
    Seed 1: p2 operator round 1 running (p2s1_n005, p2s1_n010), then the dc loop.
41. Value-corrected loop (dc), zb RMSE days 0-25 vs truth [cm] (target = 1.5 x referee):
      n005 (1.2):    inv3 1.31 -> it1 0.96 -> it2 1.16 -> it3 1.03   (step/sigma 0.43, 0.33: not < 0.2)
      n010 (1.8):    inv2 .... -> it1 1.77 -> it2 2.14               (step/sigma 0.64)
      clean (0.5):   inv2 .... -> it1 0.36 -> it2 0.35               (step/sigma 0.72)
      s1_n005 (1.31): it1 2.54 (step/sigma 5.1);  s1_n010 (3.23): it1 2.75 (step/sigma 2.5)
    Distance to the REFEREE ESTIMATE (same data, exact optimum; scripts/dist_to_referee.py, scoring only), days 0-25:
      n005 0.65 -> 0.25 -> 0.50; n010 2.33 -> 0.67 -> 1.06; clean 1.16 -> 0.36 -> 0.35; s1_n005 2.83 -> 2.31;
      s1_n010 1.99 -> 0.67.
    -> the loop gets within 0.3-1 cm of the exact optimum, then JITTERS (does not reach the 0.2 sigma stop). Two causes:
      (a) forward-PINN value error ~1e-4 in theta (members differ: val_pde 0.0086 vs 0.0189, theta sd 6e-5..1.1e-4)
          projects to 0.15-0.35 cm in zb (the clean-case 0.35 cm is exactly this);
      (b) the step uses the operator Jacobian: fixed point J_op^T (y - theta(c)) = 0 is up to ~0.4 sigma from the exact
          optimum (scales with the noise) and the iteration overshoots (predicted spectral radius 0.78).
    Forward PINN at one path trains in ~22 min (25k Adam + 15 L-BFGS blocks; L-BFGS gains little: val 0.0118 -> 0.0101).
42. Coarse-to-fine multi-start (param_invert --coarse <wide-stage members>): every start except 'previous' is first
    inverted through the WIDE-stage ensemble (right basin), then through the refined, corrected model. n010 had a
    60 cm start spread (far starts leave the region where the refined operator is trained); with --coarse all 5 starts
    reach the identical optimum in n005, n010 and clean (spread 0). Adopted (dc_loop COARSE=...).
43. Multi-secant (Broyden least-change) slope correction from earlier forward PINNs (param_invert --fwd-prev,
    parametric.secant_correction, dc_loop SEC=1): REJECTED. it2 vs truth: n005 1.40 (plain 1.16), n010 2.76 (plain
    2.14), clean 0.37 (0.35). Near convergence the offset differences are at the forward-PINN noise level
    (|dc| = 0.19, d-offset 1.9e-4), and in the raw coefficient metric the secant is dominated by unidentified end
    coefficients (n010 |dc| = 3.8). Code kept, off by default.
44. First-order (exact Gauss-Newton) correction: theta_model(c) = theta_op(c) + offset + (J_fwd(c_k) - J_op(c_k))
    (c - c_k) (param_invert --fwd-jac; invert_lm lin=; dc_loop FJ=1). Needs a forward PINN that also carries
    d h / d c at c_k: sampling.mode=point with train.n_tan > 0 (sensitivity equations at c_k). Tests: value and
    Jacobian of the corrected model equal the forward net's at c_k; 15/15 parametric tests pass (job 15434772).
    Test in progress: sx_m0/m1 at dc_n005_it2 (accuracy of the sensitivity-PINN Jacobian vs solver FD, scoring only).
    dc_loop also gained WARM=1 / INIT= (member k continues from member k of the previous round: the path moved little).
45. Tests at ONE path (centre dc_n005_it2), scored by param_validate --jac-at (scoring only):
    (a) baseline = plain forward members of the loop (dc_n005_f3_m0/m1, from scratch, 25k Adam + 15 L-BFGS):
        ensemble theta RMSE 1.05e-4 (members 5.5e-5 with val_pde 0.0062, 1.79e-4 with val_pde 0.0208) -> projected zb
        bias 0.16 cm days 0-25. Member quality differs 3x and follows the held-out physics loss -> optimisation-
        limited, seed-dependent.
    (b) sx = forward + sensitivity PINN with the 6 SHARED path channels (train.n_tan=4096, from scratch, ~70 min on
        V100): NOT good enough for a Gauss-Newton step. Value: theta RMSE 3.64e-4 (3.5x worse than plain; the
        sensitivity term slows the value fit), projected bias 0.66 cm days 0-25 (up to 1.46 sigma). Jacobian vs
        solver FD: column cos 0.92-1.00, ratio 0.50-0.99, rel error by direction (strong -> weak) 0.11 0.31 0.34 0.46
        0.41 0.63 0.76 1.37 0.82; predicted contraction 0.76 (operator: 0.78) -> no gain.
        Reading: with 6 shared channels d h / d c_j = sum_i a_i(xi,t) psi_ij(t) (a 4-exponential kernel common to
        all coefficients); late in the window 7-9 coefficients are active but only 6 free functions exist.
    (c) FIX under test (sxp): per-coefficient causal channels chi_j(t; c) = E_taumax,j(t) c_j (model.percoef_memory,
        off by default; spline.CausalPathFeatures percoef=). Each sensitivity gets its own first-layer weight
        column; still causal. load_state_zoom_aware extends a saved first layer by zero columns, so a plain forward
        member can be continued as a per-coefficient member with the identical function. 18/18 parametric tests
        pass (job 15435645).
    (d) sw = warm continuation of the baseline members at lr 2e-4 (dc_loop WARM=1, WLR): sw_m1 held-out val_pde
        0.0208 -> 0.0084 (bump to 0.056 at step 1500, then monotone); no blow-up at this lr. Validation pending.
46. (2026-11-10 13:00) Status of the zeroth-order (value-corrected) loops with COLD forward members, and what limits them:
    zb RMSE days 0-25 vs truth [cm] (step/sigma max); distance to the referee estimate days 0-25 [cm] (scoring only):
      n005     it1 0.96  it2 1.16 (0.43)  it3 1.03 (0.33)  it4 1.16 (0.28)  it5 0.96 (0.46)   dist 0.25 0.50 0.31 0.46
      n010     it1 1.77  it2 2.14 (0.64)  it3 1.95 (0.36)                                     dist 0.67 1.06 0.84
      clean    it1 0.36  it2 0.35 (0.72)  it3 0.21 (0.41)  it4 0.30 (0.37)                    dist 0.36 0.35 0.21
      s1_n005  it1 2.54 (5.12)  it2 2.18 (1.59)  it3 1.74 (1.09)                              dist 2.31 1.76
      s1_n010  it1 2.75 (2.51)  it2 2.73 (0.75)  it3 2.59 (0.36)                              dist 0.67 0.61
    Targets 1.2 / 1.8 / 0.5 / 1.31 / 3.23 cm. NONE has reached the stop rule (step < 0.2 sigma): n005 and clean hover at
    0.3-0.45 sigma (jitter), s1_n005 is still contracting (distance ratio ~0.76 per round, poor operator at its
    centre: fwd offset rms 2-3.3e-3).
    (a) sw (warm continuation of f3 members at lr 2e-4, validation job 15436842, centre dc_n005_it2): ensemble theta
        RMSE 1.05e-4 -> 4.89e-5 (max 1.51e-4); projected zb bias days 0-25 0.16 -> 0.08 cm (bias/sigma <= 0.14);
        members' val_pde 0.0062 -> 0.0036 and 0.0208 -> 0.0084. CONFIRMED: continuing a member halves the value error
        and makes the error of consecutive rounds correlated (same member) instead of independent.
        -> all five loops restarted with WARM=1 (+ COARSE = wide-stage members, so far starts are tested) at their
        next iteration boundary (/tmp/switch_warm.sh; s1_n005 after it3, s1_n010 after it3, n005 after it5, clean
        after it4, n010 after it4).
    (b) sxp (per-coefficient channels, from scratch) tracks sx almost exactly through step 10000 (val_pde_c 0.123 vs
        0.127 at 7500; 0.101 vs 0.106 at 10000): the sensitivity fit is limited by optimisation / accuracy, not by the
        6-channel representation.
    (c) Accuracy model for a PINN Jacobian (working hypothesis, matches (45b) and the operator numbers): a 1 sigma_post
        move of c changes theta_5cm by ~1.9e-4 rms per observation (= sigma_theta / sqrt(N)); the relative J error in a
        whitened direction u is ~ eps / (kappa_u x 1.9e-4), eps = absolute accuracy of the net's theta / tangent,
        kappa_u = tangent scale in posterior-sigma units (~3 for weak directions, 5-20 for strong ones through the
        5 cm floor). sx: eps 3.6e-4 -> 0.1-0.3 strong, 0.6-1.4 weak (observed). Operator: eps 2.2e-4 -> 0.05-0.4.
        Weak directions are intrinsically hard: the surface response is a small tail of a field-scale interior change.
    (d) Tests running (centre dc_n005_it2, lr 2e-4 continuation, n_tan 4096): sx2_m0 (sx_m0 continued), so_m0/so_m1
        (operator members p2_n005_r4_m0/m1 continued as forward + sensitivity members with per-coefficient channels).
        Decision rule: J rel error <~ 0.25 in all but the last 1-2 directions and predicted contraction <~ 0.4 ->
        FJ=1 loops; otherwise finite-difference Jacobian from warm-started forward PINNs at c_k +- 3 sigma_post u_i.
47. (2026-11-10 13:20) R4 = finite-difference slope correction, implemented and under test; sensitivity-PINN (sx) read.
    (a) sx validations at centre dc_n005_it2 (jobs 15437121/22/29): value 3.6e-4 / 1.4e-4 / ens 2.4e-4; rel J error by
        direction (strong->weak) ens 0.09 0.30 0.31 0.44 0.40 0.65 0.81 1.44 0.93 (operator r4: 0.05 0.18 0.14 0.26
        0.29 0.30 0.40 0.78 0.30); predicted contraction 0.56 (operator 0.78). A sensitivity-trained forward PINN is
        NOT a better Jacobian than the operator in 7 of 9 directions -> FJ=1 is not the route (so/sxp still to read).
    (b) R4 design: 18 paths c0 +- 3 sd_i u_i, (sd_i^2, u_i) = eigenpairs of cov_post in local_center.npz (3 = radius of
        the local refinement cloud, param_invert --inflate; no new number). One ordinary forward PINN per path
        (sampling.mode=point), all continued from the SAME parent forward PINN at the centre (common-mode error cancels
        in the differences). dJ = slope of the least-squares affine fit of o(c) = theta_fwd(c) - theta_op(c) over the
        design + the current path (= central difference for the symmetric design); inversion model
        theta_op(c) + o(c_k) + dJ (c - c_k) (existing lin=(dJ, ct) of invert_lm): the operator keeps the curvature.
        Chord iteration: the FD set is kept between rounds (dc_loop FD="<dirs>").
        Code: parametric.affine_slope; param_invert --fwd-fd (writes fwd_slope.npz, fd_* fields in summary.json);
        scripts/fd_design.py; hpc/fd_set.sh; param_validate --jac-corr (scores J_op + dJ against the solver J);
        tests/test_parametric.py::test_affine_slope_is_central_difference_on_fd_design. Tests job 15437668: 19 PASS.
    (c) First set fd_n005 (centre dc_n005_it2, parent sw_m0, launched 13:10): |dc| by direction 7.00 (c[8]) 0.94 (c[0])
        0.35 0.14 0.12 0.065 0.038 0.035 0.021. WATCH: directions 0 and 1 are +-7 m in c[8] and +-0.9 m in c[0]
        (softplus saturates / WT far outside the admissible range) -> the secant slope there may not be the local
        slope; the even part of o along each pair (curvature diagnostic) decides whether to shrink those two.
    (d) Ops: a slow controller left `sbatch` of the dc_clean WARM loop hung for 14 min (no f5 job ever submitted).
        dc_loop.sh / fd_set.sh now submit through submit(): timeout 180 + retry + cancel-by-name of a possibly accepted
        orphan. dc_clean restarted 13:15 (it 5..12). The other four loops still run the old script (un-timed sbatch).
    (e) 47c WATCH resolved (fd_design.py --dry-run --config, job 15438078): in zb terms all 18 paths are moderate and
        inside the prior range [0.2, 1.5] m. Max |zb - centre| by direction (+/-): 0.387/0.247 m at day 30 (the last
        coefficient has weight 1/6 at t = T, hence the large c[8] move); 0.022 at day 2.1; 0.067/0.064 at day 24.4;
        0.033; 0.038; 0.014; 0.007; 0.009; 0.005. No shrink needed.
    (f) Seed 1: dc_s1_n005 it4 1.39 cm (step/sigma 1.07, offset rms 3.6e-3, one start at a local minimum, cost 2.40
        vs 0.945) -> the r1 operator is too coarse there. Operator round p2s1b_n005 r2 at the dc centre launched
        13:22 (symlinks p2s1b_n005_inv1 -> dc_s1_n005_it4, p2s1b_n005_r1_m* -> p2s1_n005_r1_m*).
    (g) sxp (per-coefficient sensitivity channels) validations (jobs 15438310-12): rel J error by direction ens
        0.09 0.33 0.27 0.42 0.41 0.71 0.60 1.19 0.70, contraction 0.68 -> no better than sx; FJ=1 closed.
48. (2026-11-10 13:50) Far start trapped (start spread 1.1 cm seed 1; 60 cm n010 without --coarse): cause + fix.
    Diagnostic (job 15438309, wide members q005s1 only): from constant:1.0 the first LM step is |4.5| in coefficient
    space and ends at c[8] = +8.80 (zb(30) = 1.76 m) with cost 1.36 vs 0.984; the other 8 coefficients agree with the
    good solution. Marquardt damping (lambda diag(JtJ)) does not bound the step of a weakly identified coefficient,
    and the wide members were trained only on the prior, whose support is exactly the box c_k in
    [inv_softplus(0.2 - zb_min), inv_softplus(1.5 - zb_min)] = [-1.821, 1.183] (knot depths clipped to 0.2-1.5 m).
    Fix (no new number): the coarse stage is a projected LM on that box (CPrior.box; invert_lm(box=); start and every
    trial step clamped). The refined stage is unchanged (its local cloud covers c[8] outside the box).
    Verified: tests job 15438505 20 PASS; re-inversions with the box (jobs 15438494 / 15438503 / 15438523):
    s1_n005, s1_n010, n010 all five starts end at the same coefficients to 3 decimals (cost 0.94476 / 0.94548 /
    0.98489), start spread 0. dc_n010 it5 (first loop inversion with the box): spread 0 cm.
    dc_n010 "converged" at it5 (step/sigma 0.18) at 1.81 cm vs target 1.8, referee 1.21: the fixed point of the
    value-corrected loop is biased by the operator Jacobian error, as predicted (finding 46) -> needs R4 (FD slope).
49. (2026-11-10 14:05) Loops after the box fix (WARM=1, value-corrected, operator Jacobian); ops.
    (a) zb RMSE days 0-25 [cm] (step/sigma max), all with start spread 0 cm:
        n005 it6 1.05 (0.34), fwd offset rms 1.5e-4; clean it5 0.12 (0.58); s1_n005 it5 1.23 (0.89, r1 operator,
        offset rms 3.6e-3); n010 it5 1.81 (0.18, loop ended "converged"); s1_n010 it4 2.37 (0.31, pre-box).
        Targets 1.2 / 0.5 / 1.31 / 1.8 / 3.23. RMSE is at or near target everywhere; what is missing is the
        stop rule (< 0.2 sigma) in four cases and the biased fixed point of n010.
    (b) Ops: the dc_n005 loop (old un-timed script) hung 17 min in `sbatch` for dc_n005_it6 (sacct: no such job).
        Killed, restarted at it 6 on the hardened script. dc_loop.sh now also resumes: a member of the round whose
        model.pt exists is reused ("already trained, reused"), so a restart costs no training.
        dc_s1_n005, dc_s1_n010 and fd_set fd_n005 still run old un-timed scripts (watched).
50. (2026-11-10 14:10) so = OPERATOR member continued as forward + sensitivity PINN at one path: decision rule MET.
    Test (finding 46d): p2_n005_r4_m0/m1 continued at centre dc_n005_it2 with train.n_tan=4096,
    model.percoef_memory=true (zero new columns: same function at the start), lr 2e-4, 25k Adam + 15 L-BFGS,
    ~65 min on a V100. Validation vs solver (scoring only; jobs 15438789 / 15439238 / 15439239):
                      theta RMSE   rel J error by direction (strong -> weak)            contraction  bias d0-25
      operator r4     2.2e-4       0.05 0.18 0.14 0.26 0.29 0.30 0.40 0.78 0.30         0.78         -
      sx / sxp        3.6e-4       0.09 0.30 0.31 0.44 0.40 0.65 0.81 1.44 0.93         0.76 / 0.68  0.66 cm
      sx2 (sx cont.)  1.3e-4       0.07 0.32 0.27 0.39 0.43 0.56 0.69 1.15 0.61         1.16         0.25 cm
      sw (plain fwd)  4.9e-5       (no Jacobian)                                         -            0.08 cm
      so_m0           3.7e-5       0.04 0.16 0.08 0.12 0.09 0.25 0.08 0.58 0.17         0.30         0.06 cm
      so_m1           3.5e-5       0.04 0.12 0.07 0.13 0.10 0.24 0.09 0.50 0.17         0.24         0.05 cm
      so ensemble     3.4e-5       0.04 0.14 0.07 0.13 0.10 0.25 0.08 0.53 0.17         0.27         0.05 cm
    Held-out physics loss: so_m0 val_pde 6.8e-4, val_pde_c 3.1e-3 (plain warm members 3.6e-3..8.4e-3; sxp from
    scratch val_pde_c ~0.1). Reading: the sensitivity PINN fails from scratch (optimisation), but an operator member
    already carries d h / d c from its training over the cloud; specialising it to one path with the tangent term
    gives both the best value and a Jacobian 3x better than the operator's. The two members agree (errors are
    systematic: column ratio 0.81-0.82 for c[2], 0.91-0.93 for c[0], c[1] in both -> early window).
    -> NEW LOOP "gn" = dc_loop.sh FJ=1 WARM=1 COARSE=... INIT=<trained members>, TSET + model.percoef_memory=true:
       every round is a Gauss-Newton step with the forward+sensitivity PINN's own value and Jacobian at c_k
       (param_invert --fwd-jac); no new code. Launched 14:06-14:08 (launcher /tmp/gn_launch.sh, logs/dc_gn_<case>.log):
         gn_n005    centre dc_n005_it2, round 1 = so_m0/so_m1 (symlinks gn_n005_f1_m{0,1})
         gn_n010    centre dc_n010_it5, INIT so_m0/so_m1 (same forcing q005), operator p2_n010_r2
         gn_clean   centre dc_clean_it5, INIT p2_clean_r2_m0/m1
         gn_s1_n005 centre dc_s1_n005_it5, INIT p2s1_n005_r1_m2/m0 (the two lowest held-out physics losses)
         gn_s1_n010 centre dc_s1_n010_it4, INIT p2s1_n010_r1_m3/m2 (same rule)
       dc_s1_n005 / dc_s1_n010 (value-corrected, old script) stopped to free GPUs; dc_n005, dc_clean, fd_n005 and
       the p2s1b_n005 r2 round keep running (baseline / alternative).
    (50b) gn_n005 it1 (centre dc_n005_it2 -> GN step with so_m0/so_m1): zb RMSE days 0-25 0.89 cm (referee 0.78,
       target 1.2), start spread 0, step/sigma max 0.69 (first step from an old centre), posterior sigma 0.804 cm
       (noise-only 0.793; referee 0.756 -> +6% / +5%), forward-member theta sd 1.2e-5, J rel sd between the two
       members 1-3%.
    (50c) p2s1b_n005 r2 (operator round at the dc_s1_n005_it4 centre) is WORSE than r1: val_pde 0.15-0.51 (r1
       0.015-0.106), operator-only inversion 2.02 cm, step 3.5 sigma. Not used. (Cloud built from the it4
       local_center of the dc loop; cause not investigated - the gn loops do not need it.)
51. (2026-11-10 14:35) FD slope correction (18 plain forward PINNs on a +-3 sd design): Jacobian to 1-3%; ADOPTED.
    Design fd_n005 (hpc/fd_set.sh, scripts/fd_design.py): c0 +- 3 sd_i u_i from cov_post of dc_n005_it2, every
    member continued from the forward PINN sw_m0 at the centre (errors common -> cancel in differences), 18 V100
    jobs, all trained by 14:16. param_invert --fwd-fd: dJ = affine LSQ fit of theta_fwd - theta_op over the design.
    Validation vs solver (scoring only, centre dc_n005_it2, job of fdtest_n005):
                      rel J error by direction (strong -> weak)            contraction
      operator r4     0.05 0.18 0.14 0.26 0.29 0.30 0.40 0.78 0.30         0.78
      so (GN, 50)     0.04 0.14 0.07 0.13 0.10 0.25 0.08 0.53 0.17         0.27
      FD slope        0.01 0.02 0.01 0.02 0.01 0.02 0.02 0.03 0.10         0.03
    J cos 1.000, J ratio 0.997-1.016, affine-fit residual / signal 0.17.
    One slope-corrected step from each case's last value-corrected centre (all starts identical, spread 0):
      case   operator      centre        fit resid  zb RMSE d0-25   referee   sigma (noise-only)   referee sigma
      n005   p2_n005_r4    dc_n005_it2   0.17       0.84 cm         0.78      0.773 (0.765)        0.756
      n010   p2_n005_r4    dc_n010_it4   0.16       1.22 cm         1.21      1.552 (1.548)        1.536
      n010   p2_n010_r2    dc_n010_it4   0.21       1.26 cm         1.21      1.584 (1.565)        1.536
    n005: 0.08 cm rms from the referee ESTIMATE on days 0-25 (<= 0.13 sigma_ref in every segment); gn_n005_it1
    0.16 cm, dc_n005_it6 0.34 cm, dc_n005_it2 0.50 cm (logs/cpu_15439481.out).
    n010 was "converged" at 1.81 cm with the value-corrected loop (biased fixed point, finding 47); the slope
    correction removes that bias. The design centred on the n005 estimate serves n010 too (same forcing q005).
    Operator choice for n010 without the truth: the smaller affine-fit residual (0.16 vs 0.21) = the operator whose
    training cloud is centred on the design -> p2_n005_r4.
    dc_clean (value-corrected loop) CONVERGED at it6: 0.06 cm (target 0.5), step/sigma 0.197, spread 0 - a
    zero-residual problem has no Jacobian bias.
    -> FINAL DESIGN = "fdl" loops: dc_loop.sh FD="<18 design members>" WARM=1 COARSE=... INIT=<members at centre>
       (launcher /tmp/fdl_launch.sh <case> <operator> <start run>, logs/dc_fdl_<case>.log):
         fdl_n005  start fdtest_n005,      operator p2_n005_r4, design fd_n005, INIT sw_m0/sw_m1       (14:29)
         fdl_n010  start fdtest_n010_op5,  operator p2_n005_r4, design fd_n005, INIT dc_n010_f5_m0/m1  (14:33)
       New design sets (14:29): fd_s1 (centre dc_s1_n005_it4, parent dc_s1_n005_f5_m0; serves both seed-1
       cases) and fd_clean (centre dc_clean_it5, parent dc_clean_f6_m0); watchers fdtest_s1_n005,
       fdtest_s1_n010_own / _op5, fdtest_clean, then fdl loops for those three cases.
       Stopped by hand: dc_n005 (at it7), gn_clean (redundant). gn_n005 / gn_n010 / gn_s1_* keep running as an
       independent cross-check (Gauss-Newton with the sensitivity PINN's own Jacobian).
    (51b) n010 slope tests scored vs the solver (scoring only; centre dc_n010_it4, design centred on dc_n005_it2):
                             rel J error by direction (strong -> weak)            contraction  dist to referee d0-25
      p2_n005_r4 uncorrected 0.05 0.21 0.15 0.27 0.28 0.32 0.49 0.78 0.51
      p2_n005_r4 + FD slope  0.02 0.02 0.02 0.03 0.03 0.04 0.08 0.08 0.28         0.14         0.04 cm (<= 0.03 sigma_ref)
      p2_n010_r2 uncorrected 0.09 0.23 0.15 0.20 0.16 0.32 0.93 0.87 1.87
      p2_n010_r2 + FD slope  0.02 0.02 0.02 0.05 0.08 0.06 0.12 0.07 0.53         0.22         0.17 cm
      The solver-free choice (smaller affine-fit residual) picked the better operator. A design centred on another
      case's estimate (1-2 sigma away) still gives J to 2-8% in the identified directions.
52. (2026-11-10 15:52) Gauss-Newton cross-check loops: n005 CONVERGED; 0.01-noise cases poor; posterior fix; GPUs.
    gn loops (finding 50), one or two rounds each, then stopped by hand (/tmp/gn_stop.sh) to free GPUs:
      case      it  zb RMSE d0-25  target  step/sigma  spread   sigma (noise-only)  referee sigma  offset rms
      n005      1   0.89 cm        1.2     0.69        0        0.804 (0.793)       0.756          2.4e-4
      n005      2   0.85 cm        1.2     0.10        0        0.797 (0.786)       0.756          1.8e-4   CONVERGED
      n010      1   2.45 cm        1.8     0.92        0        2.069               1.536          1.2e-3
      s1_n005   1   1.03 cm        1.31    0.47        0        1.746 (1.042)       0.786          3.6e-3
      s1_n010   1   3.81 cm        3.23    0.84        0.34 cm  -                   1.712          1.5e-3
    gn_n005_it2 by segment [cm] 1.27 1.15 0.29 0.08 0.50 0.98 0.49 1.29 vs referee sigma 0.92 0.82 0.23 0.29 0.77
    1.34 2.08 5.46 -> all within 2 sigma (referee's own error: 1.21 0.96 0.31 0.09 0.49 0.91 0.45 1.37); 0.12 cm rms
    from the referee estimate on days 0-25 (fdtest_n005: 0.08 cm; logs/cpu_15443982.out, cpu_15443983.out).
    => n005 (seed 0) meets RMSE, stop rule, start spread, sigma (+5%) and the 2-sigma segments with a network-only
       inversion. The two Jacobian routes (sensitivity PINN / finite differences) agree: 0.85 vs 0.84 cm.
    The sensitivity-PINN Jacobian is NOT good enough at 0.01 noise (offset 1e-3, J error -> 2.45 / 3.81 cm) ->
    the FD slope loop stays the one final design; gn is the cross-check for n005 only.
    Posterior error model in --fwd mode (param_invert.py): Sigma_e was the OPERATOR members' spread at c_hat, but the
    fitted value at c_k is the forward members' mean (operator value cancels) -> now the forward members' spread on
    their own path. Check, same iterate gn_s1_n005_it1 re-inverted (runs/chk_se_gn_s1_n005_it1, job 15443966):
    estimate identical (1.0312 cm), sigma 1.746 -> 1.044 cm (noise-only 1.042), Sigma_e sd 1.7e-3 -> 4.3e-5,
    step/sigma 0.47 -> 0.67 (the stop rule gets stricter). For seed 0 the change is ~1% (operator r4 is tight).
    The noise-only sigma of that iterate (1.04 vs referee 0.79, +33%) is the gn Jacobian error, not Sigma_e.
    Ops: GPU partitions full cluster-wide from ~14:40 (V100 78/80, A100 32/32, RTX6000 116/120 in use), all my GPU
    jobs pending on Priority for ~1 h; the 8 gn jobs held the V100s meanwhile. Measures: gn loops stopped after
    their pending inversion; fd_clean's 14 unstarted jobs on hold (scontrol hold, ids in /tmp/fd_clean_held.txt;
    release when fd_s1 is through); CPU fallback hpc/run_script_cpu_mt.slurm (threads = cpus-per-task) being
    timed on fd_s1_p0 (jobs 15443543 with 24 threads, 15443975 with 8).
    (52b) CPU timing of one forward member (fd_s1_p0, 8 threads, cpu-small): 4.4 Adam steps/s vs 23.4 on a V100
       (5.3x slower, ~2.2 h per member against 22 min); the 24-thread job never started (whole-node request).
       GPUs came back at ~15:45 (16 of 18 fd_s1 members started by 16:02), so the CPU fallback was not used; both
       timing jobs cancelled, fd_clean released at 16:03. hpc/run_script_cpu_mt.slurm stays as the fallback.
53. (2026-11-10 16:06) FD-slope loop (the final design), n005 seed 0: CONVERGED at round 1.
    fdl_n005 (start fdtest_n005, operator p2_n005_r4, design fd_n005, new forward-spread error model):
      it 1: zb RMSE d0-25 0.79 cm (target 1.2, referee 0.78), step/sigma max 0.134 (stop rule 0.2), start spread 0,
            sigma 0.763 cm (noise-only 0.762; referee 0.756, +1%), fwd offset rms 2.1e-4, fwd member sd 1.9e-5.
      by segment [cm] 1.26 0.95 0.31 0.09 0.48 0.92 0.52 1.80 vs referee sigma 0.92 0.82 0.23 0.29 0.77 1.34 2.08
      5.46 -> all within 2 sigma; referee's own error 1.21 0.96 0.31 0.09 0.49 0.91 0.45 1.37.
      distance to the referee ESTIMATE on days 0-25: 0.03 cm rms, <= 0.10 sigma_ref in every segment
      (gn_n005_it2 0.12 cm, fdtest_n005 0.08 cm). logs/cpu_15444676.out, cpu_15444677.out.
    => n005 meets every acceptance item with the final design, and the PINN estimate IS the referee's estimate to
       0.03 cm: the remaining 0.79 cm is the noise realisation, not the method.
54. (2026-11-10 16:10) FD-slope loop, n010 seed 0: CONVERGED at round 1.
    fdl_n010 (start fdtest_n010_op5, operator p2_n005_r4 chosen by the smaller affine-fit residual, design fd_n005):
      it 1: zb RMSE d0-25 1.23 cm (target 1.8, referee 1.21), step/sigma max 0.044, start spread 0,
            sigma 1.546 cm (noise-only 1.545; referee 1.536, +1%), fwd offset rms 1.9e-4, fwd member sd 3.1e-5.
      by segment [cm] 1.62 1.80 0.56 0.16 0.66 1.49 1.20 3.70 vs referee sigma 1.83 1.71 0.47 0.57 1.55 2.74 4.17
      8.75 -> all within 2 sigma; referee's own error 1.59 1.82 0.57 0.16 0.62 1.33 0.92 3.26.
      distance to the referee estimate on days 0-25: 0.06 cm rms, <= 0.08 sigma_ref in every segment.
      logs/cpu_15444810.out, cpu_15444811.out.
    => both seed-0 noisy cases meet every acceptance item with one design; open: seed 1 (fd_s1 training), clean.
    (54b) Solver-scored validation of the two converged iterates (scoring only; jobs 15444868, 15444869):
                       rel J error by direction (strong -> weak)            contraction   J cos   J ratio
      fdl_n005_it1     0.01 0.02 0.02 0.02 0.02 0.02 0.04 0.04 0.09         0.03          1.000   0.980-1.010
      fdl_n010_it1     0.01 0.02 0.02 0.02 0.02 0.02 0.03 0.04 0.10         0.03          1.000   0.985-1.006
      (operator alone: 0.05 0.20 0.15 0.28 0.31 0.33 0.43 0.83 0.23)
    fd_s1 design: 18 trained, 0 missing at 16:27 -> fdtest_s1_n005 / _n010_own / _n010_op5 running.
55. (2026-11-10 16:38) Seed-1 slope tests (design fd_s1, centred on dc_s1_n005_it4; one slope-corrected step each):
      test                 operator       fit resid/signal  zb RMSE d0-25  spread   step/sigma  rel J error (strong -> weak)                    contraction
      fdtest_s1_n005       p2s1_n005_r1   0.05              0.88 cm        0        1.51        0.02 0.02 0.01 0.02 0.03 0.02 0.05 0.04 0.10    0.02
      fdtest_s1_n010_own   p2s1_n010_r1   0.07              2.23 cm        0        0.53        0.03 0.03 0.05 0.08 0.12 0.03 0.23 0.05 0.19    0.15
      fdtest_s1_n010_op5   p2s1_n005_r1   0.05              2.21 cm        0.39 cm  0.52        0.03 0.02 0.05 0.07 0.13 0.04 0.21 0.05 0.47    0.17
    targets 1.31 / 3.23 cm, referee 0.87 / 2.16 cm. fdtest_s1_n010_own is 0.12 cm from the referee estimate
    (dc_s1_n010_it4 was 0.62 cm). Uncorrected operator J error: 0.14-3.17 (s1_n005), 0.06-1.43 (s1_n010 own).
    Operator choice for s1_n010: the fit residuals are nearly equal (0.05 / 0.07), but with the n005 operator the
    four far starts stop at a second minimum (cost 0.94742, c[8] = +0.3) while "previous" reaches 0.94530 with
    c[8] = -7.5; with the own operator ALL five starts reach the same point (cost 0.94536, c[8] = -7.0). The n010
    estimate has its last (unidentified) coefficient far outside the n005 operator's cloud. Solver-free rule,
    refined: first require that all starts agree (the stability criterion itself), then the smaller fit residual.
    -> fdl_s1_n005 (operator p2s1_n005_r1, start fdtest_s1_n005) launched 16:35;
       fdl_s1_n010 (operator p2s1_n010_r1, start fdtest_s1_n010_own) launched 16:37.
56. (2026-11-10 16:58) Clean slope test (design fd_clean centred on dc_clean_it5, operator p2_clean_r2):
      fdtest_clean: fit resid/signal 0.07, zb RMSE d0-25 0.03 cm (target 0.5; dc_clean_it6 was 0.06), step/sigma
      0.18, spread 0, sigma 0.760 cm (noise-only 0.758; referee 0.757), offset rms 1.1e-3, fwd member sd 2.5e-5.
      rel J error (strong -> weak) 0.02 0.01 0.01 0.02 0.02 0.01 0.03 0.04 0.12 (uncorrected 0.09 .. 2.23),
      contraction 0.08. -> fdl_clean launched 16:56 (start fdtest_clean) so all five cases end with one method.
    The FD slope has now been validated against the solver in all five cases / three forcings: J error 1-5% in
    the identified directions everywhere (worst: s1_n010, 3-23%, estimate 1-2 sigma off the design centre).
57. (2026-11-10 17:05) FD-slope loop, s1_n005 (seed 1, noise 0.005), unconstrained: CONVERGED at round 1.
    fdl_s1_n005_it1 (operator p2s1_n005_r1, design fd_s1): zb RMSE d0-25 0.87 cm (target 1.31, referee 0.87),
      step/sigma max 0.071, start spread 0, sigma 0.814 cm (referee 0.786, +4%); days 25-30 at 2.2 / 2.6 sigma, as for
      the referee itself (the unconstrained estimate dives at the window end: referee max error 15.9 cm).
    validation against the solver (scoring only, job 15447419): theta RMSE at truth 3.1e-3; rel J error by direction
      (strong -> weak) 0.03 0.02 0.02 0.03 0.07 0.05 0.09 0.07 0.31; contraction (spectral radius) 0.10.
58. (2026-11-10 17:15) s1_n010 (seed 1, noise 0.01): start spread 0.63 cm in fdl_s1_n010_it1 -> NOT accepted.
    Diagnosis: the last spline coefficient c[8] (acts on days 25-30 only, weight <= 1/6 at t = 30) is not identified
    by the data, and unconstrained ML is ill-posed along it:
      fdl_s1_n010_it1     four starts: cost 0.94534, c[8] = -7.09 (2.21 cm); hydrostatic: stalled after 60 iterations
                          at cost 0.95269, c[7] = -0.215, c[8] = +2.967 (1.86 cm)
      gn_s1_n010_it1      two starts stalled at c[8] ~ +2.9
      fdtest_s1_n010_op5  four far starts -> c[8] ~ +0.3 (cost 0.94742); "previous" -> c[8] = -7.5 (cost 0.94530)
      diag_wide_s1        unconstrained wide inversion ran to c[8] = +8.8
    c[8] = -7 is zb = 0.05 m + softplus(-7): the water table "dives" to fit noise at the window end, and |c[8]| >> 3
    is far outside the prior box [-1.821, 1.183] on which every network was trained. The solver does the same
    (scoring only; jobs 15448392/96/404): from constant 1.0, constant 0.3 and hydrostatic_flat it ends at cost
    0.94523, c[7] = +0.31, c[8] = -7.55, whole-window RMSE 4.94 cm, max error 22.6 cm - one minimum for the solver,
    but in a region where the networks extrapolate, so for them it is start-dependent.
59. (2026-11-10 17:35) Fix: the estimate is ML WITHIN the support of the path prior (--box). No new number: the box
    is CPrior.box, softplus^-1(zb - 0.05) for zb in [0.2, 1.5] m = [-1.821, 1.183], the range the wide stage samples.
    invert_lm: active-set projected LM (a coefficient on a bound whose descent direction points out of the box is
    held; the others solve the reduced LM system; trial points clamped) -> converges to the KKT point. Without a box
    the code path is the old LM. Test test_lm_box_converges_to_the_constrained_minimum; 21 tests pass (job 15448340).
    The posterior sigma stays the unconstrained Laplace value (conservative when a bound is active).
    scripts/classical_inversion.py got the same optional --box (scoring only; default unchanged).
    Re-inversion of all five cases with the EXISTING networks (runs/box0_<case>), zb RMSE in cm:
      case      d0-25  whole  start spread  sigma (referee)  step/sigma  on bound  cost      boxed solver d0-25 / whole / cost
      n005      0.79   0.91   6e-5          0.763 (0.756)    0.134       -         0.98389   0.78 / 0.82 / 0.98387
      n010      1.25   1.39   6e-5          1.549 (1.536)    0.072       c[8]      0.98405   1.20 / 1.33 / 0.98399
      s1_n005   0.90   1.45   3e-5          0.828 (0.786)    0.408       c[8]      0.94675   1.02 / 1.26 / 0.94912
      s1_n010   2.46   2.86   1e-5          1.662 (1.712)    0.613       c[8]      0.94919   2.46 / 2.85 / 0.94894
      clean     0.03   0.03   0             0.760 (0.757)    0.180       -         0.00001   -
    All five starts agree to < 1e-4 cm in every case (was 0.63 cm for s1_n010). The boxed solver estimate is start
    independent too (s1_n010 from hydrostatic / constant 1.0 / constant 0.3 / hydrostatic_flat: cost 0.948936 +- 1e-6,
    d0-25 2.44-2.47 cm; jobs 15448382/93/97/405). Cost of the box for the solver itself on days 0-25: n005 0.78 ->
    0.78, n010 1.21 -> 1.20, s1_n005 0.87 -> 1.02, s1_n010 2.16 -> 2.46 cm; whole window 2.78 -> 1.26 and 4.94 ->
    2.85 cm for seed 1 (no dive). s1_n010 PINN coefficients -1.342 -0.247 +0.503 -0.210 -1.444 -0.298 +0.666 -0.357
    -1.821 vs boxed solver -1.296 -0.257 +0.504 -0.210 -1.446 -0.296 +0.668 -0.375 -1.821.
    n005, n010, clean already satisfy the stop rule; s1_n005 / s1_n010 moved from the old (out-of-box) centre ->
    boxed chord loops fdb_s1_n005 / fdb_s1_n010 launched 17:34 (forward members retrained at the in-box c_k).
    (59b) Scoring of the boxed estimates (solver for scoring only):
      error by segment [cm] vs referee sigma (unconstrained referee; jobs 15448710/32/33/56/57):
        box0_n005     1.26 0.95 0.31 0.09 0.48 0.92 0.52 1.80   sigma 0.92 0.82 0.23 0.29 0.77 1.34 2.08 5.46
        box0_n010     1.63 1.81 0.56 0.17 0.77 1.57 0.87 2.60   sigma 1.83 1.71 0.47 0.57 1.55 2.74 4.17 8.75
        box0_clean    0.03 0.05 0.00 0.01 0.02 0.05 0.03 0.05   sigma 0.98 0.79 0.23 0.29 0.76 1.30 2.18 6.47
        box0_s1_n005  0.60 0.89 0.13 0.39 0.47 2.26 3.51 2.21   sigma 0.99 0.82 0.23 0.28 0.75 1.41 2.44 2.79
        box0_s1_n010  1.72 2.17 0.23 0.90 1.79 6.27 5.58 2.69   sigma 2.08 1.74 0.47 0.56 1.62 3.28 6.01 4.87
        -> every segment within 2 sigma in all five cases (seed 1 unconstrained was 2.2 / 2.6 sigma on days 25-30,
           errors 5.3 / 7.4 cm and 10.6 / 10.4 cm there; worst now s1_n010 days 22.5-25 at 1.9 sigma).
      distance to the BOXED solver estimate, |zb - zb_ref| / sigma_ref by segment (jobs 15449627/28, 15449138):
        box0_n005     0.05 0.02 0.01 0.01 0.03 0.03 0.10 0.10   (0.03 cm rms on days 0-25)
        box0_n010     0.04 0.01 0.02 0.01 0.05 0.07 0.03 0.01   (0.07 cm)
        box0_s1_n010  0.06 0.01 0.03 0.06 0.05 0.02 0.05 0.05   (0.07 cm)
      Jacobian of the corrected model AT the boxed estimate vs the solver (jobs 15449139, 15449220):
        box0_n005  rel J error (strong -> weak) 0.01 0.02 0.02 0.02 0.03 0.02 0.04 0.04 0.09, contraction 0.03
        box0_n010  rel J error                  0.01 0.02 0.02 0.02 0.02 0.02 0.03 0.04 0.09, contraction 0.03
60. (2026-11-10 18:15) Clean case, FD-slope loop: CONVERGED at round 1 (fdl_clean_it1, operator p2_clean_r2, design
    fd_clean): step/sigma max 0.047, zb RMSE d0-25 0.01 cm (target 0.5), whole window 0.02 cm, start spread 0,
    sigma 0.759 cm (referee 0.757). Boxed re-inversion with the same networks (runs/boxf_clean, job 15449930): no
    coefficient on a bound, identical estimate (coefficients -1.501 -0.193 +0.403 -0.193 -1.476 -0.193 +0.402 -0.191
    -1.528), i.e. the box is inactive when the data do not push c[8] out.
    GPU queues: seed-1 forward members (fdb_s1_*_f1) pending on Priority from 17:35 to 18:12 (Slurm "job queue
    full" at 17:42-17:50); time limits of the pending jobs lowered to 1 h (A100/H200) / 2 h (V100) for backfill.
61. (2026-11-10 18:40) Boxed chord loops for seed 1 (forward members retrained at the in-box estimate, design fd_s1).
    fdb_s1_n010 (operator p2s1_n010_r1): CONVERGED at round 1.
      it 1: step/sigma max 0.049, start spread 2e-5 cm (five starts, cost 0.94904 each), zb RMSE d0-25 2.47 cm
            (target 3.23; boxed solver 2.46, unconstrained solver 2.16), whole window 2.87 cm (unconstrained solver
            4.94), sigma 1.672 cm (referee 1.712, -2%), fit resid/signal 0.059, offset rms 2.2e-3, member sd 1.5e-5.
      coefficients -1.294 -0.258 +0.507 -0.211 -1.444 -0.298 +0.668 -0.366 -1.821
      boxed solver -1.296 -0.257 +0.504 -0.210 -1.446 -0.296 +0.668 -0.375 -1.821 (cost 0.948936)
      by segment [cm] 1.68 2.23 0.23 0.89 1.81 6.30 5.50 2.86 vs referee sigma 2.08 1.74 0.47 0.56 1.62 3.28 6.01
      4.87 -> all within 2 sigma (worst 1.92 on days 22.5-25; the unconstrained referee itself is at 1.68 there).
      distance to the boxed solver estimate: <= 0.05 sigma_ref in every segment, 0.04 cm rms on days 0-25.
      validation at the estimate (job 15451111): rel J error (strong -> weak) 0.03 0.02 0.04 0.06 0.09 0.04 0.15
      0.08 0.25 (uncorrected 0.06 .. 1.74), contraction 0.15 -> design fd_s1 is still adequate at the in-box centre.
    fdb_s1_n005 (operator p2s1_n005_r1): round 1 step/sigma max 0.42 (from box0, whose offset came from members at
      the old out-of-box centre), start spread 4e-6 cm (cost 0.94918), zb RMSE d0-25 1.03 cm (target 1.31; boxed
      solver 1.02), whole 1.29 cm, sigma 0.828 cm (referee 0.786, +5%); distance to the boxed solver estimate
      <= 0.08 sigma_ref, 0.02 cm rms (box0: 0.19 cm). Round 2 running to meet the stop rule.
    (jobs 15451109/10/12; coarse stage with the box: three starts end at c[8] = lower bound, constant:0.3 at the
    upper bound, and the fine boxed stage brings all five to the same point.)
    (61b) Operator-swap check for s1_n010 (job 15451181; same forward members and design, operator p2s1_n005_r1
      instead of p2s1_n010_r1, boxed): all five starts agree (spread 2e-5 cm, cost 0.94901; unboxed the same swap
      gave two solutions and 0.39 cm spread, finding 55). Estimate 2.31 cm d0-25 / 2.68 cm whole, sigma 1.708 cm;
      it differs from the own-operator estimate by at most 0.16 sigma (below the 0.2 sigma stop tolerance) - the size
      expected from a 3-25 % Jacobian error acting on a noise-level residual. So the operator-choice rule of finding 55
      is no longer needed for stability; the choice moves the estimate by < 0.2 sigma.
62. (2026-11-10 19:25) fdb_s1_n005 CONVERGED at round 2 -> all five cases meet every acceptance item.
    it 2 (operator p2s1_n005_r1, design fd_s1, boxed): step/sigma max 0.111 (round 1: 0.42; late segments
      0.42 / 0.34 -> 0.11 / 0.09), start spread 1.4e-5 cm (five starts, cost 0.94900), zb RMSE d0-25 1.01 cm
      (target 1.31; boxed solver 1.02, unconstrained solver 0.87), whole window 1.24 cm (unconstrained solver 2.78),
      sigma 0.837 cm (referee 0.786, +7%).
      coefficients -1.392 -0.223 +0.445 -0.199 -1.463 -0.239 +0.509 -0.230 -1.821 (c[8] on the lower bound)
      by segment [cm] 0.64 0.92 0.12 0.42 0.68 2.59 2.62 1.29 vs referee sigma 0.99 0.82 0.23 0.28 0.75 1.41 2.44
      2.79 -> all within 2 sigma (worst 1.84 on days 22.5-25; job 15452243).
      distance to the boxed solver estimate <= 0.13 sigma_ref in every segment, 0.06 cm rms on days 0-25 (job 15452244).
    Solver validation at the estimate (scoring only, job 15452245): rel J error (strong -> weak) 0.03 0.03 0.03 0.05
      0.18 0.14 0.29 0.10 1.15 (uncorrected 0.14 .. 3.51), predicted contraction over all 9 coefficients 1.36.
      -> looked like a diverging loop, but the weakest direction is c[8], which the box holds fixed. The boxed
      inversion only moves the free coefficients, so the contraction has to be computed over those:
      scripts/param_validate.py --fixed <coef_on_bound> (new option; also writes val_jac.npz with J_net, J_ref).
      With c[8] fixed (jobs 15452346-48):
                          rel J error by direction (strong -> weak)   contraction (spectral radius / 2-norm)
        fdb_s1_n005_it2   0.03 0.03 0.03 0.05 0.18 0.14 0.28 0.10     0.38 / 0.79
        fdb_s1_n010_it1   0.03 0.02 0.04 0.06 0.09 0.04 0.15 0.08     0.11 / 0.55
        box0_n010         0.01 0.02 0.02 0.02 0.02 0.02 0.03 0.04     0.02 / 0.27
      The predicted 0.38 agrees with the observed step reduction of fdb_s1_n005 (0.42 -> 0.11, ratio 0.26).
      s1_n005 is the weakest case: its operator had one refinement round (theta RMSE vs solver at the truth 3.1e-3;
      p2s1_n010_r1 1.9e-3, p2_n005_r4 3.3e-4), so the forward-PINN offset and slope correction carry more of the
      accuracy there. It still passes the decision rule of finding 46 (rel J error <~ 0.25 in all but the last 1-2 directions, contraction <~ 0.4).
    FINAL ACCEPTANCE TABLE (zb RMSE in cm against the truth; solver used for scoring only)
      case      run               d0-25   target  whole   start spread  sigma (referee)   step/sigma  on bound
      n005      box0_n005         0.79    1.20    0.91    6.2e-5        0.763 (0.756)     0.134       -
      n010      box0_n010         1.25    1.80    1.39    6.2e-5        1.549 (1.536)     0.072       c[8]
      s1_n005   fdb_s1_n005_it2   1.01    1.31    1.24    1.4e-5        0.837 (0.786)     0.111       c[8]
      s1_n010   fdb_s1_n010_it1   2.47    3.23    2.87    1.6e-5        1.672 (1.712)     0.049       c[8]
      clean     boxf_clean        0.01    0.50    0.02    1.2e-9        0.759 (0.757)     0.047       -
      every segment within 2 referee sigma in all five; sigma within -2 .. +7 % of the referee's.
    Figures (one per case, line plots): output/compare/final_{n005,n010,s1_n005,s1_n010,clean}.png (job 15452247).
63. (2026-11-10 19:40) Twin v4 (non-sinusoidal zb, smooth flux, reference solver, air-entry soil): wide stage only.
    configs/twin_synth_v4.yaml, output/twin_synth_v4. Operator = 4 wide members (prior paths), no refinement yet:
      noise 0.005 (runs/v4_n005_inv0): zb RMSE d0-25 4.84 cm, whole 4.93 cm, start spread 0.082 cm, c[8] on bound
      clean       (runs/v4_clean_inv0): d0-25 4.83 cm, whole 5.14 cm, bias -0.6 cm, max 15.5 cm, start spread
                  0.116 cm, theta RMSE vs truth 1.6e-3, no coefficient on a bound
    Same level as v3 before its refinement rounds (5.80 cm). Refinement / FD chord loop for v4 NOT run: the user
    asked for the FEM / GLDAS twin instead (finding 64); v4 stays a side case.
64. (2026-11-10 20:40) GLDAS-forced twin from the wtd_pipeline FEM with a prescribed, time-varying water table
    (user request: "synthetic from fem ... forced by gldas, prescribed a fixed wtd and simulate sm ... just a non
    const wtd"). scripts/make_twin_fem.py, configs/twin_gldas_fl.yaml (job 15454621, 8.5 min CPU).
    Setup: ForwardSensitivitySolverPyTorch as generate_training_continuous.py (L 12, 101 nodes, Picard tol 1e-5,
      omega 0.5, max 50), site G02101 (27.375 N, 81.625 W), 1 Apr - 1 May 2021, calibrated Florida sand, plain
      VG-Mualem; q_surf = E_soil - I at the surface, T_p as Feddes root sink (grass defaults), h = 0 at zb(t).
      zb(t): 0.50 -> 0.93 m (day 14) -> 0.55 m (day 21) -> 0.80 m, spline truth vs prescribed points 0.85 cm RMSE.
      Forcing: 2 mm rain day 0, dry d1-9, rain d10-11 (21 + 12 mm), d14-15, d18-20 (14 + 6 + 9 mm), d23, then dry;
      sum q -66 mm, sum T_p 76 mm. Daily values, linear in between (q and T_p).
    Departures from the sweeps, both needed for a PINN test case:
      - hourly FEM steps (sweeps: daily). Daily backward-Euler steps differ from hourly by 5.3e-3 RMSE / 1.35e-2 max
        in theta_5cm = the noise level, i.e. the daily-step run is not the solution of the PDE to twin accuracy.
      - hydrostatic start over zb(0) (sweeps: spin-up). The PINN's hard initial condition.
    Two twins from one batched solve: output/twin_gldas_fl (with root sink, as the sweeps) and
      output/twin_gldas_fl_nosink (T_p = 0; the PDE the PINN had so far).
    Checks (d theta_5cm, RMSE / max):
      FEM converged in 100 % of steps (max 13 Picard iterations hourly; 19 daily; 10 at 15 min / 201 nodes)
      FEM vs finer FEM (15 min, 201 nodes)          2.8e-4 / 8.7e-4 (sink), 2.0e-4 / 7.5e-4 (no sink)
      FEM no sink vs this project's reference solver 2.8e-4 / 1.3e-3 (finer FEM vs reference 1.8e-4 / 5.2e-4)
        -> two independent solvers agree: Landau form, flux sign and observation operator are the same in both.
      with sink - without sink                      1.76e-2 / 3.2e-2 (mean -1.6e-2) -> the sink is 3.5 x the noise:
        the PINN needs the sink term for the sweep-like twin.
      air-entry soil (v3 / v4 option) vs plain VG    1.04e-2 / 1.8e-2 -> the PINN must use plain VG on this twin.
      FEM's own node-mean theta_surf vs the Gauss-Legendre operator 4.8e-4 / 2.2e-3
      surface head min -0.97 m (sink) / -0.71 m (no sink): admissible.
    Identifiability (reference solver, no sink, job 15454929; output/twin_gldas_fl_nosink/identifiability.*):
      hourly obs, sigma_theta 0.005: sigma_zb median 2.1 cm, rms 9.8 cm (blows up after day 27), window mean 2.4 cm
      hourly obs, sigma_theta 0.01:  median 4.2 cm, rms 19.5 cm
      daily obs,  sigma_theta 0.005: median 9.8 cm, rms 35 cm
      singular values (hourly) 0.508 0.345 0.144 0.139 0.058 0.035 0.021 0.009 0.001
      v3 for comparison:        0.738 0.502 0.426 0.271 0.172 0.124 0.048 0.012 0.002 -> median 0.78 cm, rms 2.1 cm
      -> the GLDAS case is 3-6 x less informative in the middle directions (deep water table 0.8-0.94 m on days
      5-17 under dry forcing). No method can do better than ~2-5 cm there with noise 0.005; the noise-free
      reproduction is the first PINN target, noisy results are judged against this bound.
65. (2026-11-10 20:50) GLDAS/FEM twin: error bound from the FEM itself, root sink in the PINN, diagnostics queued.
    a) Bound from the FEM (scripts/fem_identifiability.py, new; jobs 15455089, 15455334): J = d theta_5cm / d c by
       central differences (delta 0.02) in one batched solve of the wtd_pipeline FEM, 38 members (19 paths x with /
       without sink), Picard tol 1e-8 (fem.jacobian in the config; the twin keeps the sweeps' 1e-5), 100 % of steps
       converged (max 125 iterations). Shared helpers bound_from_jacobian / save_bound moved to scripts/common.py
       (identifiability.py output unchanged, job 15455091).
       Checks: unperturbed member vs stored twin 6.3e-5 (sink) / 2.9e-5 (no sink) RMSE = the twin's Picard error;
         FEM Jacobian vs reference-solver Jacobian (no sink): column cosine >= 0.9999, norm ratio 1.000-1.005
         (last column 0.975) -> the two bounds agree (hourly, 0.005: median 2.1 cm both).
       With sink (the sweep-like twin), output/twin_gldas_fl/identifiability_fem.*:
         hourly obs, sigma_theta 0.005: sigma_zb median 1.6 cm, rms 11.8 cm (after day 27), window mean 2.9 cm;
           3-4 cm on days 8-16 (deep water table, dry), < 1 cm on days 0-4 and 18-25
         hourly obs, sigma_theta 0.01: median 3.2 cm; daily obs, 0.005: median 7.8 cm, rms 42 cm
         singular values (hourly) 0.645 0.519 0.196 0.178 0.084 0.045 0.023 0.011 0.000
       Twin's own discretisation error as a zb error (no sink): the zb change that explains FEM - reference theta
         (2.8e-4 RMSE) is 0.64 cm rms / 1.1 cm max on days 0-25. A method that solves the continuous equation exactly
         cannot do better than this on the clean FEM twin -> clean target on this twin ~0.6 cm + method error.
    b) run_fem (scripts/make_twin_fem.py) generalised to a batch of members; regression run into a scratch
       directory gave identical twin_summary.json for both twins (job 15455090; scratch removed).
    c) Root water uptake in the PINN residual: src/zbinv/roots.py (RootSink: Feddes stress with the T_p-dependent
       h3, Gale-Grigal root density, T_p(t) linear between the daily values as in the FEM run), physics.pde_residual
       adds S L / Ks when model.sink is set, parametric.build_param_model sets it for twins made with the sink
       (twin_config fem.root_sink). No weights, no state-dict entries (saved models load unchanged).
       tests/test_roots.py: equals wtd_pipeline RootWaterUptakeTorch.compute_sink to 1e-12 over depth 0-3.5 m,
       h +0.05 .. -158 m, T_p 0-9 mm/d; residual gains S L / Ks in model units (independent numpy). Suite: 43 PASS.
    d) Gate G1 on both twins (DIAGNOSTIC: operator trained around the true path, compared with the twin), baseline
       design unchanged: g1gn_base / g1gn_p05 (no sink), g1gs_base / g1gs_p05 (with sink); p05 = time periods
       down to 0.5 d for the 1-day rain pulses. Queued on h200 / a100 / v100 (pending on priority since 20:10).
       Decision rule: baseline first. Held in reserve, only if it falls short: causal memory features of q(t)
       (q is piecewise linear with daily kinks, the time features are smooth), K(h_eq) input feature or mixed
       (h, q) form for the unbounded dK/dh of plain VG at the water table (finding 11a).

66. (2026-11-10 22:25) GLDAS/FEM twin with root sink: baseline pipeline through the first inversion; forcing-memory
    features added (default off) because the true-path diagnostic is ~10x short of the sinusoidal twin.
    a) Scheduler (primary controller down, backup up: squeue / sbatch / scancel hang for minutes, sacct answers):
       hpc/param_pipeline.sh submit() retries with a time limit (as dc_loop.sh); hpc/race.sh rewritten with time
       limits and retries on every scheduler call (a hung squeue had left duplicate copies of running members queued
       -> they could have started and overwritten the same run directory; cancelled by hand); a watchdog kills
       squeue children of old race.sh instances older than 4 min (/tmp/race_watchdog.sh). CPU hedges (24 threads,
       ~0.2 s/step) for the diagnostics while the GPU queue was stuck on priority.
    b) Wide stage (runs/gs_wide_w0..3, A100, ~22 min each): val_pde 0.048-0.056, val_flux 0.017-0.031, theta_5cm RMSE
       at the true path 6.2-6.7e-3 = the sinusoidal wide members (6.6e-3).
    c) First inversion runs/gs_clean_inv0 (clean hourly theta, 4 starts): zb RMSE 5.82 cm (days 0-25 5.82, max 12.8,
       bias -2.0 cm; by 5-day segment 6.7 3.7 6.3 6.4 5.7 5.8 cm); all starts on the same path (spread 5e-8 m);
       theta fit 1.8e-3 (largest after the rain pulses, days 12-15, 18-19, 22). Same level as the first stage on the
       sinusoidal twins (5.4 / 4.8 cm) -> on the v3 trajectory, not yet a reproduction.
       Figure output/twin_gldas_fl/pinn/gs_clean_inv0.png (scripts/twin_result_figure.py, plots.plot_twin_result).
    d) Gate G1 on the sink twin (operator trained around the TRUE path; diagnostic only), CPU copies, theta_5cm RMSE
       at the true path vs Adam step:        5000     10000    14500    22000   (v3: 1.5e-3 5.3e-4 ~3.5e-4 2.5e-4)
         base (periods 2..64 d)              5.6e-3   4.4e-3   3.8e-3   3.1e-3   val_flux 0.019
         p05  (periods 0.5..64 d)            6.0e-3   3.6e-3   2.6e-3   1.9e-3   val_flux 0.007
       Needed for cm-level zb with this twin's sensitivity (singular values in 65a): <~ 3e-4. Short periods help
       -> the limit is the time resolution of the response to the kinked forcing (rain pulses, daily T_p), which the
       smooth time inputs (Fourier, path features) have to build by themselves.
    e) Reserve (i) implemented, default off: src/zbinv/forcing.py ForcingMemoryFeatures. Inputs of the head: causal
       exponential memories of the KNOWN forcing, tau g' = u - g, g(0) = 0 (no forcing before the window, as the
       hydrostatic initial condition), u piecewise linear -> closed form on every segment, exact at any t, C1 in t
       (d h / d t stays continuous), slope kinks where the forcing has them. Ladder model.forcing_memory_taus_days
       (runs: 1/64 1/16 1/4 1 4 16 d = the path-memory ladder extended two steps down). Series: surface flux
       (model.forcing_split_sign: upward and downward parts as two series, each on its own scale; split_by_sign adds
       the zero crossings as nodes, exact) and T_p on twins with the sink. No weights, no state-dict entries, not a
       function of c; placed before the path features so the zoom warm start still appends columns
       (load_state_zoom_aware unchanged). Raw u(t) is deliberately not an input (it would put kinks in h itself).
       Still a physics-only PINN: the forcing is the given boundary condition, theta data never enter training.
       tests/test_forcing.py (7): step / ramp closed forms, the ODE by autograd at random t, continuity of g and
       g' across nodes, hold after the last node, exact split, model wiring (per-sample c, jacfwd = FD, d h / d t
       total derivative, reload, zoom warm start, refusal of a state without the features), build switch on both
       twins. Suite 48 PASS + these 7 (jobs 15458972, 15459190).
    f) Running: G1 with the features on the sink twin, runs/g1gs_fm(_cpu) and runs/g1gs_fms(_cpu) (split sign);
       baseline refinement round 1 (runs/gs_clean_r1_m0..3) queued on GPU. Decision rule: if fm/fms is clearly below
       p05 at equal steps, rerun wide + rounds with it (new tag); else next reserve (K(h_eq) input, mixed form).

67. (2026-11-10 23:10) What limits the operator at the true path on the GLDAS twin: a residual of one sign that adds
    up, not the rain events. Forcing memory split by sign helps a lot; water-balance box term added (default off).
    a) G1 finals (DIAGNOSTIC, operator trained around the TRUE path), theta_5cm RMSE / max at the true path:
         no sink:   g1gn_base 1.09e-3 / 2.9e-3 (val_pde 0.021, val_flux 0.0065)   g1gn_p05 8.1e-4 / 2.5e-3
         with sink: g1gs_fm (forcing memory, one flux series) 1.01e-3 / 2.6e-3 (val_pde 0.013, val_flux 0.0026)
                    base 2.8e-3 @28500, p05 1.7e-3 @28000 (CPU copies, before L-BFGS)
         with sink, g1gs_fms (flux split into upward / downward parts): 2.7e-3 @5000, 7.0e-4 @11000, 6.8e-4 @15500
           -> at 11000 steps already below every other variant's final. Needed: <~ 3e-4.
    b) Error anatomy at truth (scripts/truth_error_profile.py, scoring only; logs/cpu_15460185.out, cpu_15460328.out,
       output/twin_gldas_fl/pinn/truth_error_g1gn.npz), g1gn_p05: the theta error is NOT at the rain events; it is a
       slow drift over the whole column in the quiet periods. Water balance of the column per day: surface-flux error
       <= 0.005 mm/d, but the column-integrated PDE residual keeps one sign for days (-0.5..-0.8 mm/d on days 1-13,
       +0.7..+1.1 mm/d on days 15-18; evaporation itself is 0.1-0.3 mm/d), ~75% of it leaves through the water
       table, the rest is a storage error up to -1.2 mm (day 9). That coherent part is ~17% of the RMS residual
       = ~3% of the pointwise mean square, and it carries the state error. (v3 G1: <= 0.2 mm/d, 0.12 mm.)
       -> the pointwise mean square is blind to exactly the component of the residual that the solution error
       follows (a parabolic problem integrates the residual: error ~ residual in a weak norm, not in L2).
    c) Water-balance box term (physics.box_water_defect, ParamTrainer train.n_box, default 0 = off). Boxes
       [-depth, 0] x [t0, t0 + dur] in (xi, t), one zb path each. Exact identity of the layer:
         W(t1) - W(t0) = int [ -q_surf + (q - M xi zdot)(xi_a) - z int S dxi ] dt,   W = z int M dxi
       and the model's defect of it is exactly B = int [ z int R dxi - (q(0,t) - q_surf) ] dt (zero for the exact
       solution whatever the quadrature -> consistent, same minimiser). Estimated on stratified points (box_n_t 8 x
       box_n_xi 4). Scale: the defect as a mean water content of the layer, against sigmas.theta (the precision of
       the data the operator will be fitted to) -> loss term 'box' = mean((theta* B / (z depth) / sigma_theta)^2).
       Box sizes log-uniform from the data's own smallest scales (time step of theta_obs; obs depth / deepest
       admissible water table) to the whole window and column. No new weight, no data, still physics-only.
       Held-out set gets 4 n_box boxes ('box' enters val_total, so L-BFGS selection sees it); L-BFGS 4 n_box.
       Tests (tests/test_parametric.py): defect from residuals = storage change - boundary supply computed from
       storage and fluxes separately (untrained model, 1e-3 of the terms); off by default (batch 7th item None,
       losses {'pde','flux'}); box geometry, finite loss with gradient, chunked held-out = unchunked.
       Suite 52 PASS (job 15460969).
    d) Result of the warm-start test on the no-sink twin (continue g1gn_p05: Adam 10000 at 3e-4 + 6 L-BFGS blocks;
       theta_5cm RMSE / max at the true path; twin's own floor 2.8e-4 / 1.3e-3):
         g1gn_ctlw  (no box)                         8.0e-4 / 2.4e-3   val_pde 0.0161 val_flux 0.0042
         g1gn_boxw  (box v1: 256 boxes, 8x4 points)  3.4e-4 / 9.8e-4   val_pde 0.0272 val_flux 0.0375 val_box 0.78
         g1gn_boxw2 (box v2: 128 boxes, 16x4 points) 3.6e-4 / 1.07e-3  val_pde 0.0206 val_flux 0.0071 val_box 0.026
       -> more pointwise training does nothing (8.1e-4 -> 8.0e-4); the box term takes the same network to the
       twin's floor (x2.3). Pointwise losses are equal or slightly worse, so the gain is the coherent part alone.
    e) Box v1 was mis-scaled (val_box 66 -> ~1 against a pointwise loss of 0.02, noisy training, v1 from scratch on
       the sink twin stuck at 7e-4..9e-4 with val_box ~1). Scan of what the defect measures
       (scripts/box_defect_scan.py, scoring only; logs/cpu_15461321.out, cpu_15461419.out), g1gn_p05 at truth:
         - windows <= 1-2 d: defect ~ actual change of the layer-mean theta error (ratio 0.7-1.5, all layers);
         - longer windows: the defect overstates it (x2 at 16 d without sink, up to x8.8 with sink): the layer
           hands the defect on to the soil below within the window, so it is no longer an error of that layer;
         - sampling error of the 8x4-point estimate is as large as the defect for windows >= 2 d
           (1.35-1.9 sigma at 16 d); with 16 time points it is 0.02-0.08 sigma at 1-2 d (defect 0.08-0.19 sigma).
       v2 rule (no new number): longest window = theta* zb_max / Ks = T_days * clip_m[1] / L = 2.45 d, the time
       gravity flow at Ks takes to drain the storage of the deepest admissible column (beyond it a defect has left
       the layer); 16 time strata per box. Same cost as v1 (128 x 16 x 4 points; ~ +100% per step, V100 ~0.1 s).
       Tests updated (box_dt_max, geometry): test_parametric 0 failed (job 15461418).
    f) Sink twin finals (G1 diagnostic): base 1.99e-3 / 5.6e-3, p05 1.54e-3 / 4.8e-3 (CPU copies), fm 1.01e-3,
       fms (split sign) 6.8-7.2e-4 from step 11000 to 28500 while val_pde / val_flux kept falling, final after
       L-BFGS 8.6e-4 / 2.7e-3 (selection by pointwise loss made it worse) -> same plateau as on the no-sink twin.
       fms + box v2 from scratch (g1gs_fmsb2, V100 15461549): 2.8e-3 @3000, 1.3e-3 @5000, 6.9e-4 @9500,
       4.4e-4 @12500, still falling (lr 1.5e-4) -> already below every pointwise-only variant's final.
    g) Baseline pipeline on this twin (gs_clean = v3 design, no forcing features, no box): inv0 5.82 cm; round-1
       members 4.4-6.5e-3 theta at truth; inv1 8.53 cm RMSE days 0-25 (max 20.0 cm, coefficient 8 on the bound,
       step/sigma 11.0); inv2 19.3 cm (max 42 cm, round-2 members 4.8-6.8e-3 at truth) -> diverging
       (v3 twin: 5 -> 2.0-2.4 cm at inv1). Baseline stops here.
       New operator = split-sign forcing memory + box v2, everything else as v3:
         hpc/param_pipeline.sh gb_clean 2 "<TW> forcing_memory_taus_days=[1/64..16] forcing_split_sign=true
         train.n_box=128 train.box_n_t=16" "--box" gb_wide      (logs/pipe_gb_clean.log, launched 23:15)
    h) What the box term buys in zb (DIAGNOSTIC: one operator trained around the TRUE path, no-sink twin, clean data,
       4 far starts, --box bounds; runs/g1gn_boxw2_inv vs runs/g1gn_ctlw_inv):
         without box (g1gn_ctlw):  zb RMSE days 0-25 1.14 cm, max 2.2 cm
         with box    (g1gn_boxw2): zb RMSE days 0-25 0.39 cm, max 0.92 cm, bias -0.17 cm, all starts identical
       -> an operator of this accuracy supports an inversion at the twin's own floor (0.64 cm rms / 1.1 cm max);
       what is left to show is that the truth-free pipeline (wide -> rounds -> chord loop) gets the operator there.
       Error anatomy after the box term (logs/cpu_15462676.out, cpu_15462677.out): storage error of the column
       <= 0.45 mm (was 0.94 mm), layer-mean theta error 0.06-0.08 sigma (was 0.13-0.19); the column residual
       that remains (-0.5..-0.9 mm/d) leaves through the water table and does not reach the surface layer.
    i) Sink twin, same diagnostic (g1gs_fmsb2: split-sign forcing memory + box v2, from scratch, 30000 Adam + 15
       L-BFGS; V100 56 min): theta at truth 5.0e-4 / 2.3e-3 (L-BFGS blocks 4.3-6.7e-4; val_pde 0.0138, val_flux
       0.0033, val_box 0.016); inversion with it (runs/g1gs_fmsb2_inv): zb RMSE days 0-25 1.20 cm, max 2.1 cm,
       bias -0.04 cm, all starts identical. Pointwise-only variants on this twin: 8.6e-4 .. 2.0e-3.
       Wide stage: new members at truth 6.0-7.7e-3 @22000, the same as the baseline's wide members (6.3-7.1e-3)
       -> the wide stage is limited by the breadth of the prior, not by the drift; the gain has to come in the
       rounds / forward PINNs. Forward-PINN floor at the true path being measured (g1gs_ptb2, tube 0.1 mm).
    j) gb pipeline, wide stage (4 x V100, 40000 Adam + 10 L-BFGS, 67-70 min): members at truth 3.9-6.0e-3
       (baseline 6.3-7.1e-3), val_pde 0.046-0.080, val_flux 0.031-0.066, val_box 0.09-0.13.
       gb_clean_inv0: zb RMSE days 0-25 5.53 cm, max 9.9 cm, bias -3.2 cm, no coefficient on a bound, theta fit
       9.7e-4, ensemble theta sd 1.4e-3, start spread 1.5e-7 m (baseline inv0: 5.82 cm, max 12.8 cm, coef 8 on bound).
    k) Round 1 contracts where the baseline diverged. gb_clean_r1 members at truth 1.05-1.77e-3 (baseline round 1:
       4.4-6.5e-3); gb_clean_inv1: zb RMSE days 0-25 1.84 cm (inv0 5.53), bias -0.6 cm, theta fit 4.8e-4,
       ensemble theta sd 6.1e-4, start spread 0.87 mm; whole window 2.7 cm / max 12.6 cm (window end, coefficient
       8 on the bound: past day 27 the data carry no information), step/sigma 8.9 -> not converged yet.
       Forward-PINN floor at the TRUE path (g1gs_ptb2, DIAGNOSTIC, tube 0.1 mm, 25000 Adam + 15 L-BFGS):
       3.8e-4 / 1.9e-3 (val_pde 0.0083, val_flux 0.0019, val_box 9.6e-4); twin's own floor 2.8e-4 / 1.3e-3.
    l) (02:15) Round 2: THE PINN REPRODUCES THE PRESCRIBED WATER TABLE ON DAYS 0-25 (clean GLDAS twin with root sink,
       truth-free pipeline wide -> 2 rounds, networks trained on physics only).
       gb_clean_r2 members at truth 5.1-8.8e-4; gb_clean_inv2: zb RMSE days 0-25 0.44 cm (5.53 -> 1.84 -> 0.44),
       by 5-day segment 0.54 0.71 0.34 0.13 0.20 | 6.22 cm; theta fit 3.9e-4; start spread days 0-25 1.0 mm;
       ensemble theta sd 5.8e-4; posterior sigma median 1.04 cm; step/sigma max 1.92 (stop rule 0.2 not met yet).
       Days 26-30: starts split into two branches (+9 / -16 cm at day 30), coefficient 8 on the bound -> the tail
       is not identified by theta_5cm (FEM Cramer-Rao bound blows up after day 27; same on the v3 twin).
       Twin's own floor on days 0-25: 0.64 cm rms / 1.1 cm max. Figure output/twin_gldas_fl/pinn/gb_clean_inv2.png.
       Value-corrected loop gbdc_clean started 02:13 from inv2 (operator r2 members, COARSE wide members, WARM=1,
       forward PINNs with the same options; logs/dc_gbdc_clean.log) to confirm convergence under the stop rule.

## REAL DATA (E5), started 2026-11-11
User, 2026-11-11: "ok lets skip and move on to the real data case" -> the open synthetic items (stop rule of the
chord loop, noisy GLDAS twin, new design on the old sinusoidal twin) are skipped, not done.

68. Survey of the AmeriFlux Florida sites (read-only; fem_validation/uploaded_data, half-hourly BASE files).
    a) Files: AMF_US-PAS_BASE_HH_2-5.csv (2021-2022), AMF_US-ONA_BASE_HH_4-5.csv (2016-2022). Sites 1.2 km apart
       (US-PAS 27.3944 N -81.9510 grazed bahiagrass pasture; US-ONA 27.3836 N -81.9509 native pasture), Spodosols.
       Local standard time; missing = -9999; CRLF line ends; WTD negative below ground; SWC in %.
    b) SENSOR DEPTH: SWC_1_1_1 is a TDR at -0.10 m at both sites (AmeriFlux measurement-height table), not the
       top 5 cm. US-PAS SWC_2/3_1_1 (also -0.1 m, other positions) read up to 60-72% -> not usable as VG theta.
    c) The processed daily NPZ of the forward validation (US-PAS_2021-04-01_2021-11-30.npz) uses qsurf = ET - P
       with total ET and total P (no E/T split, no runoff), sink off; its `smap_sm` is the daily mean of the
       in-situ SWC_1_1_1.
    d) DATA ERROR, US-PAS: the rain record P_RAIN_1_1_1 is 23.5-24 h LATE against soil moisture, water table and
       flux sensors over all of 2021 (lag scan of d theta vs P: sharp single peak at +48 half-hours Apr-Jul, +47
       Aug-Dec; single storms: rise 5 Oct 22:00 <-> P 6 Oct 21:30; 7 Oct 18:30 <-> 8 Oct 18:00; 28 Oct 15:00 <->
       29 Oct 14:00; LE spikes / gaps at the time of the soil response). US-ONA rain is in sync (lag -1 half-hour).
       This also touches the existing daily US-PAS forward validation (rain a day late in q_surf).
    e) The water table answers rain within 30-60 min by 0.4-0.6 m (US-PAS 7 Oct 18:30: WTD -0.86 -> -0.26, SWC
       17 -> 41.5% on ~29 mm): not representable by a 5-day-knot spline -> windows are RECESSIONS between storms.
       US-PAS WTD has a ceiling at ~0.21-0.25 m with SWC(10 cm) saturated at 0.41-0.43.
    f) Availability: US-PAS complete (LE, P, WTD, SWC1) only Apr-Nov 2021; Apr-Jun WT 1.0-1.95 m (below the
       box), Jul-Nov 0.15-1.1 m. US-ONA 2021-01..2022-12 mostly complete; WT at the surface in Jul-Sep 2021,
       Sep and Nov 2022; SWC gap 14-26 Oct 2021; LE missing 1-18 Oct and 10-15 Nov 2022; LE has gaps on most days.
    g) RETENTION MISMATCH (hand calculation, to be confirmed by a forward run): observed US-PAS recession pairs
       zb -> theta(10 cm): 0.25 -> 0.425, 0.50 -> 0.363, 0.78 -> 0.271, 1.00 -> 0.172, 1.10 -> 0.136; hydrostatic
       VG(alpha 3.6, n 1.56) at the sensor: 0.39, 0.33, 0.25, 0.22, 0.21 -> too dry near saturation, too wet at
       the dry end; theta 0.105 (US-ONA, zb 1.08 m) needs h = -4.6 m on this curve. With the calibrated Florida
       soil an inversion is pushed far outside the admissible box at the dry end.
    h) Admissibility: K(h = -1 m) ~ 1.35e-3 Ks ~ 0.34 mm/d: total ET of 2-3 mm/d cannot leave as a pure surface
       flux once the WT is below ~0.8-1 m -> transpiration as the root sink; there is no measured E/T split.
    i) Candidate recessions (start wet and near hydrostatic, as the initial condition assumes):
       US-ONA 2021-11-13 -> 12-15 (zb 0.23 -> 1.08 m, theta 0.37 -> 0.105, rain-free, rain timing clean);
       US-ONA 2022-11-21 -> 12-14 (zb 0.08 -> 0.87), 2022-10-19 -> 11-09 (0.47 -> 0.91), 2021-09-25 -> 10-05;
       US-PAS 2021-10-09 -> 11-04 (zb 0.22 -> 1.10, small rises 25 and 28-30 Oct; needs the -23.5 h rain shift),
       US-PAS 2021-09-25 -> 10-05, 2021-11-14 -> 12-10.

Real-case spec (first slice):
  inputs    half-hourly LE, P, SWC_1_1_1 (10 cm), WTD of one site and window
  forcing   ET = LE / (2.45e6 * 1000), gaps filled with the window's mean diurnal cycle (fill fraction reported);
            daily nodes (24-h centred means at midnight, linear in between: the set-up validated on the twins);
            q_surf = (1 - f_T) ET - P at the surface, T = f_T ET as the root sink (f_T: no measurement -> variants)
  data      hourly theta at 0.10 m (point sensor: obs.depth_m = [0.10, 0.10]); sigma_theta from the data
            (rms of theta minus its 24-h running mean: what daily forcing cannot explain)
  truth     observed |WTD|, hourly -- scoring only; truth_coef = spline fit to it (scoring / diagnostics only)
  window    30 d (9 coefficients, 5-day knots, as validated), start on the first dry day after a wetting
  checks    (1) forward run: solver with the observed WT -> theta(10 cm) vs observed (model error, per soil set)
            (2) referee inversion (scoring only): what this model can give at best
            (3) PINN pipeline (wide -> rounds), unchanged method
  success   PINN zb vs observed WTD: RMSE, bias, r on the window; PINN vs referee (method error) reported apart
            from referee vs observed (model/data error)

69. Real case, first slice built (2026-11-11).
    a) Windows (US-ONA, rain in step): TEST candidates and CALIBRATION candidates are the two November recessions.
       2021: rain 1.5 + 8.5 mm on 11/12 Nov (WT -0.55 -> -0.17 m), then dry to 15 Dec (0.1 mm traces on 18, 19, 29
       Nov). Daily WTD / theta(10 cm): 13 Nov 0.235 / 0.369, 15 Nov 0.466 / 0.347, 17 Nov 0.593 / 0.329,
       20 Nov 0.673 / 0.306, 22 Nov 0.711 / 0.280, 24 Nov 0.769 / 0.224, 26 Nov 0.818 / 0.193, 28 Nov 0.847 / 0.175,
       2 Dec 0.902 / 0.151, 6 Dec 0.961 / 0.133, 9 Dec 1.000 / 0.122, 13 Dec 1.050 / 0.110.
       WTD record missing 25 Nov 05:30 - 26 Nov 12:00 (62 records; bridged linearly in the scoring truth).
       2022: rain 11.7 mm on 20 Nov (WT to 0.08 m), dry to 14 Dec. 23 Nov 0.165 / 0.358, 26 Nov 0.350 / 0.348,
       28 Nov 0.471 / 0.344, 30 Nov 0.579 / 0.324, 2 Dec 0.657 / 0.275, 4 Dec 0.722 / 0.219, 6 Dec 0.785 / 0.190,
       9 Dec 0.805 / 0.169, 12 Dec 0.845 / 0.157, 14 Dec 0.872 / 0.146. No WTD/SWC gap; LE missing 131 (2021) and
       205 (2022) of the padded half-hours.
    b) Reading of (a) IF the profile were hydrostatic: theta(10 cm) nearly flat (0.37 -> 0.33) while h at the
       sensor goes -0.13 -> -0.49 m, then steep: an air-entry-like curve (alpha ~1.5 1/m, n ~5-6, theta_s ~0.37),
       far from the calibrated Florida set. To be checked by the forward run, since root uptake (1-2 mm/d, a third
       of the roots in the top 10 cm) dries the sensor depth below hydrostatic.
       Year-to-year repeatability of theta at the same WT in the steep part: 0.02-0.05 (zb 0.77: 0.224 in 2021,
       0.203 in 2022; zb 0.69: ~0.30 vs 0.242) = 4-5 cm in zb -> realistic real-data accuracy with a soil
       calibrated on the other year is ~5 cm, not the 0.4 cm of the twin.
    c) E/T split: the GLDAS pixel next to the site has T_p / (E_soil + T_p) = 0.80-0.90 in the window (total
       1.5-2.2 mm/d vs tower 1-2 mm/d) -> forcing.transpiration_fraction: gldas (daily ratio applied to tower ET),
       with fixed fractions 0 / 0.5 / 1 as a sensitivity check. No tuned number.
    d) Code (defaults unchanged; jobs of the synthetic loop still import the source):
       - obs.depth_m may be [top, bottom] (point sensor when equal): data.obs_layer, used by obs_xi_points,
         physics.theta_obs_model, train.initial_coef (mid depth), parametric box_d_min (layer bottom).
       - src/zbinv/sitedata.py: half-hourly reader (csv + numpy; no pandas in the env), rain time shift,
         mean-diurnal-cycle fill, centred means (forcing nodes, hourly data), short-gap fill, unresolved rms.
       - scripts/make_real.py + configs/real_ona_nov21.yaml (13 Nov 2021 12:00 + 30 d), real_ona_nov22.yaml
         (23 Nov 2022 12:00 + 20 d): data set in the twin format (real.npz + twin_config.yaml) and the forward
         check (FEM with the observed WT: spline / record x f_T gldas / 0 / 0.5 / 1; hydrostatic curve; finer FEM).
         Window starts at local noon: daily nodes = calendar-day means at noon.
       - make_twin_fem.run_fem accepts q per member; plots.plot_real_check; tests/test_sitedata.py.
       Jobs: tests 15468567, builds 15468568 (2021), 15468569 (2022).

70. Real case, forward check: the Florida soil set cannot give the 10-cm sensor even with the OBSERVED water table
    (2026-11-11; builds 15468568 / 15468569, tests 15468567 -> 1 wrong expectation fixed, 15468615: 0 failed).
    FEM with the observed WT (spline fit) and tower forcing vs observed theta(10 cm):
      window                 f_T=gldas (rmse / bias / r)   f_T=0     0.5      1      hydrostatic   FEM vs finer FEM
      2021 (13 Nov + 30 d)   0.0598 / +0.039 / 0.947      0.0495*   0.0546   0.0617   0.0697        7.6e-4
      2022 (23 Nov + 20 d)   0.0522 / +0.035 / 0.907      0.0455    0.0494   0.0532   0.0535        3.5e-4
      (* inadmissible: surface head to -1035 m.) Hourly WT record instead of its spline: 0.0600 (2021), no change.
    Data: WT 0.239 -> 1.076 m (2021), 0.165 -> 0.871 m (2022); theta 0.376 -> 0.110, 0.358 -> 0.152; ET 48.0 / 30.6
    mm, rain 0.3 / 0 mm; LE filled 6.3 / 12.5 %; sigma_theta (rms about the 24-h running mean) 0.0015 / 0.0022;
    spline (5-day knots) vs hourly WT record: rmse 1.6 cm, max 3.6 / 4.4 cm (the WT has a 2-4 cm diurnal cycle)
    = representation floor of the scoring.
    Reading (output/real_ona_nov21/real_check.png): observed theta stays 0.35 -> 0.30 for 9 days, then drops to
    0.11 in daytime steps; the model declines smoothly to 0.20. The E/T split and spline-vs-record change almost
    nothing -> the retention curve at the sensor is the mismatch (the Florida set was calibrated against the daily
    record with a different reading of the sensor depth).
    DECISION: soil of the site from a CALIBRATION recession whose water table is known (forward FEM, bounded least
    squares on theta_s, theta_r, alpha, n, Ks; scripts/calibrate_soil.py, 4 starts), used UNCHANGED on the other
    year's recession where the WT is the unknown. Calibration is not part of the inversion; the PINN pipeline is
    unchanged. Alternative if that fails: soil parameters as operator inputs (not started).
    Jobs: cal21_s0..3 = 15468748..51 (output/soil_ona_nov21), cal22_s0 / s3 = 15468752 / 53 (output/soil_ona_nov22).

71. Real case, set-up of the held-out test (2026-11-11, 03:00-03:40; no PINN result yet).
    a) Calibration, interim (jobs still running, ~5.5 min per LM iteration at Picard tol 1e-8; results are written
       only at the end, the "[cal] solve" log lines carry the soil):
         2021, hydrostatic-fit start (cal21_s3): rmse 0.00553 -> 0.00463; theta_s 0.3498, theta_r 0.1113,
               alpha 1.546 1/m, n 9.42, Ks 2.38e-6 m/s
         2022, hydrostatic-fit start (cal22_s3): rmse 0.00645 -> 0.00433; theta_s 0.3514, theta_r 0.130,
               alpha 1.915, n 9.99 (upper bound 10), Ks 4.1e-7
         Florida / C&P starts (s0, s1, s2): rmse 0.034-0.069 after 30 min, n 1.7-1.9, descending slowly -> will not
         converge inside the 2-h limit; "all starts agree" can not be shown tonight (say so in the report).
       The sensor follows a very sharp retention curve (n ~ 9-10): saturated plateau until the WT is ~0.65 m, then
       desaturates within ~20 cm of WT fall. theta_r <-> n trade off along the limb (theta_r 0.05 <-> n ~ 4.4).
    b) Sensitivity of theta(10 cm) to zb in the 2021 data, three regimes: plateau (zb 0.24 -> 0.65 m) 0.14 per m;
       steep limb (0.65 -> 0.82) 0.76 per m; tail (0.82 -> 1.08) 0.31 per m, partly root-uptake drying. The water
       table is identified by this sensor on the steep limb only; on the plateau the reading says "shallower than
       ~0.65 m" and nothing more. Score by regime.
    c) Cross-year offset: at the same WT, 2022 theta is 0.02-0.03 below the 2021-soil model (plateau reading 0.376
       in 2021, 0.358 in 2022; alpha 1.55 vs 1.9 = ~9 cm shift of the desaturation depth). Expected cross-year zb
       error 5-10 cm on the limb, more on plateau / tail. This is model + sensor error, not PINN error.
    d) Provisional held-out forward checks (soil = hydrostatic retention fit of the OTHER year, Ks 2.9e-6):
         output/real_ona_nov22_prov  (26 Nov 2022 12:00 + 18 d, 2021 soil 0.3501 / 0.1029 / 1.517 / 8.856):
             FEM(observed WT) vs observed theta rmse 0.0313, bias +0.028, r 0.988 (Florida soil: 0.052, r 0.907);
             f_T 0 / 0.5 / 1: 0.0302 / 0.0308 / 0.0314; hydrostatic 0.0296. WT 0.336-0.884 m, 7 coefficients.
         output/real_ona_nov21_prov  (15 Nov 2021 12:00 + 30 d, 2022 soil 0.3496 / 0.1226 / 1.684 / 9.024):
             rmse 0.0197, bias -0.011, r 0.983. WT 0.488-1.101 m, 9 coefficients.
         (output/real_ona_nov22_provb, 24 Nov + 20 d: truth c_0 = -4.15, outside the prior box, not used.)
    e) TEST WINDOWS WERE CHOSEN WITH KNOWLEDGE OF THE OBSERVED WT (disclose): right after rain the WT is at
       0.17-0.24 m and its spline has c_0 = -7.2 / -4.9, far outside the path prior's box [-1.82, 1.18]
       (= zb 0.2-1.5 m at the knots). Windows start 2-3 days later (c_0 ~ -2.1; first knot value slightly below
       the box, so the first days carry a small representation error). A truth-free rule for the same choice:
       start when the sensor has left the saturated plateau reading by more than its noise.
    f) FEM clamps (wtd_pipeline/fem/fem_pytorch.py, read only): Se in [1e-6, 1 - 1e-6] (l. 739, 753), Kr >= 1e-12
       (l. 240), h_from_Se clamps Se >= 0.01 (l. 175). The "surface head min" it reports bottoms out at
       h(Se = 0.01): -1035 m for the Florida soil, -1.05 m for n ~ 9. So reference.h_surf_admissible_m (-10 m) is
       blind for the sharp soil, and the FEM silently limits evaporation / uptake when the surface dries. The PINN
       solves the un-clamped PDE: for the sharp soil, prior paths deeper than ~1 m under prescribed E_soil and
       uptake are stiff. -> rehearsal with two priors: default (clip 0.2-1.5 m, mean 0.3-1.2) and soil-derived
       (clip upper = |h(Se = 0.05)| ~ 0.96 m -> clip 0.2-1.0, mean 0.3-0.8).
    g) Rehearsal pipelines on output/real_ona_nov22_prov (2 rounds, K 4, --box, sigmas.theta 0.005 as validated on
       the twin; forcing memories [1/64 .. 16] d, split sign, n_box 128): rp22a = soil-derived prior, rp22b =
       default prior. First members started 03:22 on V100 (init: val_flux ~105, val_pde 0.22, theta_rmse_truth
       0.028 = the hydrostatic curve vs the OBSERVED theta, i.e. it contains the model error from here on).
    h) Referee for real data: scripts/fem_referee.py (new; scoring only): the LM of classical_inversion.py with the
       wtd_pipeline FEM (root sink) as forward model, one start per job, same box rule. Jobs ref22b_0..3 =
       15469778 / 79 / 92 / 93 (default box; starts hydrostatic, hydrostatic_flat, 1.0, 0.3), ref22a_0 / 1 =
       15469813 / 14 (restricted box) -> output/referee/prov22_*. Separates PINN error (PINN vs referee) from
       model / data error (referee vs observed WT).
    i) Synthetic chord loop gbdc_clean stopped by hand at 03:25 to free GPUs (it 1: zb rmse days 0-25 0.57 cm,
       step / sigma max 0.42, start spread 0; before the loop 0.44 cm). Its f2 jobs 15469258-63 cancelled.
72. Real case, calibration finals, rehearsal results and the final runs (2026-11-11, 04:30-05:00).
    a) Soil calibration, dynamic FEM with the observed WT (scripts/calibrate_soil.py), static-start jobs finished:
         2021 (cal21_s3): rmse 0.0055 -> 0.0046 (bias 0.0000, r 0.999), no bound, xtol stop, 63 min:
               theta_s 0.34978, theta_r 0.11138, alpha 1.5464, n 9.4328, Ks 2.3826e-6; scaled Jacobian singular
               values 1.68 1.17 0.79 0.36 0.21.
         2022 (cal22_s3): rmse 0.0064 -> 0.0043 (r 0.998), n ON THE UPPER BOUND (10), 44 min:
               theta_s 0.35143, theta_r 0.12992, alpha 1.9164, n 10, Ks 4.1272e-7; singular values 1.88 1.05 0.54
               0.25 0.11.
         Other starts hit the 2-h limit (no result files): 2022 Florida start (cal22_s0) at solve 156: rmse 0.00448,
         0.3516 / 0.1278 / 1.936 / 10 / 3.8e-7 = the same soil -> two starts agree for 2022. 2021 starts 0 / 1 / 2 at
         solves 85 / 32 / 89: rmse 0.020 / 0.039 / 0.019, n 2.9 / 2.3 / 3.5 and rising -> NOT converged; for 2021
         "the starts agree" is not shown (one converged start only).
    b) Final held-out data sets (soil of the OTHER year, nothing fitted on the test window):
         T22 = output/real_ona_nov22_cal21 (26 Nov 2022 12:00 + 18 d, 7 coefficients, WT 0.336-0.884 m):
             FEM(observed WT) vs observed theta: rmse 0.0273, bias +0.0243, r 0.989 (record WT: 0.0281);
             hydrostatic 0.0250, bias +0.021; f_T 0 / 0.5 / 1: 0.0263 / 0.0267 / 0.0275. Nearly hydrostatic window.
         T21 = output/real_ona_nov21_cal22 (15 Nov 2021 12:00 + 30 d, 9 coefficients, WT 0.488-1.101 m):
             FEM(observed WT) rmse 0.0273, bias -0.0125, r 0.966; hydrostatic 0.0507, bias -0.030 (NOT hydrostatic:
             root uptake dries the sensor layer on the tail); f_T 0 / 0.5 / 1: 0.0315 / 0.0268 / 0.0270.
         -> the model error at the sensor is ~0.027 in both directions, 5x the sigma (0.005) used in the cost.
    c) FIRST PINN INVERSIONS ON SITE DATA (rehearsal set output/real_ona_nov22_prov, wide stage only):
         rp22b_inv0 (default prior): 4 starts -> one estimate (spread 1e-7 m); cost 0.29302; theta rmse vs obs
             0.00271; zb RMSE vs OBSERVED WT 6.21 cm (bias +5.60, max 19.3; below-saturation hours 5.21 cm,
             r 0.985); no coefficient on the box; posterior sigma median 0.18 cm (model error NOT in it).
         rp22a_inv0 (restricted prior): cost 0.31007, 6.81 cm, c_0 and c_6 on the box bound -> not used.
         FEM referee, default box: all 4 starts -> cost 0.29272, coefficients +0.885 -0.909 -0.107 +0.184 +0.207
             +0.337 +0.154, zb RMSE 6.58 cm (bias +6.04, max 19.7). Restricted box: 6.42 cm, c_0 on the bound.
         PINN vs referee: 1.05 cm RMSE (max 3.1 cm, at the two ends) -> the PINN's own error is ~1 cm; the 6 cm
             is model / sensor error (b), common to PINN, referee and the hydrostatic reading (6.19 cm).
         Operator residuals after the wide stage are far above the twin's (val_flux 2.5-7.3 vs 0.07; val_pde
             0.5-1.7 vs 0.06): the n ~ 9 soil is much stiffer than the Florida soil. The estimate still agrees
             with the referee because the response in this window is nearly hydrostatic. Rounds 1-2 running.
    d) Decisions: default prior for the finals (the validated configuration; lower cost, nothing on the box).
       rp22a stopped by hand (driver + race processes killed by PID, its r1 jobs cancelled; confirmed gone).
    e) Final runs launched 04:50-05:00: rf22 (T22) and rf21 (T21): hpc/param_pipeline.sh <tag> 2, same train sets
       as rp22b, --box; referees reff22_0..3 (15471779 / 80 / 91 / 92 -> output/referee/cal21on22_s*), reff21_0..3
       (-> output/referee/cal22on21_s*, 8-h limit). Scoring: scripts/score_real.py (new) -> result_<tag>.json /
       .png beside the data set: PINN, referee and hydrostatic reading vs the observed hourly WT, all hours and
       hours with the sensor below saturation (theta_obs < theta_s - 3 sigma), PINN vs referee, start spread.
    f) Results so far (07:50):
         rehearsal round 1 (rp22b_inv1): 5 starts -> one estimate; cost 0.30102; coefficients +0.843 -0.900 -0.114
             +0.180 +0.224 +0.323 +0.492 (referee +0.885 -0.909 -0.107 +0.184 +0.207 +0.337 +0.154: only the last,
             weakly identified one differs); zb RMSE vs observed 6.74 cm; PINN vs referee 0.59 cm (wide stage
             1.05 cm) -> the local rounds move the PINN onto the referee. Members after round 1: val_pde 0.29-0.73,
             val_flux 1.9-2.8, val_box 0.5-0.7 (wide: 0.5-1.7 / 2.5-7.3 / 1.2-2.4). step/sigma max 11.5.
             All 4 referee starts (default box) agree: cost 0.292723, 6.58 cm.
         FINAL T22, wide stage (rf22_inv0; w2 waited 1.5 h for a GPU): 4 starts -> one estimate (spread 2e-8 m);
             cost 0.30390; coefficients +0.365 -0.796 -0.178 +0.150 +0.218 +0.335 +0.420; zb RMSE vs observed
             6.12 cm (bias +5.48, max 18.8, r 0.984); sensor below saturation (359 of 433 h): 5.08 cm.
             Referee, 4 starts agree: cost 0.28540, +0.771 -0.883 -0.114 +0.167 +0.186 +0.329 +0.125, 6.13 cm
             (bias +5.47, max 19.5); below saturation 5.07 cm. Hydrostatic reading: 5.69 cm (bias +4.93); below
             saturation 4.43 cm. PINN vs referee 1.28 cm (max 3.4 cm). Ensemble theta sd 0.0034.
             -> on T22 every method is ~6 cm with a +5 cm offset = the cross-year soil offset (72b).
    g) Rehearsal, final round (rp22b_inv2, 07:47): 5 starts -> one estimate; cost 0.30837; coefficients +1.032
       -0.938 -0.098 +0.174 +0.219 +0.322 +0.406; zb RMSE vs observed 6.64 cm (bias +6.13); PINN vs referee
       0.38 cm (max 2.0 cm; wide 1.05 -> round 1 0.59 -> round 2 0.38); step / sigma max 1.5 (sigma ~0.2 cm).
       The cost rises slightly with the rounds (0.2930 -> 0.3010 -> 0.3084; referee 0.2927; in theta rmse
       0.00271 -> 0.00278): not understood, small; the FEM has clamps the PINN does not (71f).
    h) T21 AS A 30-DAY WINDOW IS NOT A VALID TEST OF THE 2022 SOIL (found from the referee, 07:50):
       the 2022 calibration window only saw theta 0.152-0.358, and its theta_r came out at 0.130 (theta_r <-> n
       trade-off, n on the bound). In T21 the sensor reads 0.147 on day 18 and 0.105 on day 30: after day ~22 the
       reading is BELOW theta_r, no water table can reproduce it. Referee on the 30-day window: starts 1.0 m and
       0.3 m stop at once (zero gradient: all coefficients on the box, cost 395); hydrostatic starts after 18
       iterations at cost 3.5, |step| 3 (jumping between bounds), zb RMSE 14-15 cm.
       Rule (uses the sensor reading and the calibration data only, no water table): invert only while the sensor
       reads inside the range the soil was calibrated on. T21 -> first 17 days (theta 0.349 -> 0.152).
       T21b = output/real_ona_nov21_cal22_17d (15 Nov 2021 12:00 + 17 d, 7 coefficients, WT 0.488-0.929 m,
       truth c_0 -1.98): FEM(observed WT) vs observed theta rmse 0.0340, bias -0.028, r 0.961; hydrostatic 0.0662,
       bias -0.058 (Ks 4.1e-7: the profile lags the falling WT -> this window is NOT hydrostatic).
       The 2022 soil reads 0.028 too dry on 2021, the 2021 soil 0.024 too wet on 2022: one cross-year offset.
       T22 is inside its soil's calibrated range (2021 window: 0.110-0.376; T22: 0.147-0.348).
       rf21 (30 d): driver stopped by hand after the wide stage (no refinement rounds); its inv0 is run once for
       the record. rf21b (17 d) + referees reff21b_0..3 (15475777-80 -> output/referee/cal22on21_17d_s*) launched
       07:57.
    i) FINAL T22, rounds 1-2 (rf22, pipeline done 09:02; score output/real_ona_nov22_cal21/result_rf22.json/.png):
       round 1 (rf22_inv1): 5 starts -> one estimate; cost 0.31382; zb RMSE vs observed 6.41 cm (bias +5.86);
       step / sigma max 12.6.
       round 2 (rf22_inv2, FINAL): 5 starts -> one estimate (spread 7e-8 m); cost 0.30176; coefficients +0.830
       -0.901 -0.108 +0.164 +0.199 +0.342 +0.283; no coefficient on the box; step / sigma max 2.2; posterior sigma
       0.18 cm; ensemble theta sd 0.0009; members val_pde 0.24-0.34, val_flux 1.7-2.7, val_box 0.3-1.5.
                                    all 433 h                    sensor below saturation (359 h)
         PINN vs observed WT        6.33 cm (bias +5.78, max 19.5, r 0.987)   5.41 cm (bias +5.14, max 8.5)
         FEM referee vs observed    6.13 cm (bias +5.47, max 19.5, r 0.987)   5.07 cm (bias +4.73, max 8.5)
         hydrostatic reading        5.69 cm (bias +4.93, max 17.3, r 0.987)   4.43 cm (bias +4.12, max 9.1)
         PINN vs referee            0.59 cm (bias +0.31, max 2.1, r 0.9996)   0.64 cm
       theta at the sensor, PINN at its estimate vs observed: rmse 0.00275 (FEM with the observed WT: 0.0273).
       The 7-coefficient spline of the observed WT is 1.58 cm from the hourly record (max 4.9): the record's
       steps are not representable, so ~1.6 cm is the floor of any spline estimate.
       -> PINN = solver inversion to 0.6 cm, start-independent; the 6 cm to the observed WT is the +5.5 cm
          cross-year soil offset (72b), shared by all three methods. In this slow, nearly hydrostatic recession the
          hydrostatic reading is as good or slightly better (4.4 vs 5.4 cm): T22 does not show an advantage of
          the dynamic inversion, only that the PINN reproduces it.
       PINN vs referee by round: 1.28 -> (round 1 not scored) -> 0.59 cm.
    j) T21 30 d, for the record (72h: not a valid test). PINN rf21_inv0 (wide stage only, job 15476385): the
       4 starts end at costs 3.66 / 3.92 / 27.5 / 4.02 with zb RMSE 20.8 / 11.1 / 36.7 / 21.4 cm, spread 0.31 m;
       members val_flux 9.5-13.8. Referee: hydrostatic 16.75 cm at cost 3.48; hydrostatic_flat 15.3 cm at cost
       3.51 (iteration 22, still running 09:06); 1.0 m and 0.3 m starts stalled at cost 394.6.
       -> where the sensor leaves the calibrated range neither method has a defined answer; the instability
          there is the model's, visible in both.
    k) T21b referee (17 d, 2022 soil), 09:06: hydrostatic and hydrostatic_flat finish at the same point: cost
       0.342555, coefficients -1.821 -0.366 -0.556 -0.005 +0.179 -0.325 +1.183 (c_0 on the lower and c_6 on the
       upper bound of the box), zb RMSE 9.16 cm (days 0-13: 6.61), bias -7.86, max 20.3; theta rmse 0.00293.
       0.3 m start: iteration 16 at cost 0.342559, 9.15 cm (same point, still iterating). 1.0 m start: stalled
       at once (cost 688.7, zero gradient: the sensor sits on the residual plateau for a 1 m water table).
       The observed WT's own c_0 (-1.98) is outside the box (-1.821): the box clips the first coefficient.
       Bias sign as expected (2022 soil reads too dry on 2021 -> the fit puts the WT shallower).
    l) 2021 calibration from two more starts, 12-h jobs (cal21L, 09:06, still descending):
       start 0 (Florida soil): solve 214 rmse 0.0102, theta_s 0.363, theta_r 0.056, alpha 1.87, n 7.40, Ks 1.4e-7;
       start 2 (sandy loam):   solve 213 rmse 0.0074, theta_s 0.357, theta_r 0.090, alpha 1.81, n 9.99, Ks 1.7e-7.
       Accepted soil (start 3): rmse 0.0046, 0.3498 / 0.1114 / 1.546 / 9.43 / Ks 2.4e-6. Retention is converging
       to the same curve; Ks is 14x lower on both -> Ks of the 2021 soil is NOT yet confirmed by a second start.
    m) T21b, PINN wide stage (rf21b_inv0, 09:08; members val_pde 0.9-1.9, val_flux 8.2-10.6, val_box 3.4-6.4):
       starts hydrostatic, hydrostatic_flat and 1.0 m -> one estimate: cost 0.54665, coefficients -1.821 -0.373
       -0.582 +0.010 -0.077 +0.024 -0.436, zb 0.470-0.726 m, zb RMSE vs observed 10.55 cm (days 0-13: 8.43),
       bias -9.70. Start 0.3 m ends elsewhere: cost 6.51, ... +1.183 +1.183 (c_5, c_6 on the upper bound),
       14.2 cm. Start spread 0.197 m rms -> NOT start-independent after the wide stage on this window.
       Against the referee (cost 0.34256; c_4..c_6 = +0.179 -0.325 +1.183): c_0..c_3 agree to 0.03, the last
       three differ: the window end, where the sensor nears the residual plateau (theta 0.152, theta_r 0.130).
       The PINN's 1.0-m start converges (23 iterations) where the referee's stalled (zero FEM gradient).
       Referee 0.3-m start: iteration 17 at cost 0.3425558 = the point of the two hydrostatic starts.
       Rounds 1-2 running.
    n) T21b FINAL of the 2-round protocol (rf21b_inv2, 10:24; score output/real_ona_nov21_cal22_17d/
       result_rf21b.json/.png). Round 1: 4 of 5 starts agree (cost 0.39918), the 1.0-m start ends at cost 2.67
       (c_5, c_6 on the upper bound). Round 2: all 5 starts -> one estimate (spread 5e-8 m); cost 0.42297;
       coefficients -1.821 -0.362 -0.572 +0.044 -0.084 -0.149 +1.183 (c_0, c_6 on the box, as the referee);
       step / sigma max 3.5; posterior sigma 0.25 cm; ensemble theta sd 0.0023.
                                    all 409 h                     sensor below saturation (382 h)
         PINN vs observed WT        10.50 cm (bias -9.45, max 20.1, r 0.938)  10.83 cm
         FEM referee vs observed     9.17 cm (bias -7.86, max 20.3, r 0.913)   9.45 cm
         hydrostatic reading        11.60 cm (bias -11.08, max 16.6, r 0.980) 11.96 cm
         PINN vs referee             3.17 cm (bias -1.59, max 7.0, r 0.968)    3.28 cm
       theta at the sensor: PINN at its estimate 0.00325, referee 0.00293, FEM with the observed WT 0.0340.
       Reading: (1) the -8 to -11 cm offset is the cross-year soil error again (2022 soil too dry on 2021).
       (2) Both inversions agree to < 1 cm up to day ~8 and then part: after day 9 both turn back up (WT
       "rising" 0.79 -> 0.71 m while the observed one falls to 0.93 m) -- model error absorbed in the weakly
       informed window end (sensor near the dry plateau), and the PINN's wiggle differs from the solver's by up
       to 7 cm. (3) the posterior sigma (0.25 cm) does not cover that 3 cm: it has no term for operator bias.
       Solver-free signs that the operator is NOT converged here: step / sigma 3.5, and the round-2 members'
       held-out residuals are WORSE than round 1's (val_pde 0.43-0.76 -> 0.79-8.1 (m2), val_flux 5.0-8.1 ->
       4.8-7.7, val_box 1.5-2.7 -> 2.1-3.7). Cause: between inv0 and inv1 c_6 jumped -0.44 -> +1.18, and the
       local sampling covariance has the term dc dc^T, so round 2 trained on a much wider set than round 1.
       T22 for comparison: val_pde 0.50-0.58 -> 0.24-0.34, step / sigma 2.2, PINN vs referee 0.59 cm.
    o) 2021 soil confirmed by a second start (cal21L_s0, Florida start, solve 310, 10:27): rmse 0.00464, theta_s
       0.3497, theta_r 0.1115, alpha 1.547, n 9.449, Ks 2.368e-6 = the accepted soil (start 3) to 3 digits.
       Start 2 (sandy loam): solve 309, rmse 0.00547, alpha 1.62, n 9.90, Ks 5.7e-7, still moving.
       -> the 2021 and 2022 soils differ for real (alpha 1.55 vs 1.92, Ks 2.4e-6 vs 4.1e-7), each confirmed by
          two starts: the cross-year offset is not a calibration artefact.
    p) Stopping rule of the plan (stop when the estimate moves < 0.2 sigma between rounds) was not met on either
       window after 2 rounds (step / sigma 2.2 and 3.5). hpc/param_pipeline.sh: optional START_ROUND (default 1 =
       unchanged behaviour) continues a finished pipeline. Rounds 3-4 launched 10:31 for BOTH windows with the
       same settings (logs/pipe_rf21b_c.log, logs/pipe_rf22_c.log); scores -> result_rf21b_r4, result_rf22_r4.
    q) ROUNDS 3-4 on both windows (pipelines done 11:48; scores result_rf22_r4, result_rf21b_r4).
       T22: inv3 all 5 starts one estimate, cost 0.30238, 6.33 cm, step / sigma 0.42; inv4: 4 starts one estimate
            (cost 0.30334, +0.883 -0.905 -0.112 +0.166 +0.194 +0.336 +0.312, 6.27 cm, step / sigma 0.65), the 0.3-m
            START TRAPPED at cost 21.8 (-1.821 -1.821 +0.195 -0.057 +0.582 +1.183 +1.183, 23.5 cm).
            Members r4: val_pde 0.18-0.29, val_flux 1.8-2.4, val_box 0.45-0.73 (no gain over r2).
            PINN vs referee 0.59 (r2) -> 0.52 cm (r4); vs observed 6.33 -> 6.27 cm.
       T21b: inv3 all 5 one estimate, cost 0.39311, c_5 -0.287, 11.4 cm, step / sigma 1.48; inv4 all 5 one
            estimate, cost 0.42899, -1.821 -0.365 -0.571 +0.054 -0.085 -0.171 +1.183, 10.63 cm, step / sigma 0.76.
            Members r4: val_pde 0.45-2.45, val_flux 4.0-6.9, val_box 1.1-2.2 (no gain over r1).
            PINN vs referee 3.17 (r2) -> 3.19 cm (r4): more rounds do NOT close the gap; c_5 wanders
            -0.149 / -0.287 / -0.171 (referee -0.325), c_4 stays at -0.06..-0.09 (referee +0.179).
       -> (1) the trapped far start is the known failure of findings 42 / 48 (far starts leave the region where the
          refined members are trained); the adopted fix, param_invert --coarse <wide members>, was missing from
          hpc/param_pipeline.sh (the real-data runs used the pipeline as it was). Re-inversion of round 4 with
          --coarse (rf22_inv4c 15483378, rf21b_inv4c 15483377): ALL 5 STARTS -> ONE ESTIMATE on both windows
          (spread 2e-8 / 8e-8 m), same estimates as above. param_pipeline.sh now passes --coarse in every round.
          (2) the stop rule (0.2 sigma) is not reachable by operator rounds: the estimate wobbles by 0.4-1.5 sigma
          (0.1-0.4 cm) from round to round = training noise of the operator, which sigma does not contain.
          (3) T21b's 3 cm from the referee is operator bias (held-out residuals 3-10x T22's on this soil, Ks 4e-7,
          n 10). The stage built for it on the twins is the value correction (finding 41: forward PINN at the
          estimate, hpc/dc_loop.sh). Launched 11:52 for both windows from inv4: tags rf21bdc / rf22dc, operator =
          r4 members, COARSE = wide members, WARM=1, 2 forward PINNs per iteration, up to 4 iterations
          (logs/dc_rf21bdc.log, logs/dc_rf22dc.log).
    r) 2021 SOIL: ALL THREE STARTS AGREE (cal21L finals, 2026-11-11 12:17). output/soil_ona_nov21/start{0,2,3}.json:
       start 0 (data set's own soil)  rmse 0.004635, theta_s 0.34975, theta_r 0.11146, alpha 1.5465, n 9.444, Ks 2.379e-6
       start 2 (sandy loam)           rmse 0.004635, theta_s 0.34975, theta_r 0.11146, alpha 1.5465, n 9.444, Ks 2.379e-6
       start 3 (hydrostatic fit)      rmse 0.004635, theta_s 0.34978, theta_r 0.11138, alpha 1.5464, n 9.433, Ks 2.383e-6
       none on a bound. -> the 2021 calibration is unique within the bounds; the soil used on T22 stands.
    s) VALUE-CORRECTED LOOP, T21b iteration 1 (rf21bdc_it1, 13:10; rf22dc still waits for a GPU: f1_m0 pending).
       Forward PINNs at the round-4 estimate (rf21bdc_f1_m0/1): val_pde 0.052 / 0.066, val_flux 0.021 / 0.007,
       val_box 0.11 / 0.18 (operator members on this soil: val_pde 0.45-2.45, val_flux 4-7) -> a single-path PINN
       solves this stiff soil 10-100x better than the operator. Operator value error at the estimate: theta rms
       0.0021 (max 0.0052), the two forward members differ by 0.0004.
       Corrected inversion: 5 starts -> one estimate (spread 5e-9 m); cost 0.34558 (round 4: 0.42899; referee
       0.34256); theta rmse 0.00294 (referee 0.00293); coefficients -1.821 -0.356 -0.574 +0.055 -0.063 -0.140
       +1.183 (referee -1.821 -0.366 -0.556 -0.005 +0.179 -0.325 +1.183).
                              vs referee (rms / max)   vs observed WT
         round 4 (inv4c)        3.19 / 7.14 cm           10.63 cm
         value-corrected it 1   2.79 / 6.21 cm           10.04 cm        (referee vs observed 9.16 cm)
       zb (m) at days 0, 4, 8, 12, 16: observed 0.490 0.647 0.775 0.855 0.915; referee 0.474 0.547 0.689 0.785
       0.715; round 4 0.473 0.546 0.691 0.715 0.722; it 1 0.475 0.546 0.692 0.724 0.735.
       -> the cost is now the referee's to 1%, the estimate equals the referee's to 0.5 cm up to day 9 and differs
       after day 10 (c_4, c_5: 3-5 cm), a direction the data barely constrain (cost difference 0.003). Step / sigma
       2.26 (segments 0.58 0.23 0.24 1.64 2.26) -> not converged, loop continues (up to 4 iterations).
73. Real case: what "reliable" means on site data, and the next test: soil from the PREVIOUS RECESSION
    (2026-11-11 13:30; criteria and plan written before any result of it).
    a) Where it stands (finding 72): the PINN returns the solver's inversion (0.5 cm on T22, 2.8 cm on T21b) and one
       estimate from every start. What is left to the observed water table (6.3 cm, bias +5.7 on T22; 10 cm, bias
       -9.0 on T21b) is the soil taken from ANOTHER YEAR, not the network: the referee has the same error.
    b) Site record, no model (daily means, AmeriFlux file; WT depth m / sensor theta):
         Oct 2022 recession   0.565/0.342  0.621/0.322  0.674/0.292  0.711/0.254  0.742/0.221  0.766/0.201
                              0.790/0.188  0.813/0.177  0.830/0.168  0.849/0.161  0.858/0.155  0.877/0.148  0.889/0.142
         T22 (26 Nov-14 Dec)  0.513/0.339  0.579/0.324  0.625/0.305  0.657/0.275  0.693/0.242  0.722/0.219
                              0.770/0.203  0.785/0.190  0.789/0.175  0.805/0.169  0.824/0.160  0.845/0.156  0.872/0.146
       At the same theta the WT of T22 is 0-4 cm shallower than in October (4 cm at 0.32, 3 at 0.25-0.29, 0 at
       0.19-0.20, 2.5 at 0.16-0.18, 1 at 0.15), against ~9 cm between Nov 2021 and Nov 2022 (finding 71). Both fall
       at 3-4 cm/d on the steep limb. -> within one season the sensor-WT relation repeats to ~2 cm.
       Not single valued in general: 6 Nov 2021, one day after a 0.7-m rise, the sensor reads 0.333 with the WT at
       0.34 m (0.369 a week later at 0.24 m: entrapped air / wetting limb); 25 Sep-4 Oct 2021 the WT falls 7 cm/d and
       theta lags (0.325 at 0.71 m; 0.325 at 0.60 m in Nov 2021). Only drying limbs after a wet plateau compare.
    c) RELIABILITY CRITERIA for site data (all four must hold):
         R1 one estimate from all starts (spread < 0.2 cm);
         R2 the PINN's own error: distance to the referee (solver inversion, same data and soil) <= 1 cm rms;
         R3 accuracy: vs the observed WT over the whole window RMSE <= 3 cm, |bias| <= 2 cm, r >= 0.98
            (3 cm = what the site allows: 1.6 cm is the distance of the 5-day spline from the hourly record, ~2 cm
            the repeatability of the sensor reading in b));
         R4 it repeats: R1-R3 on a second held-out window with the same protocol.
       Uncertainty: the posterior sigma (0.2 cm) has no soil error in it. Reported next to it: the distance between
       the estimates with independently calibrated soils (previous recession, other year), not a pass/fail.
    d) PROTOCOL "previous recession": soil calibrated on the last recession with a known WT BEFORE the test window
       (scripts/calibrate_soil.py, all starts, theta range must contain the test's), then the adopted PINN method
       unchanged (wide stage, 4 members -> inversion -> 4 refinement rounds with --coarse -> value-corrected loop
       -> score). Nothing is fitted on the test window. Test window unchanged, T22 = 26 Nov 2022 12:00 + 18 d, so
       the soil is the only difference from finding 72. Calibration window: configs/real_ona_oct22.yaml,
       17 Oct 2022 12:00 + 15 d (WT 0.39 -> 0.89 m, theta 0.351 -> 0.142, contains T22's 0.348 -> 0.147; no rain;
       LE missing 17-19 Oct, filled with the mean diurnal cycle as everywhere).
       DISCLOSED: the protocol was chosen after the other-year soil left 6-10 cm, and T22's WT had been seen; the
       other-year results stay in the report next to it.
    e) CPU checks before any GPU time: (1) calibration starts agree; (2) FEM with the OBSERVED WT on T22 with the
       new soil (other-year soil: theta rmse 0.0273, bias +0.024); (3) the same check of the two existing soils on
       the October window (how far each soil carries: output/real_ona_oct22_soilnov21, _soilnov22); (4) referee on
       T22 with the new soil. The PINN is trained only if (2) is clearly below 0.0273.
    f) Recessions of the record with complete WT and sensor and no rain inside (scan of the daily means, 2021-2022):
         A 23 Sep- 4 Oct 2021  WT 0.05 -> 0.83 m (7 cm/d), theta 0.380 -> 0.218, LE complete
         B 13 Nov-   Dec 2021  theta from 0.369; WT gap 25-26 Nov (31 h)
         C 15 Oct- 1 Nov 2022  WT 0.34 -> 0.89 m, theta 0.362 -> 0.142; LE missing 17-19 Oct
         D 21 Nov-14 Dec 2022  WT 0.08 -> 0.87 m, theta 0.356 -> 0.146
       (29 Sep-11 Oct 2022 has no LE at all: not usable.) Four recessions, two per year: all there is.
       Data sets (scripts/make_real.py): A configs/real_ona_sep21.yaml (25 Sep 2021 12:00 + 9 d, theta 0.379 ->
       0.221), C configs/real_ona_oct22.yaml (17 Oct 2022 12:00 + 15 d, spline vs hourly record 1.6 cm).
    g) Check e2/e3, FEM with the OBSERVED WT and the soil of another recession (theta at the sensor, rmse / bias):
         soil \ window     A Sep 2021      B Nov 2021      C Oct 2022      D Nov-Dec 2022
         S_B (Nov 2021)    0.0385/-0.036   0.0046 (fit)    0.0161/+0.013   0.0273/+0.024
         S_D (Nov 2022)    0.0563/-0.051   0.0273/-0.013   0.0073/-0.003   0.0043 (fit)
       -> C and D (same season, 5 weeks apart) carry each other's soil at 0.007, the level of the fit itself; the
       other year is 2-4 times worse, and the fast recession A is reproduced by neither soil (the sensor lags a WT
       that falls 7 cm/d: 0.04-0.06 too dry in the model).
    h) Soil of October 2022, S_C (output/soil_ona_oct22, start 3, 31 solves): theta_s 0.35107, theta_r 0.12610,
       alpha 1.6607, n 10 (ON THE BOUND, as S_D), Ks 2.0511e-06; fit 0.0051, r 0.998. S_D has alpha 1.92 with
       Ks 4.1e-07: the two trade against each other on a recession (same forward check, g). Starts 0-2 running
       (jobs 15488766 / 15488767 / 15488773): the soil counts as calibrated only when they agree.
    i) Second held-out window of the same season (for R4), T22o = output/real_ona_oct22_cal22nov_13d: 17 Oct 2022
       12:00 + 13 d (cut where the sensor leaves S_D's range, 0.1524), 6 coefficients, soil S_D of the LATER
       recession ("nearest recession", backwards in time; an estimate for October could not have been made in
       October, it tests the transfer, not a forecast). Forward check 0.0085 / -0.0041, r 0.995; hydrostatic
       0.050. Full method: hpc/real_full.sh rf22o (new driver: param_pipeline.sh with 4 rounds, then dc_loop.sh,
       flags of rf22 / rf21b); referee output/referee/cal22novon22oct_s0..3 (jobs 15489045-15489049).
    j) NEXT PROTOCOL, written before its results: "leave one recession out". One soil fitted to ALL recessions
       with a known WT except the tested one (scripts/calibrate_soil.py now takes several windows after --data and
       fits one soil to all hourly values; one window: as before). No choice of calibration window is left, and
       the fit residual by window says how well ONE soil explains the site at all. Tests: D <- A+B+C
       (output/soil_ona_loo_nov22), B <- A+C+D (soil_ona_loo_nov21), C <- A+B+D (soil_ona_loo_oct22),
       A <- B+C+D (soil_ona_loo_sep21); start 3 first (jobs 15489540 / 15489586 / 15489587 / 15489613), start 0
       for D (15489614). Regression of the edit: --data output/real_ona_nov22 --start 3 must return S_D
       (output/soil_chk_nov22, job 15489539; same starting soil and starting rmse 0.006445 so far).
       DISCLOSED: three protocols now (other year, previous / nearest recession, leave one out); every one is
       reported, none is chosen on the test window's forward check.
    k) Previous-recession test, set-up (14:05): T22 with the October soil S_C = output/real_ona_nov22_caloct22
       (26 Nov 2022 12:00 + 18 d, 7 coefficients; sensor 0.147-0.348 inside S_C's 0.1435-0.3521). Check e2: FEM with
       the OBSERVED WT gives theta rmse 0.0103, bias +0.0088, r 0.9975 (other-year soil S_B: 0.0273 / +0.024) ->
       clearly below, the PINN is trained: hpc/real_full.sh rf22c (logs/full_rf22c.log), referee
       output/referee/caloct22on22_s0..3 (jobs 15490072 / 15490073 / 15490076 / 15490099).
       The model is 0.01-0.02 too wet on the steep limb (days 3-10) -> expected estimate ~2 cm too deep there;
       days 0-2.5 the sensor reads saturation (WT above ~0.5 m): no information on the WT in that part.
    l) What each retention curve says with NO dynamics (hydrostatic profile, zb = 0.1 m + |h(theta)|, the 13 daily
       pairs of T22 in b; awk on the login node, context only, nothing of the method uses it):
         S_C (Oct 2022)   est - obs  +4.4 +2.0 +0.7 +1.3 +1.4 +1.2 -1.6 -1.2 +1.0 +0.6 +0.9 +0.0 +1.1   rmse 1.7 cm
         S_B (Nov 2021)              +6.1 +4.5 +3.6 +4.7 +5.1 +5.0 +2.3 +2.6 +4.7 +4.2 +4.2 +3.0 +3.0   rmse 4.2 cm
         S_D (Nov 2022, own window)  -1.5 -4.5 -6.3 -6.1 -6.5 -7.0 -10. -9.8 -7.8 -8.2 -8.0 -9.0 -7.8   rmse 7.4 cm
       -> S_C and S_B are close to a static relation between sensor and WT (S_B with the +4 cm of the other year,
       the bias the PINN and the referee found in finding 72: +5.7). S_D is NOT: its curve alone puts the WT 7 cm
       too shallow on its own window; it matches the sensor through slow drainage (Ks 4.1e-07, five times lower
       than the other three recessions: S_A 1.6e-06, S_B 2.4e-06, S_C 2.1e-06; alpha 1.92 against 1.43-1.66).
       With such a soil the sensor follows the drainage rate, not the WT: the inversion is badly conditioned.
       S_D came from ONE start (3); its other starts were never finished -> it does not meet the rule "accepted
       only if the starts agree". Soil of Sep 2021, S_A (output/soil_ona_sep21 start 3): theta_s 0.3686,
       theta_r 0.0984, alpha 1.433, n 10 (bound), Ks 1.62e-06, fit 0.0042.
    m) Full transfer matrix (14:20; FEM with the OBSERVED WT, theta at the sensor rmse / bias; windows as calibrated:
       A 25 Sep 2021 + 9 d, B 13 Nov 2021 + 30 d, C 17 Oct 2022 + 15 d, D 23 Nov 2022 + 20 d; soils of start 3):
         soil \ window     A Sep 2021      B Nov 2021      C Oct 2022      D Nov-Dec 2022
         S_A (Sep 2021)    0.0042 (fit)    0.0301/+0.025   0.0528/+0.049   0.0617/+0.055
         S_B (Nov 2021)    0.0385/-0.036   0.0046 (fit)    0.0161/+0.013   0.0273/+0.024
         S_C (Oct 2022)    0.0546/-0.049   0.0149/-0.008   0.0051 (fit)    0.0099/+0.008
         S_D (Nov 2022)    0.0563/-0.051   0.0273/-0.013   0.0073/-0.003   0.0043 (fit)
       -> ORDERED IN TIME: a soil from an earlier recession is too wet on every later window, a soil from a later one
       too dry on every earlier window, and the misfit grows with the distance in time (neighbours C-D 0.007-0.010,
       B-C 0.015-0.016, B-D 0.027, A-D 0.06). The sensor-WT relation of the site DRIFTS (at the same WT the sensor
       reads lower as time goes on: sensor, well datum or soil), it is not a random year effect. Consequences:
       the nearest recession in time is the soil to use (the previous one, in operation); pooling recessions that
       are far apart averages over the drift; A (fast, 7 cm/d, and first in time) is the far end.
    n) Value-corrected loops of the other-year tests, iteration 1 (finding 72): rf22dc step / sigma max 1.44
       (segments 0.32 1.12 0.40 0.31 1.44), cost 0.282, vs observed 6.19 cm (bias +5.58), all 5 starts one estimate;
       rf21bdc 2.26 (0.58 0.23 0.24 1.64 2.26), cost 0.346, vs observed 10.0 cm (bias -9.0), coefficients 0 and 6 on
       the box. Both continue (limit 4 iterations).
    o) New check of a calibrated soil, written before its results (14:30): IN-SAMPLE INVERSION. The calibration fits
       the sensor with the WT given; the method needs the opposite direction. A soil that matches the sensor through
       slow drainage (S_D, l) can pass the forward fit and still leave the WT undetermined. Check, CPU and solver
       only (calibration side, no PINN run reads it): the referee inversion on the calibration window itself with
       its own soil must return the WT that was used to calibrate. Sets output/real_ona_<win>_own, referee
       output/referee/own_<win>_s0 (hydrostatic start) and _s1 (flat start), win = sep21, nov21, oct22, nov22
       (jobs 15490763-15490818). If S_D fails it and S_C, S_B pass, the rule for a usable soil becomes: starts agree
       AND the in-sample inversion returns the WT (stated here before the numbers).
       Remaining starts of the Nov 2022 calibration submitted (15490546 / 15490547 / 15490596).
    p) REFEREE (solver inversion, scoring only) on the two tests of the nearest-recession protocol (14:35):
       T22o with S_D (output/referee/cal22novon22oct_s0/_s1, hydrostatic and flat start, same minimum, cost 0.378):
         zb RMSE 9.70 cm (inner 4.83), bias -3.4, max 25.1; coefficients 0 and 5 ON THE BOX
         (+1.183 -1.267 -0.246 +0.485 -0.760 +1.183). est - obs by day (cm): +7.0 +2.7 -1.2 -7.2 -4.3 -1.8 +3.7
         +6.4 +7.1 +0.6 -8.5 -16.4 -23.9 -23.6: the estimate RISES after day 8 while the observed WT falls.
         Start constant:1.0 stays at its start (cost 789, no sensitivity); constant:0.3 slow (cost 57 at it 5).
         -> with S_D the data do not determine the WT (as l said): rf22o cannot meet R3, whatever the PINN does.
       T22 with S_C (output/referee/caloct22on22_s0/_s1, same minimum, cost 0.2797, 7 iterations, no coefficient
         on the box, theta rmse 0.0026, sigma median 0.17 cm): zb RMSE 4.37 cm whole window (inner 4.92), bias
         +2.81, max 18.8. est - obs by day (cm), d0..d18: +18.8 +10.9 +6.3 +6.0 +4.3 +3.2 +5.0 +4.1 +4.3 +0.2 -0.0
         +3.9 +2.6 +1.8 +2.2 +1.0 +1.2 +1.9 -1.1.
         Hours with the sensor below saturation by > 3 sigma (359 h, from day 3.08): RMSE 2.44 cm, bias +1.69 cm.
         -> the whole-window miss is days 0-3: the sensor reads saturation (theta_obs >= theta_s - 3 sigma), which
         only says "WT above the air-entry depth"; the estimate sits at that depth (0.52 m), the WT is at 0.34 m.
         From day 3 on the solver inversion is inside the R3 thresholds (2.4 cm, bias +1.7). R3 as written (whole
         window) is NOT met by the referee, so it cannot be met by a PINN that reproduces it; both scores are
         reported, R3 is not changed after the fact.
       Hydrostatic reading of the Oct 2022 daily pairs (calibration window of S_C; est - obs, rmse / bias cm):
         S_C 1.6 / -1.3; S_B 2.0 / +1.7; S_A 7.0 / +6.9; S_D 9.4 / -9.4 (S_D reads ~10 cm too shallow everywhere).
    q) Calibration starts, state 14:40. Regression check of the multi-window calibrate_soil.py on Nov 2022 alone
       (output/soil_chk_nov22/start3.json): identical to S_D (rmse 0.0043278, same five values) -> the edit did not
       change the single-window result. Other starts on the short windows end in a DIFFERENT, much worse basin
       (broad retention curve): Oct 2022 start 0 rmse 0.0160 (theta_s 0.458, n 3.40) against 0.0051 of start 3;
       Sep 2021 start 0 rmse 0.0207 (theta_s 0.447, n 1.72) against 0.0042. Starts 1, 2 are heading the same way
       (0.020-0.023). Only the 30-day window (Nov 2021) brought all starts to one soil. So on the short windows
       "the starts agree" does NOT hold: the soil taken is the lowest-misfit one (start 3, 3-5 x lower), and the
       in-sample inversion (o) is the check that decides whether it can be used.
    r) Decisions 14:42, written before the results.
       - rf22o (T22o with S_D) STOPPED (driver group killed, GPU jobs 15489060-62 cancelled; w1 had finished): the
         solver inversion with S_D misses by 9.7 cm with coefficients on the box (p), so no PINN on that data set
         can meet R3; the GPUs go to the tests below. Also S_D is the LATER recession, which protocol d) does not
         allow ("the last recession BEFORE the test window"); i) departed from d) and is withdrawn.
       - Protocol d) applied to EVERY recession that has a previous one (all three are reported):
           D (Nov 2022) with S_C (Oct 2022)   rf22c, running; referee caloct22on22 (p)
           C (Oct 2022) with S_B (Nov 2021)   NEW: rf22ob on output/real_ona_oct22_soilnov21 (17 Oct 2022 12:00 +
                                              15 d, 6 coefficients; S_B's range 0.110-0.376 contains 0.144-0.352;
                                              forward check 0.0161 / +0.0133, m); referee calnov21on22oct_s0..3
                                              (jobs 15491620-27); hpc/real_full.sh as rf22c
           B (Nov 2021) with S_A (Sep 2021)   S_A's range 0.221-0.379 -> window cut at day 11 (theta 0.2256):
                                              output/real_ona_nov21_calsep21_11d, referee calsep21on21_s0/_s1
                                              first (jobs 15491647-51); forward check on the full window was
                                              0.0301 / +0.025 (m), so a miss is expected; the PINN run follows
                                              only if the solver inversion itself is inside R3.
       - In-sample inversions (o), state 14:40, not finished: Oct 2022 with S_C zb RMSE 3.8 cm at iteration 2
         (cost 0.336), Sep 2021 with S_A 6.4 cm at iteration 4 (cost 0.695). Both windows START with the sensor at
         saturation, so the whole-window number contains the undetermined days; to be read by day and for the hours
         below saturation.
    s) IN-SAMPLE INVERSIONS (check o), solver only, each soil on its own calibration window (14:55). Whole window /
       hours with the sensor below saturation by > 3 sigma (vs the observed WT, cm):
         S_A on Sep 2021  (own_sep21_s1)  6.44, bias +2.97, max 20.7 | from day 3.4: rmse 1.37, bias -0.00, r 0.975;
                                          est - obs by day +20.7 +10.5 +3.2 -0.9 -1.4 -1.2 -0.8 -0.6 +0.2 -0.8
         S_C on Oct 2022  (own_oct22_s0)  3.59, bias +1.14, max 12.5 | from day 3.6: rmse 1.63, bias -0.01, r 0.981;
                                          +12.5 +8.8 +4.7 -2.4 -1.3 -1.4 +0.7 +0.8 +1.2 -1.4 -2.7 -2.2 -3.3 -1.0
                                          -0.2 +0.5; coefficient 0 on the box
         S_D on Nov 2022  (own_nov22_s0)  7.03, bias +2.20, max 21.8 | from day 6.1: rmse 3.40, bias -0.68, r 0.933;
                                          +21.8 +18.5 +15.4 +9.4 +4.4 +0.8 +1.0 +0.1 +0.4 +3.9 +4.1 +4.0 -1.4 -3.4
                                          -1.1 -3.3 -3.9 -2.4 -1.5 +2.6 +9.3; coefficients 0 and 6 on the box,
                                          sigma 11 -> 233 cm on the last four days
         S_B on Nov 2021  running (30 d, 9 coefficients).
       -> as stated in o) before the numbers: S_D does not return its own WT (3.4 cm with r 0.93 below saturation,
       no information at the end of the window) and is NOT a usable soil; S_A and S_C return theirs to 1.4-1.6 cm
       with zero bias once the sensor is below saturation. Rule for a usable soil from here on: lowest-misfit
       calibration AND in-sample inversion within 2 cm, r >= 0.975 below saturation.
       The same picture in every window: on the first 3-6 days the sensor is at saturation and the estimate sits
       at the air-entry depth (0.5 m), 10-20 cm below a WT that is still at 0.2-0.4 m. The posterior sigma there is
       1-7 cm, far too small: it is the linearised sigma at the estimate and knows nothing of the flat side. The
       estimate on those hours is a BOUND (WT at or above it), which is how it has to be reported.
    t) rf21bdc iteration 2 (14:42): step / sigma max 0.72 (segments 0.03 0.04 0.21 0.40 0.72), cost 0.342, all starts
       one estimate, vs observed 10.1 cm (bias -9.0): converging to the solver's answer for that soil (S_D, not
       usable, s). Continues.
       Referee, B with S_A (calsep21on21, iteration 2): 14 cm from the observed WT at cost 0.28-0.35 -> the
       previous-recession protocol FAILS on B, as the forward check (0.030 / +0.025) said it would.
    u) Whole record (15:10; daily means of the half-hourly file, 2021-2022, awk on the login node; WT known ->
       calibration-side reading, no inversion):
       - WT depth (m) at the same sensor reading, every drying episode of the record
           episode                 theta 0.30   theta 0.22
           A   25 Sep-4 Oct 2021     0.76         0.83
               13-14 Oct 2021        0.69          -
               6-11 Nov 2021         0.66          -
           B   13 Nov-14 Dec 2021    0.68         0.77
               1-11 Oct 2022         0.63         0.755
           C   17 Oct-1 Nov 2022     0.66         0.74
               11-19 Nov 2022        0.65          -
           D   26 Nov-14 Dec 2022    0.63         0.72
         Seven of the eight lie within 0.63-0.69 (0.72-0.77): the relation repeats to about +-3 cm, within 2022 to
         +-1.5 cm. A is 8-10 cm deeper than every other episode, including the one ten days after it with the same
         sensor installation (13-14 Oct 2021) -> A is the outlier: S_A cannot transfer (t), and no soil transfers
         to A. It is the only drainage that starts from weeks of standing saturation and has complete forcing.
       - Sensor with the WT above it (0.04-0.10 m): 0.386-0.391 in Sep 2021 and in Sep 2022 (no drift of the
         sensor). After a fast rise from a dry profile it reads lower: 21-22 Nov 2022 WT 0.65 -> 0.08 m in a day on
         11.7 mm of rain, reading 0.356; 11 Nov 2022 0.12 m, 0.369 (entrapped air). The "saturated" reading
         depends on the event by 0.03.
       - Above the knee the reading falls gently with the WT (D: 0.358 at 0.17 m, 0.348 at 0.35 m, 0.344 at 0.47 m;
         A: 0.376 at 0.20 m, 0.359 at 0.60 m). A van Genuchten curve with n at 10 (flat, then a sharp knee) has no
         such slope and reads that range as "WT at the knee": days 0-3 of the D test (p). This is model structure
         and hysteresis, not a sensor offset, so a misfit without the mean (considered) would not remove it: not
         run. It is what the bound for theta_obs >= theta_s - 3 sigma (s) is for.
       - 1-11 Oct 2022 has no forcing: LE is missing from 26 Sep 12:00 to after 13 Oct (tower down after the
         193 mm storm of 28 Sep), and the WT record stands at 0.324 m through that storm. Not usable as a window.
         Soil moisture is missing 14-26 Oct 2021.
       - The WT record itself: jumps of 5-40 cm within hours in Sep 2022 without rain (26 Sep 06:00 0.54 m,
         12:00 0.15 m), +-2-3 cm from day to day inside the windows. R3's 3 cm is at the repeatability of the site.
       Expected accuracy of a test, RULE WRITTEN BEFORE THE RUNS (hindcast): for a test with soil S_k,
         sigma_hind = below-saturation RMSE of the inversion, with S_k, of the most recent earlier recession with
         complete forcing other than the calibration one (its WT is known: calibration side, solver referee),
       on a window whose soil moisture stays inside S_k's calibration range. It is reported next to the realised
       error, and the uncertainty of the estimate is sqrt(sigma_post^2 + sigma_hind^2). No number is tuned.
         D with S_C  -> B (15 Nov 2021 + 17 d, the window of T21b) inverted with S_C
         C with S_B  -> A (25 Sep 2021 + 9 d) inverted with S_B   (A is the outlier above: expect about 10 cm)
    v) Results that arrived after (u) (15:45). Solver inversions = referee, scoring/calibration side only.
       - Protocol (d), referee, all starts finished or at the same estimate:
           test / soil    starts   cost     whole window              below saturation
                                           rmse  bias  max    r      hours from day  rmse  bias   r
           D / S_C        4 of 4   0.2797  4.37  +2.81 18.8  0.985   359   3.08      2.44  +1.69  0.983
           C / S_B        4 of 4   0.3345  4.41  +3.37 12.3  0.987   275   3.58      3.23  +2.69  0.975
           B / S_A        2 of 2   0.2767  14.46 +11.92 35.2  -      -     -         11.34 +9.91  0.909
         (cm). The far starts (1.0 m, 0.3 m) reach the estimate of the hydrostatic ones on D and on C (C: s2, s3 at
         cost 0.33445 after 13-15 iterations, still polishing): the solver inversion is start-independent, so the
         distance to the well is the model's (soil, structure), not a local minimum.
         C / S_B by day (cm): +12.3 +9.0 +5.5 -0.8 +1.0 +1.6 +4.2 +4.5 +4.9 +2.1 +0.6 +0.9 -0.5 +1.0 +0.5 -0.9,
         coefficient 0 on the bound of the box. R3 in range: D meets it (2.44 / +1.69 / 0.983), C misses it
         narrowly (3.23 / +2.69 / 0.975), B / S_A fails (A is the outlier of (u)).
       - In-sample, S_C on C: second start = first (3.59 cm whole; below saturation 1.63, bias -0.01, r 0.9805).
       - Hindcast for the C / S_B test (A, 25 Sep 2021 + 9 d, inverted with S_B; two starts, one estimate):
         whole window 9.75 cm, bias -8.73, cost 5.13, coefficients 0 and 1 on the bound; below saturation (99 h
         from day 4.92) 6.96 cm, bias -6.75, r 0.935  ->  sigma_hind(C | S_B) = 7.0 cm. Realised on C: 3.23 cm.
         The hindcast is pessimistic here because its only available window is the outlier; it is kept as written.
       - Hindcast for the D / S_C test (B, 15 Nov 2021 + 17 d, inverted with S_C): two starts at cost 0.3269,
         3.49 cm over the whole window, not finished; the in-range number follows.
       - Calibration starts. Oct 2022: starts 0/1/2 stop at rmse 0.0160 / 0.0203 / 0.0196 (n 3.1-3.4, theta_s
         0.46-0.54), start 3 at 0.0051 (n at its bound): one basin is lower by a factor 3-4, the others are not
         competitive. Sep 2021: 0.0207 / 0.0224 / 0.0215 against 0.0042 for start 3: the same picture. The rule of
         (s), lowest misfit AND in-sample inversion within 2 cm, selects start 3 in both.
         One soil for B + C + D (A left out; start 3): theta_s 0.3501, theta_r 0.1193, alpha 1.635, n 9.33,
         Ks 1.97e-6; rmse 0.0115; by window B 0.0122 / -0.0076, C 0.0066 / +0.0022, D 0.0133 / +0.0097. The three
         episodes that share one theta-WT relation (u) share one soil to 0.012, and it is S_C within 0.01 in every
         parameter but theta_r. Not used for a test (it contains every test window).
       - PINN, D with S_C (rf22c), first inversion with the wide members only (inv0, no refinement yet):
         one estimate from all starts (spread 2e-9 m), cost 0.2777 (referee 0.2797); whole window 5.24 cm, bias
         +3.98; below saturation 4.02 cm, bias +3.10, r 0.977.
           by day, estimate - observed (cm):  +18.1 +11.1 +6.5 +5.5 +3.0 +1.3 +3.0 +2.4 +3.4 +0.4 +1.3 +6.4 +6.1
                                              +5.8 +6.2 +4.5 +4.5 +5.8 +4.9
           by day, PINN - referee (cm):       -0.2 +0.3 -0.1 -0.8 -1.6 -2.0 -1.9 -1.3 -0.4 +0.7 +1.9 +3.0 +3.8
                                              +4.1 +3.8 +3.4 +3.5 +4.7 +6.0        (rms 2.57 cm)
         The wide operator is within 0.3 cm of the solver inversion on days 0-2 and drifts deep by 3-6 cm in the
         dry half, the same pattern as every earlier inv0 (T21b, T22): the refinement rounds and the value
         correction are what remove it. Nothing is concluded from inv0.
       - Scoring: scripts/score_real.py --sigma-hind <m> (optional; default output byte-identical, job 15493875)
         adds sqrt(sigma_post^2 + sigma_hind^2): share of the hours below saturation within 1 and 2 sd, share of
         the saturated hours with the observed WT not below estimate + 2 sd, and the +-2 sd limits in the figure
         as two thin dotted lines.
    w) PINN against the solver inversion on site data, after two value-correction iterations (16:00; scoring only):
           run (test / soil)             starts spread   PINN - referee, whole window        vs observed (PINN | referee)
           rf22dc_it2  (D / S_B, 2021)   2.5e-8 m        rms 0.19 cm, max 0.89 cm, r 0.9999  6.16 | 6.13 cm
           rf21bdc_it2 (B17 / S_D, 2022) 1.3e-8 m        rms 2.67 cm, max 5.97 cm            10.07 | 9.17 cm
       - D with the 2021 soil: R1 and R2 are met on site data. All five starts give one path and it is the
         solver's to 2 mm; the 6 cm to the well is what the soil of the year before costs (both inversions alike).
         output/real_ona_nov22_cal21/result_rf22dc_it2.{json,png}
       - B with the 2022 soil (S_D, which fails its own in-sample test, (s)): the two agree to 0.6 cm on days 0-8
         and part on days 9-17 (-1.8 -3.7 -5.4 -5.9 -4.8 -2.5 +0.2 +2.7 +3.7 cm), with coefficients 3-5 moved
         against each other (+0.05 -0.05 -0.16 | -0.00 +0.18 -0.33) at equal cost (0.34235 | 0.34255) and both
         end coefficients on the box: a flat direction of a problem whose soil does not fit (10 cm from the well
         either way). The loop is at step/sigma 0.72 and continues. Read as: where the model fits the data
         (D) the PINN is the solver inversion; where it does not, neither is determined.
         output/real_ona_nov21_cal22_17d/result_rf21bdc_it2.{json,png}
    x) Hindcast for the D / S_C test, finished (16:05): B (15 Nov 2021 + 17 d, theta 0.152-0.349 inside S_C's
       calibration range 0.144-0.352) inverted with S_C by the solver: cost 0.3269, coefficient 6 on the bound;
       whole window 3.49 cm, bias -2.44, max 7.2, r 0.979; below saturation (382 h from day 1.12) 3.49 cm, bias
       -2.80, r 0.979  ->  sigma_hind(D | S_C) = 3.5 cm. Realised on D (referee): 2.44 cm in range.
           test / soil   sigma_hind (rule of u)   realised in range (referee)
           D / S_C       3.5 cm  (B with S_C)      2.44 cm, bias +1.69
           C / S_B       7.0 cm  (A with S_B)      3.23 cm, bias +2.69
       Both realised errors are inside 1 sigma_hind. The sign differs between hindcast and test in both pairs
       (-2.8 | +1.7, -6.8 | +2.7): the soil-transfer error is not a fixed offset that could be subtracted.
       - The C / S_B far starts have stopped at the same estimate (cost 0.334452 x 4, 4.41 cm); S_D in-sample,
         second start = first (7.03 cm whole window; below saturation 3.40 cm, r 0.933: still fails (s)).
       - Soils fitted to three recessions, the fourth left out (start 3; rmse by window in theta):
           left out   theta_s  theta_r  alpha   n     Ks        A       B       C       D
           A          0.3501   0.1193   1.635   9.33  1.97e-6   -       0.0122  0.0066  0.0133
           C          0.3565   0.1146   1.700   9.82  4.36e-7   0.0344  0.0073  -       0.0223
           D          0.3578   0.1136   1.629   10.0  6.60e-7   0.0320  0.0058  0.0200  -
         Whenever A is in the fit it keeps a bias of -0.03 and drags the 2022 windows to +0.02: A does not share
         a soil with the rest (u). (The run leaving B out is not finished.)
       - Reading note for every "by day" row of estimate - observed in (p)-(v): the value at the start of each day
         of the window (12:00), not a daily mean; the scores (rmse, bias, r) are over all hours. The well has a
         daily cycle of 1-2 cm, so the rows read about 1.5 cm more negative than the bias on B.
    y) Loop of the B / S_D test, iteration 3 (rf21bdc_it3, 16:30), and solver-side jobs that ended (16:40):
       - step / sigma max 2.26 -> 0.72 -> 0.73 over iterations 1-3 (segments 0.02 0.08 0.39 0.62 0.73): the step
         does not shrink. PINN - referee over the window 2.79 -> 2.67 -> 2.45 cm rms; the offset of the forward
         PINNs from the operator grows (rms 0.0021 -> 0.0023 -> 0.0028); coefficients 3-5 move towards the
         referee's by a few 0.01 per iteration. Iteration 4 (the last) is queued.
         HYPOTHESIS (not tested): the value correction replaces the operator's value at the estimate and keeps
         its Jacobian, so its fixed point solves J_op^T r = 0 and the solver's J^T r = 0. With a residual r that
         is not noise (cost 0.34, a soil that does not fit) the two differ by (J^T J)^-1 (J_op - J)^T r, large
         only along directions the data hardly constrain (posterior sd on days 9-17: PINN 0.3-23 cm, solver
         0.8-68 cm). On D with the 2021 soil the same loop ended 0.19 cm from the solver (w).
         Test of it, if the two primary runs (rf22c, rf22ob) stall the same way: the chord loop with the
         Jacobian from forward PINNs on a finite-difference design (hpc/fd_set.sh, --fwd-fd; findings 60-61).
         Not started: the baseline is not shown to fail on a soil that passes its in-sample test.
       - Hindcast B with S_C, second start (hydrostatic_flat): 3.494 cm, cost 0.32688 = first start (3.494 cm,
         0.32688) -> sigma_hind(D | S_C) = 3.5 cm stands on two starts.
       - Calibration of D, start 0 (the data set's own soil) = start 3: theta_s 0.35142, theta_r 0.12991,
         alpha 1.9164, n 10 (bound), Ks 4.127e-7, rmse 0.00433 from both. Starts 1 and 2 still run (rmse 0.039
         and 0.008 so far). S_D is reproducible and still fails its in-sample inversion (s).
       - Queue: GPU partitions are full; rf22c round 1 has one member training since 16:25, three pending since
         15:43-16:08; rf22ob has three of four wide members.
    z) Loop of the D / S_B (2021 soil) test CONVERGED at iteration 3 (rf22dc_it3, 16:38; stop rule step < 0.2 sd):
       step / sigma max 1.44 -> 0.62 -> 0.07 (segments 0.03 0.07 0.04 0.02 0.02), forward offset rms 0.0013,
       cost 0.28431 (solver 0.28540 from four starts), start spread 2.3e-8 m (5 starts).
           PINN - solver inversion, whole window: rms 0.10 cm, max 0.31 cm, r 0.99997
           by day (cm): -0.1 +0.0 +0.1 +0.1 +0.0 -0.0 -0.0 +0.0 +0.0 +0.0 -0.0 -0.0 -0.1 -0.0 +0.0 +0.1 +0.2 +0.3 +0.3
           vs observed, whole window:      PINN 6.15 cm, bias +5.51, r 0.988 | solver 6.13 cm, +5.47, r 0.987
           vs observed, below saturation:  PINN 5.10 cm, bias +4.78, r 0.982 | solver 5.07 cm, +4.73, r 0.982
       -> on this window the full PINN method (wide members, four rounds, three value-correction iterations)
          ends on the solver's estimate to 1 mm from every start: R1 and R2 met, with the loop's own stop rule
          and no truth used. R3 is not met with the soil of the year before (5-6 cm), for the PINN and the
          solver alike. output/real_ona_nov22_cal21/result_rf22dc_it3.{json,png} (job 15496703).
    aa) Primary PINN runs, intermediate inversions (17:30; nothing is concluded before the loops end):
           run / stage                 starts   cost     vs observed: whole | below saturation     PINN - solver
           rf22c  inv0 (wide)          one      0.2777   5.24 / +3.98      | 4.02 / +3.10 / 0.977   2.57 cm rms
           rf22c  inv1 (round 1)       one      0.3131   4.61 / +3.43      | 2.97 / +2.46 / 0.983   1.02 cm rms, max 3.7
           rf22ob inv0 (wide)          one      0.3240   4.51 / +3.42      | 3.33 / +2.77 / 0.984   1.28 cm rms, max 2.4
           solver, D / S_C             one      0.2797   4.37 / +2.81      | 2.44 / +1.69 / 0.983
           solver, C / S_B             one      0.3345   4.41 / +3.37      | 3.23 / +2.69 / 0.975
        rf22c by day, PINN - solver after round 1 (cm): +0.0 -0.1 -0.3 -0.3 -0.3 -0.2 -0.0 +0.2 +0.4 +0.7 +1.0
        +1.2 +1.3 +1.2 +1.0 +0.9 +1.4 +2.7 +3.7 (deep at the dry end, as in every earlier round 1). Round 2 of
        rf22c and round 1 of rf22ob are training.
    ab) Calibrations that ended (17:30; files went in with commit c7ce04e, not named in its message):
        - D, start 2 (sandy loam) = starts 0 and 3: theta_s 0.35142, theta_r 0.12991, alpha 1.9164, n 10 (bound),
          Ks 4.127e-7, rmse 0.00433 -> S_D is the same from three of four starts (start 1 still runs, 0.039).
        - Pooled soil with B left out (A + C + D, start 3; output/soil_ona_loo_nov21): theta_s 0.3724,
          theta_r 0.0588, alpha 2.547, n 10 (bound), Ks 1.0e-7 (bound); rmse by window A 0.0304 (bias -0.026),
          C 0.0112, D 0.0150 (bias +0.011). Completes the leave-one-out table of (x): every pooled fit that
          holds A keeps A's bias of -0.03 and two parameters end on bounds here. No pooled soil is used.
    ac) IN-SAMPLE INVERSION OF S_B, finished (check o; solver only; own_nov21_s0, 21 iterations, 3.2 h; the second
        start is at the same cost 0.6644 and still iterating): Nov 2021, 30 d, 9 coefficients, cost 0.6643,
        coefficient 0 on the box. Whole window 5.74 cm, bias +0.73, max 23.7; below saturation (645 h from day 3.2)
        3.74 cm, bias -0.62, r 0.9635  ->  S_B FAILS the rule of (s) as written (2 cm, r >= 0.975).
        By day (start of day, cm): +23.4 +14.2 +6.2 +0.6 -1.9 -2.0 -1.1 -2.1 -2.6 -0.2 -3.0 -2.5 +0.9 -1.5 -2.2
        -1.7 -3.0 -3.0 -4.0 -4.2 -4.1 -2.3 -1.9 -0.3 +0.8 +1.0 +0.8 -2.2 -6.9 -12.3 -20.6; posterior sd 0.1-0.6 cm
        to day 23, then 1.1 2.9 5.5 8.1 10.8 16.9 29.7 cm: the miss is the last five days, where the inversion
        itself reports no information. Diagnostic only (the rule is NOT changed after the fact), the same scores
        without the last five days of each window:
            soil / window      below saturation          without the last 5 d
            S_A  Sep 2021      1.37, +0.00, r 0.975      1.36, +0.31, r 0.922  (15 h)
            S_B  Nov 2021      3.74, -0.62, r 0.964      1.96, +0.12, r 0.988  (525 h)
            S_C  Oct 2022      1.63, -0.01, r 0.981      1.62, -0.07, r 0.968  (155 h)
            S_D  Nov 2022      3.40, -0.68, r 0.933      2.82, -0.66, r 0.941  (215 h)
        Consequences, stated before the PINN results of rf22ob:
        - Usable soils by the rule: S_A and S_C only. The C / S_B test (rf22ob) runs on a soil the rule rejects and
          whose hindcast error is 7 cm (u); it is finished and reported as a test of PINN = solver (R1, R2), and
          it cannot count as the second window of R4 whatever it scores against the well.
        - The B / S_A test fails although S_A passes the rule (v): the rule is necessary, not sufficient.
        - So at US-ONA one held-out test has a usable soil and a solver inversion inside R3 below saturation
          (D / S_C). R4 needs a second one, and this site has no further recession with complete forcing.
          Next: look for recession pairs at a second site with the same instruments (US-PAS, same soil order,
          record in fem_validation, read-only), solver side first (calibration, in-sample, hindcast, referee);
          a PINN run there only if the solver inversion is inside R3, as in (r).
    ad) Loop of the B / S_D test, iteration 4 (the last; rf21bdc_it4, 17:57): step / sigma max 0.78 (segments 0.02
        0.06 0.49 0.29 0.78) after 2.26 -> 0.72 -> 0.73: NOT converged in 4 iterations. PINN - solver 2.58 cm rms
        (2.79 -> 2.67 -> 2.45 -> 2.58), max 5.8 cm, all of it on days 9-14 (daily means -1.6 -3.5 -5.1 -5.8 -4.9
        -2.9; days 0-8 within 0.6 cm). Against the well both miss by 10 cm (PINN 10.25 / bias -9.11, r 0.928):
        S_D is the soil that fails its in-sample inversion (s), so this run is no test of accuracy. It stays the
        one case where the loop does not reach the solver's estimate (hypothesis in y, untested; the two primary
        runs decide whether the chord loop is needed).
    ae) SECOND SITE, US-PAS (grazed bahiagrass pasture, 1.2 km north of US-ONA, Spodosol; finding 68: same TDR at
        0.10 m, WT negative below ground, rain stamped 23.5 h late Aug-Dec 2021 -> p_shift_hours -23.5).
        PROTOCOL AND WINDOWS, WRITTEN BEFORE ANY MODEL RUN AT THIS SITE (18:20). Everything as at US-ONA (c, d,
        o, s, u): criteria R1-R3 unchanged, scores for the whole window and below saturation, previous-recession
        soil, usable-soil rule (lowest misfit of the starts AND in-sample inversion within 2 cm, r >= 0.975 below
        saturation), hindcast sigma. Solver side first; the PINN is trained on a test only if the solver
        inversion of that test is inside R3 below saturation (r).
        Record (daily means of the half-hourly file, awk; LE, rain, WT, sensor complete Apr - 20 Dec 2021;
        2022 not usable: nothing Jan - mid Nov, and from 16 Nov the rain record reads 0 while the WT rises by
        0.4-0.8 m three times and the sensor stays at 0.27-0.30 for WT 0.05-0.88 m).
        Recessions of autumn 2021 (WT m / sensor), rain in true time:
          P0  19 Aug 12:00 + 6 d   0.23 -> 0.68 / 0.432 -> 0.31   3 mm on 20 Aug
          P1  25 Sep 12:00 + 9 d   0.24 -> 0.98 / 0.400 -> 0.168  no rain (next: 5 Oct 22:00)
          P2  14 Oct 12:00 + 9 d   0.31 -> 1.02 / 0.423 -> 0.163  no rain (next: 24 Oct)
          P3  14 Nov 12:00 + 20 d  0.24 -> 0.97 / 0.410 -> 0.168  0.5 mm on 18 Nov
        P3 goes on to 14 Dec (1.04 m, 0.135); it is cut at 20 d, where the sensor leaves P2's calibration range
        (d: the calibration range must contain the test's; the cut uses the sensor only, never the WT).
        P1 and P2 end one day before the next rain so that no forcing node holds it.
        Tests (nothing fitted on a test window):
          T_P2 = P2 with S_P1 (soil calibrated on P1);  hindcast: P0 inverted with S_P1
          T_P3 = P3 with S_P2 (soil calibrated on P2);  hindcast: P1 inverted with S_P2
        Configs configs/real_pas_{aug21,sep21,oct21,nov21}.yaml = configs/real_ona_sep21.yaml with the site block,
        the rain shift and the window replaced; same GLDAS pixel for the transpiration fraction (G02101, next to
        both sites). Data sets output/real_pas_<win> (jobs 15500022-15500025).
        What would count for R4: a test at this site with a usable soil whose PINN run meets R1-R3 - disclosed as
        a second site 1.2 km away, not a second season of US-ONA. If no US-PAS soil is usable or the solver
        inversion is outside R3, R4 stays open and is reported so.
    af) 18:45. PINN runs at US-ONA, and US-PAS before any inversion there.
        rf22ob (C with S_B) inv1: cost 0.32914 (solver 0.33445); PINN - solver by day -0.1 -0.1 -0.1 -0.0 +0.0
        +0.1 +0.2 +0.3 +0.4 +0.5 +0.7 +0.9 +1.1 +1.4 +1.7 +1.9, rms 0.74 cm, max 1.91 cm (inv0 1.28 cm); starts
        agree to 2e-15 m; vs well 4.67 / +3.84 / r 0.986, below saturation (275 h) 3.74 / +3.32 / 0.978.
        rf22c (D with S_C, primary) inv2: PINN - solver by day -0.0 +0.0 +0.0 -0.0 -0.1 -0.1 -0.1 -0.1 +0.0 +0.2
        +0.4 +0.7 +0.8 +0.7 +0.5 +0.5 +0.9 +2.4 +3.6, rms 0.74 cm, max 3.60 cm (inv0 2.57, inv1 1.02 cm); starts
        agree to 4e-9 m; vs well 4.52 / +3.20 / r 0.983, below saturation (359 h) 2.74 / +2.15 / 0.983
        (solver 2.44 / +1.69 / 0.983). What is left is the window end (last knot), as in every run before the
        value correction.
        US-PAS data sets (make_real, default Florida soil, hourly): P0 sensor 0.3185-0.4386, WT 0.225-0.679 m,
        ET 19.7 mm, P 3.5 mm; P1 0.1691-0.4000, 0.242-0.986 m, 28.2 mm, 0.2 mm; P2 0.1619-0.4230, 0.305-1.024 m,
        22.7 mm, 0; P3 0.1680-0.4164, 0.239-0.971 m, 31.1 mm, 0.7 mm. Unresolved sensor noise 0.0022-0.0066.
        THE SENSOR HAS TWO LEVELS WHEN WET (hourly record, no model): a reading of 0.41-0.44 that differs by
        event and does not follow the water table (WT 0.23-0.25 m: 0.432 in Aug, 0.400 in Sep, 0.411 in Nov),
        and a level of 0.38-0.40 it steps down to within one hour (P2 day 1.04 -> 1.08: 0.420 -> 0.387; P3 day
        3.5 -> 3.75: 0.411 -> 0.392; P0 flips 0.435 / 0.408 on days 1-3, then 0.394). P1 has no upper level.
        Calibrations (calibrate_soil.py, jobs 15500140-15500148), start 3 finished for both:
          S_P1 (P1): theta_s 0.394148, theta_r 9.7e-7 (bound), alpha 1.393142, n 4.843617, Ks 2.597021e-06;
                     rmse 0.00453, bias 0.0000, max 0.0103; starts 0 and 2 at 0.00456 (same theta_s, alpha, n;
                     Ks 1.7-1.8e-6), start 1 (C&P sand) still far (0.05).
          S_P2 (P2): theta_s 0.429899, theta_r 5e-9 (bound), alpha 1.678704, n 3.180866, Ks 2.661683e-06;
                     rmse 0.00605, max 0.0216; starts 0 and 2 at 0.00605 (same soil), start 1 still far (0.037).
        n is not on its bound here (US-ONA: 10). S_P2 holds P2's upper level (theta_s 0.430 against 0.394).
        The solver side goes on with the start-3 soils; if a start still running ends lower, it is redone.
        WINDOWS BY RULE d, FIXED BEFORE ANY INVERSION AT THIS SITE (the sensor only; as at US-ONA the range is
        checked on both sides, i 'sensor 0.147-0.348 inside S_C's 0.1435-0.3521'). A window starts at the first
        hour at which the sensor is inside the calibration range of the soil used and lasts the largest whole
        number of days that stay inside it:
          T_P2 (S_P1, range 0.1691-0.4000): P2 reads 0.420-0.423 until day 1.04 and 0.1683 from day 8.125
               -> 15 Oct 14:00 + 7 d (0.387 -> 0.169), not 14 Oct 12:00 + 9 d.
          T_P3 (S_P2, range 0.1619-0.4230): P3 0.1680-0.4164 is inside -> unchanged, 14 Nov 12:00 + 20 d.
          hindcast of T_P2 (P0 with S_P1): P0 is above 0.400 until day 3.21 -> 22 Aug 18:00 + 2 d (0.396 ->
               0.357, WT 0.34 -> 0.57 m): short, and said so wherever its sigma is used.
          hindcast of T_P3 (P1 with S_P2): P1 0.1691-0.4000 is inside -> unchanged.
        Expected weak point, written before the result: in P3 the upper level (0.411-0.415 for 3.5 d, WT
        0.24-0.31 m) is below S_P2's theta_s, so the model must read it as drained (hydrostatic: about 0.35 m);
        the rule stays as it is and the outcome is reported.
    ag) 19:25. US-PAS, SOLVER SIDE (rules of ae and af; nothing changed after the results).
        Calibrations, all starts: S_P1 start 3 lowest (0.004530); starts 0, 1, 2 end at 0.004559-0.004566 on a
        neighbouring soil (theta_r 0.012-0.025, alpha 1.46-1.50, n 5.0-5.2, Ks 1.5-1.7e-6): a flat direction,
        misfit within 1 %. S_P2: all four starts end on the same soil (0.006050).
        Forward check of the transfer (solver with the OBSERVED water table, sensor RMSE): own 0.0045 (P1),
        0.0060 (P2); T_P2 0.012 (bias +0.003); T_P3 0.015-0.016 (+0.006); P0 cut with S_P1 0.0048; P1 with S_P2
        0.018-0.019. (sigma_theta of the inversion: 0.005.)
        Solver inversions (fem_referee.py --box, est - obs in cm; all starts of a set agree to the last digit):
                                   cost    whole window (rmse/bias/max/r)   below saturation      on the box
          own P1 (S_P1 on P1)      0.798   3.84 / +1.03 / 12.0 / 0.994      3.37 / -0.53 / 0.981  c0, c1
          own P2 (S_P2 on P2)      1.471   2.26 / -0.55 /  6.2 / 0.997      2.33 / -0.80 / 0.994  c0, c4
          T_P2  (S_P1, 15 Oct+7 d) 0.504   4.50 / +0.19 /  7.9 / 0.994      4.13 / -0.64 / 0.992  c0, c4
          hind. P1 with S_P2       1.476   13.40 / +8.15 / 33.0 / 0.936     (all hours)           c0, c3, c4
          hind. P0 cut with S_P1   0.297   6.52 / -5.11 / 10.3 (49 h)       1.18 (8 h only)       c0, c1, c3
        T_P2 by day: +6.5 +5.2 +2.7 +0.8 -1.2 -4.5 -6.8 -6.5 (too deep early, too shallow late); own P1 by day
        +0.8 +3.1 +5.7 +5.8 +1.2 -1.3 -0.2 +1.6 -2.4 -12.0.
        BY THE RULES: neither soil passes the usable-soil rule (in-sample within 2 cm below saturation: 3.37 and
        2.33 cm), and the solver inversion of T_P2 is outside R3 (4.13 cm). T_P3 is still running; its soil
        already fails and its hindcast is 13 cm. So US-PAS gives NO test that can count for R4; R4 stays open.
        Is the prior box the cause (estimates on it at both ends; the observed paths' own spline coefficients
        lie outside it at the window start: -2.6, -2.4, -4.1 against the bound -1.82)? Diagnostic, solver
        without --box on T_P2: cost 0.479, 4.27 / +0.15, below saturation 4.02 / -0.57 / 0.993, by day +5.0
        +5.0 +2.9 +0.9 -1.4 -4.5 -6.5 -5.8. No: the box costs 0.1-0.2 cm. The error is in soil and sensor.
        Reading: at this site the sensor - water table relation of one recession does not carry to the next
        within 3 cm (two wet levels of the sensor, af), and not even within its own recession for P1. That is
        a limit of the model (one van Genuchten soil, no hysteresis, this sensor), the same for any inversion.
        NEXT, DECLARED BEFORE THE RUN: the PINN is run on T_P2 all the same (tag rfp2, method unchanged), for
        R1 and R2 ONLY - does it reach the solver's estimate at a second site, with another soil (n 4.8, not
        10), a recession twice as fast (0.62 m in 7 d) and the estimate on the box at both ends. This departs
        from the sentence in ae (PINN only if the solver is inside R3) in purpose, not in what counts: the run
        cannot count for R3 or R4 whatever it gives. It is queued behind rf22c's last round.
    ah) 19:35. PINN runs against the solver, and two diagnostics that close open ends.
        rf22ob (C with S_B) inv2 (job 15501578): cost 0.33261 (solver 0.33445), PINN - solver 0.52 cm rms, max
        1.30 cm (inv1 0.74); vs well 4.57 / +3.68 / r 0.987, below saturation (275 h) 3.50 / +3.08 / 0.978.
        rf22c (D with S_C, the primary test) inv3 (job 15503028): cost 0.28839 (solver 0.27966); starts agree
        to 1e-8 m; PINN - solver 0.66 cm rms, max 3.44 cm (inv0 2.57, inv1 1.02, inv2 0.74). By day:
        +0.0 -0.1 -0.1 -0.1 +0.0 +0.0 +0.0 -0.0 -0.0 +0.1 +0.3 +0.5 +0.6 +0.6 +0.4 +0.3 +0.8 +2.2 +3.4:
        within 0.1 cm for days 0-9, the whole distance is in the last two days, where the posterior sd is
        0.7-1.4 cm (the end of the window is weakly seen by the sensor) - the part the value correction is for.
        vs well 4.49 / +3.13 / r 0.984, below saturation (359 h) 2.69 / +2.09 / 0.983 (solver 2.44 / +1.69 /
        0.983): RMSE and r inside R3, bias 0.09 cm outside it until the last days are closed.
        T_P3 (S_P2 on P3), solver without the box (diagnostic of ag): cost 0.626, 5.92 / +3.04 / max 11.7 /
        r 0.992; by day +11.1 +8.9 +7.5 +6.9 +8.2 +11.0 +8.1 +7.4 +9.1 +3.1 -4.2 -4.4 -1.7 -1.6 -0.6 +0.5 -1.5
        -1.4 -0.2 +1.3 +4.8. Outside R3, as written in advance in af: the wet level of the sensor (0.41) is
        read as a drained soil, 7-11 cm too deep for nine days. The boxed starts are still running.
        One soil for all three earlier US-ONA recessions (A + B + C pooled, output/soil_ona_loo_nov22, start
        0; asked: would a pooled soil carry to D better than the last recession's?): sensor RMSE 0.0164
        (by window 0.0203 / 0.0152 / 0.0162, bias -0.010 / -0.001 / +0.008), Ks on its lower bound (1e-7).
        Three to five times the misfit of a single-recession soil (0.003-0.005) and three sigma_theta: one
        van Genuchten soil does not describe the three recessions together, so a pooled soil is NOT an
        alternative to rule d, and it is not used. It is the same fact as S_B/S_D failing the usable-soil
        rule and as US-PAS: the sensor - water table relation shifts from one recession to the next.
        rfp2 (PINN on T_P2, R1/R2 only, as declared in ag) started 19:27 behind rf22c's round 4.
    ai) 19:55. rf22ob inv3 (job 15503241): cost 0.33635 (solver 0.33445), all five starts on the same
        coefficients (spread 1e-8 m); PINN - solver 0.40 cm rms, max 0.86 cm (inv1 0.74, inv2 0.52); by day
        +0.0 x9, +0.2 +0.4 +0.6 +0.7 +0.8 +0.8 +0.7. vs well 4.54 / +3.62 / r 0.987, below saturation (275 h)
        3.45 / +3.01 / 0.978 (solver 3.23 / +2.69 / 0.975).
        US-PAS solver side, rest (est - obs in cm):
          T_P3 (S_P2 on P3, with the box; starts 0 and 2 identical, 1 and 3 still running): cost 0.627,
               5.89 / +2.97 / max 11.7 / r 0.993 (479 of 481 h below saturation: 5.88 / +2.95); c6 on the box;
               by day +11.1 +9.0 +7.5 +6.9 +8.2 +11.0 +8.1 +7.4 +9.1 +3.1 -4.2 -4.4 -1.7 -1.7 -0.7 +0.5 -1.4
               -1.3 -0.3 +0.6 +2.9. OUTSIDE R3; days 0-8 are the wet sensor level read as a drained soil (af).
          own P2 without the box: cost 1.387 (box 1.471) but 3.76 / -1.20 / max 13.6 (box 2.26 / -0.55 /
               6.2): the lower cost is bought with a last day 13.6 cm off. The box is not what fails the rule.
          T_P2 without the box, all four starts: 0, 1, 2 identical (cost 0.4786, 4.27 cm); start 3
               (constant 0.3 m) RUNS AWAY: coefficients -15890 ... +1671, water table hundreds of metres
               deep, cost 1.45 after 20 iterations and still falling. Without the support of the prior the
               inversion of the solver itself is not stable from every start; with it all four starts agree
               (ag). The box is part of the method for that reason, and it costs 0.1-0.2 cm here.
        US-PAS closes as written in ag: no test inside R3, no usable soil; R4 stays open.
    aj) 20:02. rf22c inv4 (job 15503662; pipeline done 19:50, value-correction loop rf22cdc started):
        cost 0.28426 (solver 0.27966), five starts identical (spread 6e-9 m); PINN - solver 0.66 cm rms, max
        3.53 cm; by day +0.0 -0.1 -0.1 -0.0 +0.0 +0.0 -0.0 -0.1 -0.1 +0.0 +0.2 +0.5 +0.6 +0.6 +0.3 +0.2 +0.7
        +2.3 +3.5. The refinement rounds have stopped gaining (inv3 0.66): what is left is the last coefficient
        (PINN +0.393, solver -0.112; posterior sd of the last day 1.5 cm), i.e. the value error of the
        operator at the window end - the job of the loop. vs well 4.48 / +3.10 / r 0.984, below saturation
        (359 h) 2.67 / +2.05 / 0.983: RMSE and r inside R3, bias 0.05 cm outside.
    ak) 20:50. rf22ob inv4 (job 15504498; pipeline done 20:24, loop rf22obdc started): cost 0.32945 (solver
        0.33445), starts identical (2e-9 m); PINN - solver 0.44 cm rms, max 1.01 cm (inv3 0.40): the rounds
        have stopped gaining here too. vs well 4.50 / +3.59 / r 0.987, below saturation (275 h) 3.43 / +2.99 /
        0.978.
        rf22cdc it 1 (job 15505064; two forward PINNs at the round-4 estimate, value offset 0.0013 rms):
        step 3.02 sigma (last segment; 0.22-0.91 before it), cost 0.27273, five starts identical (6e-9 m).
        PINN - solver 0.43 cm rms, max 2.66 cm (inv4 0.66 / 3.53); by day -0.0 +0.0 x8 +0.1 +0.1 +0.1 -0.0
        -0.2 -0.4 -0.4 +0.2 +1.5 +2.7. vs well 4.38 / +2.87 / r 0.985; below saturation (359 h) 2.46 / +1.76 /
        r 0.983 (solver 2.44 / +1.69 / 0.983) -> INSIDE R3 on all three numbers. Not converged yet (stop
        rule 0.2 sigma): the loop goes on.
    al) 21:25. rfp2 inv0 (US-PAS test T_P2, wide members only; job 15505271): four starts identical (1e-8 m),
        cost 0.50623 (solver 0.5043); coefficients -1.821 -0.650 +0.327 +0.286 +0.783 (solver -1.821 -0.684
        +0.414 +0.157 +1.183): c0 on the same bound as the solver's. PINN - solver 1.11 cm rms, max 1.81 cm;
        by day +0.0 -0.8 -1.6 -1.6 -0.5 +0.9 +1.1 +0.3. vs well 3.99 / -0.15 / max 7.6 / r 0.998, below
        saturation (150 h) 3.47 / -1.03 / 0.997 (solver 4.13 / -0.64 / 0.992). R1/R2 only, as declared.
        rf22obdc it 1 (job 15506962; value offset 0.0008 rms): step 0.37 sigma (segments 0.10 0.31 0.37
        0.37), cost 0.32789, starts identical (1e-9 m). PINN - solver 0.16 cm rms, max 0.37 cm (inv4 0.44 /
        1.01); by day +0.1 +0.0 x9 +0.1 +0.3 +0.3 +0.3 +0.2 +0.1. vs well 4.46 / +3.47 / r 0.987, below
        saturation (275 h) 3.29 / +2.80 / 0.976 (solver 3.23 / +2.69 / 0.975): the PINN gives the solver's
        answer; that answer is outside R3 (soil S_B, not usable by the rule), as known since 73 v.
    am) 21:48. rf22cdc it 2 (job 15507993; value offset 0.0016 rms): step 0.89 sigma (segments 0.01 0.03 0.18
        0.21 0.89; it 1: 3.02), cost 0.27353, starts identical (2e-8 m). PINN - solver 0.27 cm rms, max
        1.48 cm (it 1: 0.43 / 2.66); by day 0.0 x11, -0.1 -0.2 -0.3 -0.4 -0.4 -0.1 +0.8 +1.5. vs well 4.36 /
        +2.79 / r 0.985; below saturation (359 h) 2.40 / +1.66 / r 0.984 (solver 2.44 / +1.69 / 0.983):
        inside R3. The loop contracts (x0.3 per step) and goes on to the 0.2-sigma stop.
    an) 22:30. rfp2 inv1 (US-PAS T_P2, refinement round 1; job 15509536): four starts identical, cost 0.49642
        (solver 0.5043; the PINN's own forward model, not yet value-corrected), coefficients -1.821 -0.681
        +0.402 +0.208 +1.152 (solver -1.821 -0.684 +0.414 +0.157 +1.183). PINN - solver 0.77 cm rms, max
        1.59 cm (inv0 1.11 / 1.81); by day -0.0 -0.1 -0.1 +0.1 +0.6 +1.1 +1.5 +1.6: the gap sits in the last
        days of the window, as at US-ONA before the value correction. vs well 3.99 / +0.62 / max 7.6 /
        r 0.997, below saturation (150 h) 3.48 / -0.15 / 0.995. R1 holds, R2 inside 1 cm after one round.
        Late arrival: output/soil_ona_nov22/start1 (calibration of S_D from the Carsel & Parrish sand start,
        387 min, finished 20:45): rmse 0.00433, theta_s 0.3514, theta_r 0.1299, alpha 1.916, n on bound 10,
        Ks 4.13e-7 - the same soil as starts 0, 2, 3: all four starts of S_D agree. Nothing changes (S_D is
        the in-sample soil of the D window, not usable for a held-out test).
    ao) 22:50. rf22obdc it 2 (job 15510025; value offset 0.0012 rms): step 0.25 sigma (segments 0.02 0.08 0.11
        0.25; it 1: 0.37), cost 0.33135 (solver 0.33445), starts identical. PINN - solver 0.13 cm rms, max
        0.64 cm (it 1: 0.16 / 0.37); by day +0.0 +0.1 +0.1 +0.0 x6 +0.1 +0.1 +0.1 -0.1 -0.4 -0.6. vs well
        4.42 / +3.38 / r 0.987, below saturation (275 h) 3.23 / +2.67 / 0.974 (solver 3.23 / +2.69 / 0.975).
        rf22cdc it 3 (job 15510622; value offset 0.0018 rms): step 0.41 sigma (segments 0.06 0.10 0.13 0.35
        0.41; it 1, 2: 3.02, 0.89), cost 0.28207 (solver 0.27966), starts identical; coefficients +0.565
        -0.841 -0.197 +0.103 +0.110 +0.311 -0.001 (solver +0.573 -0.842 -0.197 +0.102 +0.114 +0.329 -0.112).
        PINN - solver 0.19 cm rms, max 0.47 cm (it 2: 0.27 / 1.48); by day 0.0 x10, -0.1 -0.1 -0.2 -0.3 -0.4
        -0.5 -0.3 +0.0 +0.3. vs well 4.34 / +2.71 / r 0.986; below saturation (359 h) 2.37 / +1.57 / r 0.984
        (solver 2.44 / +1.69 / 0.983): inside R3. Both loops go on to their next iteration (stop 0.2 sigma).
    ap) 22:58. rfp2 inv2 (US-PAS T_P2, refinement round 2; job 15510813): five starts identical, cost 0.50122
        (solver 0.5043), coefficients -1.821 -0.690 +0.414 +0.170 +1.183 (solver -1.821 -0.684 +0.414 +0.157
        +1.183): both ends on the same bounds of the prior box as the solver's. PINN - solver 0.26 cm rms,
        max 0.50 cm (inv1 0.77 / 1.59); by day -0.1 -0.1 -0.1 +0.1 +0.2 +0.4 +0.5 +0.5. vs well 4.29 / +0.31 /
        max 7.5 / r 0.995, below saturation (150 h) 3.89 / -0.49 / 0.993 (solver 4.13 / -0.64 / 0.992).
        R1 and R2 hold at the second site after two rounds; the distance to the well is the solver's (soil
        S_P1 not usable, 73 ag), so this test counts for R1/R2 only, as declared before the run.
    aq) 23:10. THIRD SITE, US-NC2 (survey and plan, written BEFORE any inversion or calibration there). Reason:
        R4 is open at US-ONA (soil S_B not usable) and at US-PAS (soil S_P1 not usable); US-NC2 is the third
        strongly validated site of the project (CLAUDE.md) and has a well next to the sensor.
        Survey (login node, awk over the half-hourly file AMF_US-NC2_BASE_HH_19-5.csv; nothing fitted):
        - Records: LE_1_1_1, P_1_1_1, WTD_1_1_1, SWC_1_1_1. Stand harvested in 2020 (sensors moved Nov 2020):
          2019-2020 is another soil-moisture regime and is not used. Rain is missing in 2021, unreliable in 2022
          (443 mm in the year, long runs of zeros) and zero from Aug 2024; in 2023 rain and the rise of the
          water table coincide (no time shift). From 23 Oct 2023 to Apr 2024 the soil-moisture and well records
          have a gap of 4 - 17 h almost every night (66 h on 9-12 Apr 2024): outside the rule of the Florida
          tests (gaps <= 3 h). Usable period with the same data rule: Feb - 22 Oct 2023.
        - Sensor: listed at -0.30 m; the reading falls (0.553 -> 0.498) while the water table goes from 0.11 to
          0.29 m, so it is not a point at 0.30 m but rods from the surface: taken as the mean over 0 - 0.30 m
          (obs.depth_m [0.0, 0.30]; the layer mean is already in the code, src/zbinv/data.py obs_layer).
          Saturated plateau 0.553 - 0.557. Sensitivity about 0.26 per m of water table (0.01 = 3.8 cm).
        - Recessions of 2023 that start with the water table at 0.2 - 0.3 m, fall by more than 0.25 m and are
          complete: 1 - 16 May (0.288 -> 0.731 m, sensor 0.495 -> 0.398; 3.6 mm of rain on 9 May),
          1 - 7 Sep (0.218 -> 0.534 m, 0.499 -> 0.404; after 84 mm on a water table at 1.25 m),
          24 Sep - 6 Oct (0.189 -> 0.616 m, 0.537 -> 0.413). At the same water table the sensor reads within
          0.007 in May and in Sep-Oct, but 0.015 - 0.023 LOWER in the recession of 1 - 7 Sep (fast rise before
          it: probably trapped air, i.e. hysteresis) - known before the tests, so a failure of the soil
          transfer across that window would not be a surprise.
        - Transpiration fraction: the same rule as in Florida (GLDAS ratio of a nearby pixel): nearest pixel in
          the project's tables = USGS well 354227076232901 (27 km east), data/gldas_well354227076232901_2015_2024.csv.
        Plan (protocol d, the usable-soil rule and R1 - R4 unchanged; the windows follow from the rule "last
        earlier recession whose sensor range contains the test's"):
        - W1 = 1 May 2023 12:00 + 15 d, W2 = 1 Sep 2023 12:00 + 6 d, W3 = 27 Sep 2023 12:00 + 9 d (starts
          where the sensor, 0.498, is inside the range of W2). configs/real_nc2_{may23,sep23a,sep23b}.yaml.
        - T_N1 (the test): W3 with the soil calibrated on W2. T_N0: W2 with the soil calibrated on W1 (the
          sensor range of W2 is inside W1's except the first hours, 0.499 against 0.495); its solver error is
          also the hindcast sigma of T_N1.
        - Solver side first (CPU only): make_real for the three windows, calibrate_soil (4 starts) on W1 and
          W2, usable-soil rule (in-sample solver inversion within 2 cm, r >= 0.975 below saturation), solver
          inversion (4 starts, --box, layer sensor) of T_N0 and T_N1 against R3.
        - The PINN is run on a test only if its soil is usable; a test with a usable soil whose PINN run meets
          R1 - R3 counts for R4 (as a second site, disclosed). If no soil is usable the site is reported as
          such and nothing is changed to make it pass.
    ar) 23:38. rfp2 inv3 (US-PAS T_P2, refinement round 3; job 15512224): five starts identical (1e-10 m), cost
        0.49575 (solver 0.5043), coefficients -1.821 -0.686 +0.411 +0.169 +1.183 (solver -1.821 -0.684 +0.414
        +0.157 +1.183). PINN - solver 0.21 cm rms, max 0.45 cm (inv2 0.26 / 0.50); by day -0.1 -0.1 -0.1 -0.0
        +0.1 +0.3 +0.4 +0.5. vs well 4.33 / +0.27 / max 7.6 / r 0.995, below saturation (150 h) 3.93 / -0.54 /
        0.993 (solver 4.13 / -0.64 / 0.992). Round 4 and the value correction follow.
        US-NC2 data sets (jobs 15511395-97, scripts/make_real.py; forward check with the STARTING soil of
        CLAUDE.md, nothing fitted): W1 1 May 2023 + 15 d: water table 0.288 - 0.731 m, sensor 0.398 - 0.495,
        6 coefficients, ET 41.3 mm, rain 3.6 mm, f_T 0.69; W2 1 Sep 2023 + 6 d: 0.218 - 0.533 m, 0.404 -
        0.499, 5 coefficients, ET 24.5 mm, rain 0.3 mm, f_T 0.75; W3 27 Sep 2023 + 9 d: 0.292 - 0.615 m,
        0.413 - 0.498, 5 coefficients, ET 15.3 mm, rain 0.5 mm, f_T 0.58. No hour filled in the sensor or the
        well record; LE filled 14 - 16 %; spline of the well within 0.2 - 0.5 cm rms of the record; all
        admissible (surface head >= -0.8 m), FEM converged everywhere. With the starting soil the FEM over
        the observed water table reads 0.09 - 0.11 too wet (r 0.98 - 0.99): that soil belongs to the sensor
        of 2021, the calibration has to replace it. The transpiration fraction hardly matters here (bias
        changes by 0.002 between f_T 0 and 1): the profile stays near hydrostatic, the water table falls by
        drainage (ET 1.7 - 4 mm/d against 3 - 5 cm/d of water-table fall).
        Calibrations submitted (4 starts each, jobs 15511553-60): output/soil_nc2_may23 (W1),
        output/soil_nc2_sep23a (W2).
    as) 23:48. rf22obdc it 3 (C with S_B; job 15512245): step 0.27 sigma (segments 0.02 0.04 0.03 0.27; its 1,
        2: 0.37, 0.25), five starts identical, cost 0.33911 (solver 0.33445), coefficients +1.154 -1.121
        +0.103 +0.163 +0.334 +0.142. PINN - solver 0.30 cm rms, max 1.31 cm (it 1: 0.16 / 0.37; it 2: 0.13 /
        0.64); by day +0.1 x4, 0.0 x8, -0.2 -0.5 -1.0 -1.3. vs well 4.39 / +3.28 / max 12.3 / r 0.986; below
        saturation (275 h) 3.17 / +2.53 / 0.972 (solver 3.23 / +2.69 / 0.975): outside R3, as known (S_B is
        not a usable soil). it 4 (the last) is running.
        rf22cdc it 4 (D with S_C, the primary test; job 15512521): step 0.30 sigma (segments 0.04 0.09 0.09
        0.18 0.30; its 1 - 4: 3.02, 0.89, 0.41, 0.30). THE LOOP ENDED AT ITS LIMIT OF 4 ITERATIONS WITHOUT
        MEETING THE 0.2-SIGMA STOP RULE. The step no longer falls: it sits at the training noise of the
        forward PINNs (0.25 - 0.3 sigma; it 4 - it 3 0.29 cm rms, max 1.57 cm, all of it in the last days of
        the window, where the posterior is widest). Five starts identical (3e-8 m), cost 0.27790 (solver
        0.27966), coefficients +0.547 -0.837 -0.198 +0.104 +0.110 +0.313 -0.181 (solver +0.573 -0.842 -0.197
        +0.102 +0.114 +0.329 -0.112). PINN - solver 0.35 cm rms, max 1.24 cm (its 1 - 4: 0.43, 0.27, 0.19,
        0.35); by day 0.0 x11, -0.1 -0.2 -0.2 -0.4 -0.5 -0.8 -1.0 -1.2.
        The scored result is it 4, the estimate the method stops at; the loop is not continued (a fifth
        iteration would move the window end by another training-noise step, not converge it).
        FINAL SCORES of rf22c (job 15512665, scripts/score_real.py, sigma_hind 3.5 cm from the hindcast;
        output/real_ona_nov22_caloct22/result_rf22c.json / .png), 433 h, 359 h below saturation:
                                   whole window                  below saturation
          PINN vs well             4.34 / +2.63 / 18.8 / 0.9857  2.37 / +1.47 / r 0.9826
          solver inversion vs well 4.37 / +2.81 / - / 0.9850     2.44 / +1.69 / r 0.9830
          hydrostatic reading      4.18 / +2.10 / - / 0.982      1.93 / +0.84 / r 0.983
        (rmse cm / bias cm / max cm / r). theta rmse of the fit 0.00264; posterior sigma of the inversion
        0.15 cm (median; holds no soil error); with sigma_hind 3.5 cm: below saturation 81.6 % of the hours
        within 1 sd and 100 % within 2 sd; in the 74 saturated hours the well is never below the estimate
        + 2 sd (a saturated sensor bounds the water table from below only).
        Against the criteria of 73 c: R1 met (starts within 3e-8 m). R2 met (0.35 cm rms <= 1 cm). R3 met
        below saturation (2.37 cm <= 3, +1.47 cm <= 2, r 0.983 >= 0.98) and NOT met over the whole window
        (4.34 cm, +2.63 cm): the first three days the sensor reads saturated and the well stands 10 - 19 cm
        above the estimate; the solver inversion misses the same days by the same amount, so this is the
        limit of the sensor, not of the PINN. R4 open (C / S_B and US-PAS cannot count; US-NC2 is being
        screened). Not hidden: on this recession the hydrostatic reading of the sensor (no flow model at all)
        scores as well as both inversions (1.93 cm below saturation), i.e. this window does not show that
        the dynamics add skill over the retention curve alone.
    at) 00:32 (12 Nov). rf22obdc it 4 (C with S_B; job 15514472): step 0.06 sigma (segments 0.01 0.02 0.06
        0.05) -> THE LOOP CONVERGED BY ITS STOP RULE (its 1 - 4: 0.37, 0.25, 0.27, 0.06). Five starts
        identical (4e-8 m), cost 0.33792 (solver 0.33445), coefficients +1.148 -1.121 +0.102 +0.163 +0.332
        +0.083. PINN - solver 0.45 cm rms, max 1.95 cm (its 1 - 4: 0.16, 0.13, 0.30, 0.45); by day 0.0 or
        +0.1 on days 0 - 10, then -0.1 -0.3 -0.7 -1.5 -2.0: as in rf22cdc the two inversions part on the
        last three days only, where the sensor constrains the path least (posterior sd 0.5 - 1.1 cm there).
        it 4 - it 3 0.15 cm rms, max 0.64 cm.
        FINAL SCORES of rf22ob (job 15514557, sigma_hind 7.0 cm; output/real_ona_oct22_soilnov21/
        result_rf22ob.json / .png), 361 h, 275 h below saturation (rmse cm / bias cm / max cm / r):
                                   whole window                  below saturation
          PINN vs well             4.38 / +3.21 / 12.3 / 0.9859  3.15 / +2.45 / r 0.9694
          solver inversion vs well 4.41 / +3.37 / 12.3 / 0.9865  3.23 / +2.69 / r 0.9746
          hydrostatic reading      4.89 / +2.60 / 25.4 / 0.964   2.65 / +1.87 / r 0.974
        theta rmse of the fit 0.00291; posterior sd 0.16 cm (median); with sigma_hind 7.0 cm all hours below
        saturation are within 1 sd (the hindcast sigma of this soil is wide), saturated hours 86, bound
        holds in all of them.
        Criteria: R1 met, R2 met (0.45 cm rms), R3 NOT met (3.15 cm, +2.45 cm, r 0.969 below saturation),
        as the solver inversion (3.23 / +2.69 / 0.975) and as stated before the run: S_B is not a usable
        soil, the test cannot count for R4. What it adds: the PINN is again the solver inversion to within
        0.5 cm, from every start, on a second window.
        rfp2 inv4 (US-PAS T_P2, refinement round 4; job 15513888): five starts identical (7e-11 m), cost
        0.50222 (solver 0.5043), coefficients -1.821 -0.692 +0.414 +0.166 +1.183. PINN - solver 0.20 cm rms,
        max 0.34 cm (inv3 0.21 / 0.45); vs well 4.32 / +0.23 / max 7.6 / r 0.995, below saturation (150 h)
        3.94 / -0.57 / 0.992 (solver 4.13 / -0.64 / 0.992). Pipeline done 00:10; value correction rfp2dc
        started (it 1 forward PINNs on V100).
        hpc/real_full_obs.sh: hpc/real_full.sh with the sensor depth as third argument (for the layer sensor
        of US-NC2); real_full.sh itself untouched (rfp2 still reads it). Not used yet.
        US-NC2 calibrations after 75 min (not finished; 4 starts each): W2 (1 Sep 2023) starts 1 and 2 have
        met at rmse 0.00105 (theta_s 0.550, alpha 15.4, n 1.25, Ks on its upper bound 1e-3 m/s: the fit
        asks for a profile in equilibrium with the water table), start 3 at 0.0025, start 0 at 0.0038
        (different basins, Ks 1.6e-5 and 3e-6). W1 (1 May 2023) starts at 0.0015 - 0.0017, theta_s 0.52 -
        0.54, alpha 4 - 6, n 1.2 - 1.4, Ks 2e-4 - 9e-4, theta_r anywhere in 0 - 0.13 (not identified).
    au) 01:25. rfp2dc it 1 (US-PAS T_P2, value correction; job 15515881): step 0.51 sigma, five starts
        identical (6e-10 m), cost 0.50617 (solver 0.5043), coefficients -1.821 -0.683 +0.412 +0.157 +1.183
        (solver -1.821 -0.684 +0.414 +0.157 +1.183). PINN - solver 0.03 cm rms, max 0.05 cm (inv4 0.20 /
        0.34): after one correction the PINN estimate IS the solver's. vs well 4.52 / +0.17 / max 7.9 / r
        0.995, below saturation (150 h) 4.14 / -0.67 / 0.992 (solver 4.13 / -0.64 / 0.992). it 2 running.
        US-NC2, record of 2024 looked at again (awk, sensor / well / rain only; no model run): complete from
        27 Apr to 12 Jul 2024 (the survey of aq stopped at the gaps of Apr 2024). Water table 0.55 - 1.47 m,
        sensor 0.40 -> 0.15: below the range of every 2023 window. Rule d inside 2024 gives two tests only,
        both with the soil of 2 - 9 May 2024 (sensor 0.372 -> 0.326): 19 - 23 May (4 d) and 1 - 5 Jun (4 d,
        after a 7.5-h sensor gap). Windows of 4 - 5 d have 4 spline coefficients, a case the method has never
        been run on (all tests: 5 - 9 coefficients). And at one water table (0.845 m) the sensor reads 0.372
        on 2 May, 0.359 on 20 May, 0.333 on 4 Jun: with the water table below 0.8 m the top 0.30 m follows
        rain and ET history, not the water table. NO TEST IS DEFINED IN 2024 (not run, not counted, said
        here so that the choice is on record: it was made from the records, before any model run there).
        US-NC2 PREVIEW with the calibrations still iterating (they stop at 80 function evaluations, 03:30 -
        06:30; nothing below is a result, the pre-registered numbers come from the finished calibrations).
        Soils at 01:10: S_W2 (starts 1, 2 at rmse 0.00104) 0.550 / 0.164 / 15.31 / 1.252 / Ks 1e-3 (bound);
        S_W1 (all four starts within rmse 0.00148 - 0.00150, heading to one soil) 0.538 / 0.001 / 5.58 /
        1.219 / Ks 1e-3 (bound). Data sets output/real_nc2_{sep23a,sep23b}_prev (S_W2),
        output/real_nc2_{may23,sep23a,sep23b}_prevW1 (S_W1). FEM over the OBSERVED water table against
        the sensor (forward check; rmse / bias):
                          W1 (May)           W2 (1 Sep)         W3 (27 Sep)
          S_W1            0.0015 / -0.0001   0.0219 / +0.0214   0.0038 / +0.0003
          S_W2            -                  0.0010 / +0.0000   0.0232 / -0.0229
        -> exactly the picture of the survey: W1 and W3 share one retention relation, W2 (after the storm)
        is 0.02 drier at the same water table. Solver inversions so far (preview): W2 with S_W2 (in-sample)
        0.62 cm, bias +0.04, max 1.8 cm, starts 0 and 3 identical (cost 0.1512) -> S_W2 passes the in-sample
        rule; W3 with S_W2 (= T_N1) 9.5 cm (cost 0.132: the sensor is fitted, the water table is wrong by
        the hysteresis offset). So T_N1 fails R3 by a wide margin in the solver inversion itself, as
        announced in aq; T_N0 (W2 with S_W1) will fail the same way with the other sign.
        Decision written BEFORE the solver inversion of it: "W3 with S_W1" (soil of May, skipping the
        post-storm recession) is run as a DIAGNOSTIC. It is NOT the pre-registered test (rule d takes the
        last earlier recession, W2), the choice was made after seeing the forward check above, so it can
        NOT count for R4 whatever it scores. What it can show: whether the inversion returns the well at a
        third site when the retention relation carries over, and whether the PINN equals the solver there.
        Preview solver jobs 15516206-13 (W1 in-sample, T_N0, W3 with S_W1).
        01:20: calibration of W2 (output/soil_nc2_sep23a), starts 1 and 2 FINISHED (xtol; 179 / 191 member
        solves, 112 / 108 min): rmse 0.0010408 / 0.0010409, bias 0, r 0.99935; soil theta_s 0.54976 /
        0.54976, theta_r 0.1778 / 0.1775, alpha 15.111 / 15.114, n 1.2660 / 1.2657, Ks 1.0e-3 (upper bound)
        in both -> two starts, one soil. Starts 0 and 3 are still iterating (other basins, 0.0037 and
        0.0024 at 01:10); S_W2 = start 1 unless one of them ends lower.
    av) 02:05. rfp2 (US-PAS T_P2 = Oct 2021 with the soil of Sep 2021, S_P1) FINISHED. Value correction it 2
        (job 15517587): step 0.03 sigma -> the loop CONVERGED BY ITS STOP RULE at it 2 (its 1, 2: 0.51, 0.03).
        Five starts identical (8e-10 m), cost 0.50994 (solver 0.50426), coefficients -1.821 -0.682 +0.412
        +0.157 +1.183 (solver -1.821 -0.684 +0.414 +0.157 +1.183; first and last on the box in both).
        FINAL SCORES (job 15517638, scripts/score_real.py; output/real_pas_oct21_soilsep21/result_rfp2.json
        / .png; no hindcast sigma at this site), 169 h, 150 h below saturation:
                                   whole window                  below saturation
          PINN vs well             4.52 / +0.17 / 7.9 / 0.9945   4.14 / -0.66 / r 0.9919
          solver inversion vs well 4.50 / +0.19 / 7.9 / 0.9944   4.13 / -0.64 / r 0.9918
          hydrostatic reading      5.28 / -2.69 / 9.1 / 0.998    5.35 / -3.59 / r 0.998
          PINN - solver            0.03 cm rms, max 0.05 cm
        sensor fitted to 0.0036 (PINN); FEM over the observed water table 0.0124 from the sensor: the soil
        of September does not describe October (by day the estimate is +6.5 cm at the start, -6.5 cm at
        the end: the slope of the recession, not its level).
        R1 yes (one estimate from five starts), R2 yes (0.03 cm), R3 NO (4.1 cm; bias and r inside). As
        known since ak: S_P1 is not a usable soil, so this run was the R1 / R2 check at the second site and
        does not count for R4.
        US-NC2 calibrations finished so far (output/soil_nc2_*):
          W1 (May):   start 0 rmse 0.00147988, start 2 0.00147988 (lower by 1e-9), both xtol: theta_s 0.53809
                      / 0.53809, theta_r 0 (bound), alpha 5.5845 / 5.5836, n 1.21821 / 1.21822, Ks 1.0e-3
                      (bound) -> ONE soil; starts 1 and 3 are at the same point (0.00148), still iterating.
                      S_W1 = that soil (the data sets carry the digits of start 0; alpha differs from start
                      2 by 1.5e-4 relative).
          W2 (1 Sep): start 1 0.00104083, start 2 0.00104089 (one soil, see au); start 0 stopped at its 80
                      evaluations at 0.00360 (theta_s 0.523, theta_r 0.083, alpha 4.98, n 1.369, Ks 5.7e-6:
                      another basin, not converged); start 3 at 0.00112 and falling towards starts 1 / 2.
                      S_W2 = start 1 unless start 3 ends lower.
        Data sets with these soils (jobs 15516306-07, 15516532-34), FEM over the recorded water table
        against the sensor: real_nc2_sep23a_own (W2, S_W2) rmse 0.00104; real_nc2_sep23b_soilsep23a (= T_N1)
        bias -0.0230 (r 0.9996); real_nc2_may23_own (W1, S_W1) 0.00148; real_nc2_sep23a_soilmay23 (= T_N0)
        bias +0.0214 (r 0.9958); real_nc2_sep23b_soilmay23 (the diagnostic of au) rmse 0.0038, bias
        +0.0004 (r 0.9972). Solver inversions, 4 starts each, --box, layer sensor [0, 0.30] m: jobs
        15516425-44 (S_W2), 15516719-30 (S_W1).
        First solver results (zb RMSE / bias / max, cm):
          T_N0 (W2 with S_W1), final soil: starts 0, 1, 3 identical, cost 0.12278: 10.69 / +9.95 / 16.6
                      -> T_N0 FAILS R3 (the sensor is fitted to 0.0018, the water table is 10 cm too deep).
          T_N1 (W3 with S_W2), interim soil of au (preview): 9.49 / -9.28 / 12.6, r 0.998 (final pending).
          Diagnostic W3 with S_W1, interim soil (preview; three starts identical, cost 0.0631):
                      1.61 / +0.51 / 2.6, r 0.998, by day -1.8 -2.0 -1.2 +0.1 +1.5 +2.0 +1.8 +1.6 +1.7.
          W2 in-sample, interim S_W2 (preview): 0.62 / +0.04 / 1.8, r 0.998.
    aw) 02:32. US-NC2, solver side with the FINAL soils (zb RMSE / bias / max in cm, r; all hours are below
        saturation at this site, so the two scores of 73 c coincide):
          W1 calibration: all four starts finished (xtol) at ONE soil, rmse 0.00147988 (start 2 lowest by
                      1e-9; theta_s 0.53809, theta_r 0 (bound), alpha 5.583-5.584, n 1.2182, Ks 1e-3 (bound))
                      -> S_W1 is final.
          W2 calibration: starts 1, 2 at one soil (0.00104083); start 0 unconverged in another basin
                      (0.00360); start 3 (job 15511560) still running: 0.00105 at theta_r 0.199, alpha 14.9,
                      n 1.289, Ks 8.0e-4 after 214 solves, moving along the flat valley towards starts 1 / 2.
                      S_W2 = start 1 unless start 3 ends lower.
          in-sample W1 / S_W1 (starts 0, 2; cost 0.08843):      0.89 / +0.03 / 2.5, r 0.9975
          in-sample W2 / S_W2 (four starts, cost 0.15095):      0.62 / +0.02 / 1.8, r 0.9978
              -> both soils meet the usable-soil rule (in-sample <= 2 cm, r >= 0.975).
          T_N0 = W2 with S_W1 (four starts, cost 0.12278):      10.69 / +9.95 / 16.6, r 0.9981
                      by day +4.2 +5.1 +7.8 +10.8 +12.8 +14.3 +16.5       -> R3 NO
          T_N1 = W3 with S_W2 (starts 0, 3; cost 0.13207):      9.52 / -9.30 / 12.7, r 0.9982
                      by day -5.6 -7.0 -7.6 -7.9 -8.0 -9.0 -10.5 -11.7 -12.2 -11.5 -> R3 NO
          diagnostic W3 with S_W1 (four starts, cost 0.06303):  1.64 / +0.55 / 2.7, r 0.9978
                      by day -1.8 -2.0 -1.1 +0.1 +1.5 +2.1 +1.9 +1.7 +1.8 +2.6
        Reading: both PRE-REGISTERED tests at US-NC2 miss R3 by 10 cm with the solver itself, with
        opposite signs: the recession after the storm of late August (W2) lies 0.02 in theta BELOW the
        relation of May (W1) and of late September (W3) at the same water table (the forward checks of
        au), the sensor is fitted in every case (0.0015 - 0.0018) and r is 0.998. With 0.26 theta per m
        that offset is 8 - 10 cm. This is the anticipated soil-transfer failure (aq), not an inversion
        failure: W3 read with the soil of May (four months and one storm earlier) is within 1.64 cm.
        Nothing is changed to make the site pass; the diagnostic cannot count for R4 (au).
        PINN on US-NC2: the diagnostic first (it is the case whose solver inversion is near the well, so a
        PINN defect would show against the well too): tag rfnd, hpc/real_full_obs.sh rfnd
        output/real_nc2_sep23b_soilmay23 "[0.0,0.30]" launched 02:02 (first use of the layer sensor on site
        data). Wide members on V100: w0 15517740, w3 15517741, w1 15517742, w2 15517743.
        Checked before the launch (reading only): the layer mean is 4 Gauss-Legendre nodes at 0.021,
        0.099, 0.201, 0.279 m; when zb < 0.30 m a node lies below the water table (xi < -1, outside the
        trained domain). There the ansatz gives h > 0 and SoilVG blends to theta_s within 0.02 m, the FEM
        side clamps to theta_s: mismatch <= 0.0005 in theta (<= 0.5 cm). R2 will show whether that holds.
        Training of rfnd_w0 so far (physics only; steps 2000 / 4000 / 6000 / 8000 / 10000): L_pde 70 / 216
        / 121 / 24 / 8.7, L_flux 91 / 115 / 87 / 77 / 63 (held-out total 191 / 605 / 130 / 133 / 78). The
        first 6000 steps are far above the earlier sites at the same stage (rfp2_w0 L_pde 0.1, L_flux 13
        at step 3500; rf22c_w0 0.4, 63), then it comes down; this soil has Ks 1e-3 m/s (the bound of the
        calibration), 30 - 300 times the Florida soils, so the profile is close to hydrostatic and the
        residual is stiff. To be judged at the end of the wide stage by the kept held-out loss and by
        inv0 against the solver (output/referee/nc2_may23on_sep23b_s*) - no change before that.
        The PINN runs of the pre-registered tests (T_N1: output/real_nc2_sep23b_soilsep23a; T_N0:
        output/real_nc2_sep23a_soilmay23) wait for (i) S_W2 final and (ii) inv0 of rfnd, so that no GPU is
        spent on this soil class before the first check against the solver. Their R3 outcome is fixed by
        the solver (NO); they would add R1 / R2 at this site.
        US-NC3 looked at as a possible further site (reading the CSV header and columns only): WTD exists
        2019 - 2021 only, within 0.08 - 0.47 m in 2019 - 2020, LE sparse or absent 2020 - 2021 -> not a
        candidate (the sensor layer would be at the water table, the flux cannot be built).
    ax) 03:40. W2 CALIBRATION FINISHED, S_W2 CHANGES. Start 3 (job 15511560, 258 solves, xtol) ended at rmse
        0.00103759, BELOW starts 1 / 2 (0.00104083 / 0.00104089): theta_s 0.54931, theta_r 0.19853, alpha
        14.803, n 1.28956, Ks 9.997e-4 (no bound active) against theta_s 0.54976, theta_r 0.17781, alpha
        15.111, n 1.26597, Ks 1e-3 (bound). One flat valley (smallest scaled singular value of the Jacobian
        0.002; theta_r and n trade), the misfit differs by 0.3 %. By the rule (lowest misfit of the starts)
        S_W2 = the soil of start 3, as announced in av / aw. New data sets with it (jobs 15519425 / 26):
          output/real_nc2_sep23a_own_cal3        (W2 in-sample)  FEM over the record vs sensor 0.00104
          output/real_nc2_sep23b_soilsep23a_cal3 (= T_N1)        0.0233 / bias -0.0230, r 0.9996
        (the sets real_nc2_sep23a_own and real_nc2_sep23b_soilsep23a carry start 1's soil: kept, now a
        sensitivity check of the choice between the two ends of the valley). Solver inversions redone,
        four starts each: output/referee/nc2_own_sep23a_cal3_s0..3, nc2_sep23aon_sep23b_cal3_s0..3 (jobs
        15519460-67). With start 1's soil T_N1 was 9.52 / -9.30 / 12.7 (aw); the forward check is the same
        to 1e-4, so the same is expected.
        rfnd (diagnostic W3 with S_W1), WIDE STAGE AND inv0. Kept held-out physics loss of the four wide
        members 25.6 / 17.8 / 3.4 / 21.0 (w0..w3; at 40000 steps L_pde 2 - 7, L_flux 5 - 34), above the
        Florida runs (rfp2_w0 4.0, rf22c_w0 10.7): with Ks 1e-3 m/s the flux residual amplifies an error
        of the head gradient 30 - 300 times more. It does not reach the sensor: inv0 (job 15519529, wide
        members only): four starts identical (4e-9 m), cost 0.06331 (solver 0.06303), coefficients -1.821
        -1.478 -0.530 -0.246 +0.199 (first on the box, as the solver),
          PINN - solver   0.07 cm rms, max 0.17 cm (by day +0.1 +0.1 +0.0 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1 -0.2)
          PINN vs well    1.57 / +0.53 / 2.5, r 0.9979 (217 h, all below saturation)
        -> the layer sensor and this soil class work with the unchanged method; the node below the water
        table (aw) costs nothing visible. R1 and R2 hold at the third site already at inv0; the run
        continues through its rounds and the value correction unchanged (it remains a diagnostic: its 1.6
        cm cannot count for R4).
        PINN ON THE PRE-REGISTERED TEST T_N1 launched 03:36: hpc/real_full_obs.sh rfn1
        output/real_nc2_sep23b_soilsep23a_cal3 "[0.0,0.30]" (logs/full_rfn1.log). Expected from the solver:
        R1 / R2 yes, R3 no (9.5 cm too shallow). No PINN run is made for T_N0: its soil is usable too, but
        its R3 outcome is fixed by the solver (10.69 cm, four starts) and T_N1 is the designated test (aq);
        stated here so that the omission is not read as a selection.
    ay) 03:55. FOURTH SITE, US-HB2 (survey and plan, written BEFORE any calibration or inversion there).
        Reason: R4 is open at three sites and the limit is the soil transfer between recessions, not the
        inversion (av, aw). A site whose sensor reacts more strongly to the water table loses less depth
        per unit of retention drift (US-NC2: 0.26 theta per m, 0.02 of drift = 8 - 10 cm).
        Sites looked at (login node, awk over the half-hourly files of fem_validation/uploaded_data, nothing
        fitted) and set aside: US-NC3 (aw); US-NC4 (water table only 2019 - 2020 and within 0.2 m of the
        surface); US-Oho (water table 1 - 3 m, sensor 17 - 23 %, no response); US-SSH (16 - 27 m); US-TLR
        (2024 only); US-Fo1 / US-Fo2 (sparse records).
        Sensor depths: AmeriFlux measurement-height table (BASE_MeasurementHeight_20260908.csv, public;
        read in /tmp, not in the repository). It also confirms the earlier sites: US-ONA SWC_1_1_1 -0.1 m,
        US-PAS -0.1 m, US-NC2 -0.3 m (read as rods 0 - 0.30 m, aq).
        US-HB2 (Hobcaw Barony, SC, mature longleaf pine on sand, 33.3242 N 79.2440 W; file
        AMF_US-HB2_BASE_HH_2-5.csv):
        - Records: LE (29.87 m), P_RAIN (27.43 m, above the canopy: no throughfall correction, the windows
          hold at most 16 mm), WTD_1_1_1 (pressure transducer, negative below the surface; from Aug 2019,
          missing 14 Jan - 21 Feb 2021), SWC_1_1_1 (TDR at -0.1524 m, until 7 Feb 2022): taken as a point
          sensor, obs.depth_m [0.1524, 0.1524]. Saturated plateau 44 - 45 %.
        - Sensitivity: on the recession of Sep 2019 the sensor reads 36 / 30 / 20 / 12 / 9.5 / 6 / 3 % with
          the water table at 0.25 / 0.32 / 0.39 / 0.49 / 0.59 / 0.80 / 1.0 m: about 1 theta per m down to
          0.45 m, 0.15 - 0.2 per m below 0.6 m.
        - Rain and the rise of the water table coincide to the hour (5 Sep 2019, 3 Aug 2020, 11 - 12 Nov
          2020, 3 Mar 2021): no time shift.
        - Transpiration fraction: same rule as before, nearest pixel of the project's GLDAS tables = USGS
          well 330247079340300 (43 km south-west); a partition exists on every day of the windows (window
          means of f_T 0.66 - 0.84).
        Windows, by the rule of aq written out: after a rise, the window starts at 12:00 of the first day
        whose daily-mean water table is back at 0.25 - 0.30 m and runs to the last noon before the next
        day-to-day rise of the daily mean by more than 0.01 m, at most 20 d (the longest window inverted so
        far); kept if the water table falls by more than 0.25 m in it and the sensor and well records have
        no gap above 3 h. Aug 2019 - Feb 2022 holds exactly seven (half-hourly ranges; the last column is
        the highest reading of the sensor in the 7 d before the start):
          V1  7 Sep 2019 + 20 d   sensor 36.0 ->  5.5 %   water table 0.254 -> 0.820 m   rain  1.9 mm   44.4
          V2  7 Mar 2020 + 15 d          35.0 -> 10.4                  0.260 -> 0.565          7.2      44.8
          V3 13 Jul 2020 + 20 d          31.9 ->  6.2                  0.256 -> 0.733         15.5      43.6
          V4 26 Aug 2020 + 13 d          18.4 ->  8.5                  0.257 -> 0.598          3.7      25.4
          V5 14 Nov 2020 + 15 d          12.8 ->  8.8                  0.254 -> 0.526          6.3      15.3
          V6  6 Mar 2021 +  8 d          25.5 -> 14.4                  0.255 -> 0.520          0.0      31.6
          V7 23 Aug 2021 + 16 d          33.1 ->  8.1                  0.258 -> 0.703          0.4      41.4
        (LE missing 9 - 23 % of the half hours, filled with the mean diurnal cycle as before.)
        KNOWN BEFORE THE TESTS: with the water table at 0.25 - 0.26 m the sensor reads 36.0, 35.0, 31.9,
        18.4, 12.8, 25.5 and 33.1 %. V4, V5 and V6 follow a rise that did not bring the sensor back to its
        plateau (25, 15, 32 % against 44 - 45; on 3 Mar 2021 the well stood above the surface and the
        sensor read 32 %): another sensor - water-table relation (rewetting branch, trapped air), which a
        model with one retention curve cannot hold together with V1 / V2 / V3 / V7.
        Tests (protocol d unchanged: soil of the last earlier window whose sensor range contains the
        test's, to within sigma_theta = 0.005 as for T_N0; nothing fitted on a test window):
          T_H1 = V2 with the soil of V1      T_H2 = V3 with the soil of V1 (V2 ends at 10.4 %)
          T_H3 = V4 with the soil of V3      T_H4 = V5 with the soil of V4
          T_H5 = V6 with the soil of V3      T_H6 = V7 with the soil of V1 (V3 starts at 31.9 %)
        Expectation written now: T_H1 and T_H6 can pass (same branch, same relation to within 1 - 1.5 %);
        T_H2 is open (V3 reads 3 - 4 % below V1 at 0.3 - 0.4 m, in July); T_H3, T_H4 and T_H5 should fail
        R3 with the solver itself (the relation above).
        Rules, unchanged: soil = lowest misfit of the four starts of scripts/calibrate_soil.py (start 0 =
        the Florida set of CLAUDE.md), usable if the in-sample solver inversion is within 2 cm, r >= 0.975
        below saturation; R1 - R3 as in c. Solver side first (CPU): make_real for the seven windows,
        calibrations on V1, V3, V4, in-sample inversions, the six tests (four starts each, --box, point
        sensor). All six solver outcomes are reported. The PINN (hpc/real_full_obs.sh, unchanged) is run on
        every test with a usable soil whose solver inversion is inside R3 or within the R2 margin of it
        (1 cm RMSE, 1 cm bias): for the others R3 is decided by the solver, as for T_N0. A test with a
        usable soil whose PINN run meets R1 - R3 counts for R4; the report gives the full tally (passes out
        of six, and out of the tests whose window starts from the plateau: sensor >= theta_s - 3 sigma of
        the calibration soil at some hour of the 7 d before the start - an input-only check, declared here).
        If no soil is usable the site is reported as such; nothing is changed to make a test pass.
    az) 05:40. US-NC2, the solver side is COMPLETE (four starts everywhere; zb RMSE / bias / max in cm, r):
          W1 in-sample, S_W1   (nc2_own_may23_s0..3)              0.89 / +0.03 / 2.5, r 0.9975  usable
          W2 in-sample, S_W2   (nc2_own_sep23a_cal3_s0..3)        0.63 / +0.02 / 1.8, r 0.9978  usable
          T_N1 = W3 with S_W2  (nc2_sep23aon_sep23b_cal3_s0..3)   9.53 / -9.31 / 12.7, r 0.9982  R3 fails
          the same with start 1's soil (other end of the valley)  9.52 / -9.30 / 12.7           (aw)
        Every group ends at one cost to 1e-6 (0.150580 / 0.131741), first coefficient on the box.
        PINN runs:
          rfn1 (T_N1) inv0 (wide members only, job 15520550): four starts identical (2e-9 m);
            PINN - solver 0.07 cm rms, max 0.12 cm; vs well 9.53 / -9.29 / 12.8, r 0.9980
            -> R1, R2 hold; R3 fails by the 9 cm the solver has (the soil transfer W2 -> W3, aw).
          rfnd (diagnostic W3 with S_W1, not counting for R4):
            inv1 (job 15520238)  starts identical (7e-13 m)  PINN - solver 0.06 cm rms, max 0.14 cm
                                 vs well 1.58 / +0.55 / 2.5, r 0.9979
            inv2 (job 15520751)  starts identical (3e-9 m)   PINN - solver 0.06 cm rms, max 0.13 cm
                                 vs well 1.58 / +0.55 / 2.5, r 0.9979
          Both runs continue to the end of the driver (refinement rounds, value correction) unchanged.
        US-HB2, FIRST STEPS (after the push of ay, commit 7615349). GLDAS rows of well 330247079340300 in
        data/gldas_well330247079340300_2015_2024.csv; configs/real_hb2_{sep19,mar20,jul20,aug20,nov20,
        mar21,aug21}.yaml = V1..V7; make_real for the seven windows (jobs 15519848-69). With the Florida
        soil the FEM over the recorded water table reads 0.17 - 0.25 too wet at the sensor (r 0.94 -
        0.99): a much coarser sand, as the record shows (sensor at 6 % with the water table at 0.8 m).
        Calibrations on V1, V3, V4 submitted, four starts each (jobs 15519919-15520001).
        TOOL BUG, fixed (commit 54e078a): start 3 of V4 stopped at once, "Initial guess is outside of
        provided bounds": the static start takes theta_s = max(theta_obs) + 0.01 = 0.194 as its first
        guess, below the lower bound 0.30 (V4 never comes near saturation). The guess is now clipped to
        the bounds, as the starting soil of the main fit already is; windows whose guess was inside are
        unchanged. Start 3 of V4 resubmitted (job 15520572). lessons.md updated.
    ba) 08:20. US-NC2, PINN runs to the end of the pipeline (four refinement rounds), each inversion against
        its solver reference (four starts, one cost; zb RMSE / bias / max in cm against the well):
          rfnd (diagnostic W3 with S_W1, NOT counting for R4)
            inv3 (job 15521799)  starts identical (6e-10 m)  PINN - solver 0.06 cm rms, max 0.12 cm
            inv4 (job 15522578)  starts identical (3e-9 m)   PINN - solver 0.06 cm rms, max 0.11 cm
            value correction rfnddc: converged at iteration 1 (step 0.003 sigma), starts identical
            FINAL runs/rfnddc_it1: PINN - solver 0.06 cm rms, max 0.11 cm; vs well 1.59 / +0.55 / 2.6,
            r 0.9979 (solver 1.64 / +0.55 / 2.7, r 0.9978; hydrostatic reading of the sensor 1.64 / -0.33
            / 3.9). Figure output/real_nc2_sep23b_soilmay23/result_rfnd.png (job 15524846).
          rfn1 (T_N1 = W3 with S_W2, the pre-registered test)
            inv1 (job 15521579)  starts identical (2e-13 m)  PINN - solver 0.07 cm rms, max 0.09 cm
            inv2 (job 15522252)  starts identical (4e-9 m)   PINN - solver 0.06 cm rms, max 0.09 cm
            inv3 (job 15523110)  starts identical (3e-9 m)   PINN - solver 0.06 cm rms, max 0.09 cm
            inv4 (job 15524447)  starts identical (3e-9 m)   PINN - solver 0.06 cm rms, max 0.10 cm
                                 vs well 9.49 / -9.26 / 12.7, r 0.9980 (solver 9.53 / -9.31 / 12.7)
            value correction rfn1dc running. R1, R2 hold; R3 fails by the 9 cm the solver has (aw).
        US-HB2, calibrations: a member solve costs about 2.2 min here against 0.7 - 1.0 at the earlier
        sites (cause not looked into; windows of 13 - 20 d, alpha 5 - 18). Finished so far, all by xtol
        (rmse of theta; theta_s, theta_r, alpha, n, Ks):
          V1 start 0 (88 solves)  0.002807   0.3957 0.0581 5.413 3.557 1.80e-5
          V1 start 3 (99 solves)  0.002646   0.3868 0.0580 5.311 3.656 8.74e-6
          V4 start 3 (75 solves)  0.003651   0.3013 0.0016 18.74 1.653 3.40e-6
        the other nine are running (V1 starts 1, 2 at 0.00267, 0.00276; V3 0.00336 - 0.00416; V4 0.00130 -
        0.00169). TOOL (commit a216104): a calibration that reaches its time limit writes nothing, so
        scripts/calibrate_soil.py now writes every improvement to progress<start>.json and can continue a
        start from it (--resume) or from a soil given on the command line (--x0, for the jobs started
        before the change); the result records which (continued_from). Checked with three batch jobs
        (runs/_stage/caltest). Not needed so far.
        TOOL hpc/transfer_solver.sh (commits 54349a0, 47ded8d): the solver side of one soil transfer as one
        driver = soil of the lowest-misfit start (needs all four), make_real for the calibration window
        (in-sample check) and each test window, solver inversion from the four starts.
        DRY RUN of that chain for V1 (names *_pre, DOES NOT COUNT; started 07:09 with the state of start 3
        at solve 87, which is its final soil to four digits; log logs/transfer_hb2_sep19_pre.log). FEM over
        the recorded water table against the sensor (rmse / bias / r):
          V1 in-sample 0.0026 / -0.0001 / 0.9993     V2 0.0197 / +0.0139 / 0.9821
          V3 0.0138 / +0.0081 / 0.9952               V7 0.0170 / -0.0143 / 0.9923
        Solver inversions in progress (10 - 15 min per LM iteration). Seen so far:
          - V2 (T_H1): starts hydrostatic, hydrostatic_flat and 0.3 m at one cost 0.7928, zb RMSE 2.50 cm.
            The start at 1.0 m did not move (cost 622, 23 member solves, lambda 1e4): with the water table
            at 1.0 m this sand holds the sensor at theta_r, the sensitivity of the sensor to the water
            table is nil and no step is accepted. A start in the part of the box the sensor cannot see.
          - V7 (T_H6): three starts at cost 0.1835, zb RMSE 4.95 cm (days 0 - 12: 3.65); start 1.0 m moving.
          - V1 in-sample: cost 0.66 - 0.73 after two iterations, zb RMSE 1.6 - 2.0 cm on days 0 - 15 and
            7.5 - 9.5 cm on the whole window: in the last days the water table is below about 0.6 m and
            the sensor is at theta_r (same blindness; the usable-soil rule is scored on the whole window).
    bb) 09:05. US-NC2 T_N1 FINAL (rfn1, the pre-registered test: W3 with S_W2). Value correction rfn1dc
        converged at iteration 1 (step 0.012 sigma), starts identical (1.5e-8 m). runs/rfn1dc_it1:
          PINN - solver 0.07 cm rms, max 0.10 cm                      -> R1, R2 hold
          vs well: PINN 9.48 / -9.25 / 12.7 cm, r 0.9980; solver 9.53 / -9.31 / 12.7, r 0.9982;
                   hydrostatic reading of the sensor 11.1 / -11.0 / 14.0  -> R3 fails, as the solver
        Figure output/real_nc2_sep23b_soilsep23a_cal3/result_rfn1.png (job 15525891). US-NC2 is closed:
        T_N0 not run (soil range rule, R3 decided by the solver), T_N1 fails R3 through the soil
        transfer W2 -> W3 (aw), the diagnostic with S_W1 (rfnd, ba) is 1.59 cm from the well and does
        not count. On both data sets the PINN is the solver inversion to 0.06 - 0.07 cm.
        US-HB2 dry run (does not count), V2 with the state of V1 start 3, solver start hydrostatic
        (output/referee/hb2_sep19on_mar20_pre_s0, 6 iterations, cost 0.7928, coefficients 0 and 5 on the
        box): 2.50 / +1.23 / 8.0 cm, r 0.9679. By day (cm): +2.8 +0.6 +0.9 +2.0 +2.8 +4.3 +3.6 +1.8 +1.2
        +0.1 +0.1 +1.0 +1.7 +0.9 -1.5 -6.8; posterior sd 0.1 cm until day 10, 1.4 cm on the last day (sensor
        at 10 - 11 %, close to theta_r). RMSE and bias inside R3, r below 0.98 through the last day.
        A PINN dry run on the same data set (tag rfh1p, started 08:17, does not count) checks the tool on
        this soil (alpha 5.3 with n 3.7; so far US-ONA alpha 1.4 - 2.7 with n 8 - 10, US-PAS 1.4 - 1.7 with
        n 3.2 - 4.8, US-NC2 5.6 - 14.8 with n 1.2 - 1.3) while the last calibration start of V1 runs.
        (The commit message of a216104 puts the slow solves down to "n near 3.7": not established.)
    bc) 09:45. US-HB2: all four starts of the V1 calibration ended by xtol (misfit of theta 0.002807 /
        0.002667 / 0.002761 / 0.002646, nothing on a bound): S_V1 = start 3 (theta_s 0.3868, theta_r 0.0580,
        alpha 5.311, n 3.656, Ks 8.74e-6), the soil of the dry run to four digits. OFFICIAL solver chain
        for T_H1, T_H2, T_H6 and the in-sample check started 09:29 (RT=08:00:00 hpc/transfer_solver.sh hb2
        sep19 "[0.1524,0.1524]" mar20 jul20 aug21; logs/transfer_hb2_sep19.log; data sets
        output/real_hb2_sep19_own, output/real_hb2_{mar20,jul20,aug21}_soilsep19). Forward checks equal
        to the dry run's (ba). Dry run, V2: starts hydrostatic, hydrostatic_flat and 0.3 m end at one cost
        (0.792835 / 0.792835 / 0.792835) and one estimate; the start at 1.0 m stays where it is (cost 622).
        CHECKED AND DROPPED: could the misfit of the inversion on the test window pick the soil among the
        earlier recessions' (an input-only choice)? No. US-NC2, solver inversions already on disk:
          W2 with its own soil   cost 0.151   0.6 cm from the well
          W2 with S_W1           cost 0.123  10.6 cm          <- lower misfit, wrong water table
          W3 with S_W1           cost 0.063   1.6 cm
          W3 with S_W2           cost 0.132   9.5 cm
        The sensor alone does not tell the soil from the water table: a soil that is wrong for the
        window can fit the sensor better with a shifted water table. Protocol d stays as it is.
    bd) 22:10. US-HB2, OFFICIAL solver side of the three tests with the soil of V1 (transfer chain of 09:29,
        finished 17:38; S_V1 = start 3). Solver inversions from the four starts, against the well (RMSE /
        bias / max in cm, r; this sand never reads saturated, so "below saturation" is the whole window):
          in-sample V1    9.3 / -3.7 / 27.5, r 0.84   starts 0, 1, 3 at cost 0.651 - 0.653; start 1.0 m at
                          0.6507 after 21 iterations when the 8 h limit ended it. Days 0 - 15: 1.9 / +0.5.
                          Days 16 - 20 (well below 0.6 m, sensor at theta_r): -5 ... -27 cm, with a
                          posterior sd of the inversion of 5 - 27 cm there: the sensor does not see it.
                          -> by the rule written beforehand (in-sample within 2 cm, r >= 0.975) S_V1 is
                          NOT a usable soil.
          T_H1 (V2)       2.50 / +1.22 / 8.0, r 0.968  starts hydrostatic, flat, 0.3 m at one cost (0.79239);
                          the start at 1.0 m does not move (cost 622). RMSE and bias inside R3, r below 0.98
                          (last day -6.8 cm).                                                  R3 fails
          T_H2 (V3)       16.7 / +7.0 / 37.7, r 0.66   starts 0, 1, 3 at cost 1.3328 - 1.3329; start 1.0 m at
                          1.3328 after 24 iterations at the limit. Days 0 - 10 within 5 cm, then the well
                          falls below 0.6 m and the estimate is free.                          R3 fails
          T_H6 (V7)       4.96 / -4.29 / 8.9, r 0.992  all four starts at one cost (0.18350).   R3 fails
        Written beforehand (73 ay): T_H1 and T_H6 can pass, T_H2 open. Outcome: none passes; T_H1 misses on
        r only. No test of this soil qualifies for a counting PINN run (soil not usable).
        PINN on the dry-run data set of T_H1 (rfh1p; soil = S_V1 to four digits; does NOT count; solver
        reference hb2_sep19on_mar20_pre, 2.50 / +1.23 / 8.0, r 0.968):
          wide members only (inv0)   PINN - solver 0.62 cm rms (max 1.49)
          refinement rounds 1 - 4    0.33 / 0.20 / 0.19 / 0.19 cm rms (max 0.49 at round 4)
          value correction           converged at it 3 (step 0.09 sigma): 0.22 cm rms (max 0.60)
          against the well           2.57 / +1.15 / 8.0, r 0.965 (runs/rfh1pdc_it3)
          starts                     all five identical (spread 2e-9 m), INCLUDING the start at 1.0 m, which
                                     the solver's own inversion cannot leave (PINN: first step accepted,
                                     cost 362; solver: all ten trial steps rejected, cost stays 622).
                                     Why the two differ there was not examined.
        So R1 and R2 hold at the fourth site as at the other three; the distance to the well is the
        solver's. V3 calibration complete, all by xtol: starts 0, 2, 3 at misfit 0.004145 (theta_s 0.3826,
        theta_r 0.0618, alpha 6.59, n 3.17, Ks 3.6e-6), start 1 at 0.003309 (theta_s 0.3686, theta_r 0.0341,
        alpha 5.675, n 2.926, Ks 5.28e-4) -> S_V3 = start 1, nothing on a bound. Chain for T_H3 and T_H5
        started 22:01 (RT=12:00:00 hpc/transfer_solver.sh hb2 jul20 "[0.1524,0.1524]" aug20 mar21;
        logs/transfer_hb2_jul20.log). V4 calibration: starts 0 - 2 ran into the 8 h limit still moving
        (misfit 0.00128 / 0.00166 / 0.00101 after 190 / 153 / 173 solves; start 3 ended at 0.003651). They
        began before the progress file existed, so each is CONTINUED from the last soil in its log with
        --x0 (jobs cal_hb2aug20_s0c / s1c / s2c, 8 h; result key continued_from = "x0").
    be) 22:20. README: section "Results on site data" with the six scored PINN runs (PINN - solver, PINN /
        solver / hydrostatic reading against the well, figure links); numbers from the result_*.json files.
        US-HB2, soil of V3 (S_V3 = start 1), forward checks of the chain started 22:01 (FEM over the
        spline of the observed water table against the sensor; sigma_theta of the inversion is 0.005):
          V3 in-sample     rmse 0.0033, bias  0.0000, r 0.9984
          T_H3 (V4)        rmse 0.0651, bias +0.0464, r 0.9924
          T_H5 (V6)        rmse 0.0366, bias +0.0111, r 0.9928
        The model with S_V3 is 0.046 too wet over V4 on average: the sensor starts V4 at 18 % and V6 at
        25.5 % with the water table near where V3 starts at 32 %. Solver inversions running (12 h limit).
        CONSIDERED AND NOT ADOPTED (no run made; signs only): scale theta_s - theta_r of the calibration
        soil by the highest sensor reading M of the 7 days before the window (wet-end reading of the
        event; an input, no well needed), theta_s_test = theta_r + (theta_s_cal - theta_r) (M_test -
        theta_r) / (M_cal - theta_r). M_test < M_cal makes the estimate shallower, M_test > M_cal deeper.
        M (%, /tmp/hb2tools/prewin.sh), and the sign of the change against the bias the solver has now:
          US-HB2  V1 44.4  V2 44.8  V3 43.6  V4 25.4  V5 15.3  V6 31.6  V7 41.4
                  T_H1 (V2 <- V1): deeper,    bias now +1.2  -> wrong way (by 1 % of the range)
                  T_H2 (V3 <- V1): shallower, bias now +7.0  -> right way (2 %; the error there is the
                                                               blind tail, bd)
                  T_H6 (V7 <- V1): shallower, bias now -4.3  -> wrong way (7 %)
          US-ONA  sep21 38.9  nov21 37.7  oct22 36.9  nov22 36.0
                  C <- B: shallower, bias now +2.5 -> right way;  D <- C: shallower, +1.5 -> right way
          US-PAS  aug21 43.8  sep21 41.8  oct21 43.5  nov21 43.5
                  T_P2 (oct21 <- sep21): deeper, bias now -0.7 below saturation, +0.2 whole window
          US-NC2  may23 51.3  sep23a 50.6  sep23b 55.7  (layer sensor 0 - 0.30 m: M depends on how far the
                  water table rose into the layer, not on the wet end of the curve alone)
        Two of five signs against it, at the site where the rule would be judged (US-HB2) one of them by
        the largest factor. Choosing it now would be choosing a rule on tests that are already scored.
        Not adopted; protocol d stays.
    bf) 22:35. Is there a FIFTH site in fem_validation/uploaded_data? (read-only, awk over the half-hourly
        files on the login node, nothing fitted; not looked at in ay: the 17 sites below). Counted: Jun -
        Sep half-hours with the water table 0.15 - 1.2 m below the surface together with a soil-moisture
        record and LE; then the reading of the sensors by depth of the water table (mean and sd over all
        years, 0.1 m bins).
          US-Ho1 (forest on stony till; sensors at -0.05 / -0.1 / -0.2 / -0.5 / -1 m, wells 0.8 and 1.45 m
                  deep, 2008 - 2025): sensor at -0.1 m reads 31.6 / 30.2 / 29.2 / 28.6 / 28.1 / 27.0 / 26.3 %
                  with the water table at 0.0 ... 0.6 m, spread within a bin 10 - 15 %: 0.09 theta per m.
                  A retention drift of 0.01 is 11 cm of depth (US-NC2: 0.26 per m; US-HB2: 1 per m) -> no.
          CA-Mer (bog, peat; 1998 - 2020, water table 0.2 - 0.7 m): sensors at -0.1 m follow the rain, not
                  the water table (51 / 46 / 36 / 37 / 35 / 30 % with sd 5 - 17 %; another 21 / 12 / 10 / 7 /
                  5 / 4 % with sd up to 16 %). The sensors at -0.4 / -0.5 m follow it closely (SWC_3_5_1:
                  89.7 / 86.7 / 63.3 / 49.7 / 42.2 / 36.2 % at 0.2 ... 0.7 m, sd 1.6 - 9 % over eleven
                  years) but they sit 0 - 0.3 m above the water table: not the near-surface problem this
                  work is about, and peat (compressible, moss layer) is outside the rigid van Genuchten
                  soil of the model. Not taken; noted as the one record where a deep sensor would work.
          CA-ARB, CA-ARF, CA-PB2 (bog / fen): water table within 0.0 - 0.3 m (0.5 m at most) -> no range.
          CA-CF3 (2024 only, above the surface), CA-TP4 (5.5 - 7.4 m), US-Ton (1.5 - 9.4 m), CL-SDF / CL-SDP,
          US-Elm, US-Esm (water table above the surface), US-HRA / US-HRC (rice, irrigated: the flux
          cannot be built), US-StJ, US-Hsm (tidal marsh: the water table moves twice a day), US-Uaf
          (permafrost: frozen layer), US-CLF (3 - 5 m), US-CRT (drainage tiles) -> no.
        So with near-surface sensors the four sites used are all there is in this collection. R4 cannot
        be tried on further data here; it is decided by US-HB2 (T_H3 - T_H5 running).
    bg) 22:55. CORRECTION of bf: CL-SDF and CL-SDP were set aside there as "water table above the surface".
        Wrong: their WTD_PI_1 is a PI variable with the opposite sign (positive = depth below the surface;
        the reading of the sensor falls as the value rises, well 0.5 m deep, sensor SWC_PI_1 at -0.05 m,
        both from the measurement-height table). The sign had been assumed from the AmeriFlux convention
        and not checked (lessons.md). US-Elm (0.27 - 1.47 m, WTD at height 0) stays as stated; US-Esm has
        no half-hour with the sensor and the well together.
        Looked at properly (dry spells of >= 7 days, /tmp/hb2tools/spells.sh; nothing fitted):
          CL-SDP (peatland, 2014 - 2022): the sensor at 5 cm reads 20 - 46 % with the water table at 0.18 - 0.21 m
                  (45.8 Dec 2014, 33.7 and 23.0 Jan 2016, 44.3 Jan 2020, 20.0 Mar 2019): the moss layer
                  does not follow the water table, as at CA-Mer. The rain record has holes read as zero
                  (7 Apr 2021 + 28 d: 1.2 mm recorded, water table 0.53 -> 0.04 m). -> no.
          CL-SDF (forest, 2014 - 2021): nine recessions of 7 - 11 days in which the water table falls from
                  0.19 - 0.28 m to 0.36 - 0.47 m (the well ends at 0.50 m) with LE on 72 - 86 % of the
                  half-hours (Sep 2014, Apr 2015, Oct 2015, Nov 2016, Apr 2017, Dec 2019, Jan / Feb / Mar
                  2020); the sensor falls by 5 - 17 % over them (0.2 - 0.8 theta per m). At the start of
                  these recessions, water table 0.19 - 0.28 m, it reads 53.8 / 42.9 / 53.0 / 59.6 / 52.5 /
                  64.7 / 52.8 / 54.6 / 46.7 %: a spread of 22 % at one water table, more than the whole
                  fall within a recession. A POSSIBLE fifth site with a near-surface sensor, but one whose
                  recessions repeat each other no better than US-HB2's (start readings 12.8 - 36.0 % at
                  0.255 m). Not started: by the same reasoning as at US-HB2 most of its tests would fail
                  on the solver side before the PINN is asked anything. Kept as the candidate if a
                  further site is wanted.
        So bf's last sentence becomes: with near-surface sensors there is one further candidate (CL-SDF),
        of the same kind as US-HB2.
    bh) 23:25 (12 Nov). FIGURES OF THE PINN RUNS, AND A WINDOW WITH RAIN (asked by the user: every window shown
        is a monotone drydown).
        Figures against the observations only (scripts/plot_real_pinn.py, new: sensor observed / PINN, water
        table observed / PINN; no solver line), commits 8072005 and 9eae874:
          output/real_ona_nov22_caloct22/result_rf22c_pinn.png, output/real_ona_oct22_soilnov21/result_rf22ob_pinn.png,
          output/real_pas_oct21_soilsep21/result_rfp2_pinn.png, output/real_hb2_mar20_soilsep19_pre/result_rfh1p_pinn.png
        Checked for the answer: the forcing of every data set so far is DAILY nodes (centred daily means,
        linear in between), the PINN fits the hourly sensor; the diurnal part of the sensor (theta minus its
        centred 24-h mean) is 0.0012 - 0.0033 rms, 0.003 - 0.008 peak to peak, against falls of 0.2 - 0.25.
        All windows are rain-free by construction (68 e).

        RAIN WINDOW - facts from the records, before any run (awk over the half-hourly file, /tmp/hb2tools/
        daily.sh; US-ONA has sensor and well together in 2021 - 2022 only):
        - Search: 12-day windows, sensor and well complete, water table 0.15 - 1.3 m throughout, a rain day
          >= 3 mm in the middle. Found: Feb 2021 (1.09 - 1.28 m, sensor 7 - 11 %), 28 Oct - 10 Nov 2021,
          13 - 29 Dec 2021 (0.97 - 1.22 m, sensor 9 - 12 %), 5 - 22 Oct 2022 (LE on 0 - 29 % of the half-hours), 25 Oct - 9 Nov 2022,
          7 - 31 Dec 2022.
        - 28 Oct - 10 Nov 2021: dry since 13 Oct. 28 Oct 12:30-17:00, 4.6 mm: well 1.04 -> 0.85 m by 21:00,
          sensor 12.0 -> 13.9 %. 5 Nov, 25.2 mm: 10 mm until noon (1.03 -> 0.76 m, 10.9 -> 15.1 %), 15 mm from
          19:30, 13.8 mm of it in the first hour: well 0.745 -> 0.384 -> 0.264 -> 0.204 m at 20:30 / 21:00 /
          21:30, sensor 14.9 -> 33.9 %. Then a recession to 0.67 m / 29.0 % on 10 Nov. Well and sensor answer
          the rain within the hour; 27 cm of rise on 10 mm, 55 cm on 14 mm (2.5 - 3.7 mm of rain per 10 cm).
        - 2 Nov 2022 19:00-20:30, 6.8 mm: sensor 13.4 -> 18.5 % within the hour; the well stays at 0.90 m,
          steps to 0.69 m on 3 Nov 20:00-21:30 and back to 0.86 m on 4 Nov 16:00-16:30 (12 cm in one
          half-hour), no rain and no change at the sensor at either step. 15 Dec 2022, 14 mm: sensor
          14.3 -> 31.4 %, the well stays at 0.87 - 0.92 m for two days and is at 0.30 m on 17 Dec (no rain,
          sensor 30.4 - 31.1 %). In late 2022 the well moves one to two days after the rain and not with the
          sensor: steps that no reading of the sensor can time. NOT taken (cause not known, not looked for).
        - Dec 2021 and Feb 2021: sensor at 7 - 12 %, around theta_r of the calibrated soils (0.098 - 0.130):
          next to no sensitivity to the water table (hand check with S_B: 0.0008 theta per cm at 0.9 - 1.07 m).
          Not taken.
        -> window W_R1 = US-ONA 27 Oct 2021 12:00 + 14 d (configs/real_ona_rain21.yaml), the one window in which
        rain, sensor and well move together. Water table 0.20 - 1.07 m, sensor 10.8 - 33.9 %.
        What rain changes in the set-up (fixed here, before any run; nothing else differs from the recessions):
          forcing.node_hours 24 -> 1   (bursts of 30 - 60 min; daily nodes spread them over two days)
          zb_spline.knot_days 5 -> 1   (17 coefficients; the rise of 5 Nov takes two hours, so even daily
                                        knots smooth it: the fit of the spline to the record is reported)
        Soil: NO soil qualifies by protocol d. The only earlier recession is A (Sep 2021, S_A, usable), sensor
        range 0.221 - 0.379: it does not contain 0.108 - 0.221. B (S_B, 13 Nov 2021 + 30 d) is the recession
        that FOLLOWS and S_B is not a usable soil (ac); S_C is a year later. So W_R1 cannot count for R4
        whatever comes out; it is a demonstration. Soil of the runs = S_A (the one earlier, usable soil;
        extrapolated below 0.221, which is said with every number). S_B and S_C: forward check only.
        Known beforehand (retention curves, hand check, not a result): theta_r of S_B (0.1114) and S_C (0.1261)
        is above the 10.8 - 10.9 % the sensor reads before the rain; and after the burst of 5 Nov the sensor
        reads 33.9 % with the well at 0.20 m, while it read 35.6 - 37.4 % with the well at 0.17 - 0.30 m on the
        recessions (13 Nov 2021, 23 Nov 2022; theta_s 0.350 - 0.369): on re-wetting the sensor stays 2 - 4 % below the drying curve the soils were
        calibrated on (hysteresis / trapped air), and a single-valued retention curve reads that as a water
        table 0.3 - 0.4 m deeper than the well for the first days.
        Order (solver = forward check and scoring only, as always):
          1. scripts/make_real.py, W_R1 with S_A / S_B / S_C: FEM with the OBSERVED hourly water table and
             the hourly rain against the sensor (record and spline). This says whether the Richards model
             with a recession soil holds through a rain event at all; the level of the transfers that led
             to R3-level inversions is 0.007 - 0.010 rmse (g, line 1459).
          2. solver inversion (fem_referee, four starts, --box) and PINN (hpc/real_full_obs.sh, unchanged)
             on W_R1 with S_A - run whatever 1 gives (the user asked to see it), expectation written down
             from 1 before the numbers of 2.
          3. scripts/score_real.py and plot_real_pinn.py; R1 - R3 as they are, whole window and below
             saturation; additionally the scores for the hours before the burst of 5 Nov and after it.
    bi) 23:40 (12 Nov). RAIN WINDOW W_R1: THE FORWARD CHECK (step 1 of bh) AND WHAT IS EXPECTED OF THE INVERSION
        (written and committed before any inversion of W_R1 is submitted).
        Data sets (scripts/make_real.py, configs/real_ona_rain21.yaml; jobs 15556531 - 15556533 COMPLETED, each
        real_summary.json written): output/real_ona_rain21_soilsep21 (S_A), _soilnov21 (S_B), _soiloct22 (S_C).
        Common: 337 hourly values, 17 coefficients; LE filled 9.2 %; rain 31.2 mm, ET 27.2 mm; water table
        0.202 - 1.066 m; sensor 0.1077 - 0.3383; FEM converged 100 %; admissible (surface head >= -1.16 m).
        The representation: the spline with 1-day knots misses the hourly record of the well by 4.85 cm rms
        (bias -0.23 cm, max 26.3 cm at the burst, r 0.9789). The estimate is such a spline, so against the
        hourly record it cannot be inside R3 (3 cm, r 0.98) on the whole window even with a perfect model.
        Not changed (bh fixed it); the spline fit is the floor the scores are read against.
        FEM with the OBSERVED water table (hourly record), fT = gldas, against the sensor (theta rmse / bias):
                                                   S_A                S_B                S_C
          whole window (337 h)                     0.0271 / +0.0197   0.0146 / +0.0117   0.0177 / +0.0144
          before the rain of 28 Oct (24 h)         0.0044 / -0.0044   0.0012 / -0.0010   0.0070 / +0.0070
          28 Oct 12:00 - 5 Nov 04:00 (185 h)       0.0150 / +0.0109   0.0114 / +0.0096   0.0158 / +0.0148
          5 Nov 05:00 - 19:00 (15 h, first rain)   0.0184 / +0.0104   0.0213 / +0.0162   0.0313 / +0.0267
          5 Nov 20:00 - end (113 h, from burst)    0.0422 / +0.0405   0.0190 / +0.0174   0.0196 / +0.0137
        (spline instead of the record: 0.0271 / 0.0160 / 0.0172; fT 0 - 1: S_A 0.0274 - 0.0297.) The level of
        the transfers that led to R3-level inversions is 0.007 - 0.010: only the first day is at that level.
        FEM (dt 1 h, 101 nodes) against the finer FEM (15 min, 201 nodes): rmse 0.0034 - 0.0039, max 0.055 -
        0.064, in the hour of the burst (13.8 mm in one hour = 2.4 Ks of S_A).
        What the series shows (S_A; output/real_ona_rain21_soilsep21/real_check.png):
        - 28 Oct, 4.6 mm: the sensor goes 0.121 -> 0.1385 within the hour. The FEM with the observed water
          table is still at 0.1145 six hours later and reaches 0.134 - 0.146 half a day to 2.5 days later; from
          31 Oct to 5 Nov it is 0.013 - 0.021 above the sensor, which dries faster than the model.
        - after the burst the FEM has the sensor at saturation (0.3686) until 8 Nov and at 0.343 at the end.
          The sensor peaks at 0.3383 (5 Nov 23:00, well at 0.22 m) and ends at 0.3016. In the 49 h after the
          burst with the well above 0.5 m the sensor reads 0.323 - 0.338 (mean 0.331): 0.030 - 0.045 below
          theta_s of S_A, 0.012 - 0.027 below S_B's (0.3498), 0.013 - 0.028 below S_C's (0.3511). This is what
          bh wrote down from the records beforehand; no soil calibrated on a recession holds through the event.
        Hand check, NOT the inversion: the hydrostatic reading of the sensor with each retention curve
        (water table = 0.10 m + |h|(theta)), against the well (awk on real.npz):
          S_A  before 28 Oct 2.0 cm rms (bias -0.2); 28 Oct - 5 Nov 5.6 (+4.7); first rain of 5 Nov 13.7
               (+12.1); from the burst 22.7 (+19.9): 0.675 m where the well is at 0.202 m, 0.732 against
               0.674 m at the end; whole window 14.1 cm rms, bias +9.8 cm
          S_B  no reading for 36 h (sensor below theta_r 0.1114); from the burst 15.9 cm (+12.2): 0.580 m at
               the peak, 0.667 against 0.674 m at the end; the 301 h with a reading 12.6 cm rms
          S_C  no reading for 139 h (sensor below theta_r 0.1261, all of the first day); from the burst 14.3
        EXPECTATION for the inversion with S_A (solver; and the PINN if it follows the solver as it did on the
        recessions) - it differs from the hand check by the dynamics, so orders of magnitude only:
          - the rise of 28 Oct (19 cm within the hour) is seen in part: about 1.0 -> 0.95 m where the well goes
            1.04 -> 0.85 m; then 2 - 6 cm deeper than the well until 5 Nov;
          - at the burst the estimate rises to 0.65 - 0.70 m, not to 0.20 m, and stays within 0.65 - 0.75 m to
            the end while the well recedes from 0.20 to 0.67 m: about +45 cm right after the burst, +5 cm at
            the end, of the order of 20 cm rms from the burst on;
          - whole window of the order of 14 cm rms, bias about +10 cm: far outside R3. The cycle (deep - rise -
            recession) is there in sign and timing, the size of the rise is not.
          - R1 (starts agree) and R2 (PINN = solver within 1 cm): open. First window with rain, 17 coefficients
            instead of 3 - 7, and a burst the hourly solver itself resolves to 0.06 theta only.
        The miss, if it comes out as expected, is the retention curve (drying curve of a recession used on a
        re-wetting soil; S_A also extrapolated below 0.221): the same for the solver and for the PINN, and not
        something an inversion can repair.
        ADDED after the forward check (not in bh; diagnostics on the solver side, never counted for R1 - R4):
          - solver inversions with S_B and S_C too (four starts each): how much of the miss is the soil;
          - calibration of a soil ON W_R1 (scripts/calibrate_soil.py, four starts; data set
            output/real_ona_rain21 = the config's own soil, as for every calibration): is there ANY van
            Genuchten soil that holds the sensor through the event with the observed water table? A misfit
            well above the 0.004 - 0.005 of the recessions = the window is outside the single-valued model
            whatever the soil; a misfit at that level = a rain window needs the soil of a rain window (and a
            second rain window to test it on, which US-ONA does not have: bh).
        Runs: PINN tag rfr1a (hpc/real_full_obs.sh, unchanged) on S_A; solver output/referee/ona_rain21_s{A,B,C}_s0-3.
    bj) 02:52 (13 Nov). RAIN WINDOW W_R1: SOLVER INVERSIONS WITH THE RECESSION SOILS, AND THE SOIL CALIBRATED ON
        THE WINDOW. Jobs 15557247 - 15557258 COMPLETED, each summary.json written (output/referee/
        ona_rain21_s{A,B,C}_s0-3; fem_referee, --box, four starts; 2 h 18 - 3 h 06 each).
          soil start             cost    zb rmse  bias    max    coef. on a bound  iter.  theta rmse
          S_A  hydrostatic       1.9289  24.09    +17.10  48.35  4 of 17           40     0.00694
          S_A  hydrostatic_flat  1.9320  23.22    +16.25  48.38  4                 40     0.00695
          S_A  constant:1.0      1.9303  23.80    +16.79  48.33  4                 40     0.00695
          S_A  constant:0.3      1.9289  24.14    +17.12  48.35  5                 40     0.00694
          S_B  (four starts)     2.3575 - 2.3860   34.75 - 39.94  +27.6 - +32.0   5 - 9   32 - 40  0.0077
          S_C  (four starts)     7.86 - 11.85      28.28 - 40.07  +15.9 - +31.3   6 - 9   40       0.014 - 0.017
        (cm; 40 = the iteration limit of fem_referee: the S_A and S_C runs and one S_B run stopped at it.)
        S_A by part (start 0; estimate against the hourly well):
          before the rain of 28 Oct     25.2 cm rms, bias +22.7      starts apart by up to 10.5 cm
          28 Oct - 5 Nov 04:00          25.7         +17.8           up to 5.2 cm
          5 Nov 05:00 - 19:00           16.9         -16.3           0.1 cm
          5 Nov 20:00 - end             21.8         +19.2           < 0.1 cm
          whole window: r 0.794; spread between the starts 2.7 cm rms (R1 asks for < 0.2 cm: not met)
        Against the expectation of bi:
          - from the burst on AS EXPECTED: estimate 0.61 m at the burst (well 0.20 m), 0.68 - 0.73 m to the end
            (well 0.35 - 0.67 m); 21.8 cm rms where "of the order of 20" was written.
          - before the burst NOT as expected (2 - 6 cm written, 25 cm found). The estimate swings between
            0.89 and 1.50 m (the deep bound, 3 - 4 Nov) while the well stays within 0.85 - 1.07 m, and it
            starts to rise on 4 Nov, a day before the rain. The hydrostatic hand check does not predict the
            inversion where the sensor is near theta_r: there the retention curve gives 0.0016 theta per cm
            (S_A, 1.0 - 1.1 m), and the forward check has the model 0.013 - 0.021 wetter than the sensor on
            those days. The fit to the sensor is 0.0069 all the same: the water table pays for the model error.
        So with every recession soil the solver itself misses the well by 23 - 40 cm on this window. Whatever the
        PINN (rfr1a, S_A, running) gives, it cannot be better than this against the well; what rfr1a can still
        show is R2 (PINN = solver on a window with rain), with the caveat that the solver's four starts are
        themselves apart by up to 10 cm before the burst.
        CALIBRATION OF A SOIL ON W_R1 (output/soil_ona_rain21; data set output/real_ona_rain21, job 15557259:
        the Florida set gives 0.0757 / +0.0614 over the record). State 02:52:
          start 3 (static start)  0.006546, bias 0.0000, max 0.056, r 0.9976   theta_s 0.3311 theta_r 0.1057
                                  alpha 1.555 n 10.0 (bound) Ks 4.36e-6        39 min, 62 solves
          start 0                 0.018597  theta_s 0.549 theta_r 0.015 alpha 10.2 n 1.62 Ks 3.7e-6
          start 1                 0.019736  theta_s 0.597 theta_r 0.049 alpha 15.6 n 1.64 Ks 9.4e-6
          start 2                 running, 0.018963 after 66 solves
        -> a van Genuchten soil that holds the sensor through both rain events with the observed water table
        EXISTS (0.0065; the recessions' own fits are 0.0042 - 0.0051). It is the recession soils' curve (alpha
        1.43 - 1.66, n 9.4 - 10) with a lower wet end: theta_s 0.331 against 0.350 - 0.369. The window is inside
        the single-valued model; what does not carry over from a recession is theta_s after a re-wetting.
        And it does not carry over between rain events either: after the rain of 11 - 12 Nov (well to 0.07 m)
        the sensor reads 0.356 - 0.376 with the well at 0.17 - 0.30 m, above this theta_s. So a soil calibrated on
        W_R1 has no later window at US-ONA it could be tested on (bh: 2022 and Dec 2021 are out for other reasons).
        Hand check with the W_R1 soil (hydrostatic reading against the well, as in bi): 5.8 cm rms before 28 Oct,
        4.5 (bias +1.0) to 5 Nov, 8.1 on the first rain; after the burst 22 h at saturation (no reading), then
        7.5 cm to 8 Nov and 2.1 cm on the last two days. The sensor is at or above theta_s - 3 sigma (0.3161) for
        84 h (5 Nov 21:00 - 9 Nov 10:00): the well is then at 0.20 - 0.59 m, inside the capillary fringe of this
        sand, where the sensor does not see it.
        NEXT (in-sample, a DEMONSTRATION OF THE METHOD on a window with a cycle; nothing here counts for R3 / R4
        because the soil has seen the well of this window):
          1. hpc/transfer_solver.sh ona rain21 with SOIL_SET = the soil of start 3 (names *_pre, as the rule
             needs all four starts; if start 2 ends below 0.006546 the chain is redone with it): data set
             output/real_ona_rain21_own_pre, solver from the four starts (output/referee/ona_own_rain21_pre_s0-3).
          2. PINN rfr1o (hpc/real_full_obs.sh, unchanged) on that data set, launched when the data set exists
             and not after the solver result: the question for the PINN here is R2 through a rain burst, which
             needs both runs whatever the solver gives, and the user asked to see a window with a cycle.
          EXPECTATION (before either): solver 5 - 10 cm rms against the hourly well on the whole window (the
          1-day spline alone is 4.85 cm); the rise at the burst is there but its depth is open for about three
          days after it (sensor at saturation: anything above about 0.45 m fits; shallow bound 0.20 m); the last
          two days within a few cm. Before the burst the hand check says 5 cm, but it said 2 - 6 cm for S_A and
          the solver gave 25: the same low sensitivity applies, so 5 - 15 cm there would not surprise.
          R1: the starts may again differ before the burst. R2: open.
        03:06: all four calibration starts on W_R1 finished (jobs 15557260 - 15557263 COMPLETED, start?.json
        written): 0.018597 / 0.019736 / 0.018963 / 0.006546 (starts 0 - 3). Lowest = start 3, the soil the *_pre
        chain was started with (same file), so its results stand as the rule's. Data set
        output/real_ona_rain21_own_pre (job 15565909): FEM over the hourly record 0.006546 / 0.0000 (= the
        calibration), FEM against the finer FEM 0.0015 (max 0.015); with the 1-day SPLINE of the well instead
        of the record 0.0150 (max 0.157): the daily knots alone cost 0.015 in theta on this window, because
        the spline cannot time the rise of 5 Nov. PINN rfr1o launched 02:58 on that data set.
    bk) 03:21 (13 Nov). US-HB2, OFFICIAL solver side of the two tests with the soil of V3 (chain of 22:01, ended 03:17;
        logs/transfer_hb2_jul20.log; four starts each, --box, point sensor; zb RMSE / bias / max in cm, r, from
        /tmp/byday.sh on the lowest-cost start; this sand never reads saturated, so "below saturation" is the
        whole window):
          in-sample V3    1.23 / +0.15 / 4.4, r 0.9950   all four starts at cost 1.43567 (spread between starts
                          0.012 cm at most; 4 - 11 iterations; coefficient 0 on the shallow bound in all four).
                          By day within 2.3 cm except the first (+3.1) and the last (+4.4, posterior sd 1.8).
                          -> by the rule written beforehand (within 2 cm, r >= 0.975) S_V3 IS a usable soil,
                          the first at this site (S_V1 was not, bd).
          T_H3 (V4)       7.58 / +6.31 / 13.6, r 0.9942  all four starts at cost 0.042106 (spread 0.0006 cm;
                          6 - 9 iterations; nothing on a bound; theta fitted to 0.0010). By day +13.6 +13.1
                          +11.7 +11.0 +9.1 +7.7 +7.1 +6.1 +4.9 +3.1 +1.7 +1.5 +2.0 +2.5.          R3 fails
          T_H5 (V6)       4.36 / +0.05 / 8.0, r 0.9841   all four starts at cost 0.052791 (spread 0.0001 cm;
                          4 - 12 iterations; nothing on a bound; theta fitted to 0.0011). By day +6.1 +6.4
                          +3.8 +1.8 +0.6 -1.4 -4.3 -6.1 -7.4: no bias, a tilt.                    R3 fails
        Written beforehand (73 ay): T_H3 and T_H5 should fail R3 with the solver itself. Outcome: both fail,
        T_H3 with the sign the forward check gave (be: model 0.046 too wet over V4 -> estimate too deep), T_H5
        on RMSE only (4.36 against 3; bias and r inside). Neither is inside R3 or within the margin declared
        in ay for a PINN run (RMSE <= 4, |bias| <= 3): T_H5 misses that margin by 0.36 cm, and the margin is
        not moved -> NO counting PINN run for T_H3 or T_H5; R3 is decided by the solver for both, as for T_N0.
        In both tests the four starts agree to 0.001 cm and the sensor is fitted to 0.001: the estimate is
        unique and wrong, i.e. the error is the soil's (the relation sensor - water table of V3 does not hold
        in V4 and V6), not the inversion's.
        Start from the plateau (declared in ay; highest sensor reading of the 7 d before the window, be,
        against theta_s - 3 sigma of the calibration soil): S_V1 0.3868 - 0.015 = 0.372 -> T_H1 (44.8 %), T_H2
        (43.6 %), T_H6 (41.4 %) start from the plateau; S_V3 0.3686 - 0.015 = 0.354 -> T_H3 (25.4 %) and T_H5
        (31.6 %) do not.
        Tally at US-HB2 so far: five of six tests decided on the solver side, none inside R3 (T_H1 2.50 cm,
        r 0.968, with a soil that is not usable; T_H2 16.7; T_H3 7.58; T_H5 4.36; T_H6 4.96). Open: T_H4 (V5
        with the soil of V4). The V4 calibration continuations (cal_hb2aug20_s0c / s1c / s2c, jobs 15553786,
        15553790, 15553791, started 22:01:40, 8 h limit) are still running: misfit 0.00122 / 0.00163 / 0.00098
        after 112 / 101 / 108 solves of the continuation (start 3 ended at 0.003651).
    bl) 03:39 (13 Nov). RAIN WINDOW W_R1, PINN rfr1a (soil S_A) AFTER THE WIDE STAGE AND ROUND 1: THE OPERATOR IS NOT
        ACCURATE ON THIS WINDOW. Interim (the driver is in round 2 of 4; nothing here is a final score).
        PINN against the solver's estimate (S_A, start 0) and against the well, cm:
          inversion          PINN - solver rms (max)   PINN - well rms / bias   sensor fit
          rfr1a_inv0 (wide)  29.50 (61.6)              41.7 / +35.7
          rfr1a_inv1 (r 1)   27.68                     39.3 / +33.9             0.0294 (solver 0.0069)
          by part, inv1 (t < 1 d / 1 - 8.7 / 8.7 - 9.33 / from 9.33 d): PINN - solver 19.4 / 36.5 / 9.6 / 3.7
          sensor fit by part: solver 0.0026 / 0.0038 / 0.0196 / 0.0082; PINN 0.0101 / 0.0202 (bias +0.017) /
          0.1043 / 0.0210. Before the burst the PINN's theta at the sensor stays at 0.139 - 0.144 while the
          sensor falls 0.137 -> 0.109 (the solver with zb at 1.5 m gives 0.113), and the PINN's estimate sits
          on the deep bound (1.5 m) on days 2.5 - 7: the water table pays for a network that does not dry.
          All four starts give the same estimate (spread 1.8e-8 m): stable, and wrong.
        Scoring of the operator against the solver (scripts/param_validate.py, jobs 15566798 / 15566799,
        runs/rfr1a_inv{0,1}/validation.json; SCORING ONLY):
          theta at the sensor, operator - solver, round-1 members: 0.0231 at the well's path (max 0.182),
          0.017 - 0.026 on local draws, 0.0065 - 0.023 on prior draws, 0.0303 at the PINN's own estimate,
          0.0258 at the solver's; by 5-day segment about 0.011 / 0.037 - 0.046 / 0.001 - 0.010. Wide members:
          0.0253 at the well's path. On the recessions the same numbers are 0.001 - 0.002.
          Jacobian against finite differences of the solver: cosine 0.97 - 1.00 for the first two and the last
          five coefficients, 0.84 - 0.90 for coefficients 3 - 7, 0.54 / 0.25 / 0.03 / 0.03 / 0.11 for 8 - 12
          (the days around the burst); predicted contraction 2.27 (wide members 0.98): the refinement rounds
          cannot be expected to converge to the solver's estimate from here.
        What the training itself reports (metrics.csv; no solver needed to see it):
          run            L_pde  L_flux  held-out pde / flux / box
          rfr1a_w0       1.73   0.94    1.95 / 1.15 /  9.3
          rfr1a_r1_m0    3.43   0.93    3.54 / 1.22 / 20.4   (27.9 - 34.9 in the last blocks)
          rf22c_w0       1.41   5.98    1.49 / 6.14 /  3.0   (recession, for comparison)
          rf22c_r1_m0    0.45   3.08    0.41 / 3.29 /  1.1
          The held-out water balance of boxes (in sigma_theta^2) is 20 - 35 on W_R1 against 1 - 3 on a
          recession: a layer-mean defect of 4.5 - 6 sigma_theta = 0.022 - 0.030, the size of the actual error.
          The pointwise terms do not show it (L_flux 0.9 is LOWER than on the recession).
        The residual scales on this window (src/zbinv/parametric.py physics_scales, one number each for the
        whole window; computed from the data set with awk):
          sigma_q = 0.1 x RMS(q) = 1.90 mm/d. 27 of the 337 hourly nodes have q < 0 (rain; RMS 67 mm/d), the
          other 310 have mean 0.35 and RMS 0.65 mm/d: between the rains the flux condition is held to three
          times the evaporation itself. On the recessions sigma_q is 0.026 (rf22c), 0.021 (rfp2), 0.27 mm/d (rfh1p).
          sigma_pde = 0.650 (scaled time), = RMS of the day-mean d Se / dt = 0.16 /d, set by the two burst days
          (+0.553, +0.238 /d); the dry days have 0.009 - 0.020 /d, the days after the burst 0.022 - 0.039 /d.
          rf22c 0.170, rfp2 0.320, rfh1p 0.057.
        HYPOTHESIS (NOT verified; what is established is only the numbers above): with one scale for the whole
        window the rain hours set it, and between the rains neither the evaporation at the surface nor the slow
        drying is enforced by the pointwise terms; only the box term asks for the water balance there, and it
        is not brought down. The network then stays wet before the burst (+0.017) and the inversion answers
        with the deep bound. Whether the burst itself (theta 0.15 -> 0.34 within the hour in a sand with
        n = 10; the solver against its own finer grid differs by 0.055 - 0.064 in that hour) is within reach
        of the network at all is a second, separate question.
        PLAN 1 (test of the hypothesis; forward PINNs at ONE path, the cheapest run that can decide it):
          Option, off by default (no run in progress changes): sigmas.q_by_regime / sigmas.pde_by_regime.
          The scale of a residual is the size of the signal in the forcing regime of its time, the two regimes
          being net infiltration (q_surf < 0) and not (the split the forcing-memory inputs already make,
          model.forcing_split_sign):
            flux  sigma_q(t) = q_frac x RMS of q over the hourly nodes of the regime of q_surf(t)
            PDE   sigma_pde(t) = signal_rate_scale's estimate (day means, noise variance subtracted, same
                  floor) over the pairs of consecutive days of one regime; a pair is 'infiltration' if either
                  day has an infiltration hour; t belongs to the pair whose day-mean times bracket it
          A regime that does not occur (or has RMS 0) leaves the window's scale: on a window without rain both
          options give exactly the present method. No new constant; q_frac and pde_factor as they are.
          W_R1 would get sigma_q 0.065 mm/d between the rains and 6.7 mm/d in the rain hours.
          The code goes into a COPY of src/ and scripts/ (runs/_stage_regime/) and is run from there, so that
          the jobs of the two drivers in progress keep importing the present source; it moves into src/ only
          if the test supports it, after the tests (hpc/run_tests.slurm) pass on the copy.
          Runs: forward PINN as in hpc/dc_loop.sh (sampling.mode=point, 25000 Adam + 15 L-BFGS blocks, seed 0,
          the TS set of real_full_obs.sh) on output/real_ona_rain21_soilsep21 at the path of the well's spline
          (truth_coef), three variants: A as it is, B q_by_regime, C q_by_regime + pde_by_regime.
          Score (scoring only): param_validate.py with each as the one member -> theta at the sensor against
          the solver on that path, whole window and by 5-day segment; and each run's own held-out box value.
          EXPECTATION (written before the runs): A 0.010 - 0.025 with the first segment about 0.01 too wet
          (as the operator); if the hypothesis holds, B brings the first segment (days 0 - 5, no burst) below
          0.004 and the held-out box value below 5; C at least as good as B. The burst segment (days 5 - 10)
          may stay at 0.02 - 0.04 in all three (second question above).
          DECISION: the option is adopted (for every site run from now on, since it changes nothing on the
          closed recessions without q < 0 hours) only if B or C reaches the solver to 0.004 on the segments
          without the burst AND does not make the burst segment worse than A. If A is already there, the
          hypothesis is wrong for the forward PINN and the fault is in the operator stage (then: the same
          three variants as a local operator). If none gets there, the scales are not the cause.
        PLAN 2 (in parallel, solver side only, CPU): a window with cycles where the sensor stays in its
        sensitive range. US-PAS, 27 Jul - 2 Sep 2021 (daily and 6-hourly tables from the half-hourly record,
        /tmp/hb2tools/rainsearch/pas_daily.txt, pas_6h.txt; rain stamps 24 h late here too: the sensor jumps on
        27 Jul 13:00 and 4 Aug 07:00, the rain is stamped 28 Jul 13:00 and 5 Aug 07:00):
          29 Jul - 3 Aug   recession, sensor 39.7 -> 18.0 %, well 0.27 -> 0.90 m
          3 - 4 Aug        re-wetting: well 0.90 -> 0.50 m on 3 Aug with the sensor at 18.0 -> 21.4 %, then
                           0.42 -> 0.22 m and 24.4 -> 40.5 % on the morning of 4 Aug; plateau to 7 Aug (41.1 %)
          7 - 12 Aug       recession 41.0 -> 18.3 %, 0.32 -> 0.90 m
          12 - 14 Aug      well to 0.33 m within six hours, sensor to 30.4 - 31.1 % and no further for two
                           days (well 0.35 - 0.50 m; on the drying branch 0.37 - 0.44 m reads 33 - 40 %)
          25 Aug - 2 Sep   three re-wettings back to the plateau (41.0 - 41.5 %) within 12 h each, recessions
                           to 25.5 - 30 % between them; well 0.24 - 0.75 m
          Windows (hourly forcing nodes and 1-day knots as configs/real_ona_rain21.yaml; site block as
          configs/real_pas_oct21.yaml, p_shift_hours -23.5; point sensor [0.10, 0.10]):
            C1 pas_jul21   29 Jul 12:00 + 9 d   calibration (recession, re-wetting, plateau; 18.0 - 41.1 %)
            T1 pas_aug21b  7 Aug 12:00 + 6 d    test, soil of C1 (18.3 - 39.7 %)
            T2 pas_aug21c  25 Aug 12:00 + 8 d   test, soil of C1 (25.5 - 41.5 %; C1 contains it to 0.004;
                                                no window between holds its range: T1 tops at 39.7 %, the
                                                existing aug21 window of 19 - 25 Aug bottoms at about 31 %)
          Protocol d and the usable-soil rule as they are; R1 - R4 as they are.
          Steps: make_real of C1 -> calibrate_soil, four starts -> hpc/transfer_solver.sh pas jul21
          "[0.10,0.10]" aug21b aug21c (in-sample check + solver on both tests) -> PINN only where the rule
          allows it, and only once PLAN 1 has settled how the PINN is run on a window with rain.
          EXPECTATION (before any of it): the single-valued retention curve cannot follow the re-wettings in
          which the well rises before the sensor does (3 Aug: 40 cm in 12 h at 18 - 21 %; 12 - 14 Aug: the
          sensor 7 % below the drying branch for two days). The solver's estimate should be LATE by up to a day
          on 3 - 4 Aug (in-sample) and 15 - 25 cm too deep on 12 - 13 Aug (T1, last day): T1 5 - 10 cm rms,
          R3 not met. T2 has the cleanest cycles (each re-wetting completes within 12 h): 4 - 8 cm rms, with
          the error at the rises (26 Aug: well 0.74 -> 0.46 m with the sensor at 26 -> 29 %) and at the
          plateaus (sensor at saturation); r 0.9 or better. The calibration may be pulled by 3 - 4 Aug; if its
          in-sample inversion is not within 2 cm (r 0.975 below saturation) the soil is not usable by the rule
          and the tests are demonstrations only.
    73 bm. (13 Nov, 03:40 - ) PLAN 1 and PLAN 2 of 73 bl started.
        PLAN 1, code (staged copy runs/_stage_regime/{src,scripts,tests}; src/ of the running drivers untouched):
          train.py      signal_rate_scale split into day_mean_rates (day numbers, day-mean times, rate^2 and
                        noise^2 per pair) and rate_scale (noise subtraction and floor); same numbers as before
          parametric.py RegimeScales (sigmas.q_by_regime / sigmas.pde_by_regime, both off by default) and
                        ParamTrainer.sq(tb) / sp(t), which return the window-wide numbers when the option is off;
                        the five divisions by sig_q / sig_pde go through them
          A node with q < 0 on a day boundary counts for both days (the interpolated flux is negative on both
          sides of it).
          Tests added (tests/test_parametric.py): test_regime_scales (no infiltration -> exactly the window
          scales; with it -> the formulas over the nodes / day pairs of each regime, and the time -> regime
          maps), test_regime_option_off_by_default_and_same_loss_for_equal_scales (trainer: off by default;
          switched on with both regimes given the window value, every loss term and the held-out terms are the
          same numbers to 1e-12).
          Data check before relying on the sign split: in W_R1 the 27 nodes with q < 0 are all rain hours
          (days 1, 2, 8, 9; no night-time q < 0), so pairs (0,1) (1,2) (2,3) (7,8) (8,9) (9,10) are the
          infiltration regime, 6 of 13.
        PLAN 1, runs (runs/_stage_regime/abc_fwd.sh; forward PINN at the well's spline, seed 0; V100):
          rgA_truth  as it is                      job 15567457   q~ 1.355e-2, pde 0.650
          rgB_truth  q_by_regime
          rgC_truth  q_by_regime + pde_by_regime   job 15567458   q~ [4.63e-4, 4.78e-2] (0.065 / 6.7 mm/d),
                                                                  pde [0.0917, 0.952]
        PLAN 2, set-up: configs/real_pas_jul21.yaml (C1), real_pas_aug21b.yaml (T1), real_pas_aug21c.yaml (T2).
          Rain time shift checked event by event on the half-hourly record, 28 Jul - 30 Aug 2021 (stamp of the
          rain against the half-hour in which the well starts to rise): 23.5 h for 28 Jul, 3 Aug, 4 Aug, 10 Aug,
          12 Aug, 13 Aug, 14 Aug, 26 Aug, 27 Aug, 29 Aug; 24.0 h for 9 Aug and 16 Aug. p_shift_hours -23.5
          as in the other US-PAS configs.
          The well's rises are large for the rain that causes them: 12 Aug 16:00 - 17:00, 17 mm, 0.89 -> 0.33 m
          with the sensor going 18.5 -> 28.2 %; 26 Aug, 7 mm, 0.78 -> 0.44 m.
        PLAN 1, tests (03:54): staged suite, job 15567582: 0 failed, both new tests PASS (the first run, job
          15567428, failed in the new trainer test on its own set-up: n_tan = 8 is too small for the sampler,
          int(frac_rain * n) = 0; n_tan = 32 as in the existing centre-term test. Every other test passed in
          that run already, so the split of signal_rate_scale changes no number.)
        PLAN 2, data sets (make_real, jobs 15567542 - 15567544, 03:52; all admissible, solver converged on every
          step; solver vs finer solver 0.0023 / 0.0057 / 0.0035):
                     days  coef.  rain mm  ET mm  well m       sensor          1-d spline vs hourly well
          C1 jul21   9     12     37.4     28.5   0.21 - 0.92  0.180 - 0.411   rmse 3.3 cm, max 13.4, r 0.991
          T1 aug21b  6      9     21.8     21.7   0.33 - 0.90  0.183 - 0.402   rmse 5.7 cm, max 27.8, r 0.944
          T2 aug21c  8     11     40.0     23.5   0.24 - 0.77  0.257 - 0.415   rmse 6.4 cm, max 21.8, r 0.910
          Sensor range of C1 contains T1's, and T2's to 0.004 (rule: to within 0.005) -> C1 soil is admissible
          for both under protocol d.
          C1 is a full cycle in both records: sensor 0.34 -> 0.18 over 4.8 days, 37 mm on days 4.8 - 5.9,
          sensor back to 0.41, well 0.30 -> 0.92 -> 0.22 m.
          LIMIT SEEN BEFORE ANY INVERSION: the well rises within an hour or two and a spline with 1-day knots
          cannot follow that (3.3 / 5.7 / 6.4 cm rms from the hourly record at its best fit). R3 asks for 3 cm
          against the well, so on T1 and T2 no estimate of this form can meet R3 on the whole window; the score
          of the best-fit spline is the floor and is reported next to every estimate. Knot spacing is not
          changed after seeing this (1 day was fixed in 73 bl for rain windows).
          With the site's default soil the solver at the observed water table is 0.045 - 0.053 from the sensor
          on C1 (too wet by 0.07 at the dry end): the calibration has to do the work.
        NEXT: four calibration starts on C1 (output/soil_pas_jul21), then transfer_solver.sh pas jul21 on T1, T2.
    73 bn. (13 Nov, 04:40) PLAN 1 RESULT: THE REGIME SCALES ARE NOT THE CAUSE. OPTION NOT ADOPTED.
        Forward PINNs at the well's spline on W_R1 (soil S_A, seed 0, 25000 Adam + 15 x 200 L-BFGS, V100, 47 - 49
        min each), scored against the solver at the same path (param_validate, jobs 15569349 - 15569351):
        [CORRECTED in 73 bp: the reference of param_validate has no root sink, the PINN and the site solver
        have one. Against the site solver (truth_error_profile --sensor fem) the columns whole / days 0-5 /
        5-10 / 10-14 read A 0.0221 / 0.0114 / 0.0353 / 0.0010, B 0.0316 / 0.0061 / 0.0525 / 0.0009,
        C 0.0308 / 0.0080 / 0.0510 / 0.0007. Same rule, same decision: none reaches 0.004, B and C are worse
        than A in the burst segment.]
                                   theta vs solver                              held-out, last state
                                   whole    max     days 0-5  5-10    10-14     pde    flux   box
          A  as it is              0.0205   0.182   0.0074    0.034   0.0010    2.14   8.2    8.1
          B  q by regime           0.0297   0.182   0.0054    0.049   0.0012    3.97   37.2   21.3
          C  q and pde by regime   0.0294   0.181   0.0055    0.049   0.0006    4.01   36.7   20.4
        (held-out terms are in units of each run's own scales, so B and C are not comparable with A there.)
        Against the written rule (73 bl): B and C do not reach 0.004 before the burst (0.0054 / 0.0055) and make
        the burst segment worse than A (0.049 against 0.034) -> not adopted; A is not there either (0.0074) ->
        the fault is not only in the operator stage; none gets there -> the scales are not the cause.
        The expectation was wrong on the main point: B's first segment was expected below 0.004 with a held-out
        box below 5; it is 0.0054 and 21.
        What the three runs do show: training stops improving long before the end. A: held-out flux 8.3 from
        step 16000 on (12.3 at 8000, 8.9 at 16000, 8.2 at 28000), pde 2.1 - 2.5; L-BFGS changes nothing. A
        flux term of 8.2 is an rms miss of 2.9 sigma_q = 29 % of the rms flux of the window, at ONE known path.
        On the recessions this term ends well below 1. So the forward PINN does not solve the problem it is
        given on this window, and the miss is in the flux condition.
        Data, read before guessing at the reason: the forcing has one node at 331 mm/d (day 9.333; 13.8 mm in
        the hour) - 2.4 x Ks of S_A (140 mm/d). Every other rain node is below 62 mm/d.
        Reference solver, read (wtd_pipeline/fem/fem_pytorch.py, read only): it steps in Se and clamps Se to
        [1e-6, 1 - 1e-6] after every Picard update (lines 739, 753), and h to <= 0 (line 194). Rain above what
        the profile takes is therefore removed at the clamp; the PINN's head form with a pure flux condition
        has to pass all of it. NOT YET SHOWN that this is what separates the two on day 9 - checked next:
        scripts/truth_error_profile.py (staged copy, new option --sensor fem: reference at the sensor = the
        solver at the same path instead of the observations), jobs 15569464 - 15569466: error by day, by
        depth band, the worst hours, and the daily water balance of the error (column residual, surface flux
        error, remainder).
        The staged option stays in runs/_stage_regime (not moved into src/, not committed).
        04:45. WHERE THE FORWARD PINN LEAVES THE SOLVER (truth_error_profile, staged, --sensor fem; jobs
        15569464 - 15569466; run A unless said; theta at the sensor, PINN - solver at the well's spline):
          day        0      1      2      3      4      5      6      7      8      9      10 - 13
          rms        0.006  0.020  0.011  0.009  0.006  0.006  0.008  0.015  0.022  0.074  0.00002 - 0.0015
          mean       all positive (PINN wetter) up to day 9
          C (rms)    0.005  0.013  0.009  0.006  0.004  0.002  0.002  0.007  0.054  0.100  as A
        Three separate things, each read off the hourly series (not inferred):
        1. After the burst the PINN is exact (1e-5 - 1e-3): the profile is wet and near equilibrium.
        2. Day 9.17 - 9.33, the largest error (+0.10 ... +0.19): the PINN's sensor goes to saturation three to
           four hours before the 331 mm/d node, while the solver's stays at 0.174. In those hours there is no
           rain; what moves is the PATH: the 1-day spline of the well is at 0.62 -> 0.54 m (the well itself
           stays at 0.76 m until 9.33 and is at 0.32 m one hour later). Equilibrium over a water table at
           0.59 m with S_A is 0.362 at the sensor; the PINN is at 0.34 - 0.37. So the PINN follows the rising
           boundary almost in equilibrium and the solver does not (dry soil above a boundary rising 0.5 m in
           20 h). B and C do the same and start earlier (+0.12 from day 8.9).
        3. Days 0 - 8 (what matters for the inversion: +0.006 ... +0.02 too wet -> an estimate pushed to the
           deep bound, as seen in rfr1a): the water balance of the error has a column-integrated PDE residual of
           +4 ... +35 mm/d (a source), drained through the water table; the surface flux is right to 0.2 - 0.4
           mm/d on the dry days. On day 1 the PINN takes in 0.25 of the 3.9 mm of rain (flux miss 3.7 mm) and
           still wets the sensor at once (+0.028 within three hours; the solver's front arrives half a day
           later). The sensor is at Se ~ 0.1 here (theta 0.12 - 0.15), on the dry branch of a soil with n = 10
           and below the range S_A was calibrated on (0.221).
        Not shown: which of these a different design would remove. The solver is itself not converged in the
        burst hour (hourly against finer: max 0.064), and its Se clamp was not shown to matter.
        ASSESSMENT. W_R1 with S_A is outside what the single-valued model describes (solver inversion 23 - 24
        cm from the well, 73 bj), so making the PINN equal the solver on it would not give a result to show.
        No more forward-PINN work on W_R1/S_A. The order from here:
          a) solver side first on every rain window: is there a soil with which the SOLVER inversion follows
             the well through an event (ONA rain21 in-sample: running; US-PAS C1 in-sample, then T1, T2)?
             A PINN of the same equation cannot do better than that.
          b) where a) holds: forward PINN at the well's spline with that soil (same harness), then the
             inversion. The US-PAS drying limbs are at Se 0.4 - 1 with n ~ 3, the regime in which the PINN
             equals the solver on the recessions; whether the event hours hold the rest hostage there as well
             is what that run shows.
        US-PAS C1 calibration, state 04:44 (not finished, 60 - 130 solves): misfit 0.0138 - 0.0144 for all four
        starts (theta_s 0.412 - 0.416, alpha 2.30 - 2.45, n 3.07 - 3.29, Ks 1.4 - 1.6e-6). The US-PAS recessions
        calibrate to 0.0045 - 0.0061: the cycle is NOT held by one retention curve to the level of a recession.

    73 bo. (13 Nov, 05:45) US-PAS C1 SOIL: ONE SOIL FROM ALL FOUR STARTS. SOLVER CHAIN, FORWARD PINN AND PINN
        DRIVER ON THE CYCLE WINDOW: PLAN AND EXPECTATIONS (written before any of the three runs).
        C1 calibration (jobs 15567730 - 15567733, all COMPLETED, 54 - 100 min, nothing on a bound):
          start  misfit    theta_s  theta_r   alpha   n      Ks
          0      0.013872  0.4137   0.0089    2.327   3.105  1.496e-6
          1      0.013839  0.4136   0.0000    2.301   3.076  1.462e-6
          2      0.013837  0.4136   0.0002    2.300   3.075  1.469e-6
          3      0.013834  0.4136   0.0000    2.297   3.070  1.476e-6   <- lowest: S_C1 (Ks = 128 mm/d)
        The four starts give one soil (unlike several earlier windows). Fit of the solver at the well's record to
        the sensor: rms 0.0138, bias 0.0000, max 0.036, r 0.988 (output/soil_pas_jul21/start3.png): the drying
        limb, the rise on day 5.8 and the wet plateau are all followed; the misfit sits in the first day (the
        sensor drops 0.04 within hours, the solver does not) and in days 8 - 9 (the solver dries, the sensor
        stays at 0.41). 142 of the 217 hours are below saturation (theta_obs < theta_s - 3 sigma).
        US-ONA rain window, in-sample solver inversion (soil S_R1 fitted on this very window; chain _pre, jobs
        15566848/49/53/54, STATE at 05:43, not final): three starts at cost 1.170 - 1.174 with steps of 1 - 4 mm
        and 19.4 - 19.9 cm from the well; the fourth still moving (cost 1.21, 14 - 21 cm). So on W_R1 the sensor
        does not hold the water table even with the window's own soil: the usable-soil rule fails in-sample
        there, and rfr1a / rfr1o can only be shown as what they are (figures with that caveat), not as a
        reproduced water table. The US-PAS cycle windows are where the request can be met, if anywhere.
        PLAN (three things, started together; the first is the gate for what counts):
        a) SOLVER SIDE: RT=06:00:00 hpc/transfer_solver.sh pas jul21 "[0.10,0.10]" aug21b aug21c
           -> output/real_pas_jul21_own (C1, in-sample), real_pas_aug21b_soiljul21 (T1), real_pas_aug21c_soiljul21
           (T2); solver inversions from the four starts each. Rule unchanged: S_C1 is usable if the in-sample
           inversion is within 2 cm of the hourly well record with r >= 0.975 below saturation; a test window
           counts for the PINN if its solver inversion is inside R3 or within the margin (4 cm, |bias| 3 cm).
           The floor of the 1-day spline (3.3 / 5.7 / 6.4 cm on the whole of C1 / T1 / T2) is reported next to
           every score, and the score of the best-fit spline below saturation as well (not yet computed).
           EXPECTATION (uncertain): n ~ 3 puts the sensor on the steep part of the retention curve over the
           whole range of the well, so the path is held far better than on W_R1; with a misfit of 2.8 sigma the
           inversion will bend the path in the first day and in days 8 - 9. In-sample 3 - 6 cm on the whole
           window (floor 3.3), 2 - 4 cm below saturation; the rise on day 5.8 placed within a few hours. The 2 cm
           rule may therefore fail on the floor alone; if it does, the windows do not count and the figures are
           shown as demonstrations (the rule is not moved).
        b) FORWARD PINN at the well's spline on C1 with S_C1 (run fwd_pasc1; main-tree source, settings of the
           forward stage of hpc/dc_loop.sh, seed 0; local file = truth_coef of the _own data set, diagonal
           covariance that point mode never samples), scored against the solver at the same path
           (param_validate; truth_error_profile --sensor fem from the staged copy).
           RULE: operator holds through the cycle if theta at the sensor is within 0.004 rms of the solver on
           days 0 - 5 and within 0.010 on the whole window. If not: the hourly error series says where (the
           event hours, peak rain 270 mm/d = 2.1 Ks, or the dry limb), and the design change is planned here
           before it is run.
           EXPECTATION: days 0 - 5 pass (Se 0.4 - 1, n ~ 3: the regime of the recessions); the event day is
           open - on W_R1 it failed, with n = 10 and a sensor at Se 0.1, neither of which holds here.
        c) PINN INVERSION of C1, unchanged method: hpc/real_full_obs.sh rfpc1 output/real_pas_jul21_own
           "[0.10,0.10]". Started with a) and b) rather than after them because it is the run the user asked
           for and takes 7 - 10 h; in-sample soil, so it is a demonstration whatever the score, and it counts as
           evidence for R1 - R3 on a cycle only if a) holds. T1 and T2 (soil from C1, nothing fitted on them)
           are the counting runs and wait for a).
        05:50 - 06:00. STARTED, AND ONE RUN STOPPED.
        Data sets with S_C1 (make_real jobs 15571383 - 15571385; all admissible, solver converged on every step):
                                 solver at the well's record vs sensor     best-fit 1-day spline vs hourly well
                                 rms      bias      max     r              whole    below saturation (bias, r, n)
          C1 jul21  (in-sample)  0.0138   0.0000    0.036   0.988          3.3 cm   3.9 cm  (-0.3, 0.981, 142 h)
          T1 aug21b (S_C1)       0.0301   -0.0182   0.049   0.927          5.7 cm   5.7 cm  (-0.3, 0.943, 144 h)
          T2 aug21c (S_C1)       0.0637   -0.0588   0.137   0.969          6.4 cm   6.5 cm  (-1.5, 0.858, 146 h)
        Two things read off this table before any inversion result exists:
        - the spline floor BELOW saturation is 3.9 / 5.7 / 6.5 cm with r 0.981 / 0.943 / 0.858: neither the 2 cm
          of the usable-soil rule nor R3 (3 cm, r 0.98) can be met against the hourly well on these windows by
          any estimate that is a 1-day spline. The rule and the knot spacing stay as they are (fixed in 73 bl
          before this was seen), so these windows cannot count; what is reported next to every estimate is its
          distance to the hourly well, the floor, and its distance to the best-fit spline (the reachable target).
        - S_C1 does not carry to T2: the solver at the well's own record is 0.059 too dry there (0.018 on T1).
          The solver inversion of T2 will be pulled shallow by that; T2 is expected to fail on the soil alone.
        b) forward PINN fwd_pasc1 (hpc/fwd_truth.sh, new: local file assembled from truth_coef, three duplicate GPU
           submissions 15571665/70/75, then param_validate and truth_error_profile --sensor fem; log
           logs/fwd_fwd_pasc1.log) and c) PINN driver rfpc1 (logs/full_rfpc1.log) started 05:50.
        rfr1o STOPPED at 05:54 (driver process group killed, its twelve queued/running round-2 jobs 15571559 -
        15571570 cancelled and confirmed CANCELLED). Reason: its window cannot give a reproduced water table -
        the solver's own in-sample inversion sits 19 - 20 cm from the well, and the PINN's first two inversions
        are at 22.2 and 23.1 cm (bias +13.8 / +15.4, max 50) - and its jobs were queued ahead of the US-PAS
        runs on all three GPU partitions. Kept: runs/rfr1o_w0..3, rfr1o_inv0, rfr1o_r1_m0..3, rfr1o_inv1.
        rfr1a (recession soil, last refinement round) runs on to its end: it is the one US-ONA rain-window
        figure, to be shown with the caveat above.
        06:05. US-HB2 V4 (aug20) calibration: the continuations s0c / s1c / s2c (15553786 / 90 / 91) ran into the
        8 h limit again, still creeping in the fourth digit (misfit 0.00119 / 0.00161 / 0.00097 after 172 / 155
        / 169 solves; theta_s 0.301 / 0.351 / 0.300 - the last on the lower bound 0.30 - alpha 18.2 / 19.9 /
        19.2 against the upper bound 20, Ks 2.2e-5 / 4.9e-5 / 2.8e-5; start 3 ended at 0.003651). Three soils,
        not one. Continued a second time from progress<k>.json (--resume; jobs 15572178 - 15572180,
        cal_hb2aug20_s0d / s1d / s2d, 12 h; ids in logs/cal_hb2_aug20_d.ids). T_H4 waits for all four starts
        as the rule requires; it is the only open test at US-HB2 and cannot change the tally there (five of
        six decided on the solver side, none inside R3).
    73 bp. (13 Nov, 06:54) US-PAS C1: THE FORWARD PINN HOLDS THROUGH THE WHOLE CYCLE. ONE SCORING TOOL WAS WRONG
        FOR SITE DATA (FOUND HERE, FIXED, 73 bn CORRECTED).
        Forward PINN fwd_pasc1 at the well's spline on C1 with S_C1 (hpc/fwd_truth.sh; GPU job 15571675, V100,
        05:56 - 06:42; held-out pde 0.092, flux 0.246, box 0.63 - on W_R1 these were 2.1 / 8.2 / 8.1).
        Theta at the sensor against the SITE SOLVER at the same path (scripts/truth_error_profile.py --sensor
        fem, job 15573295; reference = theta_fem[0] of real.npz):
          whole window 0.0035 (max 0.0425)     days 0 - 5: 0.0015     days 5 - 9: 0.0049     profile rms 0.0023
          by day 0 .. 8: 0.0003 0.0006 0.0007 0.0008 0.0031 0.0099 0.0003 0.0001 0.0004
        RULE of 73 bo b) (<= 0.004 on days 0 - 5, <= 0.010 on the whole window): PASSED. The expectation ("days
        0 - 5 pass, the event day open") held; the event day is inside the rule too.
        Hour by hour through the event (days 4.75 - 6.05): PINN - solver within +-0.010 in every hour, the PINN
        slightly late (-0.006 .. -0.010 in the hour after rain nodes of 24 - 74 mm/d), except ONE hour, 5.833
        (the 268 mm/d node; sensor 0.257 -> 0.384 in that hour): solver 0.3666, PINN 0.3241 (-0.0425); the next
        hour 0.4008 / 0.3987. After the event the difference is <= 0.003 and goes to zero. For scale: the
        hourly solver differs from its finer version on this data set by 0.0026 rms, max 0.032.
        So, unlike W_R1 (n = 10, sensor at Se 0.1, a node at 2.4 Ks), the operator form solves a full
        drying - wetting - drying cycle here. Whatever the inversion of C1 gives is not limited by the PINN's
        forward accuracy at the 0.004 level.
        SCORING TOOL. hpc/fwd_truth.sh also ran scripts/param_validate.py, which printed 0.0081 (max 0.017; by
        segment 0.0095 / 0.0059) for the same run. Read in the code, not guessed: its reference is
        run_reference(...) of the in-house solver with win = {t_days, qsurf} only (param_validate.py line 35) -
        NO root sink, while the PINN (model.sink = RootSink) and the solver of the data set (theta_fem) have
        it. On site data and FEM twins its numbers are therefore not PINN-against-solver errors. (That the
        missing sink is the whole of the 0.008 offset is the reading of the code; no separate run isolated it.)
        Consequences:
        - 73 bn's table was made with that tool. Recomputed against theta_fem[0] from the truth_profile.npz of
          the same three runs (rgA_truth / rgB_truth / rgC_truth; whole / days 0-5 / 5-10 / 10-14):
            A 0.0221 / 0.0114 / 0.0353 / 0.0010   B 0.0316 / 0.0061 / 0.0525 / 0.0009   C 0.0308 / 0.0080 / 0.0510 / 0.0007
          The decision of 73 bn stands under the same rule (none reaches 0.004 before the burst; B and C are
          worse than A in the burst segment). Pointer added at 73 bn.
        - scripts/param_validate.py now refuses a data set with a root sink (exit with a message naming the
          two right tools). Checked: job 15573433 on runs/fwd_pasc1 -> "this data set has a root sink ..."
          exit=1, no output file. Its use on the in-house twins (no sink) is unchanged.
        - hpc/fwd_truth.sh: the param_validate step is removed; the score is truth_error_profile --sensor fem.
        - Every use of param_validate on site data, listed from the files (runs/*/validation.json of site runs
          and the grep of this file): runs/fwd_pasc1 (this entry), runs/rgA_truth / rgB_truth / rgC_truth
          (73 bn, corrected above), and runs/rfr1a_inv0 / rfr1a_inv1 (the operator scoring of 73 bl: "0.0231
          at the well's path", the 5-day segments, the Jacobian cosines and the contraction 2.27). The last
          set is against the solver without the sink too: its values are indicative only. What was concluded
          from it - the operator is 0.02 - 0.03 from the solver on W_R1 and the rounds will not converge
          there - does not rest on it alone: the forward PINN A against the site solver is 0.0221, and the
          inversions rfr1a_inv0 and inv1 themselves were 29.5 and 27.7 cm from the site solver's estimate.
          No hpc driver calls param_validate (grep of hpc/); R2 of every site result is scored by
          scripts/score_real.py against scripts/fem_referee.py, the site solver with the sink.
    73 bq. (13 Nov, 06:57) US-PAS CYCLE WINDOWS, SOLVER SIDE COMPLETE: THE SOLVER ITSELF IS 12 - 16 cm FROM THE
        WELL, IN-SAMPLE INCLUDED. NEXT CYCLE WINDOWS: US-NC2 (plan written before any run there).
        Chain of 73 bo a) (logs/transfer_pas_jul21.log, 05:45 - 06:47; soil S_C1 = start 3; four starts each;
        zb RMSE / bias / max in cm against the hourly well, coefficients on the box):
          C1 in-sample (pas_own_jul21_s0..3)     11.88 / -0.51 / 32.9  [0,4,7]  cost 1.556 (starts 0, 1, 3);
                                                 start 2 (constant 1.0): cost 1.651, 23.81 / +10.46 / 61.1
          T1 = aug21b (pas_jul21on_aug21b_s0..3) 14.5 / -8.5 / 44.3    []       cost 0.977 (all four)
          T2 = aug21c (pas_jul21on_aug21c_s0..3) 15.55 / -12.5 / 31.7  [6,10]   cost 6.593 (all four)
        C1 below saturation (theta_obs < theta_s - 3 sigma = 0.3986; 142 h; awk on predictions.npz of s0):
          13.99 cm, bias +1.05, r 0.780 (whole window r 0.897). Floor of a 1-day spline: 3.9 cm, r 0.981.
        RULES APPLIED (unchanged): usable soil needs <= 2 cm and r >= 0.975 below saturation in-sample -> S_C1
        is NOT usable; T1 and T2 are outside the margin (4 cm) -> no counting PINN run on T1 / T2. The PINN
        run rfpc1 on C1 (in progress) stays what 73 bo c) said: a demonstration, and the R1 / R2 evidence on
        a cycle (reference: the solver estimate of starts 0 / 1 / 3).
        WHERE the solver's estimate leaves the well (s0; by day: rms of solver(theta) - sensor at the well's
        spline / at the estimate; estimate - well rms and mean, cm):
          day 0  0.0269 / 0.0094   5.1  +5.0      day 5 (rain) 0.0131 / 0.0113   7.2  +1.9
          day 1  0.0096 / 0.0035   5.4  -4.5      day 6        0.0036 / 0.0037   1.9  +1.0
          day 2  0.0090 / 0.0033  20.9 +16.8      day 7        0.0063 / 0.0013   2.7  -2.3
          day 3  0.0078 / 0.0070  15.7  +3.8      day 8        0.0232 / 0.0018   9.9  -9.8
          day 4  0.0070 / 0.0067  19.3 -16.1
        - days 2 - 4 (well 0.64 - 0.92 m, sensor 0.29 -> 0.18): two water tables 16 - 21 cm apart give the
          sensor to within 0.003 - 0.007 of each other - e.g. day 2.75: well 0.75 m -> 0.236, estimate 1.05 m
          -> 0.239. With n = 3.07 the soil at 0.10 m has drained to where ET and slow drainage set it, not the
          water table. The estimate swings +17 / -16 cm around the well there: 1-day knots, no information.
        - day 0 and day 8: the solver at the well's own record is 0.027 / 0.023 off the sensor (day 0: too wet
          by 0.03 in the first 18 h; day 8: the well falls 0.28 -> 0.36 m while the sensor stays at 0.41, the
          model dries to 0.37 - a capillary fringe wider than this van Genuchten curve has). The fit takes
          the misfit away by moving the water table 5 - 10 cm. On day 8 the sensor is on its plateau, so
          nothing can be read from it anyway.
        - the rise itself (days 5 - 7) is followed within 2 - 7 cm.
        Reading: on this soil a sensor at 0.10 m constrains the water table only while it is shallower than
        about 0.6 m; the dry half of the cycle is outside that. This is the forward model and the sensor, on
        the solver side - the PINN matches that solver through the cycle to 0.0035 (73 bp).
        Two sites, four rain / cycle windows, none invertible by the solver: US-ONA W_R1 (19 - 24 cm), US-PAS
        C1 / T1 / T2 (12 - 16 cm). Both are sands (n 3 - 10) with the water table falling to 0.9 - 1.5 m.
        PLAN: CYCLE WINDOWS AT US-NC2. Reason, from what is on record (73 aq, 73 az): n 1.2 - 1.3 there, the
        sensor (rods 0 - 0.30 m) follows the water table at 0.26 per m down to about 0.8 m, and the in-sample
        solver inversions of its recessions are 0.6 - 0.9 cm from the well (r 0.998) - the one site where the
        sensor is tied to the water table. What failed there was the TRANSFER of a soil between windows
        (retention drift 0.02 = 9.5 cm), which an in-sample window does not test.
        Windows, chosen from the records only (awk on the half-hourly file, Aug - Oct 2023, this entry; rain and
        the rise of the well coincide to the half-hour in both events; no gaps in rain, well or sensor; LE
        missing in 124 of 528 and 145 of 576 half-hours, filled as everywhere):
          C_N1 = sep23c  10 Sep 12:00 + 11 d  (14 coefficients). Well 0.54 -> 0.645 m, three showers of 5 - 8
                 mm (12 Sep 16:00, 13 Sep 01:00, 13 Sep 14:00) lift it to 0.47 m, recession to 0.78 m.
                 Sensor 40.6 -> 39.5 -> 41.2 -> 38.0 %. Everything inside the box [0.20, 1.50] m, below the
                 plateau (55.5 %) and shallower than 0.8 m. A small cycle (sensor amplitude 0.03).
          C_N2 = sep23d  18 Sep 12:00 + 12 d  (15 coefficients). Well 0.68 -> 0.78 m, 170 mm in 22 h from 22
                 Sep 10:30 (up to 16 mm per half hour) lift it to 0.107 m, recession to 0.40 m.
                 Sensor 39.4 -> 37.7 -> 55.7 -> 45.9 %. A large cycle, with two known handicaps: the well is
                 ABOVE the shallow bound of the box (0.107 - 0.20 m) for about 2 days, and the sensor is on
                 its plateau for about 1 day (22 Sep 22:00 - 23 Sep).
        Set-up: configs/real_nc2_sep23b.yaml with the two settings of the rain windows fixed in 73 bl
        (forcing.node_hours 1, zb_spline.knot_days 1); sensor layer [0.0, 0.30]. Nothing else differs.
        Steps (hpc/cycle_solver.sh, new: data set with the starting soil -> four calibration starts ->
        hpc/transfer_solver.sh in-sample; solver only):
          a) soil of each window = lowest-misfit start of four (in-sample, so every result is a
             DEMONSTRATION: the soil has seen the well of the window it is used on - said with every figure).
          b) solver inversion from four starts, scored against the hourly well, whole window and below
             saturation, next to the floor of the 1-day spline.
          c) GATE for the PINN (so that no GPU time goes into a window the solver cannot invert): PINN
             inversion (hpc/real_full_obs.sh, unchanged) and forward check (hpc/fwd_truth.sh) only on a
             window whose lowest-cost solver start is within 4 cm of the well below saturation.
             The usable-soil rule (2 cm, r >= 0.975) is reported as it is; it is not the gate.
        EXPECTATION (before any run): C_N1 passes the gate (in-sample fit <= 0.003; solver 2 - 4 cm, limited
        by the 0.01 the sensor reads lower before the showers than after them at one water table, = 4 cm).
        C_N2: fit 0.005 - 0.010; the estimate is held at 0.20 m while the well is at 0.11 - 0.19 m; below
        saturation 3 - 6 cm - the gate is open either way. If both fail the gate, the statement for the user
        is that no rain window at the three sites is invertible with this forward model and one near-surface
        sensor, PINN or solver, and what would change that (a deeper or second sensor; hysteresis in the
        retention curve) is outside the inverse method.
        06:59. STARTED: hpc/cycle_solver.sh nc2 sep23c "[0.0,0.30]" and ... nc2 sep23d ... (06:58:31; logs
        logs/cycle_nc2_sep23c.log, logs/cycle_nc2_sep23d.log; configs/real_nc2_sep23c.yaml / sep23d.yaml differ
        from real_nc2_sep23b.yaml in node_hours, knot_days, the window and the output directory only - diff
        checked). Running besides: PINN rfpc1 (US-PAS C1; four wide members training since 05:55), PINN rfr1a
        (US-ONA W_R1; value-correction loop, iteration 1), US-HB2 calibration continuations.
        07:07. DATA SETS OF THE TWO US-NC2 CYCLE WINDOWS (starting soil of CLAUDE.md; no solver inversion and
        no calibrated soil exists yet). make_real jobs 15573865 (sep23c), 15573866 (sep23d): COMPLETED, both
        admissible, solver converged on every step, hourly against finer solver 0.0007 / 0.0008.
                                         C_N1 = sep23c            C_N2 = sep23d
          coefficients (1-day knots)     14                       15
          well, hourly (m)               0.472 - 0.768            0.110 - 0.783
          sensor                         0.379 - 0.4285           0.377 - 0.557
          rain / ET (mm)                 21.3 / 36.9              170.4 / 27.4
          LE filled                      23 %                     25 %
          1-day spline against the hourly well (the FLOOR of every score against the well):
            whole window                 0.81 cm, max 3.5, r 0.995   3.97 cm, max 22.2, r 0.986
            sensor < 0.54 (provisional)  0.81 cm (all 265 h)         3.96 cm, bias -0.43 (254 of 289 h)
          starting soil at the well's record against the sensor (not a result; the calibration starts here):
                                         0.1115, bias +0.111, r 0.970   0.0859, bias +0.083, r 0.967
          hours with the well above the box's 0.20 m:   0            44
          hours with the sensor on its plateau (>= 0.54): 0          35
        Where the floor of C_N2 is: days 3 and 4 only (the well rises 0.78 -> 0.11 m in 22 h; spline against
        well 6.5 cm and 11.6 cm rms, max 17.5 / 22.2 cm); days 0 - 2: 0.2 - 1.0 cm; days 5 - 6: 2.2 - 2.3 cm
        (well above 0.20 m and moving); days 7 - 11: 0.3 - 1.4 cm. Outside days 3 - 4 and below the plateau:
        1.00 cm (219 h). So on C_N2 no estimate with 1-day knots - solver or PINN - can be inside R3 (3 cm)
        against the hourly well over the whole window; R3 stays as it is and C_N2 is scored against it as it is.
        AMENDMENT OF THE GATE, written before any solver inversion or calibrated soil of these windows exists
        (the calibration starts are at their first or second solve): the gate of 73 bq was written without
        this floor. With a floor of 3.96 cm, "within 4 cm of the hourly well" would decide on the spline's
        ability to follow a 22-hour rise, not on whether the solver can invert the window, which is what the
        gate is for. The gate is therefore evaluated against the well's 1-day spline (the water table the
        unknown can represent; identical for C_N1 within 0.8 cm): lowest-cost solver start within 4 cm rms
        of the spline below saturation. Unchanged: the usable-soil rule and R1 - R3, all against the HOURLY
        well, reported next to the floor. The knot spacing (1 day, 73 bl) is not changed.
        Calibration starts (jobs 15574012/14/16/18 = sep23c s0..3; 15574013/15/17/19 = sep23d s0..3): all
        eight RUNNING, each log has its start line; first solves: sep23c s0 0.1115 (starting soil), s3
        0.00217 (start from the hydrostatic fit); sep23d s0 0.0859. Waiter on both chains: b4kiadbqy.
        07:45. US-ONA W_R1, IN-SAMPLE SOLVER CHAIN COMPLETE (dry run *_pre with the W_R1 soil of start 3;
        logs/transfer_ona_rain21_pre.log "done" 07:44; referee jobs 15566848/49/53/54 COMPLETED, 2.7 - 4.3 h
        each). Against the hourly well, all four starts:
          start              cost    rmse cm  bias cm  max cm  coefficients on the box  sensor fit
          hydrostatic        1.1741  19.42    +10.29   48.5    [7,8,9,11]               0.00542
          hydrostatic_flat   1.1695  19.68    +10.26   48.5    [0,7,8,9,11]             0.00541   (lowest cost)
          constant 1.0       1.1794  20.26    +11.82   47.0    [8,9,11]                 0.00543
          constant 0.3       1.1714  19.83    +10.98   48.5    [7,8,9,11]               0.00541
        Lowest-cost start: whole window 19.68 cm, bias +10.26, r 0.898; below saturation (sensor < 0.3161,
        253 of 337 h) 22.23 cm, bias +12.76, r 0.821. Floor of the 1-day spline: 4.85 cm whole (r 0.979),
        4.19 cm below saturation. Posterior sigma (median) 3.5 - 3.8 cm - far below the actual error.
        By two days (estimate against the well, rms / mean; well range):
          0-2: 8.5 / +3.5 (0.85 - 1.04 m)     2-4: 3.1 / +0.8 (0.88 - 0.96)     4-6: 16.0 / +9.8 (0.94 - 1.04)
          6-8: 44.7 / +44.6 (1.00 - 1.06; the estimate is at the deep bound of the box)
          8-10: 18.8 / +13.6 (0.20 - 1.07; the rise)   10-12: 4.8 / +0.3 (0.35 - 0.57)
          12-14: 1.8 / -0.6 (0.57 - 0.68)     14-16: 4.6 / -4.6 (0.67)
        EXPECTATION of 03:0x (5 - 10 cm on the whole window): NOT MET, by a factor of two. What was expected
        and found: the recession after the burst is followed (2 - 5 cm from day 10 on). What was not: with
        the well at 1.0 m and deeper (days 4 - 8) the estimate runs to the deep bound, 45 cm away, at a
        sensor fit of 0.0054 - better than the forward run at the well itself (0.0065). In this sand (n at
        its bound of 10) a sensor at 0.10 m does not see a water table below about 0.95 m.
        Decision by the rules: the W_R1 soil is not a usable soil (in-sample 22 cm below saturation against
        the 2 cm of the rule); no PINN counting run on W_R1; the stop of rfr1o at 05:54 stands (its
        reference would have been these four starts). Rain / cycle windows so far - US-ONA W_R1 with the
        recession soil (19 - 24 cm), W_R1 with its own soil (19 - 20 cm), US-PAS C1 / T1 / T2 (12 - 16 cm):
        none invertible by the solver. Open: the two US-NC2 windows (calibration fits at this time: sep23c
        0.0018 - 0.0020 after 45 - 83 solves, start 3 finished; sep23d 0.0042 - 0.0087, start 3 finished).

    73 br. (13 Nov, 10:29) US-NC2 SMALL CYCLE (C_N1 = sep23c): THE SOLVER INVERTS IT TO 1.1 cm FROM ALL FOUR
        STARTS -> FIRST RAIN WINDOW THAT PASSES THE GATE. PINN STARTED ON IT. US-ONA PINN rfr1a ENDED.
        C_N1, solver side (logs/cycle_nc2_sep23c.log, "done" 09:52):
        - soil: all four calibration starts end at ONE soil, misfit 0.001551 (jobs 15574012/14/16/18):
          theta_s 0.41935, theta_r 0 (lower bound), alpha 0.8223, n 2.747, Ks 1.923e-6
          (file output/soil_nc2_sep23c/start1.json, lowest misfit by 1e-9). Expectation (fit <= 0.003): met.
          NOTE: this theta_s is below the sensor's plateau at this site (0.555) and below 66 hourly readings
          of this very window (sensor up to 0.4285 after the showers): it is an effective curve for sensor
          readings 0.38 - 0.43, not the soil of the profile. In-sample -> every result here is a DEMONSTRATION.
        - data set with that soil (output/real_nc2_sep23c_own, job 15578663): solver at the well's record
          against the sensor 0.00155, bias 0.0000, max 0.0124, r 0.984; hourly against finer solver 0.0003.
        - solver inversion (output/referee/nc2_own_sep23c_s0..3, jobs 15578769 - 72, 12 - 34 min): all four
          starts the same estimate: cost 0.09178, sensor fit 0.00151, no coefficient on the box, posterior
          sigma (median) 1.1 cm. Against the HOURLY well (awk on predictions.npz / real.npz):
                                whole window (265 h)             below saturation (sensor < 0.4044, 199 h)
            solver - well       1.11 cm, +0.11, max 3.1, r 0.990    0.99 cm, -0.05, max 3.0, r 0.986
            solver - spline     0.76 cm, +0.12, max 1.7, r 0.996    0.77 cm, -0.03, max 1.7, r 0.992
            floor (spline-well) 0.81 cm,  0.00, max 3.5, r 0.995    0.65 cm, -0.02, max 3.3, r 0.994
          by day (solver - well, rms cm): 0.3, 1.8, 1.7, 1.7, 1.0, 0.4, 1.1, 0.7, 1.0, 0.6, 0.7 (days 1 - 3 =
          the dry-down before the showers and the showers themselves, mean +0.9 to +1.1 cm).
        RULES APPLIED (as written): GATE (73 bq, amended 07:1x: <= 4 cm of the spline below saturation):
        0.77 cm -> PASSED. Usable-soil rule (<= 2 cm and r >= 0.975 below saturation against the hourly
        well): 0.99 cm, r 0.986 -> PASSED. Solver against R3 (3 cm, |bias| 2 cm, r 0.98): inside, whole
        window and below saturation. Expectation (solver 2 - 4 cm): better than expected.
        What this window is and is not: a cycle with re-wetting (well 0.54 -> 0.645 -> 0.47 -> 0.78 m; sensor
        0.406 -> 0.395 -> 0.428 -> 0.380), small (three showers, 21 mm), and scored with a soil that has seen
        its well. A held-out test needs a later US-NC2 window whose sensor range lies inside 0.379 - 0.4285
        with this soil (rule d); C_N2 does not (0.377 - 0.557) and overlaps it by three days.
        C_N2 = sep23d, state: soil = start 1, misfit 0.004224 (three starts agree: theta_s 0.5663, theta_r
        0.20 (upper bound), alpha 6.98, n 1.461, Ks 5.63e-4; start 3 stays at 0.0138 in another minimum);
        expectation (0.005 - 0.010): slightly better. Solver inversions running.
        PLAN FOR THE PINN ON C_N1 (written before either run starts; method and settings unchanged):
          a) inversion rfnc1: hpc/real_full_obs.sh rfnc1 output/real_nc2_sep23c_own "[0.0,0.30]"
             (4 wide members -> 4 refinement rounds -> value-correction loop). Reference for R2:
             output/referee/nc2_own_sep23c_s0..3.
          b) forward check fwd_nc2c: hpc/fwd_truth.sh fwd_nc2c output/real_nc2_sep23c_own "[0.0,0.30]"
             (one forward PINN at the well's spline against the solver at the same path).
             RULE: the operator holds through the cycle if theta at the sensor is within 0.002 rms of the
             solver on the whole window (the window's sensitivity is 0.05 of sensor per 0.30 m of water
             table, so 0.002 is 1.2 cm).
          EXPECTATION (before any PINN number): forward check 0.001 - 0.003 (US-PAS C1: 0.0015 on its dry
          days, 0.0049 through its rain, at rain rates ten times these); R1 spread < 0.2 cm; R2 (PINN -
          solver) 1 - 2.5 cm, open whether <= 1 cm: the rain rate of the showers (up to 8 mm/h = 2.2e-6
          m/s) is above this soil's Ks (1.9e-6 m/s) for single hours; R3 against the hourly well 1.5 - 3 cm.
          If R2 fails with the forward check inside its rule: look at the value-correction loop first
          (its forward offsets), then the reserve list (causal weights, decomposition at the showers).
        US-ONA W_R1, PINN rfr1a (recession soil S_A), ENDED 10:24 (logs/full_rfr1a.log "done"): refinement
        rounds inv0 .. inv4 = 41.7, 39.3, 42.4, 38.4, 36.0 cm from the well; value-correction loop it 1 - 4 =
        36.5, 39.2, 39.2, 39.3 cm (bias +31 to +33, max 59 - 65), step/sigma 4.8, 2.8, 1.8, 2.5 -> NOT
        converged after its four iterations; nine of 17 coefficients on the box; sensor fit 0.0225 - 0.030.
        Solver with the same soil: 23.2 - 24.1 cm, sensor fit 0.0069. So on this window the PINN is not at
        the solver's estimate either (R2 fails by a wide margin; scored below). Not followed up now: the
        window failed the solver side twice (S_A 23 - 24 cm, own soil 19 - 20 cm) and no result on it can
        count; noted as an open PINN question (a window where the soil is 0.027 off at the well's record).
        (Consistent with 73 bl: its interim table had inv0 / inv1 at 41.7 / 39.3 cm from the well and 29.5 /
        27.7 cm from the solver's estimate. The distance to the well did not close in the later rounds or
        the loop; the final distance to the solver's estimate comes from scripts/score_real.py, below.)
        10:29:30. STARTED: PINN driver rfnc1 (process group 3839010, logs/full_rfnc1.log) and forward check
        fwd_nc2c (process group 3839430, logs/fwd_fwd_nc2c.log) on output/real_nc2_sep23c_own. Scoring of
        rfr1a submitted (jobs 15581634 score_real, 15581635 plot_real_pinn; final directory runs/rfr1adc_it4).
        10:30. rfr1a SCORED (scripts/score_real.py job 15581634, plot job 15581635; final directory
        runs/rfr1adc_it4; files output/real_ona_rain21_soilsep21/result_rfr1a.json|png, result_rfr1a_pinn.png):
          PINN - hourly well      39.30 cm, bias +33.42, max 65.4, r 0.844 (337 h, all below saturation)
          PINN - solver (start 0) 28.21 cm, bias +16.33, max 61.0, r 0.773
          solver - hourly well    24.09 cm, bias +17.10, max 48.3, r 0.794
          starts of the PINN      spread max 1.85 cm; posterior sigma (median) 11.8 cm
          sensor fit              PINN 0.0225, solver 0.0069; solver at the well's record 0.0271
        -> R1 fails (1.85 cm against 0.2), R2 fails (28 cm against 1), R3 fails. The figure is kept as the
        record of a window that neither method inverts; it is not shown as a reproduction.
        11:21. US-NC2 LARGE CYCLE (C_N2 = sep23d), SOLVER SIDE COMPLETE (logs/cycle_nc2_sep23d.log "done" 11:19;
        data set output/real_nc2_sep23d_own, job 15578682; referee jobs 15578820 - 23, 1.3 - 2.0 h):
        solver at the well's record against the sensor 0.0042 (bias 0.0000, max 0.036, r 0.998), at the
        well's 1-day spline 0.0059; hourly against finer solver 0.0002.
        Solver inversion: all four starts ONE estimate (cost 7.3712, 9 - 10 iterations), coefficients 0, 6
        and 7 on the box, sensor fit 0.0136, posterior sigma (median) 0.3 cm. Against the well:
                                whole window (289 h)              below saturation (sensor < 0.5513, 269 h)
            solver - well       7.45 cm, +0.86, max 28.1, r 0.950    6.85 cm, +0.01, max 28.1, r 0.956
            solver - spline     5.26 cm, +1.19, max 15.5, r 0.975    4.54 cm, +0.45, max 15.5, r 0.981
            floor (spline-well) 3.97 cm, -0.32, max 22.2, r 0.986    3.95 cm, -0.45, max 22.2, r 0.985
          by day (solver - well rms cm | estimate range against well range, m | sensor fit of the estimate):
            0   2.1 | 0.66-0.73 / 0.68-0.70 | 0.0023        6   0.5 | 0.20-0.21 / 0.19-0.22 | 0.0016
            1   3.3 | 0.68-0.74 / 0.71-0.74 | 0.0039        7   0.6 | 0.22-0.25 / 0.22-0.25 | 0.0031
            2  10.4 | 0.69-0.94 / 0.74-0.77 | 0.0080        8   0.3 | 0.25-0.29 / 0.26-0.29 | 0.0020
            3  14.2 | 0.51-0.94 / 0.77-0.78 | 0.0183        9   0.1 | 0.29-0.33 / 0.29-0.33 | 0.0008
            4  17.7 | 0.24-0.49 / 0.11-0.77 | 0.0380       10   0.2 | 0.33-0.36 / 0.33-0.36 | 0.0009
            5   5.5 | 0.20-0.23 / 0.13-0.19 | 0.0183       11   0.3 | 0.36-0.39 / 0.36-0.40 | 0.0004
        RULES APPLIED (as written): GATE (<= 4 cm of the spline below saturation): 4.54 cm -> NOT PASSED.
        Usable-soil rule (2 cm, r 0.975 against the hourly well): 6.85 cm, r 0.956 -> not passed.
        -> no PINN inversion on C_N2. The gate is not moved a second time.
        Against the expectation of 73 bq: calibration fit 0.0042 (expected 0.005 - 0.010); estimate held at
        the box's 0.20 m while the well is at 0.13 - 0.19 m (as expected, day 5); below saturation 6.85 cm
        against the hourly well (expected 3 - 6: outside), 4.54 cm against the spline.
        Reading (from the table, not a new claim about the method): the six days after the event are
        followed to 0.1 - 0.6 cm. What fails is the event itself, in two ways shared by any estimate of this
        form: (1) a 1-day spline cannot follow a rise of 67 cm in 22 h and swings deeper before it (days 2
        - 3: estimate to 0.94 m with the well at 0.74 - 0.78 m); (2) the well is above the box's shallow
        bound for two days, so the estimate cannot reach the sensor's plateau (fit 0.038 / 0.018, 0.017 too
        dry on days 4 - 5; the inversion's fit, 0.0136, is worse than the solver at the well's own spline,
        0.0059, because that spline is outside the box). Both are settings fixed before this window was
        seen (knot spacing 73 bl, box since the synthetic cases) and stay as they are.
        Rain / cycle windows, solver side, tally: invertible C_N1 (1.1 cm); not invertible US-ONA W_R1 (19 -
        24 cm), US-PAS C1 / T1 / T2 (12 - 16 cm), C_N2 (7.5 cm; 4.5 cm against the spline).
    73 bs. (13 Nov, 11:30) US-NC2 SMALL CYCLE WITH THE SOIL OF THE EARLIER RECESSION (S_W2 = sep23a): PLAN, WRITTEN
        BEFORE ANY MODEL RUN. A DIAGNOSTIC BY RULE d (range condition), NOT A COUNTED TEST.
        Why: C_N1's PINN run (rfnc1) uses the soil calibrated on C_N1 itself with its well -> a demonstration.
        A result with a soil that never saw this window's well would be worth more.
        What the record says (awk on the half-hourly file; sensor, well, rain only - no model):
          mean sensor reading (%) at equal well depth (bins of 5 cm):
            well depth (m)                      0.50   0.55   0.60   0.65   0.70   0.75
            10 - 12 Sep (C_N1, before showers)     -   40.5   40.1   39.5      -      -
            14 - 21 Sep (C_N1, after showers)   40.8   40.6   40.1   39.6   39.1   38.3
            4 - 13 Oct (after the 170 mm)          -   42.2   41.7   40.9   40.1   39.5
            27 Nov - 9 Dec                      41.9   41.2   40.7   40.5      -      -
          -> after the storm of C_N2 the sensor reads 1.2 - 1.7 % wetter at the same water table (the size of the
          offset W2 -> W3: forward check of S_W2 on W3 0.0232 / -0.0229 in au, T_N1 9.53 / -9.31 cm in az). At C_N1's slope (0.10 - 0.14 % per cm)
          that is 9 - 17 cm: C_N1's soil on a window after the storm (4 - 21 Oct with showers of 3.0 and 4.6 mm,
          or 27 Nov - 9 Dec) would repeat T_N1's failure. NOT RUN. By rule d those windows take C_N2's soil
          anyway (not usable).
          Windows before C_N1: may23 (sensor 0.398 - 0.495) and sep23a (0.4045 - 0.4995; 1 - 7 Sep, 0.25 mm of
          rain; ends 3 d before C_N1 starts). C_N1's range is 0.379 - 0.4285, i.e. 0.0255 below sep23a's -> rule
          d's range condition is NOT met (sigma_theta 0.005): no earlier window gives C_N1 a soil by the rule,
          and nothing run with one can count for R3 / R4.
        Hand check done BEFORE this plan (declared: it uses C_N1's well, so the expectation below is not blind):
        hydrostatic layer mean (0 - 0.30 m, Simpson with 3 points) of S_W2 = output/soil_nc2_sep23a/start3.json
        (0.5493 / 0.1985 / 14.80 / 1.2896 / Ks 1.0e-3; lowest misfit of the four, 0.001038; the soil of the data
        sets *_cal3; in-sample 0.63 cm, az):
            well depth (m)        0.475    0.55          0.75
            S_W2, static          0.4174   0.4052        0.3831
            sensor in C_N1        0.412    0.405-0.406   0.383     (0.475 m: the 6 h after the rise, day 3.25)
          -> on the drying limbs S_W2 holds to about 0.001 although 0.75 m is 0.22 m deeper than the deepest
          water table of its calibration window; after the rise the sensor is 0.005 drier than S_W2's curve
          (3 cm at 0.16 % per cm).
        PLAN (solver side first, CPU only):
          1. hpc/transfer_solver.sh gets a switch OWN=0 (skip the in-sample part; default unchanged): the
             in-sample sets of sep23a exist (output/real_nc2_sep23a_own = start 1's soil, _own_cal3 = start 3's)
             and must not be rewritten. Check: bash -n; no transfer_solver.sh process is running (ps, 11:27).
          2. OWN=0 hpc/transfer_solver.sh nc2 sep23a "[0.0,0.30]" sep23c -> output/real_nc2_sep23c_soilsep23a,
             output/referee/nc2_sep23aon_sep23c_s0..3 (the script takes the lowest misfit = start 3 = S_W2).
             New directories; nothing that rfnc1 / fwd_nc2c read is touched.
          3. GATE of 73 bq, unchanged: lowest-cost start within 4 cm rms of the well's 1-day spline below
             saturation (theta_s - 3 sigma = 0.534 > sensor maximum 0.4285: the whole window). Passed -> PINN
             inversion with the unchanged driver (tag rfnc1t) on that data set, R2 against these four solver
             runs. Not passed -> recorded, no PINN run.
        STATUS OF ANY RESULT: DIAGNOSTIC ("C_N1 with S_W2", as "W3 with S_W1" in au). It can show the PINN
        inverting a window with rain and re-wetting with a soil that never saw that window's well; it cannot
        count for R3 / R4 (range condition; and the hand check above).
        EXPECTATION (before any solver number): solver over the well's record against the sensor 0.002 - 0.004
        (drying limbs 0.001, the day after the rise 0.005, the two shower hours 0.01 - 0.02 too dry: with Ks
        1e-3 the layer cannot hold a shower). Inversion against the hourly well: drying limbs 1 - 2 cm, days
        2.5 - 4 too deep by 2 - 4 cm; whole window 1.5 - 3 cm, bias +0.5 to +1.5 cm, r 0.97 - 0.99; all four
        starts one estimate. Gate expected to pass.
        STARTED 11:29:47 (process group 4171707, logs/transfer_nc2_sep23a_sep23c.log): OWN=0, soil start 3 as
        expected, test window only. bash -n of the edited script: ok.
    73 bt. (13 Nov, 11:53) US-NC2 SMALL CYCLE C_N1: THE FORWARD PINN HOLDS THROUGH THE SHOWERS (rule of 73 br PASSED).
        Forward PINN fwd_nc2c at the well's spline with C_N1's soil (hpc/fwd_truth.sh, unchanged; GPU job
        15581622, 28000 steps, 47 min; held-out pde 0.029, flux 0.027, box 0.0021 - US-PAS C1: 0.092 / 0.246 /
        0.63). Theta of the layer 0 - 0.30 m against the SITE SOLVER on the same path (scripts/
        truth_error_profile.py --sensor fem, job 15585571; reference = theta_fem[0] of
        output/real_nc2_sep23c_own/real.npz):
          whole window 0.00043 (max 0.0045, in the shower hours of day 2)      profile rms 0.00033 (max 0.0058)
          by day 0 .. 10: 0.00014 0.00008 0.00129 0.00041 0.00004 0.00007 0.00011 0.00021 0.00014 0.00018 0.00013
          (surface flux of the day, mm/d: 2.2 1.1 -10.9 -6.4 1.0 0.9 1.2 0.5 1.0 0.8 0.8)
        Cross-check that the layer sensor is the one scored: the run's own summary gives 0.00154 against the
        OBSERVATIONS, the solver on the same path 0.00155 (73 br).
        RULE (<= 0.002 rms on the whole window): PASSED, by a factor of 4.7. Against the expectation (0.001 -
        0.003): better than the expected range; the two days with showers are at 0.0013 and 0.0004.
        0.00043 of sensor is 0.3 cm of water table at this window's sensitivity (0.05 per 0.30 m), so the
        inversion rfnc1 is not limited by the PINN's forward accuracy at the level of R2 (1 cm).
    73 bu. (13 Nov, 13:51) DIAGNOSTIC "C_N1 WITH S_W2", SOLVER SIDE: ONE ESTIMATE FROM ALL FOUR STARTS, 3.05 cm FROM
        THE WELL. GATE PASSED (2.96 cm against the spline). THE SOLVER ITSELF IS JUST OUTSIDE R3.
        Chain of 73 bs (11:29 - 13:50; make_real 15584490, solver inversions 15584807-10, 43 - 133 min).
        Data set output/real_nc2_sep23c_soilsep23a (14 coefficients, admissible; hourly solver against the
        finer one 0.0006). Solver over the well against the sensor: record 0.0039 / bias -0.0019 / max 0.0090 /
        r 0.959; spline 0.0039 / -0.0020 / 0.0123.
        Solver inversion, four starts: ONE estimate (cost 0.083249 to 1e-7, 4 - 9 iterations, no coefficient
        on the box, sensor fit 0.00144, posterior sigma median 0.9 cm). Lowest cost: start 1.
            solver - well (hourly)   3.05 cm, bias -1.64, max 5.1, r 0.948
            solver - spline          2.96 cm, bias -1.64, max 4.9, r 0.953
            floor (spline - well)    0.81 cm, bias  0.00, max 3.5, r 0.995
          by day (solver - well rms cm (bias) | estimate range / well range, m):
            0  1.8 (-1.7) | 0.53-0.56 / 0.54-0.59       6  4.3 (-4.3) | 0.58-0.62 / 0.62-0.66
            1  2.6 (-2.1) | 0.56-0.63 / 0.59-0.63       7  4.5 (-4.5) | 0.62-0.63 / 0.66-0.68
            2  2.1 (+0.8) | 0.55-0.63 / 0.52-0.64       8  4.6 (-4.6) | 0.63-0.67 / 0.68-0.70
            3  3.5 (+3.2) | 0.51-0.54 / 0.47-0.54       9  2.7 (-2.5) | 0.67-0.73 / 0.71-0.74
            4  0.6 (-0.2) | 0.52-0.56 / 0.50-0.56      10  1.0 (+0.2) | 0.73-0.79 / 0.74-0.77
            5  2.8 (-2.7) | 0.56-0.58 / 0.57-0.61
        RULES APPLIED (as written in 73 bs): GATE (<= 4 cm of the 1-day spline; the whole window is below
        saturation): 2.96 cm -> PASSED -> the PINN run of the plan is started.
        R3 read on the solver (not a counted test, for orientation): 3.05 cm (limit 3), bias -1.64 (limit 2),
        r 0.948 (limit 0.98) -> the solver itself is outside R3, by the correlation clearly. A PINN that
        reproduces the solver will be at about 3 cm from the well too.
        Against the expectation of 73 bs: forward misfit 0.0039 (expected 0.002 - 0.004: inside, upper end);
        whole window 3.05 cm (expected 1.5 - 3: at the upper end, just outside); bias -1.64 (expected +0.5
        to +1.5: WRONG SIGN); r 0.948 (expected 0.97 - 0.99: outside); day after the rise too deep by 3.2 cm
        (expected 2 - 4: inside); drying limbs: days 0 - 1 and 9 - 10 at 1.0 - 2.7 cm, but days 5 - 8 are
        2.7 - 4.6 cm too SHALLOW (expected 1 - 2 cm: outside). The three-point static hand check of 73 bs
        did not show that; the cause of the 4.5 cm on days 6 - 8 has not been looked at and is not claimed.
        The estimate is a smooth drying curve that runs ahead of the well before the showers, is late after
        the rise and lags the well's later recession: the cycle is seen, its amplitude is 5 - 8 cm short.
        PINN ON THIS DATA SET (plan of 73 bs, step 3; nothing changed in the method):
          a) inversion rfnc1t: hpc/real_full_obs.sh rfnc1t output/real_nc2_sep23c_soilsep23a "[0.0,0.30]"
          b) forward check fwd_nc2ct: hpc/fwd_truth.sh fwd_nc2ct output/real_nc2_sep23c_soilsep23a "[0.0,0.30]"
             RULE as in 73 br: theta at the sensor within 0.002 rms of the solver on the whole window.
          EXPECTATION (before any PINN number on this data set): forward check 0.0002 - 0.001 (Ks 1e-3 is 450
          times the largest rain node: the profile is close to static, easier than C_N1's own soil at
          0.0004); R1 spread < 0.2 cm; R2 (PINN - solver) 0.1 - 1 cm (0.07 cm on sep23b with this soil,
          without rain); against the hourly well 2.8 - 3.3 cm, bias -1.3 to -2.0, r 0.94 - 0.96, i.e. as
          the solver and outside R3.
          STATUS: DIAGNOSTIC (73 bs). What it can add: R1 / R2 on a second rain-window data set, with a soil
          that never saw this window's well.
        STARTED 13:51:47: PINN driver rfnc1t (process group 819813, logs/full_rfnc1t.log) and forward check
        fwd_nc2ct (process group 819814, logs/fwd_fwd_nc2ct.log) on output/real_nc2_sep23c_soilsep23a.
    73 bv. (13 Nov, 16:12) US-PAS C1 (jul21, own soil S_C1, NOT usable): PINN rfpc1 SCORED. R1 HOLDS, R2 FAILS (2.9 cm
        FROM THE SOLVER), THE WELL IS MISSED BY 10.7 cm, AS BY THE SOLVER (11.9 cm). A DEMONSTRATION, AS SAID IN 73 bo c).
        Driver 05:50 - 16:05. Value-correction loop: four iterations, step / sigma max 0.56, 0.21, 0.31, 0.22
        (stop rule 0.2: NOT reached, the loop ended on its iteration limit); forward offsets rms 0.0010,
        0.0008, 0.0009, 0.0010; starts spread 0. Final estimate runs/rfpc1dc_it4. scripts/score_real.py (job
        15594949) -> output/real_pas_jul21_own/result_rfpc1.json|png; scripts/plot_real_pinn.py (15594950) ->
        result_rfpc1_pinn.png. zb RMSE / bias / max in cm, r:
                                 whole window (217 h)             below saturation (sensor < 0.3986, 142 h)
          PINN - well            10.73 / -1.99 / 26.5, r 0.907    12.49 / -1.20 / 26.5, r 0.799
          solver - well          11.88 / -0.51 / 32.9, r 0.897    13.99 / +1.05 / 32.9, r 0.780
          PINN - solver           2.92 / -1.48 /  9.2, r 0.9987    3.61 / -2.26 /  9.2, r 0.9983
          floor (spline - well)   3.26 / -0.09 / 13.4, r 0.991     3.9 (73 bo)
          hydrostatic reading    13.09 / -7.59 / 23.9, r 0.922    15.58 / -9.81, r 0.852
        (solver = start 3, the lowest cost 1.5563, shared by starts 0, 1, 3; start 2 ends at cost 1.6512 and
        23.8 cm, 73 bq.) Start spread of the PINN (max) 0.0002 cm; posterior sigma median 0.8 cm; sensor fit
        PINN 0.0079, solver 0.0062; coefficients on the box: PINN 0 and 7, solver 0, 4 and 7.
        By stage (PINN - well, cm): inv0 7.22, inv1 9.31, inv2 9.72, inv3 10.05, inv4 10.68, loop 11.14, 10.98,
        10.89, 10.73: the refinement moves the PINN's estimate towards the solver's, which is 11.9 cm from
        the well; the smaller number of the wide stage is not a better estimate.
        READ AGAINST THE CRITERIA: R1 holds (one estimate from all starts). R2 (<= 1 cm from the solver):
        2.92 cm -> FAILS. R3: fails, as the solver. Nothing here counts for R3 / R4 (soil not usable, 73 bq).
        What is established and what is not: the forward PINN at the well's spline passed its rule on this
        window (73 bp: 0.0035 whole window, one event hour at 0.0425), and the loop's forward offsets are
        0.001; the sensor fit of the PINN's estimate is 0.0016 worse than the solver's and one more
        coefficient is free of the box. WHY 0.001 of forward difference leaves 2.9 cm between the two
        estimates has not been examined; on a window where the solver's own answer is 12 - 24 cm from the
        well depending on the start, it is not followed up (no gate passed here; the windows that passed
        the gate are C_N1 and "C_N1 with S_W2", both running).

    73 bw. (13 Nov, 16:50) FORWARD CHECK OF "C_N1 WITH S_W2" (fwd_nc2ct): RULE NOT PASSED (0.0029 > 0.002). THE
        SURFACE-FLUX LOSS DID NOT TRAIN ON THIS SOIL. AUDIT OF THE FLUX LOSS OF EVERY SCORED PINN RUN. PLAN OF
        A STAGED TEST OF A CONDUCTIVITY-SCALED DEPARTURE (nothing changed in the main tree).
        a) Result (hpc/fwd_truth.sh, started 13:51:47; GPU job 15590527, 28000 steps, 2839 s; error profile job
           15595900; reference = theta_fem[0] of output/real_nc2_sep23c_soilsep23a/real.npz, the site solver on
           the same path). Theta at the sensor, PINN - solver:
             whole window 0.00286 (max 0.0148); profile 0.00258 (max 0.0368)
             day        0       1       2       3       4       5       6       7       8       9      10
             q [mm/d]   2.17    1.12  -10.94   -6.43    1.00    0.92    1.19    0.49    1.00    0.79    0.81
             rms       0.0011  0.0021  0.0062  0.0040  0.0010  0.0014  0.0021  0.0022  0.0020  0.0026  0.0026
             mean     +0.0007 +0.0013 -0.0024 -0.0011 +0.0004 +0.0010 +0.0016 +0.0015 +0.0016 +0.0022 +0.0022
           RULE of 73 bu (<= 0.002 rms on the whole window): 0.00286 -> NOT PASSED. The expectation of 73 bu
           (0.0002 - 0.001) was wrong by a factor of 3 - 14. The same window with its own soil gave 0.00043
           (73 bt).
        b) What the training history shows (runs/fwd_nc2ct/metrics.csv). Held-out losses in sigma units at step
           0 (the network starts hydrostatic, i.e. with zero flux) and at the end:
             fwd_nc2ct (S_W2)       flux 612.0 -> 600.9     pde 45.2 -> 26.6    (training values at the end: 575, 25.4)
             fwd_nc2c  (own soil)   flux 612.0 -> 0.027     pde  5.2 -> 0.029
           With S_W2 the flux loss ends where it started: after 28000 steps the residual of the surface flux
           is 0.991 of what a profile with no flux at all has (own soil: 0.0067). Training and held-out values
           agree, so this is a loss that did not go down, not a gap between training and held-out points. The
           0.0029 of a) is therefore the distance between the solver and a nearly hydrostatic profile under
           the moving water table, not the accuracy of a trained operator.
           The wide stage of the running driver rfnc1t shows the same: members w2, w3 (42000 steps) flux
           609.1 -> 608.8 and 604.1 -> 602.3, pde 47.8 -> 39.8 and 51.4 -> 39.3.
        c) AUDIT of every scored PINN run, same reading (member 0 of each stage; held-out flux residual, rms, as
           a fraction of the zero-flux level of the same data set, which is 79 - 163 in sigma units on the
           recession windows and 604 - 671 on the rain windows; "forward" = the forward PINN of the last
           value-correction iteration, or the forward check):
                                                 soil alpha [1/m], n, Ks [m/s]   wide    round 4   forward
             US-ONA rf22   nov22, soil 2021       1.55,  9.43, 2.4e-6            0.176   0.147     0.0062
             US-ONA rf22c  nov22, soil oct22      1.66, 10.0,  2.1e-6            0.237   0.156     0.0033
             US-ONA rf22ob oct22, soil nov21      1.55,  9.43, 2.4e-6            0.096   0.073     0.0052
             US-PAS rfp2   oct21, soil sep21      1.39,  4.84, 2.6e-6            0.129   0.062     0.011
             US-HB2 rfh1p  mar20, dry-run soil    5.31,  3.66, 8.7e-6            0.052   0.041     0.0041
             US-PAS rfpc1  C1 (rain), own soil    2.30,  3.07, 1.5e-6            0.041   0.034     0.0125
             US-NC2 rfnc1  C_N1 (rain), own soil  0.82,  2.75, 1.9e-6            0.022   0.012     0.0067 (fwd_nc2c)
             US-NC2 rfnd   W3, soil may23         5.58,  1.22, 1.0e-3            0.470   0.387     0.194
             US-NC2 rfn1   W3, soil S_W2         14.80,  1.29, 1.0e-3            0.562   0.359     0.311
             US-NC2 rfnc1t C_N1 (rain), S_W2     14.80,  1.29, 1.0e-3            0.9997 (w2), 0.9985 (w3)   0.991 (fwd_nc2ct)
           (fwd_pasc1, the forward check of US-PAS C1: 0.019.) Three groups: with Ks of 1e-6 .. 1e-5 the final
           forward PINNs carry the flux to 0.3 - 1.3 % of its zero-flux residual; with the two recession soils
           of US-NC2 (n 1.2 - 1.3, Ks at the calibration bound 1e-3) 19 - 31 % is left on a recession; with
           S_W2 on the rain window nothing is learned.
           What this changes in results already written down:
           - rfnd (W3 with the May soil, the held-out US-NC2 recession inside R3 at 1.59 cm): PINN - solver
             0.056 cm, yet 19 - 47 % of the flux residual is left, and the hydrostatic reading of the sensor
             scores 1.64 cm on that window. The score stands as a number. It does not show that the operator
             carries the surface flux on this soil; it shows that on this soil and window the sensor says
             little about the flux. The same for rfn1 (PINN - solver 0.066 cm, 31 - 56 % left; 9.48 cm from
             the well).
           - The held-out losses of the US-NC2 runs were never written into this file (the last val_flux
             quoted is for US-ONA's rf21b), and no entry compares a val_flux with its zero-flux level.
        d) NOT EXAMINED, NOT CLAIMED: why. One reading of the equations, with numbers, as a hypothesis:
           the departure of the network is d = (1 + xi) s N and the flux is q~ = -K_r(h) d_xi / z~b. The
           flux scale is q~ = rms(q)/Ks = 1.5e-4 here (0.079 with the window's own soil). K_r of S_W2 falls
           from 1 at the water table to 0.10 at 2 cm of suction, 9.1e-3 at 10 cm, 2.0e-4 at 50 cm and 6.8e-5
           at 75 cm (own soil: 1 .. 0.62 .. 0.37). So the true d_xi is of order 1e-4 near the water table and
           of order 1 near the surface, and an error of N near the water table enters the PDE loss with
           K_r/sigma_pde ~ 1/1.1e-4 ~ 9000 while the surface flux responds with K_r(surface)/sigma_q ~ 13.
           A loss that stiff in one direction would not move under Adam from the hydrostatic start. This has
           NOT been tested. The ideas held in reserve since 73 bl (causal weights, decomposition in time at
           events, runoff boundary condition) are about events; the table of c) says the same soil leaves
           19 - 56 % on recessions, so they are not what the evidence points to.
        e) STAGED TEST of the hypothesis (copy of scripts/ and src/ in runs/_stage_cond, untracked; the main
           tree and the running drivers rfnc1, rfnc1t are not touched). One change, no new constant:
             departure='conductive':  h~ = (1 + xi) [ -z~b + g(xi, t) s(t) N ],
                                      g = q~ / (K_r(-(1 + xi) z~b) + q~),   q~ = rms(q_obs / Ks)
           g is the head gradient that carries the flux scale at the local hydrostatic conductivity, capped at
           one. Where the soil conducts far more than the flux (K_r >> q~) the departure the physics asks for
           is that small, and N stays of order one; where K_r << q~, g -> 1 and the form is the present one.
           It is a rescaling of the network output by a positive function of the soil and of the flux scale
           (the same rms(q)/Ks that defines sigma_q): every profile the present form can represent it can
           too, no loss term or weight changes, h = 0 at the water table and the hydrostatic start are kept.
             model.py      h_tilde: dep = s * head; if departure == 'conductive':
                               dep = dep * q_scale / (soil.K(-(1 + xi) * zb~) + q_scale)
             parametric.py build_param_model: departure = model.departure (default 'additive'); q_scale set
                           from the data set as in physics_scales
           Four forward PINNs at the well's spline, settings of hpc/fwd_truth.sh plus
           --set model.departure=conductive, scored as there (flux residual against the zero-flux level; theta
           at the sensor against the site solver on the same path):
             F1 fc_nc2ct  C_N1 with S_W2 (rain)       now: flux 0.991, theta 0.00286
             F2 fc_nc2c   C_N1, own soil (rain)       now: flux 0.0067, theta 0.00043
             F3 fc_nc2w3  W3 with S_W2 (recession)    now: flux 0.31 (at the estimate of rfn1, not the well)
             F4 fc_pasc1  US-PAS C1, own soil (rain)  now: flux 0.019, theta 0.0035 (days 0 - 5: 0.0015)
           RULES, written before the runs:
             F1 (decides): flux residual <= 0.1 of the zero-flux level AND theta within 0.002 rms of the solver
                on the whole window (the rule of 73 bu).
             F2: the same two conditions.   F3: flux <= 0.1; theta reported.
             F4: flux <= 0.1 and the rule of 73 bo b) (<= 0.004 on days 0 - 5, <= 0.010 on the whole window).
             The form goes into the main tree (default off, then a pre-registered re-run of rfnc1t) only if
             F1 passes and F2, F4 pass. If F1 fails it is not adopted and the hypothesis of d) is not
             supported; the stage stays untracked as runs/_stage_regime did.
           EXPECTATION: F3 (no events) flux <= 0.05. F1: the dry days learned (their theta bias of +0.001 ..
           +0.002 gone), the two shower days uncertain: flux 0.1 - 0.5, theta 0.001 - 0.002, i.e. the likeliest
           outcome is short of the rule on the flux and inside it on theta. F2 and F4: within a factor of two
           of now. I would not be surprised by a failure of F4's event hour.
        f) rfnc1t (diagnostic, 73 bu) continues unchanged. Its wide operator carries no flux, so whatever it
           returns is the inversion of a nearly hydrostatic network; with a forward difference of 0.0029 and
           S_W2's slope of 0.11 - 0.16 % per cm (73 bs) I now expect PINN - solver 1.5 - 3 cm, R2 failed (the
           expectation of 73 bu, 0.1 - 1 cm, was written before a) and b)). The number says what the unlearned
           flux costs on a rain window, which decides how much the form of e) matters for the inversion.
        g) (16:55) Stage built, before any run. runs/_stage_cond = copy of scripts/ and src/ of 3dd38ba plus
           (diff -r src runs/_stage_cond/src, complete):
             model.py   InversePINN.__init__: 'conductive' accepted; every departure other than 'additive' needs
                        base='current'; self.q_scale = None
                        h_tilde, after zb_base:
                          dep = s * self._head(xi, t_days, c)
                          if self.departure == 'conductive':
                              dep = dep * self.q_scale / (self.soil.K(-(1.0 + xi) * zb_base) + self.q_scale)
                          return (1.0 + xi) * (-zb_base + dep)
             parametric.py build_param_model: InversePINN(..., departure=mc.get('departure', 'additive'), ...);
                        if model.departure == 'conductive':
                            model.q_scale = float(np.sqrt(np.mean((ds['q_obs'] / ds['soil'].Ks) ** 2)))
           runs/_stage_cond/fwd_truth.sh = hpc/fwd_truth.sh with the two script paths pointing at the stage.
           The Python edits could not be syntax-checked on the login node: the first job of each run is the check.
           Numbers of the gain g = q~ / (K_r + q~) from the soils (awk, 73 bw d): S_W2 on C_N1 q~ = 1.5e-4, g from
           1.5e-4 at the water table to 0.69 at 75 cm of suction; own soil q~ = 0.079, g 0.073 .. 0.18.
           F3 has no present-form number at the well's path, so a fifth run is added as its baseline:
             F3b fwd_nc2w3  W3 with S_W2, hpc/fwd_truth.sh of the main tree (additive). Expectation: flux
                 0.2 - 0.4 of the zero-flux level, theta within 0.001 rms of the solver.
           STARTED 16:54 (each a driver of its own, setsid nohup; GPU jobs h200 / A100 / V100, raced):
             F1  fc_nc2ct   pgid 1919126   jobs 15596463 15596469 15596481   logs/fwd_fc_nc2ct.log
             F2  fc_nc2c    pgid 1919127   jobs 15596461 15596468 15596480   logs/fwd_fc_nc2c.log
             F3  fc_nc2w3   pgid 1919128   jobs 15596460 15596467 15596483   logs/fwd_fc_nc2w3.log
             F4  fc_pasc1   pgid 1919129   jobs 15596462 15596466 15596471   logs/fwd_fc_pasc1.log
             F3b fwd_nc2w3  pgid 1919130   jobs 15596464 15596465 15596478   logs/fwd_fwd_nc2w3.log (main tree)

    73 bx. (13 Nov, 17:32) PINN ON THE SMALL US-NC2 CYCLE C_N1 WITH ITS OWN SOIL (rfnc1): R1 - R3 PASSED, AT THE
        SOLVER'S ESTIMATE. A DEMONSTRATION (the soil has seen this window's well), not a held-out result.
        Driver hpc/real_full_obs.sh rfnc1, 10:29:25 -> 17:28:50; four wide members, four refinement rounds,
        value-correction loop converged at its first iteration (step / sigma max 0.02; forward offset rms
        2.6e-5 in theta). Final estimate runs/rfnc1dc_it1; scripts/score_real.py (job 15597323) ->
        output/real_nc2_sep23c_own/result_rfnc1.json, figure result_rfnc1_pinn.png (job 15597324).
                                              whole window (265 h)           below saturation (199 h)
                                              rms    bias   max    r         rms    bias   max    r
          PINN - well (hourly) [cm]           1.10  +0.11   3.16  0.9905     0.97  -0.06   2.90  0.9869
          solver - well                       1.11  +0.11   3.13  0.9904     0.99  -0.05   3.02  0.9862
          PINN - solver                       0.065  0.00   0.30             0.056  0.00   0.30
          hydrostatic reading - well          3.41  +1.16  36.4   0.928      1.75  +1.39   5.31  0.984
          1-day spline of the well - well     0.81   0.00   3.46  0.9948     (the floor of the path)
        R1 spread of the four starts 5e-7 cm -> passed.  R2 0.065 cm (<= 1) -> passed.  R3 1.10 cm (<= 3),
        bias +0.11 (<= 2), r 0.9905 (>= 0.98) -> passed, also below saturation. R4 does not apply: the soil
        (output/soil_nc2_sep23c/start1.json) was calibrated on this window with its well.
        Sensor fit 0.00134 (solver 0.00151); the PINN's own sigma, median 1.11 cm, is the size of its error.
        Physics losses, as 73 bw c) (held-out flux residual, rms, as a fraction of the zero-flux level of the
        wide stage, 576 - 612 in sigma units): wide members 0.021 - 0.023 (flux 0.26 - 0.31, pde 0.41 - 0.44);
        round 4 0.012 - 0.014 (0.082 - 0.121, pde 0.14 - 0.29); forward PINNs of the loop 0.0074 - 0.0077
        (0.031 - 0.036, pde 0.028 - 0.032). So on this soil the flux condition IS learned through the showers.
        Against the expectation written at 73 bt (R2 1 - 2.5 cm, "open whether <= 1 cm"; R3 1.5 - 3 cm): both
        better than expected; R2 by a factor of 15 or more. I had taken the rain rate above Ks for single
        hours as a limit of the operator; it was not one here.
        What this is: the first window with rain and re-wetting (three showers, 21 mm; well 0.54 -> 0.645 ->
        0.47 -> 0.78 m) on which the PINN returns the well's path inside R3, and on which the physics is
        doing work: the static reading of the sensor is 3.4 cm off (1.75 below saturation) where the PINN is
        1.10 (0.97). What it is not: held out. The same window with the soil of the earlier window (rfnc1t,
        diagnostic) is still running and its operator does not carry the flux (73 bw); the staged test of 73
        bw e) decides whether that can be repaired.

    73 by. (13 Nov, 18:06) US-HB2 V4 (aug20) CALIBRATION: STARTS 0 - 2 RAN INTO THE LIMIT A THIRD TIME (8 + 8 + 12 h).
        Jobs 15572178 - 15572180 (cal_hb2aug20_s0d / s1d / s2d): TIMEOUT at 12:00:15, all three still moving in
        the fifth digit of the misfit and none ended by xtol (1e-6):
          start 0  misfit 0.00097 (06:05: 0.00119)  theta_s 0.3000 (lower bound 0.30)  alpha 19.25  n 1.824  Ks 2.8e-5
          start 1         0.00156 (0.00161)          theta_s 0.3497                     alpha 19.94  n 2.164  Ks 4.2e-5
          start 2         0.00093 (0.00097)          theta_s 0.3000 (lower bound)       alpha 20.00 (upper bound)  n 1.769  Ks 3.3e-5
          start 3  ended 12 Nov at 0.003651 (theta_s 0.3013, alpha 18.74, n 1.653, Ks 3.4e-6; another minimum)
        (output/soil_hb2_aug20/progress<k>.json; 221 - 260 solves in this continuation.) Starts 0 and 2 are
        closing on each other along a flat direction (alpha towards its bound of 20 1/m, Ks rising); start 1
        is somewhere else. The rule needs four finished starts, so T_H4 still cannot be run as a counting test.
        DECISION, two things, both CPU only:
        1) third and LAST continuation, --resume, 12 h (cal_hb2aug20_s0e / s1e / s2e). If a start has not ended
           by xtol after it (40 h), V4 is recorded as "calibration not converged" and T_H4 as not decided
           under the rule; no further continuation, no change of the tolerance after the fact.
        2) DRY RUN of the T_H4 chain now with the soil of the lowest misfit so far (start 2, progress2.json
           of 18:04), names ending in _pre, which do not count (as T_H1's dry run, 73 bd):
             SOIL_SET="--set soil.theta_s=0.3000064721994929 --set soil.theta_r=0.05661642541450263
               --set soil.alpha=19.996278430849358 --set soil.n=1.7690653038695743
               --set soil.Ks=0.00003319414777749387" RT=12:00:00 hpc/transfer_solver.sh hb2 aug20
               "[0.1524,0.1524]" nov20 > logs/transfer_hb2_aug20_pre.log
           It says where T_H4 will land unless the last 12 h move the soil by more than the last 12 did.
        EXPECTATION (the pre-registration of T_H4 stands: "should fail R3 with the solver itself"):
          V4 in-sample with this soil: FEM over the record against the sensor 0.001; solver inversion 1 - 2.5
          cm from the well, usable-soil rule (2 cm, r >= 0.975 below saturation) about even.
          T_H4 (V5): at a water table of 0.25 m the sensor reads 12.8 % in V5 and 18.4 % in V4, so the soil
          of V4 must put V5's water table deeper than the well: solver 8 - 20 cm from the well, bias positive,
          R3 failed.
        STARTED 18:06: continuation jobs 15597939 / 15597940 / 15597941 (logs/cal_hb2_aug20_e.ids); dry run
        process group 2322708 (logs/transfer_hb2_aug20_pre.log).
    73 bz. (13 Nov, 18:18) STAGED FORWARD CHECKS OF departure='conductive' (73 bw e): ALL FOUR RULES PASSED, F1 INCLUDED.
        THE FLUX IS NOW LEARNED ON THE Ks 1e-3 SOILS (rain window 0.99 -> 0.018, recession 0.30 -> 0.0024). ADOPTED
        INTO THE MAIN TREE, DEFAULT OFF; RE-RUN OF THE INVERSION ON "C_N1 WITH S_W2" PRE-REGISTERED BELOW.
        Five forward PINNs at the well's spline (28000 steps each; started 16:54, the last ended 18:09; no Python
        error in any job). Held-out losses from runs/<name>/metrics.csv, step 0 -> step 28000, and their rms ratio
        to the step-0 (zero-flux, hydrostatic) level; theta at the sensor against the site solver on the same path
        (truth_error_profile --sensor fem, [truth] lines of logs/fwd_<name>.log):
                                             flux loss              flux    pde loss            pde     theta - solver
                                             step 0 -> end          ratio   step 0 -> end       ratio   rms (max)
          F1  fc_nc2ct  C_N1 with S_W2       612.03 -> 0.2025       0.018   45.22 -> 5.497      0.35    0.00098 (0.0060)
              additive (fwd_nc2ct, 73 bw)    612.0  -> 600.9        0.991   45.2  -> 26.6       0.77    0.00286
          F2  fc_nc2c   C_N1, own soil       612.03 -> 0.0316       0.0072  5.249 -> 0.0185     0.059   0.00043 (0.0046)
              additive (73 bw c)                                    0.0067           0.029      0.074   0.00043
          F3  fc_nc2w3  W3 with S_W2         101.93 -> 0.0006       0.0024  1.810 -> 0.1891     0.32    0.00028 (0.0008)
          F3b fwd_nc2w3 the same, additive   101.93 -> 9.2395       0.301   1.810 -> 1.3294     0.86    0.00028 (0.0008)
          F4  fc_pasc1  US-PAS C1, own soil  671.00 -> 0.1212       0.013   2.822 -> 0.0579     0.14    0.0034  (0.043)
              additive (fwd_pasc1, 73 bp)             0.246         0.019            0.092      0.18    0.0035  (0.0425)
        F4 by day 0 .. 8: 0.0004 0.0003 0.0005 0.0006 0.0018 0.0099 0.0002 0.0001 0.0002; days 0 - 5 (days 0 .. 4,
        as in 73 bp; combined from the per-day values): 0.0009 (additive 0.0015); days 5 - 9: 0.0050 (0.0049).
        The one event hour is missed as before (0.043 against 0.0425): the new form does not touch that.
        F1 by day 0 .. 10 (rms; mean): dry days 0.0005 - 0.0009 with a mean of -0.0004 .. -0.0007 (PINN drier
        than the solver); the two shower days 0.0022 (max 0.0060) and 0.0012 (max 0.0046).
        RULES of 73 bw e), applied as written:
          F1 (decides)  flux <= 0.1: 0.018 yes.  theta <= 0.002 on the whole window: 0.00098 yes.     PASSED
          F2            0.0072 yes; 0.00043 yes.                                                      PASSED
          F3            flux <= 0.1: 0.0024 yes (theta 0.00028, reported).                            PASSED
          F4            0.013 yes; days 0 - 5 0.0009 <= 0.004; whole window 0.0034 <= 0.010.          PASSED
        AGAINST THE EXPECTATION: F3 "flux <= 0.05": held (0.0024). F2, F4 "within a factor of two of now": held
        (flux 0.0072 / 0.0067 and 0.013 / 0.019, theta unchanged). F3b "flux 0.2 - 0.4, theta within 0.001":
        held (0.30, 0.00028). F1 "flux 0.1 - 0.5, theta 0.001 - 0.002, short of the rule on the flux": WRONG on
        the flux, on the good side (0.018; the shower days were learned too); theta 0.00098, at the lower edge.
        WHAT THE NUMBERS SAY AND WHAT THEY DO NOT:
        - The hypothesis of 73 bw d) (the additive departure cannot express the small head gradients that carry
          the flux through a soil conducting 1e3 - 1e4 times that flux) is supported by a controlled pair:
          F3 and F3b differ only in the form, same data, same settings, same seed: 0.0024 against 0.30.
        - On the recession (F3 / F3b) theta at the sensor is the SAME to 5e-6 with and without the flux
          learned (0.00028 both). On such a window with such a soil the sensor does not see the flux at the
          1e-3 level: the new form will not move a recession estimate on US-NC2 by itself; it makes the
          statement "the operator solves the equation there" true, which it was not (73 bw c).
          On the rain window (F1) it does move theta: 0.00286 -> 0.00098.
        - NOT shown: the interior residual. On the two Ks 1e-3 data sets the held-out pde loss ends at 0.32 -
          0.35 of its step-0 level (F1: 5.5, i.e. an rms residual of 2.3 sigma_pde), against 0.06 - 0.14 on
          the Ks 1e-6 soils. It is outside the written rule and it is not explained; I do not claim the
          equation is solved to sigma_pde there. Whether it costs anything is read from R2 of the inversion
          (PINN against solver), not assumed.
        - F1 against the OBSERVED sensor: 0.0043 (additive 0.0033; the solver at the well's spline 0.0039, 73 bu).
          The additive net fitted the sensor better than the solver does because it was close to the
          hydrostatic profile; the new one is where the solver is. Not a regression of the PINN.
        DECISION (the rule of 73 bw e): adopted into the main tree, DEFAULT OFF (model.departure absent or
        'additive' = the code path of every result so far, the same arithmetic).
          - src/zbinv/model.py, src/zbinv/parametric.py: the staged files (the bytes that ran in jobs 15596460,
            15596461, 15596463 and 15596471, plus the docstring of the form), moved in by rename so that a job
            starting meanwhile never reads half a file. rfnc1t is still running (round 2 of 4): its later
            jobs import the new files and take the 'additive' branch.
          - tests: test_physics.py::test_hard_constraints_conductive_departure (h = 0 at the water table,
            hydrostatic start, 0 < g < 1 growing with suction, departure = g x the additive one to 1e-13,
            finite residual, base='initial' refused); test_parametric.py::test_build_param_model_departure_switch
            (default additive, q_scale = rms(q_obs / Ks) = sigma_q / q_frac, same state dict).
            The whole suite is run on the STAGED tree first (job 15598129, runs/_stage_cond/tests with links to
            configs / output / runs), then on the main tree after the move.
          - hpc/real_full_x.sh = hpc/real_full_obs.sh with a fourth argument appended to the training options
            (a new file: hpc/real_full_obs.sh is being read by the running driver and is not edited).
        PRE-REGISTERED RE-RUN (before any number of it exists):
          hpc/real_full_x.sh rfnc1tc output/real_nc2_sep23c_soilsep23a "[0.0,0.30]" "--set model.departure=conductive"
          Everything else is the adopted method. STATUS: DIAGNOSTIC, as rfnc1t (73 bs, 73 bu): the solver itself
          is 3.05 cm from the well on this data set (r 0.948), so R3 is not expected and nothing here counts for
          R3 / R4. What it decides: R1 and R2 on a rain window with a soil that never saw this window's well,
          and, against rfnc1t (additive, same data, still running), what learning the flux is worth in cm.
          Recorded per stage before any score: flux and pde loss ratios to the wide stage's step-0 level.
          EXPECTATION: flux ratio wide 0.03 - 0.10 (a prior of paths is harder than one path: on the own soil
          wide 0.022 against forward 0.007), rounds 0.02 - 0.05, forward PINNs of the loop 0.015 - 0.03.
          R1: spread < 0.2 cm. R2 (PINN - solver, limit 1 cm): 0.5 - 1.5 cm, about even - the forward
          difference is 0.001 with a dry-day mean of -0.0006, and S_W2's slope is 0.11 - 0.16 % per cm (73 bs),
          i.e. 0.4 - 0.9 cm before the correction loop, which removes what the forward PINN sees but not the
          forward PINN's own 0.001; the shower days (0.0022) are the likeliest place for a local 1 - 2 cm.
          Against the hourly well: as the solver, 2.8 - 3.5 cm, bias -1 to -2, r 0.94 - 0.96.
          rfnc1t (additive): the expectation of 73 bw f) stands (PINN - solver 1.5 - 3 cm).
          READING RULE: the form "matters for the inversion" if R2 is <= 1 cm here and > 1 cm in rfnc1t. If
          both are <= 1 cm the flux has little weight on this data set and the form is kept for the operator
          only. If this run is > 1 cm, R2 fails on this rain window and the forward offsets of the loop, by
          day, say where, before anything else is changed.
        18:23. ADOPTED AND VERIFIED. Test suite on the staged tree (job 15598129): 62 PASS, 0 failed; after the move,
        on the main tree (hpc/run_tests.slurm, job 15598151): 62 PASS, 0 failed (the 60 before + the 2 new;
        test_defaults_reproduce_a_saved_joint_run among them). diff -rq of the stage's src against src: none.
        The two forward checks F3 / F3b are a controlled pair by their config.yaml too (differences: the
        departure, the output directory and the name of the local file; seed 0 both).
        STARTED 18:22:15: PINN driver rfnc1tc (process group 2375444, logs/full_rfnc1tc.log, pipe_rfnc1tc.log).
        To check when its first job starts: runs/rfnc1tc_w0/config.yaml carries model.departure: conductive.
    73 ca. (13 Nov, 19:38) rfnc1tc, WIDE STAGE (departure='conductive', C_N1 with S_W2; DIAGNOSTIC, 73 bz): THE FOUR
        MEMBERS CARRY THE FLUX (kept states: 4 % of the zero-flux level left; additive rfnc1t: 100 %). Recorded
        before any water-table score is used.
        Jobs 15598190 (w2), 15598191 (w0), 15598192 (w3), 15598193 (w1) COMPLETED 19:29 - 19:31, no Python error;
        config.yaml of the four carries model.departure: conductive. inv0: job 15600234 COMPLETED.
        Held-out losses (runs/<member>/metrics.csv), rms ratio to step 0. "kept" = the evaluation with the lowest
        val_total; it is the state in model.pt (parametric.py lines 452 - 454; log of w0: "[lbfgs] kept held-out
        physics loss 3.1198e+01" = 30.209 + 0.988 + box; summary.json of the four holds the values of that row).
          member  step 0 pde / flux   kept at   pde (ratio)      flux (ratio)      last row pde (ratio)  flux (ratio)
          w0      48.942 / 612.03     40800     30.209 (0.786)   0.9883 (0.0402)   493.21 (3.174)        0.8272 (0.0368)
          w1      52.485 / 576.18     41800     11.759 (0.473)   0.8812 (0.0391)    18.05 (0.586)        0.8486 (0.0384)
          w2      47.766 / 609.09     41600     14.919 (0.559)   1.0046 (0.0406)    21.72 (0.674)        0.8884 (0.0382)
          w3      51.433 / 604.12     41800     13.565 (0.514)   0.8597 (0.0377)    15.82 (0.555)        0.7708 (0.0357)
          additive rfnc1t_w0..3 (same data, same held-out sets: step-0 values identical), kept states:
                  pde 38.30 / 38.95 / 39.43 / 38.74 (ratio 0.885 / 0.861 / 0.909 / 0.868),
                  flux 611.46 / 575.59 / 608.82 / 602.53 (ratio 0.9995 / 0.9995 / 0.9998 / 0.9987).
        EXPECTATION (73 bz): flux ratio of the wide stage 0.03 - 0.10. ACTUAL 0.038 - 0.040 (kept states): inside.
        The interior residual was not given a number beforehand; it ends lower than with the additive form in all
        four members (0.47 - 0.79 against 0.86 - 0.91), and is still far from small: kept val_pde 12 - 30, i.e.
        3.4 - 5.5 sigma_pde rms.
        w0, CHECKED: its last row (val_pde 493, 3.2 x the step-0 level) is NOT the kept state. Its held-out pde
        jumps between evaluations of the closing L-BFGS blocks (steps 39500 - 42000: 153 183 178 134 58 30.2 108
        49 115 546 195 493) while the training-batch L_pde of the same rows is 5.2 - 11.3; over the run 13 of its
        91 evaluations are above 3 x step 0 (w1 - w3: none). Not new with the form: the additive members have
        11 / 23 / 19 / 13 of 91 evaluations above 3 x step 0 on the same held-out sets. NOT CHECKED: where in
        (depth, time, path) the held-out residual of w0 sits (scripts/residual_map.py was not run) and why the
        held-out value is 5 - 100 x the training-batch value. w0 enters round 1 as its kept state (pde ratio 0.79,
        the weakest of the four); whether it recovers is read from the round-1 rows of rfnc1tc_r1_m0.
        inv0 (four members, four starts; against the record, runs/<tag>_inv0/summary.json):
          rfnc1tc_inv0  3.30 cm rms, bias -1.93, max 5.65; spread of the starts 1.9e-9 m; theta vs sensor 0.00139;
                        sigma median 0.85 cm; no coefficient on the box.
          rfnc1t_inv0   2.51 cm rms, bias -1.41, max 4.82; spread 2.5e-9 m; theta vs sensor 0.00208; sigma 0.87 cm.
          (solver on this data set: 3.05 cm from the well, bias -1.64.) Distances to the well do not give the
          distance PINN - solver; that is what R2 asks.
        EARLY LOOK (no decision attached: the reading rule of 73 bz applies to the final estimates only):
        scripts/score_real.py on the two inv0 against the four solver inversions nc2_sep23aon_sep23c_s0..3,
        outputs output/real_nc2_sep23c_soilsep23a/result_rfnc1tc_inv0 / result_rfnc1t_inv0.
        EXPECTATION, written before the jobs are submitted: PINN - solver rms 0.8 - 2 cm for rfnc1tc_inv0 (wide
        stage only, no refinement, no value correction) and 1.5 - 3 cm for rfnc1t_inv0.
        Driver (process group 2375444) is in refinement round 1 since 19:32.
        19:40. EARLY LOOK, RESULT (jobs 15600414 / 15600415, COMPLETED, exit 0; hourly, 265 h; "solver" = the
        solver inversion with the lowest cost, nc2_sep23aon_sep23c_s1, the four starts equal in cost to 7 digits):
                          PINN - solver: rms    bias     max     r        vs well: rms   bias    r       theta vs sensor
          rfnc1tc_inv0    (conductive)   0.55   -0.29    2.46    0.9989            3.30  -1.93   0.948   0.00139
          rfnc1t_inv0     (additive)     1.30   +0.23    3.36    0.9855            2.51  -1.41   0.966   0.00208
          solver                                                                   3.05  -1.64   0.948   0.00144
          (cm; files output/real_nc2_sep23c_soilsep23a/result_rfnc1tc_inv0.json, result_rfnc1t_inv0.json)
        EXPECTATION: 0.8 - 2 cm and 1.5 - 3 cm. ACTUAL 0.55 and 1.30: both below the range (wrong on the good
        side; the wide stage alone is closer to the solver than I assumed for either form).
        READ FROM THE FIGURE result_rfnc1tc_inv0.png: the two curves lie on each other except in the last day,
        where the PINN ends at 0.76 m and the solver at 0.79 m (the 2.46 cm), and slightly on days 1 - 2. Neither
        follows the well's rise after the showers (well 0.47 m on day 3.3, both estimates 0.515 m).
        What this is and is not: wide stage only, one data set, DIAGNOSTIC. It says that with the flux learnt
        the wide-stage estimate is the solver's to 0.55 cm (and has the solver's fit to the sensor, 0.00139
        against 0.00144), while the additive form's is at 1.30 cm with a worse fit (0.00208). The additive
        estimate is closer to the WELL (2.51 against 3.30 cm): being nearer the well by departing from the
        solver is not a merit under the criteria (R2), and nothing on this window counts for R3. The reading
        rule of 73 bz is applied to the final estimates of the two runs, not to these.
74. PAPER-STANDARD RESULT SET, built from the US-NC2 rain window (C_N1, own soil, PINN run rfnc1)
    (14 Nov, 00:17; request: "lets work from there and try produce standard results a pinn paper would use".
    PLAN, written before anything is built or submitted.)
    a) What the case is, and what it is not. Window: 11 days, Sep 2023, hourly, sensor = mean over 0 - 0.30 m, 14
       spline coefficients. Soil calibrated ON THIS WINDOW (in-sample), so the case shows that the network recovers
       the water table from flux and soil moisture when the soil is right; it is not a held-out test. No site has
       passed R4, and the held-out windows fail R3 on the solver side too (73). Every table of this set says so.
    b) Results from runs that exist (no new training). One script, scripts/paper_figures.py (CPU batch job), output
       output/paper/nc2_sep23c/:
         P1 fig1_setup      forcing and data: surface flux q_surf and root uptake (mm/d), sensor soil moisture,
                            observed water table (record)
         P2 fig2_forward    FORWARD verification, PINN against the FEM solver, same soil, forcing and water table
                            (the well's spline): theta(z) at six times, and the error over depth against time.
                            Network: runs/fwd_nc2c (form of the rfnc1 run). Table: rms / relative L2 / max of
                            theta over the whole (z, t) domain and at the sensor; flux boundary condition; residual
         P3 fig3_inverse    INVERSE result: sensor theta observed / PINN; water table observed / PINN with +-1
                            posterior sd (dotted lines) / FEM inversion / hydrostatic reading of the sensor; error
                            against the well by hour. Table: rmse, bias, max, r of the three against the well,
                            theta misfit, PINN - FEM, start spread, floor of the spline (0.81 cm)
         P4 fig4_training   held-out PDE and flux losses of the four members against the step (log axis), wide
                            stage then the four refinement rounds
         P5 fig5_stages     the estimate by stage (wide, rounds 1 - 4, value-corrected) and its distance to the well
                            and to the FEM inversion by stage; four starts (initial paths and where they end)
         P6 fig6_physics    physics check at the estimate: prescribed against network surface flux; rms residual
                            over depth against time and over time against xi (final members and forward PINN)
         P7 tables.md       the tables above + cost (training seconds per stage from metrics.csv, inversion wall
                            time from summary.json, FEM inversion wall time) + soil and settings
         P8 synthetic       inverse verification with a known truth: the FINAL ACCEPTANCE TABLE of finding 62 and
                            output/compare/final_*.png as they are (clean, noise 0.005, 0.01, second noise seed)
         P9 limits          the held-out and other-site numbers of 73 as a table (not a success)
       EXPECTATIONS, before the script is run. P2: theta rms over the domain 3e-4 - 1e-3 (at the sensor 4.3e-4 is
       known from truth_profile.npz and has to be reproduced: check A3); relative L2 0.1 - 0.3 %; max error
       <= 0.01, at the surface during the showers; flux at the surface within 1 - 6 % rms of the prescribed.
       P5: flat. summary.json of the six inversions gives 1.093 / 1.111 / 1.105 / 1.106 / 1.109 / 1.105 cm against
       the record (read today): on this window the wide stage is already there, and the figure will say that.
       P3: the numbers of result_rfnc1.json (1.10 / 1.11 / 3.41 cm; PINN - FEM 0.065 cm).
    c) New runs a paper would add (submitted after this entry is committed):
         N0 single-member inversions: scripts/param_invert.py with ONE member of round 4 each (4 CPU jobs; --box,
            --prev-center inv3, --coarse the four wide members; out runs/rfnc1_inv4_m<k>). Spread of the estimate
            over independently seeded networks. EXPECTATION: each within 0.01 - 0.1 cm rms of the four-member
            estimate (member sd of theta at the sensor is 3e-5, the fit residual 1.3e-3).
         N1 independent repeats of the WHOLE pipeline, other seeds: tags rfnc1s1 (SEED0=1000) and rfnc1s2
            (SEED0=2000). hpc/param_pipeline.sh and hpc/dc_loop.sh get an optional SEED0 (default 0: commands
            unchanged), added to every training seed. EXPECTATION: final estimates within 0.3 cm rms (max 1 cm) of
            rfnc1's and of each other; each 1.0 - 1.25 cm from the well, <= 0.3 cm from the FEM inversion, start
            spread < 0.2 cm, loop converged within 2 iterations. READING RULE, fixed now: "independent of the
            training seed" is stated only if all three runs satisfy R1 and R2 and their pairwise rms is <= 0.5 cm;
            a run that does not is reported as it is and looked into before any such sentence is written.
         N2 forward sensitivity (one light GPU job each, hpc/fwd_truth.sh on the same data set, scored against the
            solver as fwd_nc2c): fs_nc2c_s1, _s2 (train.seed 1, 2), _w64, _w32 (model.width), _d3 (model.depth=3),
            _c2k (train.n_col=2048, train.lbfgs_n_col=8192). Baseline: width 128, depth 6, 8192 / 32768.
            EXPECTATION (weak, I have not varied these on site data): seeds 3e-4 - 8e-4 at the sensor; w64 within
            2 x the baseline; w32 2 - 5 x; d3 1.5 - 4 x; c2k 1 - 2 x (the points are redrawn every step).
       Not run: noise levels on site data (the sensor has its own; the noise study is the synthetic set P8), and
       the departure form on this soil (the additive form learns the flux here, ratio 0.022 in the wide stage; the
       comparison of the two forms is the pair rfnc1t / rfnc1tc on the other soil, running, 73 bz - ca).
    d) Acceptance of this set.
         A1 every number of tables.md that also stands in result_rfnc1.json or in a summary.json equals it (jq).
         A2 figures: lines only, no fills, no titles / annotations / text; axis labels with units; a legend only
            where there is more than one line. Checked on every PNG by eye and by grep of the script.
         A3 P2 reproduces the sensor rms of runs/fwd_nc2c/truth_profile.npz (4.3e-4) from the model file.
         A4 SEED0: bash -n on both drivers; config.yaml of rfnc1s1_w0 shows train.seed 1000; the diff of the two
            drivers touches the seed expressions only.
         A5 the tables name the soil as in-sample and carry the held-out numbers.
    74 a. (14 Nov, 00:19) LAUNCHED N0, N1, N2 of the plan above; and the user, while this was being set up:
        "and need more site to prove this pinn works" -> a site track is added (74 b, written before any site job).
        SEED0 (optional, default 0) added to hpc/param_pipeline.sh and hpc/dc_loop.sh: the diff is the two seed
        expressions of the first and the one of the second (new files moved in by rename; bash -n passes; the
        running drivers of rfnc1tc / rfnc1t keep their old copies). Settings of the repeat = the good run's: the
        train sets of logs/pipe_rfnc1.log line 1 and of hpc/real_full_obs.sh today are byte-identical (cmp).
          N1  rfnc1s1 (SEED0=1000), rfnc1s2 (SEED0=2000): hpc/real_full_obs.sh on output/real_nc2_sep23c_own,
              "[0.0,0.30]"; process groups 2001871 / 2001874; logs/full_rfnc1s?.log.
          N2  fs_nc2c_{s1,s2,w64,w32,d3,c2k}: hpc/fwd_truth.sh, same data set; logs/fwd_fs_nc2c_*.log.
          N0  jobs 15609737 / 15609752 / 15609768 / 15609783 -> runs/rfnc1_inv4_m0..3.
        Still to check for A4: train.seed in runs/rfnc1s1_w0/config.yaml (1000) and runs/fs_nc2c_s1 (1).
    74 b. (14 Nov, 00:29) SITE TRACK, PLAN (written before any site job; asked: "need more site to prove this pinn works").
        Two statements, kept apart as in 72 - 73:
          (i)  the PINN returns the estimate of the numerical inversion of the same model from every start (R1, R2):
               a statement about the PINN, testable on any window;
          (ii) that estimate is the observed water table (R3): a statement about the soil-column model and its
               calibrated soil at the site, which the PINN shares with the solver.
        Where it stands: (i) at US-NC2 in three runs (rfnc1 0.065 cm, rfn1 0.07, rfnd 0.06), not at US-PAS C1
        (2.92 cm); (ii) at US-NC2 on C_N1 in-sample (1.10 cm) and on W3 with the May soil (rfnd, 1.58 cm, held out
        but NOT the pre-registered transfer); the pre-registered T_N1 fails by the 9.5 cm the solver has. No site
        has passed R4.
        DATA ON DISK, looked at today (headers only: head / grep on the login node, nothing fitted):
          fem_validation/temp_extract, twelve half-hourly files not looked at in bf / bg:
            both a soil-moisture and a water-table column: US-CRT (drainage tiles) and US-Uaf (permafrost), both
            set aside in bf; water table without soil moisture: US-Los, US-Myb, US-OWC, US-Sne, US-Tw1, US-Tw4,
            US-Tw5, US-Twt, US-WPT; US-Snd (BASE 2-1): SWC_1, SWC_2 and no water-table column.
          fem_validation/uploaded_data: as bf / bg. CL-SDF stays the one candidate; the project's GLDAS tables
            hold US wells only, so its transpiration fraction would be a number set by hand -> last in line.
        => no NEW SITE can be started from the data on disk. New sites need new data.
        TRACKS
          S1 (now; further WINDOWS of US-NC2, not further sites; no new data). W1 (1 May 2023 + 15 d) and W2
             (1 Sep 2023 + 6 d) with their own soils: the solver's in-sample inversions are 0.89 and 0.63 cm from
             the well (az), both soils usable; the PINN was never run on them.
               rfnm1  hpc/real_full_obs.sh on output/real_nc2_may23_own          "[0.0,0.30]"
               rfna1  hpc/real_full_obs.sh on output/real_nc2_sep23a_own_cal3    "[0.0,0.30]"
             (SEED0 0, i.e. the seeds of every run so far.)
             EXPECTATION: four starts, one estimate (< 0.2 cm); PINN - solver <= 0.3 cm rms (three runs at this
             site: 0.06 - 0.07); from the well 0.7 - 1.1 cm (W1) and 0.5 - 0.9 cm (W2). Both soils have Ks on the
             upper bound (1e-3 m/s): the profile is then close to hydrostatic at every hour, and the hydrostatic
             reading WITH THE CALIBRATED SOIL should be about as good as the inversion (scripts/score_real.py
             prints it; said in the table whichever way it comes out). These two windows test (i) and the
             retention curve, not the dynamics; C_N1 is the window with dynamics. In-sample: no R4.
          S2 (running, agent; report -> tasks/site_candidates.md): sites with open data that carry, at one plot,
             a soil-moisture sensor in the top 0.3 m, a well with the water table at 0.2 - 1.5 m moving by
             >= 0.2 m, rain, and ET (eddy covariance or full meteorology), on mineral soil without peat / moss,
             inundation, drains, irrigation, tides or frost. Two groups: downloadable with no account; needs an
             account. It downloads and fits nothing.
          S3 AmeriFlux sites not on disk (candidates from S2): a download is logged with the user's name, e-mail
             and intended use and is sent to the site teams -> not started without the user's word.
        PROTOCOL FOR A NEW SITE (unchanged: ay, az). Record first (sign of WTD checked against the sensor, depth
        of the sensor from the measurement-height table, plateau of the sensor, shift between rain and the rise of
        the well), nothing fitted; windows by the written rule of ay; configs/real_<site>_<window>.yaml;
        hpc/cycle_solver.sh (data set, four calibration starts, in-sample solver inversions); the PINN
        (hpc/real_full_obs.sh) on a window only if its soil is usable (lowest misfit AND in-sample inversion
        within 2 cm, r >= 0.975 below saturation); held-out test = the next window with that soil; R1 - R4 as
        they are.
        S1 LAUNCHED 00:30: logs/full_rfnm1.log, logs/full_rfna1.log.
        WHAT WILL COUNT: a new site counts for (ii) only with a held-out window inside R3. An in-sample window
        inside R3 with R1 and R2 is reported as in-sample, as C_N1 is.
    74 c. (14 Nov, 00:48) RESULTS SET BUILT FROM THE RUNS THAT EXIST (P1 - P9, N0). scripts/paper_figures.py (new, reporting
        only; CPU jobs 15610384 and, after the label and layout fixes below, 15610522, exit=0) ->
        output/paper/nc2_sep23c/{fig1_setup .. fig6_physics}.png, tables.md, metrics.json; write-up output/paper/README.md
        (.gitignore: output/paper/* is now tracked).
        Against the expectations of 74, part b):
          P2 forward (runs/fwd_nc2c)  theta rms 3.27e-4 (expected 3e-4 - 1e-3: inside); relative L2 0.080 % (expected
                                      0.1 - 0.3 %: slightly better); max 5.8e-3, at the surface at the rain pulses
                                      (expected <= 0.01 there: as expected); flux 4.82 % (expected 1 - 6 %: inside)
          P3 inverse                  1.10 / +0.11 / 3.16 / r 0.9905 against the well; solver 1.11; hydrostatic 3.41;
                                      PINN - solver 0.07 (max 0.30); spline floor 0.81: as expected (= result_rfnc1.json)
          P5 stages                   1.09 / 1.11 / 1.11 / 1.11 / 1.11 / 1.10 cm from the well, 0.10 / 0.06 / 0.07 / 0.06 /
                                      0.07 / 0.07 from the solver: flat, as expected. On this window the wide stage is
                                      already at the estimate; the README says so (and names US-ONA Nov 2022, 1.28 ->
                                      0.52 -> 0.10 cm, as the window where the later stages are needed)
          N0 single members           each within 0.01 - 0.02 cm rms (max 0.05 - 0.09) of the four-member estimate of
                                      round 4 (expected 0.01 - 0.1: inside); against the well 1.10 - 1.11 cm
        Not expected, found while building, and in the write-up:
          - The training loss has THREE terms (PDE residual, surface flux, water balance of space-time boxes), and the
            network has memory features of the forcing (38 inputs, 87 681 parameters): read from
            runs/rfnc1_r4_m0/config.yaml and ParamTrainer.losses. Plan 74, part b), P4 said "PDE and flux losses": fig4 has
            three panels.
          - PDE residual on a grid clustered at the surface: rms 0.50 - 0.66 sigma_pde (forward networks), 0.87 - 1.48
            (operator members), max 44 - 129 sigma_pde at the three rain pulses and at t = 0. Larger than the square
            root of the held-out PDE losses (0.17 forward, 0.37 members of round 4; losses 0.029 and 0.134), which
            are sampled as in training.
          - Cost: 13.6 GPU hours of training (22 training runs) against 26 min CPU for one start of the solver
            inversion (1.59 CPU h for four).
        Fixes between the two jobs: stage label 'final' -> 'value correction' (the task log's name); y labels of fig6
        on two lines (they ran into each other); --single added (table 9).
        ACCEPTANCE: A1 met (table 3 = result_rfnc1.json: 1.10 / +0.11 / 3.16 / 0.9905; 0.07 / 0.30; 3.41; 0.81; sigma
        1.11; theta 0.00134). A2 met (grep of the script for title / annotate / .text( / fill: no hit; the six PNGs
        looked at: lines and markers only, axis labels with units, legends only with more than one line). A3 met
        (sensor rms 4.26e-4 = 4.3e-4 of truth_profile.npz). A4 for N1: train.seed 1000 (rfnc1s1_w0), 1003 (_w3), 2000
        (rfnc1s2_w0); for N2 still open (fs_nc2c_* pending on Priority since 00:19, no config.yaml yet). A5 met (data
        table: "calibrated on this window (in-sample)"; README sections "What this case shows, and what it does not"
        and "Other windows and sites").
        P9 cross-site table (README): final estimate of the 11 scored runs, labels checked against w, z, as, at, ag,
        73 bv, the rfh1p and rfr1a entries. (i) PINN = solver to <= 0.5 cm in 8 of 11 (starts < 1e-5 cm apart); fails
        at US-PAS C1 (2.92), US-ONA B17 / S_D (2.67), US-ONA rain window (28.2). (ii) whole window only at US-NC2
        (rfnc1 in-sample, rfnd diagnostic); rf22c below saturation only. No site has passed R4. Unchanged.
        STILL RUNNING at 00:48: N1 (rfnc1s1 four wide members, rfnc1s2 three of four, on V100), N2 and S1 (all
        pending on Priority), rfnc1tc round 3, rfnc1t, US-HB2 calibration continuation, site-search agent.
    74 d. (14 Nov, 01:19) SITE SEARCH REPORTED (S2 of 74 b): tasks/site_candidates.md. Checked before use; nothing
        fitted, no data file downloaded, no licence accepted, no account used.
        WHAT THE REPORT SAYS
          - Open with no account: NO site carries soil moisture, water table, rain and ET in one archive. Two have a
            sensor and a well about 1 m apart: Point Reyes (ESS-DIVE; hillslope; no rain or ET in the package) and
            Shale Hills RTH3 (HydroShare, 2009 - 2012; valley floor by the stream; the rain and LE entries are dead
            links).
          - AmeriFlux: nothing new on mineral soil. The one new site with both columns that is not peat, muck or
            marsh is BR-Xpw (Pantanal pasture, under water half the year).
          - NEON (token since 30 Jun 2026) and EDI / LTER (login since 30 Jul 2026) answer HTTP 403 without one.
          - ICOS (Europe): no account; the ICOS data licence (CC BY 4.0) is accepted at each download. FR-Bil
            (Bilos, pine on sandy podzol) first.
        MY CHECKS (01:12 - 01:19), from the cached samples and open ICOS metadata (meta.icos-cp.eu landing pages as
        JSON, SPARQL); written into the last section of tasks/site_candidates.md:
          - held as written: ICOS column lists and periods of FR-Bil / BE-Bra; the three FR-Bil objects resolve
            (19, 37, 62 MB; "ICOS CCBY4 Data Licence"); ESS-DIVE licence and file list; Shale Hills well 2 by month.
          - corrected: the BE-Bra Meteosens link was the final 2020 - 2023 file, not the interim one; Shale Hills
            well 1 is at its bottom in Feb and Jun - Aug, and November has no accepted value.
          - NEW, and the main point: the ICOS metadata lists every sensor with position and depth. At FR-Bil the
            co-location the report had only inferred is there: plot 1 well 1.8 m from a profile at 0.05 / 0.08 /
            0.12 / 0.24 / 0.47 / 1.05 / 1.97 m, plot 2 well 2.5 m from one at 0.05 / 0.08 / 0.10 / 0.16 / 0.54 /
            0.94 / 1.25 m; plots 90 m apart (two columns at one station); pressure sensors at 1.95 and 1.70 m
            (1.85 m since Mar 2026). The same layout (well 1 - 4 m from a 0.05 m sensor, rain gauge and eddy
            covariance at the station) at IT-SR2 (pine, coastal plain near Pisa), SE-Nor (sandy till; frost),
            DK-Sor (beech), BE-Dor (grassland), FR-Fon / FR-Hes (wells 0.8 - 0.96 m deep: perched winter water
            table, inferred), BE-Bra. Wells 3.75 - 9.6 m deep at IT-Cp2, DE-RuS, DK-Vng, NL-Loo: set aside.
          - NOT checked, because it needs the data: the water-table range of any ICOS station, the ditches and the
            hardpan at FR-Bil. The 0-0.5 m winter / 1-1.5 m summer range in the report is a regional figure.
        READING
          - The site track rests on one decision that is the user's: accepting the ICOS data licence at download
            (attribution; the same licence on every ICOS object, so one decision covers FR-Bil and the next
            stations). Asked at the end of this turn; default recommended: yes, FR-Bil first.
          - Without it: Point Reyes and Shale Hills are the only sites that can be opened, and both need rain (and
            ET) borrowed from somewhere else, on a hillslope and a valley floor. A result there would test (i) and
            say little about (ii). Not started; a rain source for either has not been looked for.
          - BR-Xpw (AmeriFlux account) stays last: flooded half the year, no rain in the dry half.
        PLAN ON A YES (protocol of 74 b unchanged; written before any ICOS number is seen)
          1. The three FR-Bil objects, each into its own new empty directory; listed with unzip -l before anything
             is extracted; read only by batch jobs (python -I), scripts kept elsewhere.
          2. Record check, nothing fitted: sign and reference of WTD_1 / WTD_2 against the 0.47 m and 1.05 m
             sensors (they must saturate when the well says the water table is above them); hours by month with
             the water table inside 0.2 - 1.5 m; plateau of the 0.05 - 0.24 m sensors; delay from rain to the rise
             of the well; a ditch signature (recessions that flatten at one level); a hardpan signature (0.47 m
             sensor saturated while the well is below 0.75 m).
          3. Sensor layer: the mean of the four upper sensors as the model's layer [top, bottom] is NOT assumed:
             the layer is fixed from the depths in the metadata before a window is chosen (0.05 - 0.24 m at plot 1,
             0.05 - 0.16 m at plot 2), one sensor first if the four disagree.
          4. Windows by the written rule of ay; the calibration window first, the held-out test = the NEXT window
             with that soil, declared before it is run; hpc/cycle_solver.sh; PINN only on a usable soil; R1 - R4.
          EXPECTATION, written now: (i) to <= 0.5 cm where the water table stays inside the box (8 of 11 so far).
          (ii) open: sand with a shallow water table is where it held (US-NC2); the ditches and the hardpan are
          the two ways it can fail here, and the record check is meant to show them before a GPU hour is spent.
        COUNTS AS BEFORE: a new site counts for (ii) only with a held-out window inside R3.
        RUNNING at 01:18: N1 (eight wide members on V100, 31 - 54 min in), N2 (six forward runs on V100, 24 - 32
        min), S1 (rfnm1: three of four wide members running; rfna1 pending on Priority), rfnc1tc round 4 pending,
        US-HB2 calibration continuation (7 h 11 min).
        A4 for N2, checked 01:21 from runs/fs_nc2c_*/config.yaml (written when the jobs started): s1 seed 1, s2 seed 2,
        w64 width 64, w32 width 32, d3 depth 3, c2k n_col 2048 (lbfgs_n_col 8192); every other one of width 128,
        depth 6, seed 0, n_col 8192 / 32768 as in runs/fwd_nc2c. Each variant differs from the baseline in the one
        setting it is named for. A4 met; scores not read (runs at 24 - 32 min).
    74 e. (14 Nov, 01:50) N2 FORWARD SENSITIVITY: ALL SIX RUNS DONE. THE LOSSES RESPOND TO THE NETWORK, THE "ERROR"
        AGAINST THE NUMERICAL SOLUTION DOES NOT. Cause not yet known; a check is planned below before anything is said.
        ORDER, stated as it was: the waiter printed the tail of each forward log, per-day error rows included, before
        I had read the held-out loss ratios. The ratios were read from metrics.csv right after and are given first
        here, but the rule (ratios before any score) was not kept in time. Lesson entry written.
        HELD-OUT LOSSES, step 0 -> 28000 (ratio), from runs/<run>/metrics.csv:
          run          val_pde                    val_flux                  val_box
          fwd_nc2c     5.249 -> 0.02912 (5.6e-3)  612.0 -> 0.02879 (4.7e-5)  23.97 -> 0.001772 (7.4e-5)   baseline
          fs_nc2c_s1   4.948 -> 0.02508 (5.1e-3)  576.2 -> 0.02442 (4.2e-5)  25.54 -> 0.002103 (8.2e-5)
          fs_nc2c_s2   4.943 -> 0.02752 (5.6e-3)  609.1 -> 0.04017 (6.6e-5)  17.72 -> 0.004546 (2.6e-4)
          fs_nc2c_w64  5.249 -> 0.07152 (1.4e-2)  612.0 -> 0.1344  (2.2e-4)  23.97 -> 0.008587 (3.6e-4)
          fs_nc2c_w32  5.249 -> 0.1220  (2.3e-2)  612.0 -> 0.1846  (3.0e-4)  23.97 -> 0.009954 (4.2e-4)
          fs_nc2c_d3   5.249 -> 0.1243  (2.4e-2)  612.0 -> 0.2247  (3.7e-4)  23.97 -> 0.01424  (5.9e-4)
          fs_nc2c_c2k  5.249 -> 0.03102 (5.9e-3)  612.0 -> 0.03233 (5.3e-5)  23.97 -> 0.002381 (9.9e-5)
          -> every run learned the flux and the PDE (ratios 4e-5 - 4e-4 and 5e-3 - 2.4e-2). Width 64, width 32 and
             depth 3 end 2.5 - 4.3 x higher in the PDE loss and 4.7 - 7.8 x higher in the flux loss than the
             baseline; seeds and the 4 x smaller collocation set are within 1.4 x.
        AGAINST THE NUMERICAL SOLUTION OF THE DATA SET ([truth] lines of logs/fwd_fs_nc2c_*.log, jobs 15611961,
        15611734, 15611794, 15611688, 15611679, 15611785):
          run        sensor RMS  max      profile RMS  max      day 2 (rain, -10.9 mm/d), sensor RMS
          baseline   4.26e-4   4.54e-3   3.27e-4    5.82e-3
          s1         4.26e-4   4.54e-3   3.27e-4    5.83e-3    1.30e-3
          s2         4.27e-4   4.54e-3   3.28e-4    5.83e-3    1.29 - 1.30e-3 in all six
          w64        4.32e-4   4.53e-3   3.31e-4    5.80e-3
          w32        4.30e-4   4.54e-3   3.30e-4    5.81e-3
          d3         4.29e-4   4.50e-3   3.29e-4    5.76e-3
          c2k        4.25e-4   4.54e-3   3.27e-4    5.82e-3
          Quiet days differ between runs (day 9: s1 1.8e-4, s2 1.9e-4, c2k 1.9e-4, w64 2.7e-4, w32 2.6e-4, d3 2.5e-4).
        AGAINST THE EXPECTATION OF 74 (c, N2): seeds inside (4.3e-4 in 3e-4 - 8e-4); w64 inside (1.01 x); c2k inside
          (1.00 x); w32 NOT met (1.01 x where 2 - 5 x was written); d3 NOT met (1.01 x where 1.5 - 4 x was written).
          The expectation assumed that the figure measures the network. Seven networks with losses 8 x apart agree
          on it to 1 %, and on its largest part (the rain day) to the last digit shown: it measures something they
          have in common.
        WHAT IS KNOWN, WHAT IS NOT:
          - Known: the data set's reference is ONE solve at 1 h steps and 101 nodes (twin_config.yaml fem.dt_hours,
            fem.n_nodes), and make_real.py compared it at the sensor with a finer solve (0.25 h, 201 nodes):
            real_summary.json fem_vs_finer_fem rmse 2.9e-4, max 3.1e-3. That is the size of the "forward error"
            (4.3e-4, 4.5e-3). The finer solve was not stored.
          - Both codes take the forcing as hourly nodes, linear in between (forcing.node_hours 1; np.interp in
            make_real.py, torch_interp in the networks).
          - Not known: whether the networks sit on the refined solution. They could also share an error of their own
            (same ansatz, same forcing-memory inputs, same sampling). Not stated either way until checked.
        CHECK (scoring only; the solver is not used by the method): scripts/forward_reference_check.py, new.
          The solve of the data set is repeated at 1 h / 101 nodes (must reproduce the stored reference), 0.25 h /
          101, 0.25 h / 201, 0.0625 h / 201, 0.0625 h / 401 and 0.015625 h / 401, same path, forcing nodes and root
          sink; the finest once more with the Picard tolerance / 100. Each of the seven networks is compared with
          every level (sensor, and profile on the stored grid), and with the baseline network.
          EXPECTATION, written before the job is submitted:
            E1 level 1 h / 101 equals the stored reference (same code and inputs): difference < 1e-10.
            E2 if the common part is the reference's time step: every network's sensor distance falls from 4.3e-4
               to <= 1.5e-4 at the finest level and day 2 from 1.3e-3 to <= 4e-4; the seeds and c2k are within
               5e-5 RMS of the baseline network, w64 / w32 / d3 within 5e-5 - 2e-4; against the finest level the
               runs are ordered by their losses (w32 and d3 worst).
            E3 if instead the distance falls by less than a factor of 2: the common part is in the networks, the
               forward table stands, and the shared design is where to look (ansatz time scale, forcing inputs).
          READING RULE, fixed now: the forward error of the results set becomes the one against the finest level
            only if the two finest levels differ from each other by less than a third of it; otherwise it is given
            as an upper bound with the reference's own difference next to it. The 1 h figures stay in the table
            either way, named as what they are.
          NOT TOUCHED by this: the inverse results. The numerical inversion and the data set keep the 1 h solve; a
            statement about them needs its own check (noted, not started).
    74 f. (14 Nov, 02:46) REFERENCE CHECK, FIRST JOB (15612223, 52 min; logs/cpu_15612223.out;
        output/paper/nc2_sep23c/forward_reference_a.json). THE 4.3e-4 WAS THE 1 h TIME STEP OF THE NUMERICAL
        REFERENCE. Against the solve at 1/16 h the networks are 4.7e-5 - 7.6e-5 at the sensor, ordered as their losses.
        E1 NOT MET as written: the 1 h / 101 level differs from the stored reference by 7e-7 RMS, 2.6e-6 max (sensor),
          not < 1e-10. Read in wtd_pipeline/fem/fem_pytorch.py 747 - 758: the Picard loop iterates a batch until every
          member has converged; the stored reference was member 0 of a batch of two (spline path, hourly record),
          here it is solved alone and stops earlier. Not re-run as a batch of two; 600 x below what is measured here.
        THE REFERENCE, each level minus the finest (1/16 h, 401 nodes), theta at the sensor, RMS (max):
          1 h, 101 nodes      3.92e-4 (4.17e-3)   day 2: 1.2e-3      <- the data set's reference
          1/4 h, 101          9.97e-5 (1.03e-3)   day 2: 2.9e-4
          1/4 h, 201          9.96e-5 (1.03e-3)
          1/16 h, 201         1.9e-7  (3.9e-7)
          -> time step: 3.9 x smaller for a 4 x smaller step (first order). Grid: 101 -> 201 -> 401 nodes changes
             nothing at this level (1e-7).
          -> Picard tolerance: the finest level with tolerance 1e-7 instead of 1e-5 moves by 2.8e-5 RMS (7.4e-5 max,
             mean +1.5e-5) at the sensor. Not expected; it is 60 % of the networks' distance to that level, so that
             level is not yet a yardstick for them.
        THE NETWORKS, sensor RMS (max):      1 h / 101          1/4 h / 101        1/16 h / 401       minus baseline
          baseline                           4.26e-4 (4.54e-3)  1.33e-4 (1.39e-3)  4.74e-5 (3.63e-4)  -
          s1                                 4.26e-4            1.34e-4            4.80e-5 (3.69e-4)  1.90e-5 (7.1e-5)
          s2                                 4.27e-4            1.35e-4            5.05e-5 (3.69e-4)  2.08e-5 (8.0e-5)
          c2k                                4.25e-4            1.33e-4            4.71e-5 (3.65e-4)  1.72e-5 (9.4e-5)
          w64                                4.32e-4            1.45e-4            6.71e-5 (3.53e-4)  5.47e-5 (2.3e-4)
          d3                                 4.29e-4            1.42e-4            6.94e-5 (3.24e-4)  5.35e-5 (2.7e-4)
          w32                                4.30e-4            1.46e-4            7.58e-5 (3.64e-4)  6.30e-5 (2.6e-4)
          Profile (stored grid), baseline: 3.27e-4 (5.82e-3) -> 1.03e-4 -> 3.63e-5 (4.71e-4).
          Day 2 (rain), baseline: 1.3e-3 -> 3.9e-4 -> 1.1e-4.
        AGAINST E2 (74 e): met. Distance <= 1.5e-4 at the finest level (4.7e-5 - 7.6e-5); day 2 <= 4e-4 (1.0 -
          1.2e-4); seeds and c2k within 5e-5 of the baseline (1.7 - 2.1e-5); w64 / w32 / d3 in 5e-5 - 2e-4 (5.4 -
          6.3e-5); order at the finest level = order of the losses, w32 and d3 swapped (their losses are 2 % apart).
          E3 does not apply.
        READING, limited to what was run:
          - The networks agree with each other (2e-5 between seeds) twenty times better than with the 1 h solve
            (4.3e-4), and they move toward the reference as its time step is refined. The forward error in the
            results set (3.3e-4 / 4.3e-4, "largest at the surface during the rain pulses") is the reference's
            time-step error, not the networks'.
          - How large the networks' own error is has NOT been measured yet: the finest level still carries a
            first-order time-step error (about 4.2e-4 per hour of step: 2.6e-5 at 1/16 h, an extrapolation) and
            the tolerance error above. By the reading rule of 74 e the figure against it is an upper bound:
            <= 4.7e-5 at the sensor for the baseline.
          - Capacity now shows where it should: width 64, depth 3 and width 32 are 1.4 - 1.6 x the baseline
            against the finest level, and 5 - 6e-5 from the baseline network. The written factors for w32
            (2 - 5 x) and d3 (1.5 - 4 x) are still high.
          - The inverse results are not touched by this (74 e). One consequence to carry into the write-up:
            the numerical inversion used for comparison runs the 1 h solve. Its forward error is larger than
            the networks'; both are far below the sensor's 0.005.
        NEXT (second job; the first job's long sibling 15612224, same tolerance, is cancelled as superseded):
          all levels with the Picard tolerance at 1e-8 (max 400 iterations), 101 nodes, time steps 1, 1/4, 1/16,
          1/64, 1/256 h, and 1/16 h at 201 nodes for the grid. scripts/forward_reference_check.py gets --tol and
          --max-iter (defaults: the data set's, so the first job's command is unchanged) and a table of the
          difference between consecutive levels.
          EXPECTATION, written before submission:
            F1 consecutive levels differ 4 +- 0.5 x less each time (first order in the step).
            F2 1/16 h at 101 and at 201 nodes: < 1e-6 RMS apart at the sensor.
            F3 the two finest levels (1/64, 1/256 h) are < 1e-5 apart (first-order estimate 5e-6).
            F4 against the finest level: baseline, seeds and c2k 2e-5 - 5e-5 at the sensor (they are 2e-5 from
               each other, so not far below 1.5e-5 unless their errors are shared); w64 / w32 / d3 5e-5 - 8e-5;
               day 2 still the largest day.
          The reading rule of 74 e applies to this job as written.
    74 g. (14 Nov, 04:05) REFERENCE CHECK, TIGHT TOLERANCE, FIRST OF TWO JOBS (15613446, 73 min; logs/cpu_15613446.out;
        output/paper/nc2_sep23c/forward_reference_b.json). Picard tolerance 1e-8 at every level, at most 400
        iterations (largest used: 24), every step converged. Finest level here 1/64 h. The 1/256 h job (15613447)
        is on its last level; F3 and the reading rule wait for it.
        THE REFERENCE AGAINST ITSELF, sensor RMS (max) | profile RMS (max), theta in m3/m3:
          each level minus the one before:
            1/4 h   - 1 h      (101 nodes)   2.95e-4 (3.14e-3) | 2.26e-4 (4.05e-3)
            1/16 h  - 1/4 h    (101 nodes)   9.75e-5 (1.04e-3) | 7.49e-5 (1.33e-3)      3.02 x less
            1/64 h  - 1/16 h   (101 nodes)   2.65e-5 (2.80e-4) | 2.04e-5 (3.58e-4)      3.68 x less
            1/16 h: 201 - 101 nodes          8.0e-7  (1.4e-6)  | 1.2e-7  (3.6e-6)
          each level minus 1/64 h: 1 h 4.18e-4 (4.47e-3) | 3.20e-4 (5.74e-3); 1/4 h 1.24e-4 (1.32e-3);
            1/16 h 2.65e-5 (2.79e-4). Third day (the rain): 1.25e-3, 3.7e-4, 7.8e-5.
          1 h level with tolerance 1e-8 minus the stored reference (1e-5): 1.8e-6 RMS, 4.4e-6 max at the sensor.
          Solve times: 53 s, 189 s, 657 s, 1242 s (201 nodes), 2254 s.
        THE NETWORKS, sensor RMS (max):      1 h                1/4 h              1/16 h             1/64 h
          baseline                           4.26e-4 (4.54e-3)  1.33e-4 (1.39e-3)  3.86e-5 (3.50e-4)  1.90e-5 (7.8e-5)
          s1                                 4.26e-4            1.34e-4            3.91e-5            1.94e-5 (7.8e-5)
          s2                                 4.27e-4            1.35e-4            4.13e-5            2.32e-5 (1.14e-4)
          c2k                                4.25e-4            1.32e-4            3.73e-5            1.70e-5 (7.4e-5)
          w64                                4.32e-4            1.45e-4            6.62e-5            5.62e-5 (2.52e-4)
          d3                                 4.29e-4            1.42e-4            6.31e-5            5.39e-5 (2.73e-4)
          w32                                4.30e-4            1.46e-4            7.20e-5            6.39e-5 (3.34e-4)
          Profile (stored grid), baseline: 3.27e-4 (5.82e-3) -> 1.03e-4 -> 3.02e-5 -> 1.55e-5 (2.06e-4).
          Networks minus the baseline network (as in 74 f): s1 1.90e-5, s2 2.08e-5, c2k 1.72e-5; w64 5.47e-5,
          d3 5.35e-5, w32 6.30e-5.
          Baseline by day against 1/64 h (1e-6): 16 15 31 14 10 10 14 12 25 24 21. s2: largest on day 9 (46);
          w32: largest on day 10 (123).
        AGAINST THE EXPECTATIONS OF 74 f:
          F1 (4 +- 0.5 x per level): NOT met at the first refinement (3.02), met at the second (3.68). The ratio
             rises toward 4; the 1 h step is not yet in the first-order range. What makes it so (the hourly
             forcing nodes, the rain pulses) has not been looked at.
          F2 (grid, < 1e-6 at the sensor): met, 8.0e-7.
          F3: waits for 15613447. From the last ratio, 1/256 h should be about 7e-6 from 1/64 h, and 1/64 h about
             9e-6 to 1e-5 from the limit (an extrapolation, not a measurement).
          F4, provisional (finest level here is 1/64 h): s2 inside 2e-5 - 5e-5; baseline, s1 and c2k just BELOW
             it (1.7 - 1.9e-5); w64 / d3 / w32 inside 5e-5 - 8e-5. "Day 2 still the largest day" holds for the
             baseline only: s2 and w32 have their largest day late in the window.
        READING, limited to what was run:
          - Against the 1/64 h solve the baseline is 1.9e-5 at the sensor, 22 x closer than the data set's own 1 h
            solve is (4.2e-4). Seeds and the 2k-point run are 1.7 - 2.3e-5; they are also 1.7 - 2.1e-5 from each
            other, so most of what is left is the networks' own, seed-dependent part, not a shared one.
          - Width 64, depth 3 and width 32 are 2.8 - 3.4 x the baseline against this level. The factors written
            in 74 e for w32 (2 - 5 x) and d3 (1.5 - 4 x) are inside now. They failed there because the yardstick
            was the 1 h solve.
          - The 1/64 h level is still an estimated 1e-5 from the limit, half the baseline's distance to it. The
            reading rule of 74 e (two finest levels less than a third of the error apart) is applied when the
            1/256 h level is in; until then the baseline's 1.9e-5 is given with that 1e-5 next to it.
        NEXT: paper_figures.py reads the reference check (--forward-reference <stem>): the forward table and fig2
          against the finest level, the 1 h figures kept and named as the data set's reference; a table of the
          reference against itself; the sensitivity table with "against the finest level" and "minus the
          baseline network". The script is finished before the job with the 1/256 h file is submitted.
    74 h. (14 Nov, 04:10) PLAN: DOES THE TIME STEP OF THE NUMERICAL INVERSION MOVE ITS ESTIMATE? (scoring only; no
        PINN run reads it). Why: the results set says "PINN minus numerical inversion 0.07 cm RMS, 0.30 cm max"
        (rfnc1), and the numerical inversion runs the 1 h solve whose soil moisture at the sensor is 4.2e-4 from
        the converged one (74 g), twenty times the forward network's distance. Which of the two that 0.07 cm
        belongs to has not been measured.
        WHAT: scripts/fem_referee.py gets --dt_hours (default: the hourly grid of the data set, so every earlier
          command is unchanged): the solver runs on the finer grid, forcing linear between its hourly nodes as
          before, the observations are read at their own times. Same data set, soil, starts, box, tolerance
          (1e-8 for the differences, as before). Three jobs on output/real_nc2_sep23c_own:
            1/4 h, start hydrostatic_flat   -> output/referee/nc2_own_sep23c_dt025_s1
            1/4 h, start constant:0.3       -> output/referee/nc2_own_sep23c_dt025_s3
            1/16 h, start hydrostatic_flat  -> output/referee/nc2_own_sep23c_dt00625_s1
        EXPECTATION, written before submission:
          H1 the two 1/4 h starts end on one estimate (< 0.01 cm apart), as the four 1 h starts do.
          H2 numerical inversion, 1/4 h minus 1 h: 0.03 - 0.15 cm RMS, largest on the rain days (days 2 - 4).
             Basis: 4.2e-4 in theta against a sensor sd of 0.005 that gives a posterior sd of 1.1 cm is about
             0.1 cm; the step error sits on the rain days.
          H3 1/16 h minus 1/4 h: less than half of H2's number (the estimate converges with the step).
          H4 against the well nothing moves at the level of the table: 1.11 +- 0.03 cm RMS at both steps.
          H5 the PINN estimate (runs/rfnc1dc_it1) is closer to the 1/4 h and the 1/16 h inversion than to the
             1 h one (0.07 cm RMS, 0.30 max). This is the uncertain one: it holds only if the step error of the
             solver is the larger part of the 0.07 cm.
        READING RULE: "the PINN is closer to the converged numerical inversion" is written only if the RMS and
          the max both fall at 1/4 h and do not rise again at 1/16 h. If they do not fall, the 0.07 cm is the
          PINN's own and the text says that. Either way the 1 h figures stay in the table, named as such, and
          no PINN run is changed.
        04:13. Jobs 15615136 / 15615137 (1/4 h) and 15615138 (1/16 h) running. fig2_forward.png of the regenerated
          set viewed: lines only, three differences in the lower panels, legends readable. Pushed as 551f8c4
          (paper_figures.py and the regenerated set; fem_referee.py --dt_hours and this plan, two commits).
    74 i. (14 Nov, 04:15) rfnc1t ENDED (additive form, C_N1 with the soil of 1 Sep 2023 S_W2; DIAGNOSTIC, 73 bu /
        73 bw f). LOSS RATIOS, read from metrics.csv (first and last held-out row) BEFORE the score job's output:
          stage                 held-out flux, last / first        held-out pde, last / first
          wide w0..w3           0.999 0.999 0.999 0.997            0.79 0.75 0.83 0.76   (flux 576 - 612 -> 576 - 611)
          round 1 m0..m3        0.998 0.999 1.000 0.962            0.96 0.95 0.98 1.09
          round 2               0.997 1.000 0.999 0.993            0.94 0.98 1.05 0.89
          round 3               0.988 0.998 0.999 0.984            1.09 1.03 0.98 1.03
          round 4               0.982 0.999 0.999 0.990            1.15 1.01 0.97 0.69
          loop, forward f1      0.68 0.48  (593 -> 405, 562 -> 268)   2.75 3.07 (15.5 -> 42.8, 15.9 -> 48.8)
          loop, forward f2      0.87 0.47                          0.023 0.011 (from 2.6e3, 7.6e3)
          loop, forward f3      0.71 0.60                          0.053 0.13
          loop, forward f4      0.73 0.62                          0.042 0.23
          (for comparison, the own-soil run rfnc1: wide w0 flux 612 -> 0.307, x 5e-4; forward f1_m0 593 -> 0.036)
        READING OF THE RATIOS: no stage of this run carries the surface flux. The operator's held-out flux loss
          is where it started (576 - 650 in sigma_q^2 units) in the wide stage and in all four rounds; the
          forward networks of the loop keep 47 - 87 % of it. So its inversion is that of networks which do not
          satisfy the flux condition, as 73 bw f) said of the wide stage, now for every stage.
        VALUE-CORRECTION LOOP (logs/dc_rfnc1tdc.log, its own step measure, no score): four iterations, largest
          step 4.42, 1.33, 1.28, 1.88 posterior sd; the stop rule (< 0.2) was not reached. Forward offset rms
          0.0058, 0.0054, 0.0054, 0.0051. The loop did not converge; the final directory runs/rfnc1tdc_it4 is its
          last iterate, not a converged estimate.
        EXPECTATION for the score (73 bw f, unchanged): PINN - solver 1.5 - 3 cm, R2 failed.
        Score job 15615202 (score_real.py, referees output/referee/nc2_sep23aon_sep23c_s0..3) -> result_rfnc1t.
        04:16. SCORE of the last iterate runs/rfnc1tdc_it4 (job 15615202; output/real_nc2_sep23c_soilsep23a/
          result_rfnc1t.json), read after the ratios above:
                                        PINN - solver (cm)            against the hourly well (cm)
                                        rms    bias    max    r       rms    bias    max     r
          rfnc1t, last iterate          3.89  +2.88   9.48  0.961     4.53  +1.24  11.19   0.862
          rfnc1t_inv0 (wide stage only) 1.30  +0.23   3.36  0.986     2.51  -1.41   4.82   0.966   (73 ca)
          solver (same data, soil)                                    3.05  -1.64   5.14   0.948
          hydrostatic reading                                         3.38  -2.37  13.20   0.953
          starts: one estimate (largest difference 1.3e-6 cm). theta against the sensor 0.0022.
        AGAINST THE EXPECTATION (1.5 - 3 cm, R2 failed): R2 failed, as written; the number is outside the range
          on the high side (3.89). What I had not expected: the stages after the wide one moved the estimate
          AWAY from the solver's (1.30 -> 3.89 cm). I had written the range from the wide operator's forward
          difference alone and had not thought about the loop, whose forward networks here keep half of the
          flux loss; a value correction taken from networks that do not satisfy the flux condition is not a
          correction. That is a reading of the ratios and the loop's step sizes, not a tested cause.
        WHAT IT SAYS, limited to this run: with the additive form on this soil the method does not return the
          numerical inversion's estimate on the rain window, and the loss ratios said so before any score. The
          run stays a diagnostic. Whether the conductive form does (rfnc1tc, value-correction loop running) is
          read with the rule of 73 bz when it ends.
        READ FROM THE FIGURE result_rfnc1t.png: the estimate leaves the solver's on days 1 - 3.5 (it deepens to
          0.72 m before the showers of day 2, the solver's stays at 0.63) and in the last day and a half (0.84
          against 0.78 m); between days 4 and 9 the two lie within a centimetre or two.
    74 j. (14 Nov, 04:17) SITE TRACK, the part that needs no answer: the ICOS question (74 d) is still open, nothing
        has been downloaded from ICOS. Started meanwhile: a search (research agent, read-only, no account, no
        licence step, no personal details, samples only under output/site_search/) for a precipitation record
        (hourly or finer) and an evaporation source for the two open sites of tasks/site_candidates.md that
        have a co-located sensor and well but no rain: Point Reyes (A1) and Shale Hills RTH3 (A2).
        WHAT IT DECIDES: whether either open site can be forced at all. A site goes on to a data set only if
        the rain record is within a few km, at hourly step or finer, and covers a wet-season window with the
        water table inside 0.2 - 1.5 m; a daily or gridded rain is recorded as "not good enough for a rain
        window" (recession windows only, and then the evaporation source carries the whole flux).
        Nothing here changes the answer needed for FR-Bil.
    74 k. (14 Nov, 04:42) OPEN SITES: RAIN SOURCES FOUND (agent report, its [V] marks are requests it made; I
        checked that the sample files exist and read their heads). Written into tasks/site_candidates.md.
          Shale Hills RTH3: hourly precipitation measured in the watershed (CZO Level-1, 2009 - 7 Nov 2012, mm,
            disdrometer 2009-10, weighing gauge 2011-12; 1224 h of the 2010 windows flagged as filled by
            correlation), served only as Wayback copies of czo.psu.edu. Hourly meteorology (T, RH, wind, short-
            and long-wave, pressure) in the MM-PIHM example input (MIT), 2008 - Jul 2012. No measured latent heat
            for 2009 - 2012 anywhere open.
          Point Reyes: nearest hourly gauge is a ridge-top fire-weather station 6.8 km away (285 m); a coastal
            gauge 7.7 km away caught 781 mm in 2022/23 against its 1482 mm. By the rule of 74 j: NOT good enough
            for a rain window. Recession windows only, evaporation from a daily gridded reference ET. Set aside.
        WHAT THIS CHANGES against 74 d: Shale Hills no longer needs borrowed rain. It still has no measured
          evaporation: the evaporation part of q_surf would be a formula (reference ET from the hourly
          meteorology), which no site so far has had. Before leaf-out (deciduous forest, valley floor) that part
          is small against the rain; in June it is not.
        PLAN, step V0 only (record check, nothing fitted, no GPU, no soil, no solver):
          1. Raw files, each source in its own new empty directory under data/sites/shalehills/ (git-ignored):
             ground water and soil moisture 10-min files of 2010 with their ReadMe (HydroShare, CC BY 4.0, no
             account), the hourly precipitation of 2010 (Wayback copy). Read only by a batch job run with
             python -I; the script lives in scripts/ and imports nothing from the data directories.
          2. scripts/site_records_shalehills.py prints, for the three nests (well 1 / sensors 1-3, well 2 /
             sensors 4-6, well 3 / sensors 7-9; sensors at 0.1, 0.3, 0.5 m):
               - flags: share of 10-min values with flag A, by month;
               - hours by month with the well inside 0.2 - 1.5 m (and at its dry cap);
               - sign and reference of the well against the sensors: the 0.3 m and 0.5 m sensors must sit on
                 their highest level while the well says the water table is above them;
               - range and plateau of the 0.1 m sensor by month;
               - rain by month, hours flagged "correlation"; delay from rain to the rise of the well and of the
                 0.1 m sensor (lag of the largest cross-correlation of hourly increments, -12 .. +12 h): this is
                 also the check of the time zones of the three files against each other;
               - frost: not in these files; air temperature comes with the meteorology, read in the next step.
             and draws rain, the three sensors and the well of each nest, Mar - Jun 2010 (lines only).
        EXPECTATION, written before the files are read beyond their first lines:
          K1 well 2 (1.96 m deep) is inside 0.2 - 1.5 m for most of Mar - Jun 2010 (the agent read 0.37 - 1.45 m);
             well 1 (1.10 m) is dry from June; well 3 mostly flagged.
          K2 the sensors are not consistent with the well as a hydrostatic column in a simple way: sensor 4
             (0.1 m) read 0.401 and sensor 5 (0.3 m) 0.331 on 1 Jan 2010 with the well at 0.655 m, the deeper one
             drier. In a uniform soil above a water table that order is impossible; rock fragments or layering
             (a 1-D uniform column is then the wrong model for more than one depth at a time).
          K3 the well rises within 0 - 3 h of rain (valley floor by the stream), i.e. faster than a column with
             the measured rain alone would lift it: lateral inflow. That does not break the inverse problem
             (the water table is the unknown) but it means storms carry most of the signal.
        WHAT IT DECIDES: Shale Hills goes on to a data set (one sensor, one well, windows by the written rule)
          only if a nest has (a) the well inside the box with flag A over at least two windows of 10 days or
          more in different seasons or years (one to calibrate, the next to test), (b) a 0.1 m sensor that moves
          with the well without sitting on a plateau, (c) rain and well aligned in time to the hour. If K2
          holds, only the 0.1 m sensor is used and the text says the deeper sensors contradict a uniform column.
          How the evaporation is computed is planned after this check, not before.
    74 l. (14 Nov, 04:51) SHALE HILLS RECORD CHECK (V0 of 74 k) DONE: THE SITE DOES NOT GO ON TO A DATA SET.
        Run: job 15615875, `python -I scripts/site_records_shalehills.py --raw data/sites/shalehills --years 2009
          2010 2011 2012 --zoom 2010-03-01 2010-07-01 --out output/site_shalehills/result_records` (16 s, exit 0;
          all four years were taken, not 2010 alone as step 1 said). Nothing fitted, no solver, no GPU.
        CORRECTION to 74 k: the precipitation hours filled by correlation are 1427, all in Oct - Dec 2010 (the
          agent's "1224 h of the 2010 windows" was not that); Mar - Jun 2010 has none. The type "No data" in that
          file is the missing precipitation type of the disdrometer, not a missing amount.
        WHAT THE FILES ARE:
          - Well value = depth to water: it goes to the well depth in Aug - Sep; it falls by 0.02 - 0.20 m in the
            6 h after an hour with >= 2 mm (all nests, all years); r with the 0.1 m sensor is negative.
          - Soil moisture exists as flagged volumetric water content in 2009 and 2010 only. The 2011 and 2012
            files have other column names, no flags and values of 0.84 - 0.95 in six of nine columns: not used.
          - Rain against the hourly rise of the well, largest correlation: lag 0 - 1 h in 2009 and 2010 (0.38 -
            0.92), lag -1 - 0 h in 2011, 0 - 3 h in 2012. Aligned within one hour; the 2011 rain (other gauge)
            is stamped about an hour later than the 2009 - 10 rain. (Sep - Nov 2009 at nest 1 gave lag -10 h:
            the values at -2 .. 6 h are all 0.12 - 0.19, no clear peak.)
        AGAINST THE EXPECTATIONS:
          K1 half right. Well 2 is inside 0.2 - 1.5 m for all of Mar - Jun 2009 and 2010 (0.42 - 1.45 m), yes.
             Well 1 at its depth from late June, yes. "Well 3 mostly flagged": wrong, well 3 has flag A
             throughout 2009 and reads 0.69 - 1.9 m. What I did not expect: the 0.1 m sensor of nest 2 (sensor
             4) has flag A for only 2533 h of 2010 (to 12 Mar), and the 0.5 m sensor of nest 2 for none of 2009.
          K2 holds, and more strongly than written. Nest 2: 0.1 m median 0.393, 0.3 m median 0.337 (wetter
             above). Nests 1 and 3: every sensor has two levels and switches between them in hours:
               nest 1 (figure, Mar - Jun 2010): 0.5 m and 0.3 m read 0.24 - 0.26, and 0.35 - 0.39 while the well
                 reads less than about 0.6 - 0.7 m; the 0.1 m sensor reads 0.16 - 0.20 for any well reading from
                 0.5 to 1.1 m and jumps to 0.42 (13 Mar) only when the well reaches 0.42 m.
               nest 3 (2009, Mar - Jun 2010): the 0.3 m sensor sits at 0.31 - 0.35 for days while the well reads
                 less than about 1.15 m, the 0.5 m sensor below it does not (short peaks to 0.36 - 0.38 only
                 when the well reaches 0.7 - 0.8 m). The well never reads above 0.69 m. A single water table
                 under a uniform column cannot wet the 0.3 m sensor to its top level from 0.85 m below while
                 the 0.5 m sensor stays down: layering with a perched zone, or the well and the sensors are not
                 one column. Not decided here, and it does not need to be.
          K3 holds: the well rises in the hour of the rain or the next.
        AGAINST THE RULE OF 74 k:
          (a) yes for all three wells (nest 2 with a usable 0.1 m sensor: 1 Jan 2009 + 204 d, 26 Jul 2009 + 33 d,
              16 Dec 2009 + 15 d, 25 Jan 2010 + 46 d).
          (b) NO at every nest. Nest 1: the 0.1 m sensor sits on its lower level (0.16 - 0.20) over the whole
              range 0.5 - 1.1 m of the well, so it carries no information on the water table there; it leaves
              that level only in the few hours when the water table is within about 0.3 m of it. Nest 2: the
              0.1 m sensor drifts from 0.40 to 0.455 between January and May 2009 while the well stays at 0.6 -
              0.9 m, steps down and up by 0.03 - 0.05 within hours, becomes erratic from July 2009 and is
              flagged from March 2010 (figure). Nest 3: as nest 1 for the 0.1 m sensor (0.15 - 0.22, one peak to
              0.43 per large storm), and the two-level sensors contradict the well as above.
          (c) yes, within one hour.
        DECISION (the rule was written before the check): Shale Hills is NOT taken on. The reason is the
          sensors, not the rain and not the missing evaporation: a near-surface sensor that reads one value for
          every water-table depth between 0.5 and 1.1 m cannot determine that depth, with any method. No
          evaporation formula is planned, no data set is built, no GPU time is spent on this site.
        WHERE THE SITE TRACK STANDS: of the four open sites of tasks/site_candidates.md, Point Reyes has no rain
          record good enough (74 k), Shale Hills fails on its sensors (here), Bondville has a daily well only,
          Kopcany has its shallowest sensor at 0.3 m and no measured evaporation. No new site can be built from
          open data that I have found. The one candidate with all four variables at one station on sand with a
          shallow water table is FR-Bil, and it waits on the user's answer on the ICOS licence (74 d); nothing
          has been requested from data.icos-cp.eu.
        FILES: scripts/site_records_shalehills.py; output/site_shalehills/result_records.json and the figures
          result_records_2009_n1..3.png, result_records_2010-03-01_n1..3.png (the 2010 - 2012 whole-year
          figures are left out of git: nothing in them is used).
    74 m. (14 Nov, 06:00) TIME STEP OF THE NUMERICAL INVERSION, 1/4 h AGAINST 1 h (74 h; scoring only). Jobs 15615136
        (start hydrostatic_flat, 7 iterations, 55 min) and 15615137 (start constant:0.3, 12 iterations, 102 min)
        COMPLETED, summary.json written, no coefficient on a bound, no solve unconverged. Compared by
        scripts/referee_step_check.py (job 15617222, exit 0) -> output/paper/nc2_sep23c/inversion_step.json.
                                         step    cost       vs well (cm): rmse   bias    max     r
          1 h, start hydrostatic_flat    1       0.091780                 1.112  +0.114  3.129  0.9904
          1 h, start constant:0.3        1       0.091780                 1.112  +0.114  3.129  0.9904
          1/4 h, start hydrostatic_flat  0.25    0.076227                 1.113  +0.116  3.178  0.9903
          1/4 h, start constant:0.3      0.25    0.076227                 1.113  +0.116  3.178  0.9903
          between the two 1/4 h starts:  0.0001 cm RMS, 0.0006 max      (1 h starts: 0.0004, 0.0014)
          1 h minus 1/4 h inversion:     0.044 cm RMS, 0.22 max at t = 0; by day
                                         0.092 0.037 0.046 0.059 0.011 0.005 0.006 0.007 0.004 0.003 0.072
          PINN (runs/rfnc1dc_it1) minus  1 h inversion:    0.065 cm RMS, 0.30 max (last hour); by day
                                         0.131 0.077 0.083 0.069 0.019 0.004 0.010 0.010 0.015 0.007 0.105
                                         1/4 h inversion:  0.027 cm RMS, 0.093 max (last hour); by day
                                         0.045 0.040 0.052 0.011 0.010 0.002 0.005 0.007 0.011 0.005 0.032
        AGAINST THE EXPECTATIONS OF 74 h:
          H1 holds: the two 1/4 h starts are 0.0001 cm apart.
          H2 holds for the size (0.044 cm; expected 0.03 - 0.15) and is wrong about the place: the largest
             differences are on the first day (0.092, maximum at t = 0) and the last day (0.072), then days
             1 - 3 (0.037 - 0.059); days 4 - 9 are at 0.003 - 0.011. I had written "largest on the rain days".
             Why the two ends carry it has not been looked at and is not stated.
          H3 waits for the 1/16 h inversion (job 15615138, running, 16 h limit).
          H4 holds: 1.112 -> 1.113 cm against the well, bias and r unchanged at the level of the table.
          H5 holds at 1/4 h: PINN minus numerical inversion falls from 0.065 to 0.027 cm RMS and from 0.30 to
             0.093 cm max. The PINN estimate is closer to the 1/4 h inversion (0.027) than the 1 h inversion is
             (0.044).
        READING RULE of 74 h: first half met (RMS and max both fall at 1/4 h). The sentence "the PINN is
          closer to the converged numerical inversion" is NOT written yet: it needs the 1/16 h result not to
          rise again. Until then the write-up gives both rows, named by their step.
        ALSO SEEN, stated as what it is: the misfit of the numerical inversion falls by 17 % with the step
          (0.0918 -> 0.0762), i.e. part of the 1 h inversion's misfit is its own time-step error; and one start
          costs 55 - 102 min at 1/4 h against 12 - 34 min at 1 h. The cost comparison in the write-up
          ("about 30 times") is against the 1 h inversion and has to say so.
    74 n. (14 Nov, 06:06) N1 SEED REPEATS ENDED (rfnc1s1 SEED0=1000, rfnc1s2 SEED0=2000; whole pipeline on
        output/real_nc2_sep23c_own, in-sample soil). Both drivers print "full ... done" (05:41, 06:01).
        LOSS RATIOS, read from metrics.csv (first and last held-out row) BEFORE any score of these runs:
          stage             held-out flux, last / first (values)             held-out pde, last / first (values)
          rfnc1s1 wide      5.3e-4 5.2e-4 6.4e-4 5.4e-4 (565-624 -> 0.30-0.39) 0.053 0.048 0.053 0.055 (8.1-8.5 -> 0.41-0.45)
          rfnc1s1 round 1   0.53 0.55 0.58 0.52 (-> 0.16-0.18)                0.18 0.31 0.35 0.27 (-> 0.20-0.27)
          rfnc1s1 round 2   0.77 0.78 0.83 0.75 (-> 0.12-0.16)                0.77 0.75 0.77 0.81 (-> 0.16-0.21)
          rfnc1s1 round 3   0.94 0.72 0.86 0.86 (-> 0.10-0.12)                0.79 0.80 0.99 0.79 (-> 0.13-0.20)
          rfnc1s1 round 4   0.84 0.87 0.85 0.94 (-> 0.096-0.116)              0.97 0.95 0.72 0.93 (-> 0.12-0.16)
          rfnc1s1 forward   3.6e-5 7.7e-5 (614, 636 -> 0.022, 0.049)          4.7e-3 6.1e-3 (5.0, 5.1 -> 0.023, 0.031)
          rfnc1s2 wide      6.0e-4 4.6e-4 4.6e-4 4.9e-4 (580-603 -> 0.27-0.36) 0.051 0.048 0.044 0.050 (8.2-8.6 -> 0.37-0.44)
          rfnc1s2 round 1   0.49 0.52 0.49 0.52 (-> 0.13-0.15)                0.35 0.31 0.32 0.30 (-> 0.17-0.26)
          rfnc1s2 round 2   0.74 0.74 0.80 0.83 (-> 0.10-0.13)                0.71 0.75 0.84 0.79 (-> 0.15-0.19)
          rfnc1s2 round 3   0.79 0.80 0.89 0.83 (-> 0.095-0.106)              0.84 0.76 0.89 0.89 (-> 0.12-0.15)
          rfnc1s2 round 4   0.85 0.96 0.88 0.95 (-> 0.083-0.106)              0.99 0.92 1.02 0.99 (-> 0.12-0.22)
          rfnc1s2 forward   6.6e-5 6.8e-5 (560, 587 -> 0.037, 0.040)          5.1e-3 4.9e-3 (5.2, 5.1 -> 0.026, 0.025)
          (rfnc1, seed 0, for comparison: wide w0 flux 612 -> 0.307, x 5e-4; forward f1_m0 593 -> 0.036.)
        READING OF THE RATIOS: every stage of both repeats carries the surface flux and the PDE as the seed-0
          run does: the wide stage takes the flux loss down by 5e-4 and each round lowers both losses further
          (round 4 by 4 - 16 % in flux); one member (rfnc1s2 r4 m2) ends its last round with the PDE loss 2 %
          above its start, the flux loss 12 % below. The forward networks of the loop reach 0.02 - 0.05 in
          flux and 0.02 - 0.03 in PDE. Nothing in the ratios says a stage did not train.
        VALUE-CORRECTION LOOP (logs/dc_rfnc1s<k>dc.log, step measure only): one iteration each, largest step
          0.010 and 0.012 posterior sd (stop rule < 0.2 met at once; rfnc1: one iteration too), forward offset
          rms 2.6e-5 in both. Final directories runs/rfnc1s1dc_it1, runs/rfnc1s2dc_it1.
        EXPECTATION for the scores (73/74 plan N1, unchanged): final estimates within 0.3 cm rms (max 1 cm) of
          rfnc1's and of each other; each 1.0 - 1.25 cm from the well, <= 0.3 cm from the numerical inversion,
          start spread < 0.2 cm, loop converged within 2 iterations (met: 1). READING RULE: "independent of the
          training seed" only if all three runs satisfy R1 and R2 and their pairwise rms is <= 0.5 cm.
        06:07. SCORES of the two repeats (jobs 15617300 / 15617301, exit 0; output/real_nc2_sep23c_own/
          result_rfnc1s1.json, result_rfnc1s2.json), read after the ratios above:
                              PINN - numerical inversion (cm)    against the hourly well (cm)       starts
                              rms     bias     max               rmse    bias    r                  spread (cm)
          rfnc1   (seed 0)    0.065   -0.004   0.303             1.105   +0.111  0.9905             4.9e-7
          rfnc1s1 (seed 1000) 0.062   -0.003   0.288             1.104   +0.111  0.9905             9.3e-7
          rfnc1s2 (seed 2000) 0.063   -0.004   0.279             1.105   +0.110  0.9905             2.0e-6
          (numerical inversion against the well: 1.112; posterior sd, median: 1.111 cm in all three)
          AGAINST THE EXPECTATION: each 1.0 - 1.25 cm from the well: yes (1.10). <= 0.3 cm from the numerical
          inversion: yes (0.06). Start spread < 0.2 cm: yes (1e-6). Loop within 2 iterations: yes (1).
          R1 and R2 hold in all three runs. The distance to the 1 h numerical inversion is the same in all
          three (0.062 - 0.065 cm): it does not come from the training seed (74 m: against the 1/4 h inversion
          the seed-0 run is at 0.027).
        06:10. RUN-TO-RUN DIFFERENCE (paper_figures.py --repeats, job 15617310, exit 0; metrics.json "repeats",
          tables.md section 8, fig7_seeds.png viewed: lines only, three estimates on one curve in the upper
          panel, differences in the lower one):
            seed 1000 minus first run: 0.0047 cm RMS, 0.021 cm max;  seed 2000 minus first run: 0.0059, 0.024.
            seed 1000 minus seed 2000 was not computed; it is at most 0.0047 + 0.0059 = 0.011 cm RMS (triangle
            inequality), 0.045 cm max.
          READING RULE of N1 met: R1 and R2 hold in all three runs and every pair is within 0.011 cm (rule:
          <= 0.5). The sentence "the estimate is independent of the training seed (three runs of the whole
          pipeline, 0.006 cm RMS apart)" is written, for THIS window and its in-sample soil only. It says
          nothing about held-out windows.
    74 o. (14 Nov, 06:10) FORWARD REFERENCE, SECOND JOB WITH THE 1/256 h LEVEL (15613447, 3 h 19, exit 0;
        output/paper/nc2_sep23c/forward_reference.json; plan 74 f, first job 74 g). Picard tolerance 1e-8, every
        step of every level converged (largest iteration count 24 at 1 h, 17 at 1/256 h); 1/256 h solve 7527 s.
        THE REFERENCE AGAINST ITSELF, sensor RMS (max) | profile RMS (max), theta in m3/m3:
            1/4 h   - 1 h                    2.95e-4 (3.14e-3) | 2.26e-4 (4.05e-3)
            1/16 h  - 1/4 h                  9.75e-5 (1.04e-3) | 7.49e-5 (1.33e-3)      3.02 x less
            1/64 h  - 1/16 h                 2.65e-5 (2.80e-4) | 2.04e-5 (3.58e-4)      3.68 x less
            1/256 h - 1/64 h                 6.77e-6 (7.09e-5) | 5.21e-6 (9.00e-5)      3.92 x less
            1/16 h: 201 - 101 nodes          8.0e-7  (1.4e-6)  | 1.2e-7  (3.6e-6)
          each level minus 1/256 h, sensor | profile: 1 h 4.24e-4 | 3.25e-4 (max 5.83e-3); 1/4 h 1.31e-4 |
            1.00e-4; 1/16 h 3.32e-5 | 2.56e-5; 1/64 h 6.77e-6 | 5.21e-6.
        THE NETWORKS AGAINST 1/256 h, sensor RMS (max) | profile RMS (max)        (against 1/64 h, sensor)
          baseline   1.72e-5 (7.8e-5)  | 1.42e-5 (2.01e-4)                        1.90e-5
          s1         1.75e-5 (7.8e-5)  | 1.40e-5 (1.87e-4)                        1.94e-5
          s2         2.16e-5 (1.13e-4) | 1.78e-5 (3.20e-4)                        2.32e-5
          c2k        1.52e-5 (4.7e-5)  | 1.32e-5 (2.35e-4)                        1.70e-5
          w64        5.53e-5 (2.52e-4) | 4.26e-5 (6.19e-4)                        5.62e-5
          d3         5.35e-5 (2.71e-4) | 4.28e-5 (7.82e-4)                        5.39e-5
          w32        6.35e-5 (3.33e-4) | 4.89e-5 (6.92e-4)                        6.39e-5
          Baseline by day against 1/256 h (1e-6): 17 14 22 10 10 10 14 12 24 24 20.
        AGAINST THE EXPECTATIONS OF 74 f:
          F1 (4 +- 0.5 x per level): met at the second and third refinement (3.68, 3.92), not at the first
             (3.02), as recorded in 74 g. First order in the step from 1/4 h on.
          F3 (two finest levels < 1e-5 apart): met, 6.8e-6 at the sensor (74 g's estimate from the ratio: 7e-6).
          F4 against the finest level: s2 inside 2e-5 - 5e-5; baseline, s1 and c2k below it (1.5 - 1.8e-5);
             w64 / d3 / w32 inside 5e-5 - 8e-5. "Day 2 still the largest day": NOT met for the baseline, days
             8 and 9 (24) are above day 2 (22) once the reference's own step error is out of the comparison.
        READING RULE OF 74 e ("the forward error of the results set becomes the one against the finest level
          only if the two finest levels differ from each other by less than a third of it; otherwise it is
          given as an upper bound with the reference's own difference next to it"): the two finest levels
          differ by 6.77e-6 at the sensor, the baseline is 1.72e-5 from the finest: 0.39, not below a third
          (profile: 5.21e-6 against 1.42e-5, 0.37). NOT MET, by a small margin. So the write-up does not say
          "the forward error is 1.7e-5". It says: 1.7e-5 at the sensor against the 1/256 h solve, which is
          itself 6.8e-6 from the 1/64 h solve; and it keeps the 1 h figures (4.3e-4), named as the data set's
          reference. A first-order extrapolation puts the 1/256 h solve about 2e-6 from the limit; that is an
          extrapolation, not a measurement, and is given as such or left out.
        WHAT STANDS, limited to what was run: the three width-128 networks are 1.5 - 2.2e-5 from the finest
          solve at the sensor, 25 x closer than the 1 h solve of the data set is (4.24e-4); width 64, depth 3
          and width 32 are 3.1 - 3.7 x the baseline.
        NEXT, decided at 06:11: one more level, 1/1024 h, because the rule is missed by a small margin and the
          level costs one CPU core for about 7 h (solve times so far 53, 191, 661, 2261, 7527 s: x 3.4 per
          level). Same script, same members, same tolerance, all earlier levels again so that one file holds
          the whole table -> output/paper/nc2_sep23c/forward_reference_d.{json,npz} (job D's file stays).
          EXPECTATION, written before submission:
            G1 1/1024 h - 1/256 h: 1.5e-6 - 2.0e-6 RMS at the sensor (3.5 - 4.3 x less than the step before).
            G2 baseline against 1/1024 h: 1.6e-5 - 1.75e-5 at the sensor (it went 1.90 -> 1.72e-5 when the
               reference moved by 6.8e-6; this step moves the reference by a quarter of that).
            G3 the 74 e rule is then met (about 0.1 against a third).
          DECISION RULE: if G1 holds, the forward error is stated against the 1/1024 h level, with the level
            difference next to it. If the level difference does NOT fall by at least 3 x (the Picard tolerance
            of 1e-8 per step over 270 000 steps may set a floor), the stall is reported and the statement
            stays the bounded one of this entry. No network run depends on it.
    74 p. (14 Nov, 06:15) RESULTS SET REBUILT WITH THE 1/256 h REFERENCE AND THE SEED REPEATS; README UPDATED.
        Job 15617329 (pfig2), 53 s, exit=0. output/paper/nc2_sep23c/tables.md read whole: section 2 names
        1/256 h as the finest level and gives the three columns (PINN - 1/256 h | PINN - 1 h | 1 h - 1/256 h):
        all depths RMS 1.42e-5 | 3.27e-4 | 3.25e-4, max 2.01e-4 | 5.82e-3 | 5.83e-3; sensor RMS 1.72e-5 |
        4.26e-4 | 4.24e-4, max 7.77e-5 | 4.54e-3 | 4.54e-3. Same numbers as 74 o read from the json.
        Section 7 (seven forward networks) and section 8 (seed repeats) as in 74 n / 74 o.
        fig2_forward.png looked at: lines only, legends, axis labels with units, no title. In its lower panels
        "PINN - 1 h" lies on "1 h - 1/256 h" in time, and over depth except close to the water table (both
        below 1e-5 there): what was called the network's forward error against the data set's solution is
        that solution's step error. The 5.8e-3 "largest error at the surface during the rain pulses" of the
        earlier README was the solver's; by the triangle inequality the network's share of it is <= 2.0e-4.
        output/paper/README.md: forward table replaced by the three columns with the bounded statement of
        74 o (1.7e-5 against a reference itself 6.8e-6 from the level before); sensitivity line; fig7 row;
        seed sentence limited to this window and its in-sample soil; point 5 under "What to say plainly";
        "Not yet" list (1/1024 h level, 1/16 h inversion, rfnm1 / rfna1, rfnc1tc); both commands.
    74 q. (14 Nov, 06:15) US-HB2 V4 (aug20) CALIBRATION AFTER THE THIRD AND LAST CONTINUATION: NOT CONVERGED.
        T_H4 IS NOT DECIDED UNDER THE RULE OF 73 by. NO FURTHER CONTINUATION.
        Jobs 15597939 / 15597940 (s0e / s1e): TIMEOUT at 12:00:22; 15597941 (s2e): COMPLETED after 9:11:36.
          start 0  not ended (40 h)  misfit 0.000940  theta_s 0.30002  theta_r 0.0573  alpha 19.87  n 1.777  Ks 3.24e-5
          start 1  not ended (40 h)         0.001395          0.3428           0.0835        19.95    2.156     2.78e-5
          start 2  ended by xtol            0.000933          0.30000 (lower bound)  0.0568  20.00 (upper bound)  1.770  3.34e-5
          start 3  ended by xtol, 12 Nov    0.003651          0.3013           0.0016        18.74    1.653     3.40e-6
        (output/soil_hb2_aug20/progress0..2.json, start2.json, start3.json.) Two of four starts did not end by
        xtol in 40 h, so by the rule written before this continuation V4 is "calibration not converged" and
        T_H4 is not run as a counting test. What the four starts do show: the lowest misfit (start 2) sits
        on two bounds of the box (theta_s at its lower, alpha at its upper), start 0 is next to it (misfit
        0.7 % higher, same n and Ks to 3 %), and starts 1 and 3 are elsewhere at 1.5 x and 3.9 x the
        misfit. Whether a soil on two bounds of the box would be usable is not covered by a written rule
        and is not decided here.
        The dry run of the T_H4 chain (names _pre, does not count) is still in its waiter; its numerical-
        inversion job rf_hb2_aug20on_nov20_pre_s0 (15598032) ran into its 12 h limit. Recorded when it reports.
    74 r. (14 Nov, 06:18) S1: rfna1 (06:11) AND rfnm1 (06:16) ENDED. LOSS RATIOS, read from metrics.csv (first and
        last held-out row) BEFORE any score job is submitted. Both soils have Ks on the upper bound (1e-3 m/s);
        both runs use the default (additive) form, as launched in 74 b.
        Held-out flux loss starts at 98 - 105 in every wide member and forward network (zero-flux start with
        sigma_q = 10 % of the flux rms), so "x r" below is (rms flux error / rms prescribed flux)^2.
          rfna1 (W2, 1 Sep 2023 + 6 d, soil cal3)   flux, end value (x of the wide start)   pde, end value
            wide, 4 members, 42000 steps            1.84 - 6.09  (x0.019 - 0.062)            0.68 - 2.89 (start 0.65 - 0.68)
            round 1                                 2.03 - 4.26                              0.77 - 1.03
            round 2                                 1.66 - 3.61                              0.77 - 1.00
            round 3                                 1.28 - 2.86  (member 1 rose 2.01 -> 2.79) 0.72 - 0.96
            round 4                                 1.45 - 2.51  (x0.015 - 0.026)            0.73 - 0.91
            forward nets of the loop, 28000 steps   0.69, 1.30   (x0.007, 0.013)             0.37, 0.44 (start 1.3)
            loop: one iteration, step 0.018 sd, forward offset rms 4.5e-5
          rfnm1 (W1, 1 May 2023 + 15 d, own soil)
            wide                                    41.0, 50.5, 6.44, 25.1 (x0.41, 0.49, 0.066, 0.25)   7.27, 6.30, 0.89, 5.09 (start 0.81)
            round 1                                 24.7, 26.5, 6.89, 14.4                   4.51, 3.78, 0.79, 2.24
            round 2                                 15.5, 20.7, 6.78, 11.6                   3.65, 2.61, 0.74, 1.67
            round 3                                 11.7, 18.7, 6.18, 9.35                   1.91, 1.87, 0.88, 1.21
            round 4                                 10.1, 16.5, 5.98, 9.21 (x0.10, 0.16, 0.061, 0.090)  1.35, 1.80, 0.82, 0.99
            forward nets of the loop                4.25, 5.40   (x0.041, 0.051)             0.75, 0.63 (start 1.0)
            loop: one iteration, step 0.057 sd (by segment 0.01 0.01 0.02 0.06), forward offset rms 3.8e-5
          for comparison, rfnc1 and its seed repeats: wide x4.6e-4 - 6.4e-4, forward nets x3.6e-5 - 7.7e-5.
        READ BEFORE THE SCORES:
          - Neither run satisfies the surface-flux condition as rfnc1 does. In rms terms the forward networks
            at the estimate miss the prescribed flux by 8 - 11 % (rfna1) and 20 - 23 % (rfnm1) of its rms on
            held-out points, against 0.6 - 0.9 % for rfnc1; the operator members after round 4 by 12 - 16 %
            (rfna1) and 25 - 41 % (rfnm1). In rfnm1 three of four wide members ended with the flux at x0.25 -
            0.49 and the PDE loss 6 - 9 times its starting value.
          - This is the weakness 73 bz found for the additive form on Ks 1e-3 soils (flux not learned), here in
            a milder degree than in rfnc1t (x0.96 - 1.00 there). It is not as bad as that run, and not solved.
          - The loop's stop (one iteration, 0.02 and 0.06 sd) says the operator and the forward networks give
            the same soil moisture at the estimate. It does not say either satisfies the flux condition.
          - So, whatever the scores: if the estimates are on the numerical inversion's, the rows show (i) on a
            near-hydrostatic profile (the retention curve read through the network, 74 b), with the flux
            ratio written in the row; they are not evidence that the network solved the flow problem on these
            windows. The 74 b expectation (starts < 0.2 cm apart; PINN - solver <= 0.3 cm; 0.7 - 1.1 and
            0.5 - 0.9 cm from the well) was written before any loss was seen and is kept as written; on
            these losses rfnm1 is the one more likely to miss it.
        THE NUMERICAL REFERENCE OF THESE TWO WINDOWS, before it is scored against (lesson of 74 g):
          1 h step, four starts each; cost may 0.0884301 (all four, 5 - 9 iterations), sep23a_cal3 0.150580
          (all four). Its solver did not converge in up to 2.8 % of the steps on sep23a_cal3 (summary.json,
          fem_not_converged_max 0.009 - 0.028 by start; 0 on may23). Its dependence on the time step has not
          been measured on these windows (on C_N1: 0.04 cm RMS at 1/4 h, 74 m).
          -> one 1/4 h inversion per window, start hydrostatic_flat, same command as 74 h:
               output/referee/nc2_own_may23_dt025_s1, output/referee/nc2_own_sep23a_cal3_dt025_s1
          EXPECTATION, written before submission: K1 1/4 h minus 1 h <= 0.1 cm RMS on both (near-hydrostatic
            profiles: the soil moisture at the sensor follows the water table within the step); K2 the
            not-converged fraction on sep23a_cal3 falls below 1 %; K3 nothing moves against the well at the
            level of the table (+- 0.05 cm). Uncertain: K1 on sep23a_cal3, because of the unconverged steps.
          READING: the scores of rfnm1 / rfna1 against the 1 h inversion go into the table now, named as
            1 h; the 1/4 h figures are added next to them when they are in. If K1 fails on a window (> 0.1 cm),
            "PINN - numerical" of that window is given against both steps and (i) is read against the finer.
        SCORING (CPU): scripts/score_real.py on both, referee = the four 1 h starts.
    74 s. (14 Nov, 06:18) US-HB2: DRY RUN OF THE T_H4 CHAIN ENDED 06:14 (names _pre; DOES NOT COUNT, 73 by). NUMERICAL
        SOLVER ONLY, NO PINN. Soil = start 2's of 18:04 on 13 Nov (theta_s 0.30001, alpha 19.996, n 1.7691,
        Ks 3.32e-5), which is where start 2 then ended by xtol (74 q). logs/transfer_hb2_aug20_pre.log, read whole.
          V4 (Aug 2020), in-sample: FEM over the record against the sensor 0.00094 (bias 8e-6, r 0.9993).
            Numerical inversion, four starts, cost 0.051887 (all four): from the well 3.47 - 3.68 cm RMSE,
            bias +1.25 - 1.33, max 14.6 - 15.6 cm; coefficient 0 on the box in all four (0 and 5 in one).
          V5 (Nov 2020) with that soil: FEM over the record against the sensor 0.031 (bias +0.028, r 0.998).
            Numerical inversion: starts 1 - 3 cost 0.60100 / 0.60100 / 0.60099, from the well 38.0 / 38.0 /
            35.4 cm RMSE, bias +32.9 / +32.9 / +30.5, max 66 / 66 / 60 cm, coefficients 0, 2, 4, 5 on the box;
            start 0 (15598032) ran into its 12 h limit, no result.
        AGAINST THE EXPECTATION OF 73 by:
          V4 FEM against the sensor "0.001": met (0.00094).
          V4 in-sample inversion "1 - 2.5 cm from the well, usable-soil rule about even": NOT met, 3.5 - 3.7 cm.
            By the 2 cm rule this soil is not usable even on its own window.
          V5 "solver 8 - 20 cm from the well, bias positive, R3 failed": sign and outcome as expected, size
            not: 35 - 38 cm, with four of the coefficients on the box and the starts 2.7 cm apart.
        WHERE US-HB2 STANDS: T_H4 is not decided as a counting test (calibration not converged, 74 q), and the
          dry run says the soil of V4 is not usable on V4 and misses V5 by 35 - 38 cm with the solver itself.
          No PINN run is started on it. US-HB2 has no window that passes (ii); its one PINN row stays the dry
          run rfh1p (0.22 cm from the numerical inversion, does not count).
        SITE TRACK AFTER THIS: US-NC2 in-sample only; US-ONA rf22c below saturation only; US-PAS and US-HB2
          fail with the solver itself. Open-data route exhausted (74 c, 74 l). The next site needs data that
          is not on disk, and that needs the user's word (ICOS licence, asked; not answered).
    74 t. (14 Nov, 06:20) S1 SCORED (jobs 15617459 / 15617460, exit=0; ratios in 74 r were written first).
        BOTH WINDOWS: THE PINN IS ON THE NUMERICAL INVERSION (0.03 AND 0.09 cm) AND 0.9 / 0.6 cm FROM THE WELL.
        BOTH IN-SAMPLE, BOTH RECESSIONS, FLUX CONDITION NOT LEARNED AS IN rfnc1. NO R4.
        The reference first: the data sets' own solver check (real_summary.json, fem_vs_finer_fem, at the
        observed water table, sensor layer): may23 4.3e-6, sep23a_cal3 2.8e-6 (C_N1: 2.9e-4, the 1 h - 1/4 h
        figure of 74 g). The solver's step error at the sensor is 70 - 100 x smaller here than on C_N1. The
        inversion's own step dependence (1/4 h, jobs 15617461 / 15617462) is running.
                                                rfnm1 (1 May 2023, 15 d, 361 h)   rfna1 (1 Sep 2023, 6 d, 145 h)
          PINN from the well: RMSE / bias / max, r   0.889 / +0.04 / 2.57, 0.9975     0.644 / +0.11 / 1.94, 0.9977
          numerical inversion (1 h) from the well    0.893 / +0.03 / 2.49, 0.9975     0.629 / +0.02 / 1.82, 0.9978
          PINN - numerical: RMS / bias / max         0.035 / +0.009 / 0.077           0.093 / +0.090 / 0.129
          largest difference between the 5 starts    1.8e-7 cm                        3.5e-9 cm
          posterior sd, median                       0.31 cm                          0.28 cm
          soil-moisture misfit: PINN | numerical     0.001456 | 0.001487              0.001948 | 0.001940
          hydrostatic reading, same soil             1.21 / -0.72 / 2.79, 0.9973      1.62 / -1.50 / 2.34, 0.9983
          spline fitted to the well                  0.44                             0.47
          rain / ET over the window (data set)       3.6 mm / 41.3 mm                 0.25 mm / 24.5 mm
        AGAINST 74 b: starts < 0.2 cm: met. PINN - solver <= 0.3 cm: met (0.035, 0.093). From the well 0.7 - 1.1
          (W1) and 0.5 - 0.9 (W2): met (0.89, 0.64). "The hydrostatic reading with the calibrated soil should
          be about as good as the inversion": NOT met as written. Its correlation is the same (0.997 / 0.998)
          but it is 0.7 and 1.5 cm too shallow on average, RMSE 1.21 and 1.62 against 0.89 and 0.63. Why the
          static reading is biased with a soil calibrated through the flow model was not looked at.
        AGAINST 74 r (written before the scores): rfnm1, the run with the worse flux losses, is the one closer
          to the numerical inversion (0.035 against 0.093 cm), so "rfnm1 is the one more likely to miss" was
          wrong. The reading rule of 74 r stands as written: these rows show (i) on two recessions (3.6 and
          0.25 mm of rain), with forward networks that miss the prescribed flux by 20 - 23 % and 8 - 11 % of
          its rms on held-out points; they are not evidence that the network solved the flow problem there.
          rfna1's difference from the numerical inversion is almost all bias (+0.090 of 0.093 cm).
        Figures looked at (result_rfnm1.png, result_rfna1.png): lines only, legends, axis labels with units.
          Both estimates are smooth lines through a well record that carries a daily ripple the 6- and
          15-day splines do not follow; the soil moisture falls monotonically in both.
        TALLY AFTER S1: (i) holds to 0.5 cm in 10 of 14 scored runs (three more windows cannot be counted as
          sites: all US-NC2). (ii) in-sample at US-NC2 on three windows (rfnc1 1.10, rfnm1 0.89, rfna1 0.64);
          held out only rfnd (diagnostic) and rf22c below saturation. No window has passed R4.
    74 u. (14 Nov, 06:21) ONE STATISTIC FOR "HOW WELL IS THE FLUX CONDITION MET" ACROSS WINDOWS: PLAN.
        Found while writing 74 t into the README: the held-out flux loss does not start at 100 in rfnc1 but at
        560 - 640 (runs/rfnc1dc_f1_m0 593 -> 0.0356). Read in the code: sigma_q = 0.1 x rms(q) over the hourly
        record (parametric.py, physics_scales), and the sampler draws 30 % of the times from rain intervals
        (train.py, Sampler.t, frac_rain_t). So last / first of the held-out flux loss is
        (rms flux error / rms prescribed flux)^2 ON POINTS DRAWN AS IN TRAINING, if the network has zero flux
        at step 0 (hydrostatic start; not checked separately). On that statistic: rfnc1 forward nets 0.74 and
        0.77 %, seed repeats 0.60 - 0.87 %, fwd_nc2c 0.69 %; rfna1 8.4 and 11.5 %; rfnm1 20.3 and 22.7 %.
        The results set quotes another one for rfnc1: 4.8 - 5.0 % for the forward networks, 5.2 - 5.3 % for
        the operator members (tables.md section 6, on the evaluation grid of fig6). The two differ by a
        factor of 7 for the same network; the cause was not looked at. Until the same statistic exists for
        the two recession windows, the README must not put "20 %" next to "4.8 %".
        -> scripts/paper_figures.py (unchanged, no --forward: no forward PINN at the well's spline exists for
           these windows) on rfnm1 and rfna1 -> output/paper/nc2_may23, output/paper/nc2_sep23a. CPU, ~1 min.
        EXPECTATION, written before submission: section 6, surface flux rms error / rms prescribed, forward
          networks of the loop: rfna1 8 - 16 %, rfnm1 20 - 40 % (almost no rain in either window, so the
          training points are close to uniform in time and the two statistics should be close; uncertain,
          since for rfnc1 they are not). Operator members above the forward networks in both.
        READING: the README row of each window carries the section-6 figure, the one rfnc1 is quoted with.
          If either window comes out at rfnc1's level (<= 6 %) on that grid, the sentence "flux condition
          not learned as in rfnc1" of 74 r / 74 t is withdrawn for it and the held-out figure is given as
          what it is (a statistic on training-like points).
        (06:26) Both figure jobs (15617488, 15617489) ended with exit=1 after 8 s: fig_setup plotted
        tp_daily_mmd (on the forcing nodes, t_daily) against the hourly record t_days. On C_N1 the two have
        the same length (forcing.node_hours 1), on the two recession data sets they do not
        (forcing.node_hours 24: 16 and 7 nodes). Fixed in scripts/paper_figures.py (two lines: the plot and
        the trapezoid of root_uptake_mm now use t_daily). Check on C_N1 when the results set is next rebuilt:
        setup.root_uptake_mm stays 23.6617 (t_daily = t_days there).
        NOT WRITTEN ANYWHERE IN 74 b / 74 r / 74 t / README SO FAR: the two recession data sets prescribe the
        flux on DAILY nodes (linear in between), C_N1 on hourly nodes. They are older data sets
        (make_real configs before the change of 73, line 2554). The README rows must say so.
        Second thing read in the code (parametric.py, times / make_heldout / heldout_loss): the held-out flux
        set is 2048 times, 10 % of them log-uniform in the first day (frac_early), the rest from Sampler.t
        (30 % in rain intervals), each at its own path c drawn from the member's sampling distribution - not
        at the estimate. The section-6 figure is at one path (the estimate, or c_fwd), on a half-hourly grid,
        where every second point is a forcing node (a kink of the prescribed flux) on C_N1.
        CHECK (scripts/flux_grid_check.py, new, scoring only): for the members and forward networks of
          rfnc1, rfnm1, rfna1 at the same paths as section 6: rms error / rms prescribed on uniform grids of
          1, 2, 4, 12, 60 points per hour, at the forcing nodes only, half-way between them, and by day on
          the 1-min grid.
        EXPECTATION, before submission:
          F1 rfnc1 (hourly nodes): error at the nodes > on the 1-min grid > between nodes; the 1-min figure of
             the forward networks is 2 - 4 % (half-hourly: 4.8 - 5.0 %), i.e. the half-hourly grid overstates
             the time-mean error because half of its points are kinks. Uncertain: the other candidate is the
             first day (by-day line shows it).
          F2 rfnm1, rfna1 (daily nodes): 2/h and 60/h agree within 10 % (relative); figures as in 74 u above
             (rfna1 8 - 16 %, rfnm1 20 - 40 % for the forward networks).
        DECISION RULE: if the 60/h figure of rfnc1's forward networks differs from the 2/h one by more than
          20 % (relative), section 6 of the results set is changed to the 60/h grid (a time mean; stated as
          such) and the half-hourly figure is dropped; otherwise physics_at stays as it is. Either way the
          README compares windows on one grid only.
        (06:29) RESULT of the grid check (jobs 15617546 - 15617548, exit=0; output/paper/*/flux_grid.json).
          Surface flux, rms error / rms prescribed (%), networks at the paths of section 6:
            run     network            1/h    2/h    4/h   12/h   60/h | at nodes | half-way between nodes
            rfnc1   forward 1, 2      6.30   4.99   3.81   2.69   2.12 |   7.18   |  1.39
                                      6.16   4.87   3.67   2.49   1.86 |   7.02   |  1.25
                    members (4)       6.5-6.7 5.2-5.3 4.0-4.2 2.9-3.1 2.3-2.6 | 7.5-7.6 | 1.6-1.8
            rfna1   forward 1, 2      7.78   5.64   4.20   2.83   2.30 |  34.9    |  1.21
                                      7.98   5.85   4.46   3.49   3.13 |  34.9    |  1.93
                    members (4)       7.8-8.0 5.6-5.9 4.1-4.5 2.7-3.3 2.0-2.9 | 34.9-35.0 | 1.2-2.0
            rfnm1   forward 1, 2     10.99   9.06   8.03   7.44   7.24 |  42.6    |  4.47
                                     11.66   9.93   9.14   8.69   8.54 |  43.1    |  5.09
                    members (4)       9.3-12.4 7.1-11.4 5.7-10.9 4.5-10.6 4.1-10.5 | 41.4-42.8 | 2.7-7.0
          Absolute, 1-min grid: rfnc1 forward 0.21 - 0.24 mm/d of 11.6; rfna1 0.021 - 0.029 of 0.93;
          rfnm1 0.066 - 0.078 of 0.91.
          AGAINST THE EXPECTATIONS:
          F1 (rfnc1: nodes > 1-min > between; forward networks 2 - 4 % on the 1-min grid): order MET; the
             figures are 2.12 and 1.86 %, the second just below the range. The half-hourly figure (4.9 - 5.0)
             is 2.3 x the 1-min one: half of that grid's points are the hourly nodes, where the error is
             7 %; off the nodes it is 1.6 - 1.9 % whether the points are 15, 5 or 1 min apart (solved from
             the grid figures), so the excess sits within about a minute of the nodes.
          F2 (daily nodes: 2/h and 60/h agree within 10 %): NOT MET. rfna1 5.6 -> 2.3 and 5.9 -> 3.1,
             rfnm1 9.1 -> 7.2 and 9.9 -> 8.5. The node figure is the same for all six networks of a window
             (34.9, 41 - 43 %): with 7 and 16 nodes that is one point, t = 0, where the network is
             hydrostatic by construction (zero flux) and the prescribed flux is not zero
             (0.326 mm/d x sqrt(7) = 0.86 mm/d; the window's rms is 0.93). On the 2/h grid of rfna1 that one
             point of 289 is 0.86 / sqrt(289) = 0.051 mm/d = 5.4 % of the 5.6 %. By day, the error of day 0
             is 3 - 10 x that of the other days in rfna1 and 2 - 4 x in rfnm1.
          Expectation of 74 u for section 6 (rfna1 8 - 16 %, rfnm1 20 - 40 % on the half-hourly grid):
             NOT MET, both are far lower (5.6 - 5.9 and 9.1 - 9.9 %).
          THE HELD-OUT FIGURE, in the same units (sqrt(last held-out flux loss) x 0.1 = rms error on the
          held-out points / rms of q on the nodes): rfnc1 forward 1.89, 1.77 %  (x 1.139 for the 1-min rms
          of q: 2.15, 2.02 %; the 1-min grid at the path gives 2.12, 1.86 %: the same within 10 %);
          rfna1 8.3, 11.4 % (1-min grid 2.3, 3.1 %); rfnm1 20.6, 23.2 % (7.2, 8.5 %). So on C_N1 the held-out
          set and the time mean agree, and "0.6 - 0.9 %" was that figure divided by the rain-weighted
          starting value; on the two recession windows the held-out set reads 3 - 4 x the time mean.
          Not checked: why. Candidate that fits the by-day lines: 10 % of the held-out times are log-uniform
          in the first day (frac_early_t 0.1), where the error is concentrated on these windows; the held-out
          points are also at sampled paths, not at the estimate.
          WHAT THIS CHANGES:
          - The sentence of 74 r / 74 t and of the README draft, "the forward networks miss the flux by
            20 - 23 % and 8 - 11 % (0.6 - 0.9 % in rfnc1), so the rows are not evidence that the network
            solved the flow problem", is WITHDRAWN. On one statistic (time mean, 1-min grid, at the
            estimate) rfna1 is at rfnc1's level (2.3 - 3.1 against 1.9 - 2.1 %) and rfnm1 is 3.5 - 4 x
            above it (7.2 - 8.5 %). By the reading rule written above (<= 6 % on the section-6 grid) the
            sentence falls for rfna1 and stays, reworded, for rfnm1. The draft was not committed.
          - Decision rule: 60/h differs from 2/h by 58 - 62 % on rfnc1 -> section 6 (and the forward
            table's flux line) move to the 1-min grid, stated as a time mean; the half-hourly figure goes.
        NEXT, before physics_at is changed (06:29): is 60/h itself converged, and is the first day the
          reason for the held-out figure? flux_grid_check.py gets a 600/h grid and the error on 601
          log-uniform times in [1e-3, 1] d (the frac_early sample).
          EXPECTATION: G1 600/h within 10 % of 60/h on all three windows (rfnc1 forward 1.7 - 2.1 %).
            G2 early-time rms error / rms of q over the window: rfna1 and rfnm1 30 - 70 % (the corner);
               rfnc1 below 10 % (its rms is set by the rain bursts, q(0) is small against it).
            If G2 holds, sqrt(0.1 x early^2 + 0.9 x (1-min figure)^2) should give the held-out figure of the
            two recession windows within a factor 1.5 (it ignores the sampled paths).
          RULE: G1 met -> section 6 uses 60/h. G1 not met -> section 6 uses 600/h and says so.
        (06:31) SAME QUESTION FOR THE PDE RESIDUAL of section 6 (README point 3: "RMS 0.5 - 0.7 sigma_pde
          for the forward networks, 0.9 - 1.5 for the members, maxima at the rain pulses and at t = 0"): it is
          on the same half-hourly time grid. flux_grid_check.py also gets the residual on (t, xi) with 2, 12
          and 60 times per hour (xi as in section 6), split into rows at forcing-node times and the others.
          EXPECTATION: R1 C_N1: 12/h and 60/h below 2/h, by less than for the flux (a volume mean; the
            kinks of q act at the surface only): forward networks 0.5 - 0.7 at 2/h -> 0.35 - 0.65; rows at
            node times above the others. R2 daily-node windows: all three grids within 10 %.
          RULE: if 60/h differs from 2/h by more than 20 % on a window, section 6 gives the residual on the
            finer grid too (12/h if it is within 10 % of 60/h, to keep the figure job short; else 60/h).
        (06:31) RESULT of the second flux check (jobs 15617598 - 15617600, exit=0).
            run     network        60/h   600/h | early (601 log-uniform times in [1e-3, 1] d)
            rfnc1   forward 1, 2   2.12   2.03  |   1.10
                                   1.86   1.76  |   1.28          members 2.3-2.6 / 2.2-2.5 | 1.0 - 3.1
            rfna1   forward 1, 2   2.30   2.19  |  23.7
                                   3.13   3.06  |  34.0           members 2.0-2.9 / 1.9-2.8 | 15.9 - 26.1
            rfnm1   forward 1, 2   7.24   7.20  |  68.1
                                   8.54   8.50  |  79.6           members 4.1-10.5 / 4.0-10.5 | 45 - 107
          G1 (600/h within 10 % of 60/h): MET on all 18 networks (largest change -6 %). -> section 6 uses
             the 1-min grid.
          G2 (early-time error: recession windows 30 - 70 %, rfnc1 < 10 %): rfnc1 MET (1.0 - 3.1 %);
             rfnm1 45 - 107 %, four of six above the range; rfna1 16 - 34 %, five of six below it. NOT MET
             as a range; the order (large on the recession windows, small on C_N1) holds.
          Held-out figure rebuilt as sqrt(0.1 early^2 + 0.9 (600/h)^2): rfna1 7.8 and 11.1 % (held-out
             8.3, 11.4), rfnm1 22.6 and 26.4 % (20.6, 23.2), rfnc1 1.96 % (2.15, same units). Within 14 %
             everywhere: the held-out flux loss of the two recession windows is the error of the first day,
             which the training sample weights with 10 % of its times.
          READING: on these two windows the flux error is the start corner (the network is hydrostatic at
             t = 0 by construction; the prescribed flux there is 0.86 mm/d, about the window's rms). Away
             from it the networks follow the flux to 2 - 3 % (rfna1) and 7 - 8.5 % (rfnm1) in the time mean,
             against 1.8 - 2.1 % on C_N1. How long the corner lasts was not measured (by day: day 0 carries
             the error; nothing finer was computed).
        (06:39) PLAN for a third check, written after the members' lines of the residual jobs and fig6 of
          C_N1 were seen (2/h 0.87 - 1.48, 60/h 0.25 - 0.35; node rows 1.2 - 2.1, other rows 0.17 - 0.23)
          and BEFORE any number of the first hour was computed. In fig6 of C_N1 the only row above 3 is
          t = 0 (about 20 - 35). 2.080^2 x 265 node rows = 1147 = 33.9^2: one row would give the whole
          node figure. So the node/other split may be the start corner again and not the forcing nodes.
          flux_grid_check.py gets, per grid: the row t = 0, the node rows without it, the rows of the first
          hour and those after it (rms and max), the time mean by the trapezoid rule, where the max is; by
          day on 60/h; the rows at t = 0, 1, 6, 20, 60, 120, 300, 600, 1200, 1800, 3600 s; and the same
          times and "after the first hour" for the flux error.
          EXPECTATION: S1 C_N1 members: row t = 0 at 20 - 35 sigma_pde; node rows without it within 30 %
            of the other rows. S2 C_N1: after the first hour 2/h, 12/h and 60/h agree within 10 %, at
            0.15 - 0.22 for the members. S3 C_N1: the max (77 - 130) is at t = 0, and the row rms is below
            1 by t = 600 s. S4 daily-node windows: row t = 0 at 2 - 5; after the first hour within 2 % of
            the whole window. S5 C_N1 members: trapezoid time mean on 60/h 0.25 - 0.32, below the plain mean.
          RULE (section 6 and the forward tables): the residual moves to the 1-min grid as a time mean
            (rule of R: 60/h differs from 2/h by more than 20 % on C_N1, and 12/h is not within 10 % of
            60/h). If S2 holds, the table gets a second residual column "after the first hour" and the
            README gives the row t = 0 in words; the half-hourly figure goes. If S2 fails, the residual is
            reported with its grid and the words "not converged in the time grid", and the question stays
            open. The flux gets the same second column only if "after the first hour" differs from the
            whole-window figure by more than 20 % on a window.
        (06:40) RESULT of the residual grid check so far (15617619 rfna1 exit=0; 15617617 / 15617618
          cancelled by me after 4 and 3 networks, because the third check repeats their grids and writes
          the same json; third check = jobs 15617745 fgridc, 15617746 fgridm, 15617747 fgrida).
          PDE residual / sigma_pde, rms (max) [rows at node times | other rows]:
            rfnc1 members (4)   2/h 0.874 - 1.477 (77 - 130) [1.22 - 2.08 | 0.166 - 0.172]
                                12/h 0.395 - 0.628            [same       | 0.183 - 0.189]
                                60/h 0.252 - 0.352            [same       | 0.198 - 0.230]
            rfnm1 members (3)   2/h = 12/h = 60/h to 0.001: 0.790 - 0.980 (9.6 - 16.1) [0.90 - 1.08 | 0.79 - 0.98]
            rfna1 members (4)   2/h = 12/h = 60/h to 0.001: 0.981 - 1.201 (9.5 - 10.6) [1.01 - 1.20 | 0.98 - 1.20]
            rfna1 forward (2)   0.884 / 0.882 / 0.881 and 0.973 / 0.972 / 0.972 (12.4, 13.4) [1.02 | 0.88, 0.97]
          R1 (C_N1: finer grids below 2/h "by less than for the flux"): the direction holds, the size does
             NOT: the members fall by a factor 3.5 - 4.2 from 2/h to 60/h (flux: 2.2 - 2.3), and 60/h is not
             converged (12/h -> 60/h still -36 to -44 %). Forward networks of C_N1: in the third check.
          R2 (daily-node windows within 10 %): MET (identical to 0.001 on the 9 networks read).
          RULE applied: residual of section 6 on the 1-min grid (12/h is not within 10 % of 60/h).
          Not known yet: whether the node rows of C_N1 are the forcing nodes or the single row t = 0
          (third check, S1).
        (06:52) RESULT of the third check, C_N1 (15617745, exit=0) and rfna1 (15617747, exit=0); rfnm1
          (15617746) still running. PDE residual / sigma_pde, xi grid of section 6 (101 rows clustered at
          the surface), one path (the estimate; forward networks at their own path).
                               row t = 0   node rows     after the first hour (max)       trapezoid   max is at
                               (rms)       without it    2/h     12/h    60/h             mean, 60/h
            rfnc1 members (4)  19.2-33.5   0.312-0.329   0.252-0.261 0.194-0.202 0.186-0.194 (9.3-10.4)
                                                                                          0.228-0.298 t = 0, surface
            rfnc1 forward (2)  11.7, 10.2  0.304, 0.265  0.242, 0.220 | 0.185, 0.179 | 0.179, 0.175 (11.7, 6.9)
                                                                                          0.231, 0.230 t = 2 min, 1 min
            rfna1 members (4)  0.71-0.89   1.03-1.26     0.982-1.202 on all three grids (9.5-10.6)
                                                                                          = plain mean  bottom, days 2-6
            rfna1 forward (2)  1.33, 0.90  0.96, 1.04    0.883, 0.974 on all three grids (12.4, 13.4)
                                                                                          = plain mean  bottom, day 6
          Rows of the first hour, rms over xi (C_N1 members; t = 0, 1, 6, 20, 60, 120, 300, 600, 1200, 1800,
            3600 s): 19-33, 19-32, 15-28, 5-14, 4-12, 4-9, 1.2-2.3, 0.6-1.1, 0.2-0.6, 0.2-0.3, 0.17-0.21.
            Forward networks: 10-12 up to 120 s, 1.6-2.2 at 300 s, 0.5-0.8 at 600 s, 0.14-0.17 at 1 h.
          Flux error in the first hour: at t = 0 it is minus the whole prescribed flux (C_N1 -11.26 mm/d,
            rfna1 -0.86 mm/d: the ansatz has zero flux at t = 0 by construction); within 0.1 - 0.9 mm/d
            from 120 s on C_N1, within 0.3 mm/d from 600 s on rfna1.
            After the first hour, rms error / rms prescribed (600/h grid; whole window in brackets):
            rfnc1 members 2.14 - 2.49 % (2.19 - 2.54), forward 1.96, 1.67 % (2.03, 1.76);
            rfna1 members 1.35 - 2.19 % (1.91 - 2.80), forward 1.28, 2.00 % (2.19, 3.06).
          AGAINST THE EXPECTATIONS:
          S1 first half holds for three members (25.8, 27.1, 33.5; the fourth is at 19.2, just below the
             20 - 35 I wrote). Second half FAILS: the rows at the forcing nodes without t = 0 are at
             0.31 - 0.33, the half-hour rows at 0.166 - 0.172, a factor 1.8 - 2.0, not "within 30 %". So both
             are true on C_N1: the row t = 0 carries the 2.08 I had read as "the node rows" (33.47^2 / 265
             = 2.057^2), AND the residual at the hourly forcing nodes is about twice that between them.
          S2 FAILS as written: after the first hour 2/h is 30 - 35 % above 60/h (half of its rows are node
             rows). 12/h and 60/h agree within 4 % (0.194 - 0.202 against 0.186 - 0.194), inside the
             0.15 - 0.22 I expected for the level. A finer grid than 60/h was not run for the residual.
          S3 first half holds for the members (max at t = 0 at the surface), not for the forward networks
             (max 56.5 and 46.9 at t = 2 min and 1 min, at or next to the surface). Second half FAILS for one
             member (row rms 1.14 at 600 s; the others 0.61 - 0.95); all four are below 0.6 at 1200 s.
          S4 first half FAILS: on rfna1 the row t = 0 is at 0.7 - 1.3, not above the window mean; there is
             no start corner in the residual of this window. Second half holds (after the first hour = whole
             window to 0.2 %). rfnm1: not in yet.
          S5 holds for the direction (0.228 - 0.298 against a plain mean of 0.252 - 0.352) and for two of
             four members for the range (0.25 - 0.32 expected; the others 0.228, 0.248).
          WHAT THIS SAYS ABOUT THE STATISTIC. On C_N1 the whole-window RMS of the residual is not a
            property of the network on a uniform grid of 1 min or coarser: the start corner lasts 5 - 20
            min, its rows are 5 - 30 times the rest, and a uniform grid sees it through one to twenty
            rows. From the rows above, the time integral over the first hour is of the size of the
            integral over the other 263 h (member 0, trapezoid by hand on the eleven rows: 3.5e4 against
            3.5e4 sigma_pde^2 s), i.e. a resolved time mean would be about 0.27 against 0.19 after the
            first hour. NOT computed by a job; a hand estimate from eleven rows.
          RULE, as it applies (neither branch of what I wrote covers a half-met S2, so the decision and
            its reason are stated here instead of picking the branch that reads better):
            - sections 6 / forward tables give the residual AFTER THE FIRST HOUR on the 1-min grid as the
              headline figure (12/h and 60/h agree within 4 %; the half-hourly grid reads 30 - 35 % higher
              because half its rows sit on forcing nodes), its max, and separately the row t = 0;
            - the whole-window RMS stays in metrics.json with its grid and is not put in a table: it is
              not converged in the time grid on C_N1;
            - the README says in words: start corner (row t = 0 at 19 - 33 sigma_pde for the members,
              10 - 12 for the forward networks, below 1 after 10 - 20 min), node rows twice the others,
              and that "at the three rain pulses" in point 3 was NOT a checked statement: the maxima of
              45 - 130 are at t = 0 (members) and within the first 2 min (forward networks). Where the
              maximum after the first hour (7 - 12) sits is not known; paper_figures.py will record its
              time.
            - flux: "after the first hour" differs from the whole window by 22 - 42 % on rfna1 (rule:
              more than 20 % on a window), so the flux gets the second column too.

    74 v. (14 Nov, 06:52) rfnc1tc (conductive form on the second-soil data set of C_N1; DIAGNOSTIC, 73 bz):
        THE LOOP HAS ENDED ("full rfnc1tc done"). LOSSES, written before any score of the final estimate
        is read (lesson rule). Held-out sets of training: last val_pde; last val_flux and 10 * sqrt(val_flux)
        = rms error on the held-out times / node rms of q, in % (2048 times, 10 % log-uniform in the first
        day, each at its own sampled path: NOT the statistic of section 6).
                         held-out PDE loss          held-out flux loss (% of node rms)
          wide w0-w3     493 (w0), 18, 21.7, 15.8   0.827, 0.849, 0.888, 0.771   (8.8 - 9.4 %)
          round 1        57.9, 10, 29.4, 16.8       0.556 - 0.684                (7.5 - 8.3 %)
          round 2        68.6, 78, 237, 416         0.487 - 0.617                (7.0 - 7.9 %)
          round 3        24.6, 8.38, 140, 22.6      0.395 - 0.474                (6.3 - 6.9 %)
          round 4        9.15, 25.3, 5.42, 16.1     0.526, 0.448, 0.503, 0.452   (6.7 - 7.3 %)
          forward f1     9.54, 4.16                 0.204, 0.148                 (4.5, 3.8 %)
          forward f2     5.2, 4.09                  0.271, 0.179
          forward f3     6.61, 3.41                 0.275, 0.166
          forward f4     9.37, 4.1                  0.26, 0.224                  (5.1, 4.7 %)
          (w0 ends at 10.1 times its step-0 PDE loss; flux at step 0 of the wide stage 576 - 612.)
        Value-correction loop, largest step / posterior sd: it 1 0.365, it 2 0.262, it 3 0.259, it 4 0.229.
          The stop rule (0.2) was NOT reached in the four iterations allowed; the estimate scored is it 4.
          (rfnc1t, additive form, same data: 4.42, 1.33, 1.28, 1.88.) Forward offset rms 4.4e-4, 7.7e-4,
          3.7e-4, 6.8e-4.
        AGAINST 73 bz (ratio sqrt(last flux / wide step-0 flux)): wide 0.036 - 0.038 (expected 0.03 - 0.10),
          rounds 0.027 - 0.030 (0.02 - 0.05), forward networks of the loop 0.016 - 0.021 (0.015 - 0.03): all
          three inside. The held-out PDE loss stays at 4 - 25 (rms 2 - 5 sigma_pde on the held-out points)
          in every stage: the "interior residual" caveat of 73 bz stands, and on that figure these networks
          are NOT evidence that the flow problem is solved to the level of rfnc1 (same statistic there:
          r4 0.14 - 0.29, forward 0.03; sigma_pde differs between the two soils, so the ratio is not
          like-for-like). Same statistic on rfnc1t (additive): flux 241 - 255 % (not learned), pde 38 - 44.
        THE NUMERICAL REFERENCE OF THIS DATA SET, before it is scored against (lesson of 74 g):
          1 h step, four starts, one cost (0.0832493 +- 3e-8), 4 - 9 iterations, no coefficient on a bound;
          the solver did not converge in up to 3.0 % of the steps (fem_not_converged_max 0.023 - 0.030).
          The stored 1 h solve at the record water table is 5.6e-4 RMS (5.4e-3 max) in sensor theta from
          the 1/4 h, 201-node one (real_summary.json; C_N1 own soil: 2.9e-4, 3.1e-3). Its dependence on the
          time step has NOT been measured on this data set, and rfnc1t was scored against it (3.89 cm)
          without that being written next to the score.
          -> one 1/4 h inversion, start hydrostatic (the fastest of the four at 1 h: 4 iterations, 43 min),
             same command as 74 h: output/referee/nc2_sep23aon_sep23c_dt025_s0. rfnc1tc is scored when it
             has ended, not before.
          EXPECTATION, written before submission: L1 1 h minus 1/4 h inversion 0.05 - 0.3 cm RMS (C_N1 own
            soil: 0.044; here the step error in theta is twice that and 3 % of the 1 h steps did not
            converge; I do not know the sensitivity well enough to narrow it). L2 the not-converged
            fraction falls below 1 % at 1/4 h. L3 against the hourly well the 1/4 h inversion stays within
            0.1 cm of the 1 h one (3.05 cm, bias -1.64, r 0.948).
          READING: if L1 holds, the 73 bz reading rule is applied against the 1/4 h inversion and the 1 h
            figure is given next to it. If the difference is above 0.5 cm, R2 on this data set (rfnc1t and
            rfnc1tc) is read against the 1/4 h inversion only, and the 3.89 cm of rfnc1t is re-scored.
        (06:54) Third check, rfnm1 (15617746, exit=0), same statistics:
            rfnm1 members (4)  row t = 0 1.06 - 1.71; node rows without it 0.82 - 1.08; after the first hour
                               0.767 - 0.979 on all three grids (9.6 - 16.1) = the whole window to 0.3 %;
                               max at the water table (xi = -1), days 2.6 - 15
            rfnm1 forward (2)  row t = 0 1.45, 1.20; after the first hour 0.820, 0.811 (13.7, 13.5)
            flux, after the first hour (whole window, 600/h): members 3.39 - 8.45 % (4.04 - 10.49),
                               forward 6.56, 7.74 % (7.20, 8.50): 9 - 19 % below the whole window; the flux
                               error at t = 0 is -1.50 mm/d and is still 0.2 - 1.2 mm/d at 20 min.
          S4 on rfnm1: first half FAILS as on rfna1 (row t = 0 at 1.1 - 1.7, not 2 - 5: no start corner in the
            residual of the daily-node windows); second half holds.
          So the start corner of the RESIDUAL is a feature of C_N1 (hourly nodes, q(0) = 11.3 mm/d at a
            midday start), and the start corner of the FLUX is on all three windows (the ansatz has zero
            flux at t = 0): 2 - 7 % of the whole-window figure on C_N1, 9 - 19 % on rfnm1, 22 - 42 % on rfna1.
          In the recession windows the largest residual is at the water table, not at the surface. Not
            looked at further here.
        (06:54) scripts/paper_figures.py, final form of the statistic (physics_at): flux on the 1-min grid,
          whole window and after the first hour (both over the RMS of the prescribed flux of the whole
          window); PDE residual on the 1-min grid after the first hour (RMS, max, where the max is) and the
          row t = 0; the whole-window residual RMS stays in metrics.json only. fig6, third panel: RMS over
          the times after the first hour (axis label says so). Tables of sections 2, 6, 7 follow.
          EXPECTATION for the rebuild (written before the jobs): section 6 of C_N1 gives, for the members,
            flux 2.29 - 2.62 % / after 2.1 - 2.5 %, residual after the first hour 0.186 - 0.194 (9.3 - 10.4),
            row t = 0 19.2 - 33.5; forward networks 2.12, 1.86 % / 1.9 - 2.0, 1.6 - 1.7 %, 0.179, 0.175
            (11.7, 6.9), 11.7, 10.2 -- the figures of flux_grid_check (60/h flux; its "after" was on 600/h, so
            the last digit may differ). Everything outside the physics columns is unchanged against
            /tmp/pf_before (root uptake of C_N1 23.66 mm). If a physics figure differs from the check by
            more than 2 % of itself, the two scripts disagree and the rebuild is not used until that is
            found.
