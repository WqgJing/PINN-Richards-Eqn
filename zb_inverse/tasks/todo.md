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
