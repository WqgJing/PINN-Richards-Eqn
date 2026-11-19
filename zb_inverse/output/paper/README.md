# Results set: water-table depth from surface flux and near-surface soil moisture

Built from one site window, US-NC2, 10 Sep 2023 12:00 + 11 days (hourly), PINN run `rfnc1`.
Figures and tables are written by `scripts/paper_figures.py`; every number below is in
[nc2_sep23c/metrics.json](nc2_sep23c/metrics.json) and laid out in [nc2_sep23c/tables.md](nc2_sep23c/tables.md).
The same figures and tables for the two further US-NC2 windows of "Other windows and sites" are in
[nc2_may23_b010/](nc2_may23_b010/tables.md) (`rfnm1b`) and [nc2_sep23a_b010/](nc2_sep23a_b010/tables.md)
(`rfna1b`), and for the earlier runs of the same windows on the narrower range of paths in
[nc2_may23/](nc2_may23/tables.md) (`rfnm1`) and [nc2_sep23a/](nc2_sep23a/tables.md) (`rfna1`); those sets
have no forward-problem part and no seed repeats (no fig2, fig7).

## What this case shows, and what it does not

- The soil (five van Genuchten parameters) was calibrated on this same window. The case is **in-sample**:
  it shows that the network recovers the water table from flux and soil moisture when the soil is right.
- It is **not** a held-out test. No site has yet passed a held-out window against the well
  (see "Other windows and sites").
- The numerical (FEM) solver appears here only as a reference to score against. It is not used to train the
  networks, to choose checkpoints, or in the inversion.

## Method as run

- Unknown: the water-table path zb(t), a cubic B-spline with 14 coefficients c, bounded to 0.20 - 1.50 m.
- Network: pressure head in the moving column, h = (1 + xi) [ -zb(t; c) + s(t) N(xi, t, features) ], so that
  h = 0 at the water table and the initial profile is hydrostatic exactly. N has 38 inputs (depth, time,
  Fourier features of time, causal features of the path zb(t; c), memory features of the known forcing),
  6 hidden layers of 128 units, tanh, 87 681 parameters.
- Training uses physics only, three terms and no soil-moisture data:
  mean(R^2) / sigma_pde^2 (Richards residual) + mean(f^2) / sigma_q^2 (surface flux) +
  mean(d^2) / sigma_theta^2 (water balance of space-time boxes, the integral form of the same equation).
  The three scales come from the data set (sigma_theta 0.005, sigma_q 10 % of the flux scale, sigma_pde derived);
  none is tuned.
- Operator: four independently seeded networks trained over a distribution of paths c (wide stage, 42 000 steps),
  then four refinement rounds (18 000 steps each) around the current estimate.
- Inversion: Levenberg-Marquardt over c through the networks, with exact sensitivities by automatic
  differentiation, from five starts. The fit is to the sensor's soil moisture (mean over 0 - 0.30 m).
- Value correction: two further physics-only networks are trained at the current estimate alone
  (28 000 steps); their soil moisture replaces the operator's value there, the operator keeps the slope,
  and the inversion is repeated until the step is below 0.2 posterior sd (here: one iteration).

## Figures

| file | content |
|---|---|
| [fig1_setup](nc2_sep23c/fig1_setup.png) | forcing (surface flux, root uptake), sensor soil moisture, observed water table |
| [fig2_forward](nc2_sep23c/fig2_forward.png) | forward problem: PINN against the numerical solution with the water table given |
| [fig3_inverse](nc2_sep23c/fig3_inverse.png) | inverse problem: fitted soil moisture; water table from the well, the PINN (with 1 sd), the numerical inversion and the hydrostatic reading; errors by hour |
| [fig4_training](nc2_sep23c/fig4_training.png) | held-out PDE, surface-flux and water-balance losses of the four members through all stages |
| [fig5_stages](nc2_sep23c/fig5_stages.png) | the estimate stage by stage and its distance to the well and to the numerical inversion |
| [fig6_physics](nc2_sep23c/fig6_physics.png) | physics at the estimate: prescribed against network surface flux; PDE residual over time, and over depth for the times after the first hour |
| [fig7_seeds](nc2_sep23c/fig7_seeds.png) | the whole pipeline repeated with two other training seeds: the three estimates and their differences from the first run |

## Main numbers

Forward problem (water table given; `runs/fwd_nc2c`, 47 min on one GPU). Soil-moisture difference in m3/m3.
The numerical solution stored in the data set has a 1 h time step and is itself 4.3e-4 (sensor layer) from a
solve with a 1/1024 h step ([forward_reference_d.json](nc2_sep23c/forward_reference_d.json),
`scripts/forward_reference_check.py`), so the network is given against both:

| quantity | PINN − numerical, 1/1024 h step | PINN − numerical, 1 h step (the data set's) | numerical, 1 h step − 1/1024 h step |
|---|---|---|---|
| all depths and times, RMS | 1.4e-5 (relative L2 0.0035 %, 0.022 % of the range) | 3.3e-4 (0.080 %, 0.51 %) | 3.3e-4 |
| all depths and times, max | 2.0e-4 | 5.8e-3 | 5.9e-3 |
| sensor layer, RMS | 1.7e-5 | 4.3e-4 | 4.3e-4 |
| sensor layer, max | 7.7e-5 | 4.5e-3 | 4.6e-3 |

- Surface flux of the network, RMS error / RMS of the prescribed flux, time mean on a 1-min grid: 1.65 %
  over the whole window, 1.46 % after the first hour (the network has zero surface flux at t = 0 by
  construction; point 3 below). The second figure is the same on a 6-s grid (1.45 %). The first is not
  (1.52 %): the single point t = 0 carries 22 % of its mean square on the 1-min grid. The earlier version
  of this set gave 4.8 % on a half-hourly grid. 72 % of that mean square is the point t = 0 (2.6 % without
  it); the rest of the difference is the forcing nodes, which are every second point of that grid (3.6 % at
  the nodes without t = 0, 1.1 % half-way between them).
- How well the reference itself is known: the 1/1024 h solve is 2.1e-6 (sensor layer; all depths 1.6e-6)
  from the 1/256 h solve, an eighth of the network's distance from it. The network's figures did not move
  when the reference went from 1/256 h to 1/1024 h (1.72e-5 to 1.71e-5 at the sensor). So 1.7e-5 (sensor) and
  1.4e-5 (all depths) are given as the forward error of the network, with 2e-6 as what is known about the
  reference. The change from one step to the next fell by 3.0, 3.7, 3.9 and then 3.2 per refinement: first
  order in the step up to 1/256 h, less than that at the last level, for a reason that was not examined. The
  1/1024 h solve is therefore not claimed to be closer to the limit than about 2e-6.
- At a 1/16 h step, 201 nodes instead of 101 move the solution by 8e-7: the time step limits the reference,
  not the grid.
- The difference from the 1 h solution is the 1 h solver's step error, not the network's: in fig2 (lower
  panels) the curve "PINN − 1 h" lies on the curve "1 h − 1/1024 h" in time, and over depth except close to
  the water table, where both are below 1e-5; the 5.8e-3 maximum is the solver's (5.9e-3), against 2.0e-4
  for the network.
- Other forward networks on the same problem ([tables.md](nc2_sep23c/tables.md), section 7), sensor layer,
  against the 1/1024 h solve: two other seeds 1.74e-5 and 2.2e-5; 2048 instead of 8192 collocation points
  1.5e-5; width 64 5.5e-5; width 32 6.3e-5; 3 hidden layers instead of 6 5.4e-5. Against the 1 h solution
  all seven read 4.25e-4 to 4.32e-4, so that comparison cannot tell them apart. The equations separate the
  same two groups: surface flux after the first hour 1.4 - 1.8 % and PDE residual 0.16 - 0.19 sigma_pde for
  the three seeds and the 2048-point network, 3.4 - 3.5 % and 0.29 - 0.37 for width 64, width 32 and
  3 layers (1-min grid; within 0.7 % of a 6-s grid for the flux and within 0.4 % of a 12-s grid for the
  residual, [flux_grid_fine.json](nc2_sep23c/flux_grid_fine.json)).

Inverse problem, hourly, against the well (265 hours):

| estimate | RMSE (cm) | bias (cm) | max (cm) | r |
|---|---|---|---|---|
| PINN | 1.10 | +0.11 | 3.16 | 0.990 |
| numerical inversion (same data and soil) | 1.11 | +0.11 | 3.13 | 0.990 |
| hydrostatic reading of the sensor (no flow) | 3.41 | +1.16 | 36.4 | 0.928 |
| spline fitted to the well (best a 14-coefficient path can do) | 0.81 | 0.00 | 3.46 | 0.995 |

- PINN minus numerical inversion (solver step 1 h, the one of this table): 0.07 cm RMS, 0.30 cm max.
- The numerical inversion was repeated with a 1/4 h and a 1/16 h step of the solver
  ([inversion_step.json](nc2_sep23c/inversion_step.json), `scripts/referee_step_check.py`):

  | solver step | cost at the estimate | against the well, RMSE (cm) | minus the next finer step, RMS / max (cm) | PINN minus it, RMS / max (cm) |
  |---|---|---|---|---|
  | 1 h | 0.0918 | 1.112 | 0.044 / 0.22 | 0.065 / 0.30 |
  | 1/4 h | 0.0762 | 1.113 | 0.018 / 0.06 | 0.027 / 0.09 |
  | 1/16 h | 0.0727 | 1.107 | - | 0.014 / 0.04 |

  The PINN estimate is the closer to the numerical inversion the smaller the solver's step. At 1/16 h the
  two differ by 0.014 cm RMS, which is no more than the step still moves the numerical inversion itself
  (0.018 cm between 1/4 h and 1/16 h). Below that level nothing is resolved, so this is not written as
  "the PINN's error is 0.014 cm". Against the well nothing moves (1.11 cm at all three steps). Two starts
  end 0.0004 cm apart at 1 h and 0.0001 cm apart at 1/4 h; the 1/16 h inversion was run from one start.
- Five starts end on one estimate (largest difference 5e-7 cm). Posterior sd 1.1 cm (median).
- The whole pipeline repeated with two other training seeds (`rfnc1s1`, `rfnc1s2`; fig7 and
  [tables.md](nc2_sep23c/tables.md), section 8): each is 1.10 cm from the well and 0.06 cm from the numerical
  inversion, and 0.005 and 0.006 cm RMS (0.02 cm max) from the first run. On this window, with its in-sample
  soil, the estimate does not depend on the training seed. No other window has been repeated.
- One network at a time instead of four (round 4): each estimate is within 0.01 - 0.02 cm RMS (max 0.09 cm)
  of the four-network estimate.
- Cost: 13.6 GPU hours of training (22 training runs) and 0.07 hours of inversions. The numerical inversion
  takes 26 min on a CPU for one start, 1.6 CPU hours for its four starts (1 h step; the two starts
  repeated with a 1/4 h step take 55 and 102 min, against 12 and 34 min for the same two starts at 1 h;
  the first of them with a 1/16 h step takes 160 min).

## What to say plainly in a text built on this

1. The PINN adds nothing to the distance from the well beyond what the numerical inversion has: 1.10 against
   1.11 cm. Of that, 0.81 cm is the spline's own distance from the hourly record.
2. On this window the wide stage is already at the estimate (0.10 cm from the numerical inversion); the
   refinement rounds and the value correction move it by 0.03 cm. This window does not show that those stages
   are needed. Other windows do (US-ONA Nov 2022: 1.28 cm after the wide stage, 0.52 after four rounds,
   0.10 after the value correction).
3. The PDE residual and the surface-flux error are not uniformly small, and their size depends on where
   they are measured ([flux_grid.json](nc2_sep23c/flux_grid.json) and, on finer grids and with the seven
   forward networks of sections 2 and 7, [flux_grid_fine.json](nc2_sep23c/flux_grid_fine.json);
   `scripts/flux_grid_check.py`). Unless said otherwise the figures are for the six networks behind the
   estimate: four operator members, two forward networks of the value correction.
   - The start. The network is hydrostatic with zero surface flux at t = 0 by construction, and the
     prescribed flux there is 11.3 mm/d. The residual of the row t = 0 (RMS over depth) is 19 - 33 sigma_pde
     for the members and 10 - 12 for the forward networks; it is about 1 after 10 min and 0.2 - 0.6 after
     20 min. The flux error is the full 11.3 mm/d at t = 0, below 0.9 mm/d from 2 min on and at most
     0.2 mm/d from 20 min on (sampled at 2, 5, 10, 20, 30 and 60 min).
   - After the first hour, on a 1-min grid: residual RMS 0.19 sigma_pde (members) and 0.18 (forward
     networks), maximum 7 - 12 sigma_pde, at the surface at t = 2.125 d. That is the forcing node at which
     the first rain pulse sets in (prescribed flux +1.5 mm/d at this node, -119 mm/d at the next).
     Surface flux, RMS error / RMS of the prescribed flux: 2.1 - 2.5 % (members), 2.0 and 1.7 %.
     These figures do not depend on the grid, for all 13 networks of the tables: on a 12-s grid the
     residual RMS is within 0.4 % of the 1-min figure and the maxima are the same numbers (the node is on
     every grid); on a 6-s grid the flux figure is within 0.7 %.
   - Forcing nodes. Both errors are larger at the hourly nodes of the prescribed flux than between them:
     residual 0.31 - 0.33 sigma_pde on the node rows against 0.17 on the half-hour rows, flux error
     4.5 - 4.7 % at the nodes against 1.6 - 1.8 % half-way between them (members, node t = 0 left out).
     A half-hourly grid, on which every second row is a node, reads higher than a fine one: residual after
     the first hour 0.25 - 0.26 on the half-hourly grid, 0.19 - 0.20 on a 5-min grid, 0.19 on the 1-min
     and 12-s grids.
   - The whole-window flux figure is mostly a statement about one point. For the members it is 5.2 - 5.3 %
     on the half-hourly grid, 2.3 - 2.6 % on the 1-min grid and 2.2 - 2.5 % on a 6-s grid. The point t = 0
     carries 58 - 61 % of that mean square on the half-hourly grid (3.2 - 3.5 % without it), 9 - 11 % on
     the 1-min grid and 1 % on the 6-s grid. For the forward networks of full size, whose error elsewhere
     is smaller, its share on the 1-min grid is 13 - 24 %. The tables give this figure next to the one
     after the first hour; only the second is the same on every grid it was computed on.
   - Over the whole window the residual RMS is not a converged number either: it is set by how many rows
     the grid puts into the first minutes (members: 0.40 - 0.63 on a 5-min grid, 0.25 - 0.35 on the 1-min
     grid, 0.22 - 0.27 on a 12-s grid). The tables therefore give the time after the first hour and the
     row t = 0 separately.
   - Corrections of earlier versions of this set. (a) "0.5 - 1.5 sigma_pde on the half-hourly grid, with
     maxima of 45 - 130 sigma_pde at the three rain pulses and at t = 0": the maxima over the whole window
     are at t = 0 for the members (77 - 130 sigma_pde on every grid) and within the first 3 min for the
     forward networks, where the figure still rises with the grid (the two of the value correction: 47 and
     57 on the 1-min grid, 52 and 67 on a 12-s grid; the seven of sections 2 and 7: 94 - 144 and
     95 - 149). "At the three rain pulses" had not been checked. (b) The version before this one explained
     the half-hourly flux figures by the forcing nodes alone and gave "7.5 - 7.6 % at the nodes"; 61 - 64 %
     of that mean square is the node t = 0, and the larger part of the half-hourly figure is that one
     point, as stated above.
   - The held-out losses in fig4 are sampled as in training (10 % of the times log-uniform in the first
     day) and are not comparable with any of these figures.
4. The PINN is about 30 times more expensive than the numerical inversion with its 1 h step on this window
   (13.6 GPU hours against 26 CPU minutes for one start), 8 - 15 times more expensive than the numerical
   inversion with a 1/4 h step (55 - 102 CPU minutes for one start), and 5 times more expensive than one
   start with a 1/16 h step (160 CPU minutes), the step at which the two estimates agree to 0.014 cm.
5. A forward error quoted against the data set's own 1 h solution (3.3e-4) measures that solution's time
   step. The network's distance is 1.4e-5, against a 1/1024 h solve that is itself 1.6e-6 from the level
   before.

## Synthetic verification (known truth)

Twin experiment, same method, solver used for scoring only. Water-table RMSE against the truth, in cm
(`tasks/todo.md`, finding 62; figures [n005](../compare/final_n005.png), [n010](../compare/final_n010.png),
[s1_n005](../compare/final_s1_n005.png), [s1_n010](../compare/final_s1_n010.png), [clean](../compare/final_clean.png)):

| case | days 0 - 25 | target | whole window | between starts | posterior sd (numerical reference) |
|---|---|---|---|---|---|
| noise 0.005 | 0.79 | 1.20 | 0.91 | 6.2e-5 | 0.763 (0.756) |
| noise 0.010 | 1.25 | 1.80 | 1.39 | 6.2e-5 | 1.549 (1.536) |
| noise 0.005, second noise seed | 1.01 | 1.31 | 1.24 | 1.4e-5 | 0.837 (0.786) |
| noise 0.010, second noise seed | 2.47 | 3.23 | 2.87 | 1.6e-5 | 1.672 (1.712) |
| no noise | 0.01 | 0.50 | 0.02 | 1.2e-9 | 0.759 (0.757) |

## Other windows and sites

Two statements are kept apart:
(i) the PINN returns the estimate of the numerical inversion of the same model, from every start;
(ii) that estimate is the observed water table. (ii) depends on the soil-column model and its calibrated soil,
which the PINN and the solver share.

Final estimate of every scored PINN run (`output/real_*/result_*.json`), whole window, hourly:

| site | window | soil calibrated on | run | PINN - numerical, RMS (cm) | PINN vs well: RMSE / bias (cm), r | numerical vs well (cm) | status |
|---|---|---|---|---|---|---|---|
| US-NC2 | 10 Sep 2023, 11 d | this window | rfnc1 | 0.07 | 1.10 / +0.11, 0.990 | 1.11 | in-sample |
| US-NC2 | 10 Sep 2023, 11 d | 1 Sep 2023 | rfnc1t | 3.89 | 4.53 / +1.24, 0.862 | 3.05 | diagnostic; (i) fails; loop not converged |
| US-NC2 | 10 Sep 2023, 11 d | 1 Sep 2023 | rfnc1tc | 0.82 | 3.39 / -2.02, 0.949 | 3.05 | diagnostic; the data of `rfnc1t` with the second network form (`model.departure=conductive`); within the 1 cm of (i), not within 0.5 cm (last two days 1.0 and 2.4 cm); loop not converged |
| US-NC2 | 1 May 2023, 15 d | this window | rfnm1b | 0.03 | 0.89 / +0.03, 0.998 | 0.90 | in-sample; recession (3.6 mm of rain); flux on daily nodes, 5-day spline knots; networks trained, and both inversions run, on paths of 0.10 - 1.50 m; no coefficient on the box in either estimate |
| US-NC2 | 1 May 2023, 15 d | this window | rfnm1 | 0.03 | 0.89 / +0.04, 0.998 | 0.89 | the same data on paths of 0.20 - 1.50 m; both estimates held by the 0.20 m bound of the box (without it the numerical one moves 0.45 cm RMS, see Reading); the estimate of this window is `rfnm1b` |
| US-NC2 | 1 Sep 2023, 6 d | this window | rfna1b | 0.11 | 0.56 / +0.10, 0.998 | 0.53 | in-sample; recession (0.3 mm of rain); flux on daily nodes, 5-day spline knots; networks trained, and both inversions run, on paths of 0.10 - 1.50 m; no coefficient on the box in either estimate; up to 2.1 % of the solver's steps not converged in the numerical inversion |
| US-NC2 | 1 Sep 2023, 6 d | this window | rfna1 | 0.09 | 0.64 / +0.11, 0.998 | 0.63 | the same data on paths of 0.20 - 1.50 m; both estimates held by the 0.20 m bound of the box (without it the numerical one moves 0.23 cm RMS, see Reading); the estimate of this window is `rfna1b` |
| US-NC2 | 27 Sep 2023, 9 d | May 2023 | rfnd | 0.06 | 1.59 / +0.55, 0.998 | 1.64 | held-out soil, but a diagnostic, not the declared test; both estimates held by the 0.20 m bound of the box (without it the numerical one moves 0.40 cm RMS, see Reading) |
| US-NC2 | 27 Sep 2023, 9 d | 1 Sep 2023 | rfn1 | 0.07 | 9.48 / -9.25, 0.998 | 9.53 | the declared held-out test: fails against the well; both estimates held by the 0.20 m bound of the box (without it the numerical one moves 0.42 cm RMS, see Reading) |
| US-PAS | 15 Oct 2021, 7 d | Sep 2021 | rfp2 | 0.03 | 4.52 / +0.17, 0.995 | 4.50 | run for (i) only |
| US-PAS | 29 Jul 2021, 9 d | this window (soil not usable) | rfpc1 | 2.92 | 10.73 / -1.99, 0.907 | 11.88 | (i) fails |
| US-ONA | 26 Nov 2022, 18 d | Oct 2022 | rf22c | 0.35 | 4.34 / +2.63, 0.986 | 4.37 | held-out; below saturation 2.37 / +1.47, 0.983 |
| US-ONA | 26 Nov 2022, 18 d | Nov 2021 (soil not usable) | rf22 | 0.10 | 6.15 / +5.51, 0.988 | 6.13 | (i) only |
| US-ONA | 17 Oct 2022, 15 d | Nov 2021 (soil not usable) | rf22ob | 0.45 | 4.38 / +3.21, 0.986 | 4.41 | (i) only |
| US-ONA | 15 Nov 2021, 17 d | Nov 2022 (a later window) | rf21b | 2.67 | 10.07 / -9.02, 0.937 | 9.17 | (i) fails |
| US-ONA | 27 Oct 2021, 14 d, rain | Sep 2021 | rfr1a | 28.2 | 39.3 / +33.4, 0.844 | 24.1 | (i) fails; loop not converged |
| US-HB2 | 7 Mar 2020, 15 d | dry-run soil | rfh1p | 0.22 | 2.57 / +1.15, 0.965 | 2.50 | dry run, does not count |

Reading:
- Every "PINN - numerical" figure of this table is against the numerical inversion with a 1 h solver step.
  Three rows have been repeated at finer steps. `rfnc1`: 0.07 cm at 1 h, 0.03 at 1/4 h, 0.014 at 1/16 h (see
  above); there the distance is the solver's time step. On the two recession windows it is not: the
  numerical estimate itself moves by 0.0008 cm (`rfna1`) and 0.0005 cm (`rfnm1`) between a 1 h and a 1/4 h
  step, and the PINN is as far from it at 1/4 h as at 1 h
  ([nc2_sep23a/inversion_step.json](nc2_sep23a/inversion_step.json),
  [nc2_may23/inversion_step.json](nc2_may23/inversion_step.json)).
  `rfna1`: 0.09 cm at both steps, mostly an offset: bias +0.09 cm (the PINN deeper) with 0.02 cm of spread
  around it (RMS by day 0.06 - 0.11 cm; posterior sd 0.28 cm). In that numerical inversion up to 2.8 % of the
  solver's time steps did not converge at 1 h and up to 1.7 % at 1/4 h (largest fraction in one batch of
  solves). `rfnm1`: 0.03 cm at both steps, not an offset (bias +0.01 cm); largest on the first two days
  (0.06 - 0.07 cm) and the last three (0.04 cm); posterior sd 0.31 cm; all solver steps converged.
  Where these remaining distances come from is not known.
  The data set of `rfnc1t` and `rfnc1tc` (this window, soil of 1 Sep 2023) has one numerical inversion with
  a 1/4 h step as well
  ([nc2_sep23c/inversion_step_soilsep23a_rfnc1tc.json](nc2_sep23c/inversion_step_soilsep23a_rfnc1tc.json),
  [nc2_sep23c/inversion_step_soilsep23a_rfnc1t.json](nc2_sep23c/inversion_step_soilsep23a_rfnc1t.json)).
  The 1 h and the 1/4 h inversion differ by 0.16 cm RMS: 0.05 cm over days 1 - 10 and 0.50 cm on the last
  day (largest 1.4 cm, at the end of the window). Against the well both read 3.05 - 3.07 cm. Neither is a
  clean solve of this soil: up to 2.6 % of the solver's steps did not converge at 1 h and up to 1.8 % at
  1/4 h. `rfnc1tc`: 0.82 cm at 1 h, 0.86 cm at 1/4 h; by day (1/4 h) 0.04 - 0.55 cm on days 1 - 8, then
  0.67, 1.02 and 2.41 cm. `rfnc1t`: 3.89 and 3.91 cm.
- Both inversions keep the spline coefficients in a box (water table 0.20 - 1.50 m, the range of paths the
  operator networks are trained on; 0.10 - 1.50 m in `rfnm1b` and `rfna1b`). In 10 of the 17 rows a
  coefficient of the final estimate is on that box
  (`coef_on_bound` of each `summary.json`, for the numerical inversion the start named in `result_*.json`;
  coefficients counted from 1, "shallow" = 0.20 m, "deep" = 1.50 m):

  | rows | PINN | numerical inversion |
  |---|---|---|
  | rfnc1, rfnc1t, rfnc1tc, rf22c, rf22, rfnm1b, rfna1b | none | none |
  | rfnm1, rfna1, rfnd, rfn1 | 1st shallow | 1st shallow |
  | rfp2 | 1st shallow, 5th (last) deep | the same |
  | rfh1p | 1st and 6th (last) shallow | the same |
  | rf21b | 1st shallow, 7th (last) deep | the same |
  | rfpc1 | 1st and 8th shallow | 1st and 8th shallow, 5th deep |
  | rf22ob | none | 1st deep |
  | rfr1a | 9 of 17 deep | 4 of 17 deep |

  Where the same coefficient is on the same bound in both estimates, the two agree there because the box
  holds both, and "the starts end on one estimate" includes that coefficient. Of the 12 rows in which (i)
  holds, 5 have no coefficient on the box (rfnc1, rf22c, rf22, rfnm1b, rfna1b), 6 share one or two (rfnm1,
  rfna1, rfnd, rfn1, rfp2, rfh1p) and one has one in the numerical inversion only (rf22ob); `rfnc1tc`,
  which holds (i) to 1 cm only, has none. On the two recession windows the shared bound has since been
  released (`rfnm1b`, `rfna1b`, end of this point); on the 27 Sep 2023 window (rfnd, rfn1) it has not.
  A coefficient on the box is not the path on the box: the path at the start of the window is set by the
  first three coefficients together (weights 1/6, 4/6, 1/6, before the transform to a depth). `rfna1`
  starts at 0.226 m (PINN) and 0.225 m (numerical), the well at 0.218 m; `rfnm1` at 0.301 and 0.300 m, the
  well at 0.288 m. The spline fitted to the well itself has its first coefficient outside the box in both
  windows (0.14 and 0.15 m).
  How far an estimate moves when the box is widened has been measured with the numerical inversion for the
  four US-NC2 rows that share the shallow bound: bound at 0.10 m instead of 0.20 m, 1 h step
  ([nc2_sep23a/inversion_box.json](nc2_sep23a/inversion_box.json),
  [nc2_may23/inversion_box.json](nc2_may23/inversion_box.json),
  [nc2_sep23b/inversion_box_rfnd.json](nc2_sep23b/inversion_box_rfnd.json),
  [nc2_sep23b/inversion_box_rfn1.json](nc2_sep23b/inversion_box_rfn1.json)):

  | data of row | shallow bound (m) | data misfit (cost) | coefficients on the box | start of the path minus the well (cm) | vs well: RMSE / largest error (cm) | moved from the 0.20 m estimate: RMS / largest (cm) | PINN of the row minus this estimate, RMS (cm) |
  |---|---|---|---|---|---|---|---|
  | rfna1 | 0.20 | 0.1506 | 1st | +0.74 | 0.63 / 1.8 | | 0.09 |
  | | 0.10 | 0.1314 | none | +0.18 | 0.53 / 1.3 | 0.23 / 0.75 (end of the window) | 0.25 |
  | rfnm1 | 0.20 | 0.0884 | 1st | +1.24 | 0.89 / 2.5 | | 0.03 |
  | | 0.10 | 0.0439 | none | -0.29 | 0.90 / 3.3 | 0.45 / 1.5 (start) | 0.44 |
  | rfnd | 0.20 | 0.0630 | 1st | -1.80 | 1.64 / 2.7 | | 0.06 |
  | | 0.10 | 0.0280 | none | -3.04 | 1.64 / 3.1 | 0.40 / 1.2 (start) | 0.39 |
  | rfn1 | 0.20 | 0.1317 | 1st | -5.65 | 9.53 / 12.7 | | 0.07 |
  | | 0.10 | 0.0319 | 1st, on the 0.10 m bound | -6.83 | 9.54 / 12.6 | 0.42 / 1.2 (start) | 0.44 |

  The PINN of every row here was trained and inverted on paths of 0.20 - 1.50 m. With the 0.10 m bound the
  numerical inversion ends on one estimate from all four starts on the `rfna1` and `rfnm1` data (largest
  difference 0.002 and 0.0002 cm; [nc2_sep23a/inversion_box_starts.json](nc2_sep23a/inversion_box_starts.json),
  [nc2_may23/inversion_box_starts.json](nc2_may23/inversion_box_starts.json)); `rfnd` and `rfn1` were run
  from one start, that of their row.
  The numerical estimate moves by 0.23 - 0.45 cm RMS when the bound is released, in three of the four rows
  by more than the posterior sd of the row (0.28, 0.31, 0.33, 0.23 cm), which is computed with the first
  coefficient held and does not contain this. The misfit falls by 13, 50, 56 and 76 %; why the fit gains
  that much from one coefficient is not known.
  Releasing the bound is not a move toward the well. On the two recession windows the bound held the start
  of the path too deep, and on `rfnm1` the end of the window gets worse without it (largest error
  2.5 -> 3.3 cm). On the 27 Sep 2023 window the start goes further from the well (`rfnd` -1.8 -> -3.0 cm).
  The RMSE against the well changes by 0.1 cm (`rfna1`) or less. `rfn1` fails against the well by 9.5 cm
  with either bound, and its estimate is on the bound again at 0.10 m, so it is not a finished estimate.
  So the "PINN - numerical" figures of these four rows (0.03 - 0.09 cm) are the agreement of two estimates
  held by the same bound. Against the numerical estimate of the wider box the PINN of these rows is
  0.25 - 0.44 cm away: inside the 1 cm asked of (i), but not like-for-like, since these networks never saw
  paths shallower than 0.20 m.
  The like-for-like comparison has been run on the two recession windows: networks trained on paths of
  0.10 - 1.50 m and both inversions in that box (`rfnm1b`, `rfna1b`; expectations written before the runs,
  `tasks/todo.md` 74 ac; [nc2_may23/inversion_box_rfnm1b.json](nc2_may23/inversion_box_rfnm1b.json),
  [nc2_sep23a/inversion_box_rfna1b.json](nc2_sep23a/inversion_box_rfna1b.json)):

  | run | coefficients on the box: PINN / numerical | PINN - numerical, same box: RMS / bias / largest (cm) | PINN vs well: RMSE / largest (cm) | numerical vs well: RMSE / largest (cm) | posterior sd (cm) | PINN starts, largest difference (cm) |
  |---|---|---|---|---|---|---|
  | rfnm1b | none / none | 0.034 / +0.009 / 0.13 (start of the window) | 0.89 / 3.2 | 0.90 / 3.3 | 0.31 | 7e-8 |
  | rfna1b | none / none | 0.111 / +0.098 / 0.31 (end of the window) | 0.56 / 1.5 | 0.53 / 1.3 | 0.28 | 7e-7 |

  With the bound released in training and in both inversions the two estimates still agree, to 0.03 and
  0.11 cm (expected: at most 0.3 cm), and neither has a coefficient on the box. So the agreement of the
  0.20 m rows of these two windows was not made by the bound. `rfna1b` is again mostly an offset (the PINN
  0.10 cm deeper), as `rfna1` was; where it comes from is not known, and up to 2.1 % of the solver's steps
  did not converge in that numerical inversion (none in that of `rfnm1b`). Against the well the PINN
  follows the numerical estimate (within 0.03 cm in RMSE); both windows are in-sample.
  Not measured: the 27 Sep 2023 rows (rfnd, rfn1) and the two rows at other sites that share a bound
  (rfp2, rfh1p) with networks trained on a wider range of paths.
- (i) holds to 0.5 cm or better in 12 of 17 runs, at four sites (10 of the 14 data sets: `rfnm1b` and
  `rfna1b` repeat the data of `rfnm1` and `rfna1`, `rfnc1tc` those of `rfnc1t`), and in those the starts
  end on one estimate
  (largest difference below 1e-5 cm). One more is inside the 1 cm asked of (i) but not inside 0.5 cm
  (`rfnc1tc`, 0.82 cm; next paragraph). It fails in four: US-PAS Jul 2021 (2.9 cm), US-ONA Nov 2021 (2.7 cm),
  the US-ONA rain window (28 cm; starts 1.8 cm apart, value-correction loop not converged) and the window of
  this results set with the soil of 1 Sep 2023 instead of its own (`rfnc1t`, 3.9 cm; the starts end on one
  estimate, but the value-correction loop used its four iterations with steps of 1.3 - 4.4 posterior sd and
  did not reach the stop rule of 0.2; the held-out surface-flux loss of its operator networks ended every
  stage at 0.96 - 1.00 of its starting value, so these networks do not satisfy the flux condition).
  The last one matters for how the in-sample case is read: on the same 11 days of data, with the soil
  calibrated on the window that starts 9 days earlier, the PINN is 3.9 cm from the numerical inversion, and
  the numerical inversion is 3.05 cm from the well instead of 1.11.
  The same data were run once more with a second form of the network (`rfnc1tc`,
  `model.departure=conductive`: the network's departure from the hydrostatic profile is multiplied by a
  factor between 0 and 1 that grows with suction; introduced because the first form did not learn the
  surface flux on soils with Ks near 1e-3 m/s; default off, every other row uses the first form). The
  configs of the four wide-stage members of the two runs differ in that one line. Rule and expectation
  were written before the run
  (`tasks/todo.md`, 73 bz: "the form matters for the inversion if PINN - numerical is at most 1 cm here
  and above 1 cm in `rfnc1t`"). Result: 0.82 cm against 3.89 cm (0.86 against 3.91 with the 1/4 h
  numerical inversion), so by that rule the form matters on this data set. The starts end on one estimate
  (4e-6 cm), no coefficient is on the box, and the held-out surface-flux error of the operator networks
  ends at 0.03 - 0.04 of its starting level (rms; about 0.1 % as a loss), where `rfnc1t` stayed at
  0.96 - 1.00 of the starting loss.
  What it does not show: (a) agreement at the level of the rows in which (i) holds to 0.5 cm: days 1 - 8
  are at 0.04 - 0.55 cm,
  the last three days at 0.67, 1.02 and 2.41 cm (largest 3.0 cm at t = 10.6 d); the value-correction loop
  used its four iterations and ended at a step of 0.23 posterior sd, above the stop rule of 0.2, and the
  two numerical inversions themselves differ by 0.50 cm on the last day; which of these the end of the
  window comes from has not been checked. (b) That the interior equation is solved as on the own soil: the
  held-out PDE loss stays at 4 - 25 (rms 2 - 5 sigma_pde on the held-out points) in every stage.
  (c) Anything about the well: 3.39 cm, bias -2.02 cm, r 0.949, as the numerical inversion (3.05 cm).
  It is one controlled pair, on one window with one soil.
- The two further US-NC2 windows (`rfnm1b` / `rfnm1`, `rfna1b` / `rfna1`) are recessions with their own
  soils (Ks at its upper bound of 1e-3 m/s), a surface flux given on daily nodes and 5-day spline knots;
  they are further windows of the same site, in-sample, not new sites. Their rows show (i): with the first
  coefficient on the 0.20 m bound in both estimates in `rfnm1` and `rfna1`, and with no coefficient on
  the box in `rfnm1b` and `rfna1b` (see above). How well the networks of `rfnm1` and `rfna1` satisfy the
  equations, same statistics as point 3 above ([nc2_may23/tables.md](nc2_may23/tables.md),
  [nc2_sep23a/tables.md](nc2_sep23a/tables.md), section 6; six networks each):
  surface flux after the first hour 3.4 - 8.5 % of the RMS of the prescribed flux in `rfnm1` and
  1.3 - 2.2 % in `rfna1` (`rfnc1`: 1.7 - 2.5 %); PDE residual after the first hour 0.77 - 0.98 and
  0.88 - 1.20 sigma_pde, the same on half-hourly, 5-min and 1-min grids, with maxima of 10 - 16 at the water
  table (`rfnc1`: 0.18 - 0.19, maximum at the surface). There is no start corner in the residual of these
  two windows (row t = 0: 1.1 - 1.7 and 0.7 - 1.3). sigma_pde is derived from each data set (0.059 for
  `rfnc1`, 1.6e-4 and 3.9e-4 here), so the residual figures are not like-for-like between windows.
  The networks trained on the wider range of paths, same statistics
  ([nc2_may23_b010/tables.md](nc2_may23_b010/tables.md),
  [nc2_sep23a_b010/tables.md](nc2_sep23a_b010/tables.md); expectations written before the check,
  `tasks/todo.md` 74 af): surface flux after the first hour 5.7 - 12.2 % in `rfnm1b` (three of the four
  operator networks at 10.8 - 12.2 %, the two forward networks of the value correction at 7.3 and 7.6 %)
  and 1.3 - 2.3 % in `rfna1b`; PDE residual after the first hour 0.81 - 1.17 and 0.83 - 1.38 sigma_pde,
  the same on the three grids, maxima of 11 - 29 and 8 - 14 at the water table; row t = 0: 1.8 - 2.1 and
  1.1 - 1.4. On the May window the wider range of paths costs accuracy in the flux condition (three
  operator networks at about 1.5 times the error of `rfnm1`) while the estimate is 0.03 cm from the
  numerical one; on the 1 Sep window the statistics are those of `rfna1`.
  The hydrostatic reading of the sensor with the same soil is 1.21 and 1.62 cm from the well (0.7 and
  1.5 cm too shallow on average). The numerical solution of these two data sets changes by 4e-6 and 3e-6
  at the sensor between a 1 h and a 1/4 h step (3e-4 on the window of this results set).
- (ii) holds over the whole window (3 cm RMSE, 2 cm bias, r 0.98) only at US-NC2: in-sample (rfnc1,
  rfnm1b / rfnm1, rfna1b / rfna1) and with the May soil (rfnd, not the declared test). At US-ONA Nov 2022 with the Oct 2022 soil (rf22c, held-out) it
  holds for the 359 of 433 hours in which the sensor is below saturation and not over the whole window; on
  that recession the hydrostatic reading of the sensor does as well (1.93 cm below saturation), so the
  window does not show that the flow model adds skill.
- Where (ii) fails and (i) holds, the numerical inversion misses the well by the same amount: the limit there
  is the soil-column model and its soil, not the network.
- "Soil not usable" is the written rule: lowest-misfit calibration and an in-sample numerical inversion within
  2 cm of the well.

## New stations (ICOS): data source, licence, state

Five European stations with a well next to a 0.05 m soil-moisture sensor were screened by the rule written in
`tasks/todo.md` (74 ag) before any series was read: well inside 0.15 - 1.45 m, sensor below its plateau, rain
measured at the station, no frost, rain and sensor in step. All five meet it as written. The first four in the written order
are run (FR-Bil conifer forest on a sandy podzol, IT-SR2 pine forest on sand, SE-Nor conifer forest on
sandy-loamy till, DK-Sor beech forest on moraine); BE-Dor (grassland) is not run in this first selection: its wells end at 1.40 m, inside the
accepted range, and both of its candidate windows start with the well at that depth, which the screen did not test
(the four stations taken stay at least 0.5 m above the end of their wells). The screen and the plots of the four
test windows are in `output/site_icos/<station>/`.

State (16 Nov, 00:09). Seven of the new windows have a finished PINN run: all four of the first selection and
three of the second (BE-Dor, FR-Bil 12 Dec 2020, and SE-Nor 6 Dec 2019, the one window inside the 5 cm gate);
SE-Nor's two windows, FR-Bil 12 Dec 2020 and DK-Sor 8 Dec 2024 have two runs each (without and with
`sampling.local_box`, see below). They are
reported here as run and are not yet in the counts above. All rows are in-sample (soil fitted on the same
window with the well).

| station, window (11 d from) | selection | numerical inversion: from the well | PINN: from the well | PINN minus numerical | one estimate from every start |
|---|---|---|---|---|---|
| BE-Dor, 17 Mar 2024 | second; outside the 5 cm gate | 5.41 cm, r 0.913 | 5.41 cm, r 0.913 | 0.04 cm (largest 0.17 cm) | yes, both |
| SE-Nor, 30 Oct 2023 | first | 11.48 cm, r 0.92 | 9.30 cm, r 0.91 | 9.19 cm over the window; 1.07 cm on days 2 - 10 | yes, both |
| SE-Nor, 30 Oct 2023, repeated with `sampling.local_box` | first | 11.48 cm, r 0.92 | 9.96 cm, r 0.91 | 4.33 cm over the window; 0.54 cm on days 2 - 10; the correction loop did not stop by its rule | yes, both |
| SE-Nor, 6 Dec 2019, with `sampling.local_box` | second; inside the 5 cm gate | 4.35 cm, r 0.89 | 6.87 cm, r 0.83 | 2.88 cm over the window; 0.19 cm on days 3 - 10; 4.0, 5.2 and 7.0 cm on days 0 - 2, which is the PINN's own error (started on the PINN's path, the numerical inversion returns to its own estimate, 0.01 cm) | no (four of five PINN starts and three of four numerical starts end on one estimate; the 1.0 m start ends elsewhere in both) |
| SE-Nor, 6 Dec 2019, without `sampling.local_box` (beside the row above) | second; inside the 5 cm gate | 4.35 cm, r 0.89 | 8.27 cm, r 0.80 | 4.65 cm over the window; 0.22 cm on days 3 - 10; the correction loop did not stop by its rule | no (four of five PINN starts end on one estimate; the 1.0 m start ends elsewhere, at 13.6 times the cost) |
| FR-Bil, 12 Dec 2020, with `sampling.local_box` | second; outside the 5 cm gate | 7.61 cm, r 0.88 | 10.52 cm, r 0.80 | 4.21 cm over the window; 1.6 cm on days 0 - 9 (0.02 - 0.07 cm by day on days 0 - 6); 1.38 cm over the window against the numerical estimate with 1/4 h steps | yes (five PINN starts on one estimate; the four numerical starts 0.0 - 0.2 cm apart by day on days 0 - 9, 0.3 - 1.9 cm on the last half day) |
| FR-Bil, 12 Dec 2020, without `sampling.local_box` (beside the row above) | second; outside the 5 cm gate | 7.61 cm, r 0.88 | 6.43 cm, r 0.87 | 2.39 cm over the window; 1.24 cm on days 0 - 9 (0.02 - 0.08 cm by day on days 0 - 6); 4.32 cm over the window against the numerical estimate with 1/4 h steps; the correction loop did not stop by its rule | yes (five PINN starts on one estimate) |
| IT-SR2, 13 Nov 2019 | first | 40.23 cm, r 0.11 | 22.65 cm, r 0.44 | 31.61 cm over the window; 0.3 - 3.2 cm by day on days 0 - 3 and 8 - 10; 31.63 cm over the window against the numerical estimate with 1/4 h steps | no (one of five PINN starts and one of four numerical starts end elsewhere) |
| DK-Sor, 8 Dec 2024 | first | 40.88 cm, r 0.32 | 42.71 cm, r 0.41 | 9.64 cm over the window; 0.0 - 1.2 cm by day on days 4 - 10; the PINN run did not converge | yes, both |
| DK-Sor, 8 Dec 2024, repeated with `sampling.local_box` (beside the row above) | first | 40.88 cm, r 0.32 | 41.64 cm, r 0.26 | 6.77 cm over the window; by day 1.26, 5.44, 11.07, 7.93, 1.61, 1.87, 13.85, 9.45, 1.02, 0.61, 0.48 cm (days 0 - 10); the correction loop stopped by its rule at its third iteration | yes, both |
| FR-Bil, 29 Oct 2023 | first | 52.61 cm, r -0.52 (the other three starts 55.3 - 71.2 cm) | 43.03 cm, r -0.44 | 28.4 - 56.6 cm against the five numerical end points; training marginal, the PINN run did not converge | no (PINN starts 17.6 cm apart, numerical end points 4.9 - 53.4 cm apart) |

(RMS over the hourly record; r against the hourly well. Figures: `output/real_dor_mar24_own/result_rfdor1c.png`,
`output/real_nor_oct23_own/result_rfnor1c.png` and `result_rfnor1e.png`, `output/real_nor_dec19_own/result_rfnor2e.png` and `result_rfnor2c.png`,
`output/real_bil_dec20_own/result_rfbil2e.png` and `result_rfbil2c.png`, `output/real_sr2_nov19_own/result_rfsr21c.png`,
`output/real_sor_dec24_own/result_rfsor1c.png` and `result_rfsor1e.png`, `output/real_bil_oct23_own/result_rfbil1c.png`.)

- BE-Dor (grassland; soil n 1.13, Ks at its bound 1e-3 m/s): statement (i) holds to 0.04 cm rms, the same from
  all five PINN starts and all four numerical starts, with the lowest training losses of any real-data run.
  Statement (ii) does not hold by the written criterion (5.4 cm against 3 cm; r 0.91 against 0.98): the estimate
  follows the fall of the well from 0.20 to 0.50 m, its rise after rain and the fall to 0.60 m, but smoother.
  Two limits on what the 5.4 cm says: the hydrostatic reading of the sensor scores the same in this window
  (5.30 cm), and the numerical inversion missed the 5 cm gate of this selection by 0.41 cm, so the run was made
  outside the gate (declared before it was started).
- SE-Nor: from day 2 on the PINN is within 0.3 - 1.6 cm of the numerical estimate by day. On days 0 and 1 the two
  differ by 27.8 and 12.1 cm; there the sensor does not determine the water table (posterior sigma 45 and 19 cm
  in the PINN's inversion, 442 and 21 cm in the numerical one, whose first coefficients are on the bound).
  11 - 12 % of the solver's 1 h steps do not converge in its worst solve, so the numerical inversion was
  repeated with 1/4 h steps from the flat start: it ends 0.10 cm from the 1 h estimate (largest 0.23 cm),
  11.45 cm from the well, and the PINN is as far from it as from the 1 h one (9.18 cm over the window;
  0.3 - 1.5 cm by day from day 2). The 1 h reference stands. Started on the PINN's path, the numerical
  inversion returns to its own estimate at both step sizes (to 0.0006 cm with 1 h steps, 0.0001 cm with 1/4 h
  steps): the PINN's path is not a second solution of the solver, and the difference on days 0 - 1 is the
  PINN's own error. Statement (i) is not shown over the whole window here (9.2 cm against the 1 cm criterion;
  1.1 cm on days 2 - 10). The run was made with the refinement defect described below (refinement rounds
  2 - 4 ended with a PDE loss of 139 - 411829 and 18 - 47 % of the surface-flux error left).
- SE-Nor, the same window repeated with `sampling.local_box` (same first stage, one switch changed): the PINN
  is 4.3 cm from the numerical estimate over the window instead of 9.2 cm, and 0.5 cm on days 2 - 10 instead
  of 1.1 cm (0.15 - 0.89 cm by day; 0.52 cm against the 1/4 h reference, 0.54 cm against the 1 h one). The
  five PINN starts end on one estimate (0.003 cm). Days 0 - 1 remain 11.6 and 8.4 cm from the solver (27.8
  and 12.2 cm without the switch), so statement (i) is still not shown over the whole window (4.3 cm against
  the 1 cm criterion). The solver has one estimate from five starts there, so that difference is the PINN's;
  the sensor hardly sees those days (posterior sigma 44 and 16 cm in the PINN's inversion). From the well the
  estimate is 9.96 cm (r 0.91), the solver's 11.45 cm: neither meets (ii), as for every estimate at this
  station. Two things to read beside the row. The correction loop used its four iterations without meeting
  its stop rule (moves of 0.21 - 0.32 posterior sd, limit 0.2), and the scored estimate is the last one. Over
  the four iterations the distance to the solver on days 2 - 10 was 1.20, 1.09, 0.80 and 0.52 cm: the loop
  was still approaching the solver's estimate when it ended, and the first two are no closer than the run
  without the switch. What the switch changed in training: refinement rounds 2 - 4 ended with a PDE loss of
  8 - 106 and 7.5 - 11.1 % of the surface-flux error left, and at its own estimate the fast model agrees with
  the separately trained forward networks to 0.0003 - 0.0004 in soil moisture (0.0009 - 0.0013 without).
- IT-SR2: the sensor does not determine the water table at this station in this window. The model run with the
  measured well fits the sensor to 0.0093, the numerical estimate to 0.0087, the PINN's to 0.0094; both
  estimates read each wetting of the sensor as a rise of the water table while the well rises smoothly. The two
  estimates differ on days 4 - 7, where the solver goes to the 1.50 m bound of the path range; the solver's own
  cost of the PINN's path is within 0.6 % of its best. Step check of the reference: 24.5 - 27.1 % of the
  solver's 1 h steps do not converge in its worst solve, so the numerical inversion was repeated with 1/4 h
  steps from the hydrostatic start. It ends 0.49 cm from the 1 h estimate of the same start (largest 2.09 cm,
  in the first hours; 1.39 cm on day 0, 0.02 - 0.55 cm by day on days 1 - 10; 0.45 - 0.47 cm from the two
  other 1 h starts that agree with it), 40.20 cm from the well, with the same six coefficients on the bound;
  the 1 h reference stands, and the PINN is as far from the 1/4 h estimate (31.63 cm) as from the 1 h one
  (31.61 cm). The 40 cm from the well are therefore not an effect of the solver's step. Only this start was
  repeated: whether the 1.0 m start ends elsewhere with 1/4 h steps too was not run. With 1/4 h steps 14.1 %
  of the steps still do not converge in the worst solve (expected: under 5 %); that count is reported as it
  is, not explained. Seen in that run, not an expectation: its path at iteration 9 is 25.5 cm from the well
  at a cost 0.06 % above the final one (40.2 cm from the well). On days 5 - 6, where the solver's estimate is
  at the bound, its posterior sd is 9 - 10 m (12 - 13 m with 1/4 h steps); the PINN's is 13 - 15 cm on its
  own path.
- DK-Sor: no estimate is near the well. The well rises steadily from 1.25 to 0.38 m while the sensor stays
  within 0.311 - 0.347; the numerical inversion, the PINN and the hydrostatic reading are all about 40 cm off, and
  the model run with the measured well fits the sensor worse (0.0053) than either estimate (0.0024 numerical,
  0.0031 PINN). From day 4 on the PINN and the solver draw one curve (0.0 - 1.2 cm apart by day); on days 1 - 2
  they are 22 cm apart (both fall to the 1.50 m bound, the PINN a day earlier). The PINN run did not converge:
  its last stage alternated between two paths and the scored one is the path it was on when the four
  iterations were used up. The solver's own cost of that path is 0.240, 5.2 % above its minimum (0.228), while
  the PINN's model gives 0.390 on the same path: here part of the distance is the PINN's forward model at its
  own estimate, not only the search. Started on the PINN's path, the solver goes back to its own estimate in
  nine iterations (0.05 - 0.09 cm from each of its four starts, cost 0.228): the solver has one estimate from
  five starts, and the 22 cm on days 1 - 2 are the PINN's. This row shows neither statement over the whole
  window. Step check of the reference: 9 - 11 % of the solver's 1 h steps do not converge in its worst solve,
  so the numerical inversion was repeated with 1/4 h steps from the hydrostatic start. It ends 0.11 cm from
  the 1 h estimate of the same start (largest 0.28 cm; 0.09 - 0.20 cm from the other 1 h starts), 40.82 cm
  from the well, with the same seven coefficients on the bound; the 1 h reference stands, and the PINN is as
  far from the 1/4 h estimate (9.69 cm) as from the 1 h one (9.64 cm). The 40 cm from the well are therefore
  not an effect of the solver's step or start. With 1/4 h steps 7.9 % of the steps still do not converge in
  the worst solve (expected: under 1 %); that count is reported as it is, not explained.
- DK-Sor, the same window repeated with `sampling.local_box` (same first stage, one switch differs; its own
  row beside the first, as declared before launch). The switch repairs the training and the last stage, and
  the run still does not return the solver's estimate. Training: the refinement rounds 2 - 4 ended with a PDE
  loss of 1.6 - 19.1 and 4.9 - 6.3 % of the surface-flux error left (without the switch 93 - 851 and
  16.5 - 21.9 %), the forward networks of the correction loop with 3.5 - 5.4 %. The correction loop stopped by
  its rule at the third iteration (moves of 1.51, 0.64 and 0.14 of the smaller posterior sd; limit 0.2), where
  the first run alternated between two paths. All five PINN starts end on one estimate. That estimate is
  6.77 cm from the solver's over the window (6.77 - 6.89 cm against each of its six results), against the 1 cm
  of the criterion: 1.26 cm on day 0, 5.44, 11.07 and 7.93 cm on days 1 - 3, 1.61 and 1.87 cm on days 4 - 5, 13.85
  and 9.45 cm on days 6 - 7, 1.02, 0.61 and 0.48 cm on days 8 - 10. In the figure the two curves have one shape
  (down to the 1.50 m bound, up again between days 6 and 8) and the PINN is later than the solver on the way
  down and on the way up. Beside the first run: closer on days 0 - 2 (3.0, 22.3 and 22.5 cm before), farther on days
  3 - 7 (0.0 - 3.2 cm there before); the two PINN runs are 14.4 cm apart over the window. The three
  expectations written for these scores (at most 1.2 cm by day on days 4 - 10; below 10 cm on days 1 - 2;
  1 cm over the window) are all not met. In units of the solver's posterior sd the PINN is 0.04 - 0.5 sd away
  on days 0 - 6 (sd 10 - 40 cm there: the sensor hardly determines the water table on those days) and 1.05 sd
  on day 7 (sd 9 cm); that reading was computed after the scores and is not a criterion. From the well the run
  is 41.64 cm (r 0.26), the solver 40.88 cm: statement (ii) cannot be tested at this station. By the rule
  written before launch there is no further repeat here, and DK-Sor is not a second station for statement (i).
- FR-Bil: neither statement, and the weakest training of all runs. The sensor does not determine the water table
  in this window: the well rises steadily from 1.375 to 0.35 m while the sensor (0.14 - 0.25, against a fitted
  theta_s of 0.56) follows the rain. The model run with the measured well fits the sensor to 0.0072; the
  numerical estimates fit it to 0.0058 with paths 53 - 71 cm from the well, and the four numerical starts end
  4.9 - 53.4 cm apart, so there is no single numerical estimate for the PINN to return. The soil fit ran to its
  upper bound n = 10 (1.1 - 2.8 at the other four stations): a retention curve that is a step, with the profile
  at theta_r above a 24 cm fringe whatever the water table does. On this soil the networks do not learn the
  surface-flux condition: 38 - 43 % of the node rms of the flux is left after the wide stage and 16 - 19 % in
  the correction networks, which train on one path. Over the five stations that error grows with the n of the
  soil (wide stage: BE-Dor 4 %, DK-Sor 7 %, SE-Nor 8 - 9 %, IT-SR2 13 - 14 %, FR-Bil 38 - 43 %; n 1.13, 1.10,
  1.48, 2.78, 10.0). The PINN's five starts end 17.6 cm apart (the best at a cost 16 % below the other four),
  its estimate is 28 - 57 cm from the numerical end points, its sensor misfit is 0.0077, and the last stage
  used up its four iterations (0.26 sd against the limit 0.2). Started on the PINN's path, the solver does not
  stay on it and does not go back to one of its own end points either: it gives that path a cost of 1.405
  (5.5 % above its lowest, 1.332), lowers it to 1.379 and ends on a fifth path, 35.1 cm from the PINN's and
  23.0 - 40.2 cm from its other four. That the PINN is nearer the well than the solver (43.0 against 52.6 cm)
  says nothing for it: no estimate is tied to the well here.
- IT-SR2, second selection (1 Nov 2019, well 1.06 - 1.44 m): the numerical inversion is 24.8 - 25.4 cm from the
  well with eight or nine of its 14 coefficients on the bound of the path range (posterior sigma 20 cm). The 5 cm
  gate is not met and no PINN is run there.
- DK-Sor, second selection (23 Feb 2026, well 0.17 - 0.39 m): the four numerical starts end on four different
  paths, 48.2 - 108.2 cm from the well (48.2 cm at the lowest cost), each at the iteration limit. The 5 cm gate
  is not met and no PINN is run there.
- FR-Bil, second selection (12 Dec 2020, well 0.15 - 0.38 m): the four numerical starts end on one estimate
  (0.0 - 0.2 cm apart by day on days 0 - 9; 0.3 - 1.9 cm on the last half day, which the sensor does not
  determine: posterior sigma above 2 m). It is 7.25 - 7.75 cm from the well over the window (r 0.88, bias
  2.5 - 2.6 cm), so the 5 cm gate is not met; by day it is 0.4 - 3.3 cm from the well on days 0 - 7, 7.5 and
  6.4 cm on days 8 - 9 and 21 - 23 cm on day 10. A PINN run was started here all the same, declared as outside
  the gate and after these by-day numbers were read: it is the first of these windows after BE-Dor in which the
  solver has a single estimate, which is what statement (i) needs. It cannot show (ii) by the written
  criterion. In the worst solve of each search 38 - 55 % of the solver's 1 h steps did not converge, so the
  inversion was repeated with 1/4 h steps from the lowest-cost estimate: on days 0 - 9 it stays within 0.6 cm
  of it by day (0.0 - 0.02 cm on days 0 - 6), and the 1 h estimate stands as the reference. On the last half
  day the two step sizes differ by 12.7 cm (22 and 33 cm from the well; posterior sigma 2.5 and 3.1 m): there
  the solver's own estimate depends on its step, and the PINN's distance to it on that day is given against
  both. The PINN run of this window is the next item.
- FR-Bil, 12 Dec 2020, the PINN run (with `sampling.local_box`; the run without it is the next item). All
  five PINN starts end on one estimate (the same cost, 0.0002 cm apart). On days
  0 - 6 it is the numerical estimate to 0.02 - 0.07 cm by day, and it is 0.4 - 3.4 cm from the well there
  (2.04 cm over days 0 - 7; the numerical inversion 2.11 cm). On days 7 - 9 the two differ by 1.9, 4.3 and
  1.9 cm, so the written 1 cm of statement (i) is met neither over days 0 - 9 (1.6 cm) nor over the window
  (4.21 cm; the second was expected before the run, since the sensor does not determine the last half day).
  On that half day the PINN is 12.8 cm from the numerical estimate with 1 h steps and 1.3 cm from the one with
  1/4 h steps; against the 1/4 h estimate it is 1.38 cm over the window (largest 4 cm, on day 8). Statement
  (ii) does not hold: 10.52 cm from the well over the window (r 0.80, bias 2.81 cm), with 3.5 and 5.0 cm on
  days 8 - 9 and 33 cm on the last half day (the numerical inversion with 1/4 h steps: 10.58 cm). Limits: the
  window is outside the 5 cm gate and the run was started after the solver's by-day numbers were read; the
  hydrostatic reading of the sensor is nearer the well in this window (3.44 cm, r 0.93) than either inversion;
  four of the sixteen refinement networks ended with 18.9 - 21.5 % of the surface-flux error left, above the
  18.8 % written as the limit before the run (the other twelve 16.2 - 18.8 %). The correction loop stopped by
  its rule at its third iteration.
- FR-Bil, 12 Dec 2020, the same window without `sampling.local_box` (run first; reported beside the row, as
  declared before either was scored). On the days the sensor determines, the two runs are one path: they
  differ by 0 - 0.02 cm by day on days 0 - 6, this run is the numerical estimate to 0.02 - 0.08 cm by day
  there, and 2.05 cm from the well over days 0 - 7 (with the switch 2.04 cm). They part on days 8 - 9 (1.3 and
  1.8 cm) and on the last half day (14.3 cm). Over the window this run has the lower numbers against the well
  (6.43 cm, r 0.87, against 10.52 cm) and against the numerical estimate with 1 h steps (2.39 against
  4.21 cm), and the higher one against the estimate with 1/4 h steps (4.32 against 1.38 cm). That difference
  is not evidence for either setting: it sits where the sensor does not decide (posterior sigma of the PINN
  13 and 26 cm on the last half day; the two step sizes of the solver 12.7 cm apart there). Where the two
  settings do differ measurably is in training and in the loop: without the switch the refinement rounds end
  with 32 - 156 % of the surface-flux error left (16 - 22 % with it), and the correction loop was still moving
  at its limit of four iterations (0.51 of the posterior sd in its last segment, days 7.5 - 10; limit 0.2; the
  last iteration moved the path by 4.5 and 5.0 cm on days 9 and 10), where the run with the switch stopped by
  its rule. All five starts end on one estimate here too. The written 1 cm of statement (i) is not met in
  this run either (1.24 cm on days 0 - 9, 2.39 cm over the window), nor is (ii).
- SE-Nor, second selection (6 Dec 2019, well 0.15 - 0.41 m): inside the 5 cm gate. Three of the four numerical
  starts end on one estimate (0.01 cm apart), 4.35 cm from the well over the window (r 0.89, bias 0.26 cm). The
  1.0 m start ends on a second path with a cost 1.4 % higher, 9.53 cm from the well; it differs from the other
  three on day 0 only (20.8 cm there, 0.2 - 0.5 cm on days 1 - 2, at most 0.03 cm from day 3), where the sensor
  does not determine the water table (posterior sigma 19 cm). By day the lowest-cost estimate is 10.1, 5.7 and
  7.1 cm from the well on days 0 - 2 and 0.7 - 2.7 cm on days 3 - 10, so the 3 cm criterion of (ii) is not met
  over the window by the solver itself. In the worst solve of each search 59 - 61 % of the solver's 1 h steps
  did not converge, so the inversion was repeated with 1/4 h steps from the lowest-cost estimate: it stays
  0.19 cm from it over the window (at most 0.47 cm by day, on day 0; largest 0.66 cm), 4.35 cm from the well,
  and the 1 h estimate stands as the reference. One limit: the soil fit is held by three of its bounds (theta_s
  lower, alpha and Ks upper). The PINN run of this window is the next item.
- SE-Nor, 6 Dec 2019, the PINN run (with `sampling.local_box`; the run without it is the next item). On
  days 3 - 10 the PINN returns the numerical estimate to 0.19 cm
  (0.03 - 0.46 cm by day), and the two are 1.79 and 1.75 cm from the well there (the PINN 0.7 - 3.0 cm on each
  of those days, with a posterior sigma of 0.25 - 0.56 cm). On days 0 - 2 the PINN is 4.0, 5.2 and 7.0 cm
  from the solver and 12.9 cm from the well (solver 7.8 cm): the well rises from 0.41 to 0.22 m in those days
  and both inversions swing about it (read from the figure: PINN about 0.66, 0.24, 0.51 m; solver about 0.58,
  0.24, 0.40 m); the PINN's own posterior sigma there is 18, 3.8 and 7.1 cm. Over the window the row meets
  neither statement by the written criteria: 2.88 cm from the solver against 1 cm, and 6.87 cm from the well
  (r 0.83) against 3 cm (r 0.98), where the solver has 4.35 cm (r 0.89) and the hydrostatic reading of the
  sensor 6.77 cm. Four of the five PINN starts end on one estimate; the 1.0 m start ends on a second point at a
  cost 5.0 % higher, as the solver's 1.0 m start does (1.4 % higher), so "one estimate from every start" is
  not met either. The correction loop stopped by its rule at its second iteration (moves of 0.62 and 0.03
  posterior sd, limit 0.2). Training: the refinement rounds ended with a PDE loss of 6.7 - 20.4 and
  14.0 - 16.7 % of the surface-flux error left, the forward networks of the correction loop with 9.7 - 11.5 %.
  Two limits of this window: from day 3 the well is nearly flat (0.15 - 0.22 m), so the eight days in which
  everything agrees say little about following a moving water table; and the model run with the observed
  well does not follow the sensor after day 6. The split into days 0 - 2 and 3 - 10 was made after the by-day
  numbers were read; it describes the result and is not a criterion. The difference on days 0 - 2 is the
  PINN's own error, checked as at the first window: the solver's cost of the PINN's path is 0.34053 against
  0.332434 at its own estimate (2.4 % higher), and started on the PINN's path the numerical inversion goes
  back to its own estimate in 7 iterations (0.01 cm rms over the window, largest 0.06 cm; expectation and
  rule written before the run). So the PINN's path is not a second water table that the sensor admits in the
  solver. In units of the solver's posterior sd on those days (19.1, 2.9 and 3.4 cm) the PINN is 0.2, 1.8 and
  2.1 sd away: day 0 is hardly determined by the sensor (the 4 cm there are inside what it admits), days 1
  and 2 are not (5 - 7 cm against about 3 cm). The method is not changed on these scores.
  Where that error comes from (solver only, `scripts/fem_referee.py --value_only`; expectations and rule written
  before the run): on its own path the PINN's soil moisture at the sensor differs from the solver's by 1.3e-3
  rms over the window, and the difference sits at the start (3.46e-3, 1.51e-3, 1.19e-3 and 1.53e-3 on days
  0 - 3, 0.16e-3 - 0.39e-3 on days 4 - 10). This is why the PINN's cost at its estimate is 0.267108 where the solver has
  0.340526 for the same path. Of the solver's first step back from the PINN's path (linearised: -1.81 and
  -3.08 cm on days 1 and 2) the value difference alone makes -1.75 and -2.92 cm; what is left with the PINN's
  own value is -0.06 and -0.15 cm. So the error at the start of this window is the PINN's value at the sensor,
  not its sensitivity. That path goes from the shallow to the deep bound of the range within two knots, on a
  day the sensor hardly determines; the solver itself misses its tight tolerance on 52.6 % of the steps in the
  worst of the solves around it. The same check at BE-Dor (the control) gives a value difference of 0.06e-3
  and a first step of 0.04 cm. At the first SE-Nor window (with the switch) the value difference is flat
  (0.31e-3 - 0.58e-3 on every day) and the start of the window is something else: two days the sensor does not
  determine (solver sd 441.98 and 21.41 cm).
- SE-Nor, 6 Dec 2019, the same window without `sampling.local_box` (same first stage, one switch differs;
  reported beside the row, as declared before either estimate was read). The switch does here what it did at
  the first SE-Nor window. Without it the PINN is 4.65 cm from the solver over the window instead of 2.88 cm
  (days 0 - 2: 3.6, 9.2 and 11.9 cm instead of 4.0, 5.2 and 7.0 cm), 8.27 cm from the well (r 0.80) instead of
  6.87 cm, and the correction loop used its four iterations without meeting its stop rule (last move 0.30
  posterior sd, limit 0.2) instead of stopping at the second. On days 3 - 10 the two runs give the same
  curve: 0.22 and 0.19 cm from the solver, 1.78 and 1.79 cm from the well, at most 0.12 cm from each other
  by day. In training the run without the switch shows the refinement defect in full: its refinement rounds
  ended with a PDE loss of 307 - 359284 and 28 - 69 % of the surface-flux error left, the first pair of
  forward networks of the correction loop with 74 - 77 % left; at its own first estimate the fast model was
  0.0039 in soil moisture from the separately trained forward networks, against 0.00028 with the switch. Four
  of its five starts end on one estimate; the 1.0 m start ends elsewhere at 13.6 times the cost.
- Second selection, count: one window of five inside the 5 cm gate, SE-Nor (4.35 cm at the lowest cost of four
  starts). Outside: BE-Dor 5.41 cm, FR-Bil 7.61 cm, IT-SR2 and DK-Sor far outside.
- Found on these runs (`tasks/todo.md`, 06:57): the refinement rounds drew their local paths without keeping them
  inside the path range. Where days are poorly determined (SE-Nor, IT-SR2) the local distribution is far wider
  than the range, and most refinement paths lay outside what the inversion can evaluate. A switch
  (`sampling.local_box`, default off) projects the draws onto the range; the repeat of SE-Nor with it is the
  second SE-Nor row above, and it is the setting of every PINN run started since (the rule was written before
  that repeat was read). All other rows above were made without it and stand as run. DK-Sor shows the defect at its largest: after every
  inversion the local distribution was 1.1 - 29 wide (sd, coefficient units) in all 14 coefficients against a
  range 3.0 wide, and the refinement networks ended with flux errors of 17 - 22 % (wide stage 7 %).
- Found at DK-Sor (`tasks/todo.md`, 07:2x): the stop rule of the last stage divides the move of the estimate by
  the posterior sd of the new estimate. Where the new path lies on the deep bound that sd is about 90 cm, and
  a jump across the range counted as 0.24 - 0.26 sd, just above the limit of 0.2. The rule would have reported
  a two-path cycle as converged. It was not met here, and the runs that did stop by it are not of that kind
  (BE-Dor at 0.01 sd with sigma 1.3 cm and no coefficient on the bound; SE-Nor at 0.15 and IT-SR2 at 0.14, each
  with the same coefficients on the bound in its last two iterations, sigma 3.6 and 8.0 cm). The rule has been
  changed (15 Nov, 07:4x; `src/zbinv/convergence.py`, `scripts/param_invert.py`, `hpc/dc_loop.sh`): the move is
  now measured against the smaller of the posterior sds of the previous and the new estimate, so it is small only
  if both estimates call it small; the limit 0.2 is unchanged. Re-run on the stored iterations, everything else
  in the results identical: DK-Sor 5.8, 5.3, 3.9, 3.7 sd in its four iterations (the old rule gave 0.26 and
  0.24 for the second and fourth), with a different set of coefficients on the bound each time; BE-Dor 0.01,
  SE-Nor 0.15, IT-SR2 0.14, so the three stops stand under the new rule.

Why, and what follows. The window rule of this first selection took at every station the 11-day window with the
largest range of the well. That is the window in which the well comes up from more than a metre: FR-Bil 0.35 -
1.375 m, IT-SR2 0.39 - 1.06 m, SE-Nor 0.36 - 1.27 m, DK-Sor 0.375 - 1.25 m, so that for most hours the water
table is 0.5 - 1.3 m below a 0.05 m sensor in a sand or a till. The rule ranked by the signal in the well, not in
the sensor. The four windows are finished and reported as run. None of them shows (ii), and none shows (i) over
its whole window: the PINN and the solver agree on days 2 - 10 at SE-Nor and on days 4 - 10 at DK-Sor, on parts
of the window at IT-SR2, and not at FR-Bil.

Second selection, declared as such and reported next to the first, never instead of it (rule and expectations in
`tasks/todo.md` 74 aj, written before any record was read under it). Same five stations, pairs and screen; one
change: the windows are ranked by the smallest deepest level of the well (the well stays closest to the sensor),
and a window may not overlap the first one of its station.

| station | well + sensor | window (11 d from) | well (m) | sensor (m3/m3) |
|---|---|---|---|---|
| FR-Bil | WTD_1_1_1 + SWC_1_1_1 | 2020-12-12 12:00 | 0.15 - 0.38 | 0.106 - 0.153 |
| SE-Nor | WTD_1_1_1 + SWC_2_1_1 | 2019-12-06 12:00 | 0.15 - 0.41 | 0.180 - 0.212 |
| DK-Sor | WTD_18_1_1 + SWC_18_1_1 | 2026-02-23 12:00 | 0.17 - 0.39 | 0.110 - 0.261 |
| BE-Dor | WTD_2_1_1 + SWC_2_1_1 | 2024-03-17 12:00 | 0.18 - 0.65 | 0.510 - 0.547 |
| IT-SR2 | WTD_3_1_1 + SWC_4_1_1 | 2019-11-01 12:00 | 1.06 - 1.44 | 0.092 - 0.167 |

IT-SR2 has no shallow window free of its first one; it stays in the count of five. At DK-Sor the sensor of this
pair hardly follows the well (plots in `output/site_icos/<station>/result_records_shallow_*_test.png`). In this
selection the PINN is run only where the numerical inversion comes within 5 cm of the well: a selection on the
solver's result, made before any PINN result of that window exists, and reported with its counts. Path range of
both inversions: 0.10 - 1.50 m here (the well reaches 0.15 m), 0.20 - 1.50 m in the first selection. The rows
are in-sample (soil fitted on the same window).

The station files are ICOS ETC L2 products under the ICOS CC BY 4.0 data licence
(https://data.icos-cp.eu/licence). The raw files are not in this repository; the figures and numbers derived from
them are, and any use of them must cite:

- Domec, J.C., Loustau, D., Aluome, C., Chipeaux, C., Denou, J.-L., Depuydt, J., Garrigou, C., Kruszewski, A.,
  Lafont, S. (2026). ETC L2 Meteosens from Bilos, 2018-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/z3RB-bxCZ56yThkccIDCqxuM
- Domec, J.C., et al. (2026). ETC L2 Fluxnet (half-hourly) from Bilos, 2018-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/xFrwVCA1RxiQ4UyGinPgeqSZ
- Arriga, N., Goded, I., Dell'Acqua, A., Matteucci, M. (2026). ETC L2 Meteosens from San Rossore 2,
  2018-12-31–2025-12-31, ICOS RI, https://hdl.handle.net/11676/isAlyY_8bvG704lyz5clDp9c
- Arriga, N., et al. (2026). ETC L2 Fluxnet (half-hourly) from San Rossore 2, 2018-12-31–2025-12-31, ICOS RI,
  https://hdl.handle.net/11676/WS8nG-HsUoOnF77ONczKH-pA
- Kljun, N., Molder, M., Lehner, I., Bergström, G., Båth, A., Holst, J., Linderson, M.-L., Marchenko, S. (2026).
  ETC L2 Meteosens from Norunda, 2017-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/TSarBUGLFApGPGDtep9Ji9AV
- Kljun, N., et al. (2026). ETC L2 Fluxnet (half-hourly) from Norunda, 2017-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/Klxz5lfzgiDTWAumSw--KnsR
- Ibrom, A., Lange Rønn, E., Lodsgaard Justesen, M., Møller, F., Pilegaard, K., Schaarup Sørensen, J. (2026).
  ETC L2 Meteosens from Soroe, 2020-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/fte-hEuae8Vqr4sDn3XrFYnS
- Ibrom, A., et al. (2026). ETC L2 Fluxnet (half-hourly) from Soroe, 2020-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/DF5vN_t_JS_Mx_1YU8d0pbx0
- Heinesch, B., Bogaerts, G., Chopin, H., Clément, T., Demoulin, L., Faurès, A., Hoang, K., Longdoz, B.,
  Manise, T., Orgun, A. (2026). ETC L2 Meteosens from Dorinne, 2021-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/Mf1KLjlrMSIMjJcRr2aVDmo7
- Heinesch, B., et al. (2026). ETC L2 Fluxnet (half-hourly) from Dorinne, 2021-12-31–2026-08-31, ICOS RI,
  https://hdl.handle.net/11676/wzNQuCZPRUwFL-AFiLzzVWYq

("et al." stands for the same author list as the entry above it; the full strings are in the open metadata of
each object.)

## Tried and not adopted: a smoothness term in the cost of the inversion

Where the sensor says nothing about the water table (the first days at SE-Nor, 6 Dec 2019), the inversion is
free and its end point depends on the start. One remedy without a new number was tested, in the numerical
inversion only: add to the cost the density of the paths the networks are trained on (independent steps of the
control depths of the path, sd 0.15 m per knot), so that the estimate is the most probable path given the sensor
and that prior. The expectations and the rule were written before the runs (`tasks/todo.md`, "PLAN (00:18)"):
the term is taken into both inversions only if it leaves alone what the sensor determines (at most 0.2 cm on
days whose sd without the term is 1 cm or less, and at most 0.2 cm over the window at the two control sites),
gives one estimate from both starts at SE-Nor, and is not farther from a known truth on either seed.

Sixteen solver runs are compared (16 Nov; ten of them new): five windows without and with the term, a second start at SE-Nor,
and a twin with known truth (forcing, soil and sensor of SE-Nor, 6 Dec 2019) in two noise seeds.

- Where the sensor is silent the term does what it is meant to do. At SE-Nor, 6 Dec 2019, the two starts that
  end 6.26 cm apart without the term end 0.013 cm apart with it, 1.93 cm from the well (4.35 and 9.53 cm
  without). From the hydrostatic start FR-Bil, 12 Dec 2020, goes from 7.75 to 4.2 cm from the well and SE-Nor,
  30 Oct 2023, from 11.48 to 9.79 cm. No run with the term ends with a coefficient on its bound.
- Where the sensor decides, the term moves the estimate by more than allowed: on days whose sd without the term
  is 1 cm or less by 0.4 cm at BE-Dor (day 0), by 0.8 and 0.28 cm at SE-Nor, 6 Dec 2019 (days 3 and 4), and by
  0.63 and 0.39 cm on the twins (day 10); over the window by 0.29 cm at BE-Dor and 0.43 cm at US-NC2. Distance
  from the well there: BE-Dor 5.41 -> 5.46 cm, US-NC2 1.11 -> 1.15 cm.
- With known truth one seed ends nearer (1.54 -> 1.16 cm) and one farther (1.1 -> 1.52 cm).

By the rule the term is not used: every row of this file is without it, no PINN run was made with it, and the
first days at SE-Nor, 6 Dec 2019, stay as described above (the estimate there depends on the start). The switch
(`--path_prior` in `scripts/param_invert.py` and `scripts/fem_referee.py`) stays in the code, off by default;
its unit tests are `tests/test_path_prior.py`. All lines and verdicts: `tasks/todo.md`, "Step 3 ended (10:04,
16 Nov)".

### Second test, with known truth at every window (18 - 19 Nov): dropped for good

The first check had two cases with known truth, one nearer and one farther. The term was therefore tested once
more where the truth is known at all five windows, with the cases, the three criteria and the rule written and
pushed before any run (`tasks/todo.md`, "PLAN (00:49, 18 Nov)"; the scoring scripts
`scripts/twin_test_lines.jq` and `scripts/twin_test_verdict.jq` were committed before a result existed). For
each window a twin was made: the truth is the daily-knot fit of that window's well record, the sensor series is
the solver's on that path plus noise of sd 0.005, in two noise seeds not used before. Every case was inverted
without and with the term, and three of them from a second start with the term (23 solver runs). The twins have
no model error: the solver that makes them is the solver that inverts them.

- Distance from the truth over the window, with the term minus without (cm; the two seeds): BE-Dor +0.023,
  +0.128; US-NC2 +0.05, +0.011; SE-Nor, 6 Dec 2019, +0.53, -0.565; SE-Nor, 30 Oct 2023, +3.788, -2.142; FR-Bil
  -0.387, -0.262. With the term nearer the truth in 4 of 10 cases; mean +0.117 cm, limit +0.1 cm: **not met**.
- On the days the sensor determines (sd without the term 1 cm or less; 42 case-days) the term changes nothing
  on balance: 0.614 cm from the truth without it, 0.609 cm with it (limit: not more than 0.1 cm farther):
  **met**.
- One estimate from two starts with the term: 0.001 cm apart at SE-Nor, 30 Oct 2023, and 0.002 cm at FR-Bil,
  but 77.604 cm at SE-Nor, 6 Dec 2019 (limit 0.2 cm at each): **not met**. There the search started from a
  constant 1.0 m stopped after 7 iterations near its start, at a cost of 11.47 against 0.99 from the other
  start. On the real record of that window (first check) the same start had reached the other start's estimate.
- Reported, not a criterion: the truth lies within 2 sd of the estimate on 109 of 110 case-days without the
  term and on 105 of 110 with it.

All three had to hold. By the rule the term is dropped for good: no third test, the switch stays off, no PINN
run is made with it and no row of this file changes. Read after the verdicts and in no criterion: the mean in
the first item is decided by one case (SE-Nor, 30 Oct 2023, first seed); without that window it is -0.059 cm,
with the single cases between -0.565 and +0.53 cm. All lines and verdicts:
`tasks/todo.md`, "Step 3 of the known-answer test".

## Not yet in this set (running)

- ICOS, first selection: nothing is running any more. The 1/4 h repeats of the numerical inversion at IT-SR2
  and DK-Sor are finished (see their items above). The repeat of the DK-Sor PINN run with
  `sampling.local_box` is finished and is the row beside DK-Sor's first.
- ICOS, second selection, FR-Bil (12 Dec 2020; outside the gate, see above): nothing is running any more.
  Both PINN runs are finished: the one with `sampling.local_box` is the row of this window, the one without
  it the row beside it (both started from the same first training stage, which ended with 17.4 - 18.8 % of
  the surface-flux error left; expected 15 - 25 %).
- At SE-Nor (6 Dec 2019) the PINN run was started when three of the four numerical starts were in (declared
  before launch); with the fourth the window is inside the 5 cm gate (see above). Its
  first training stage was run in both network forms: the conductive form ended with 16.9 - 18.1 % of the
  surface-flux error left and continues; the additive form ended with 155 - 163 % left and was stopped, by a
  rule written before the two were compared. The refinement of this run draws most of its paths outside the
  range the inversion evaluates (the defect found at SE-Nor's first window; round 1 ends with a PDE loss of
  307 - 2417 against 11 at the end of the first stage). The switch that keeps the draws inside the range
  (`sampling.local_box`) held its loss expectation at SE-Nor's first window (PDE 8 - 106, flux error
  7.5 - 11.1 % in rounds 2 - 4, against 139 - 411829 and 18 - 47 % without it) and is the setting of every run
  started from now on. A second run with the switch was started here from the same first stage (15 Nov,
  12:20); it is finished and is the row above, and the run without the switch is finished and is the row
  beside it. The numerical inversion started on the PINN's path is finished too (see the item of the PINN
  run above); nothing is running at this window any more.

## Regenerate

```
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_sep23c_own --tag rfnc1 --final runs/rfnc1dc_it1 \
  --referee output/referee/nc2_own_sep23c_s0 output/referee/nc2_own_sep23c_s1 \
            output/referee/nc2_own_sep23c_s2 output/referee/nc2_own_sep23c_s3 \
  --forward runs/fwd_nc2c \
  --forward-reference output/paper/nc2_sep23c/forward_reference_d \
  --variants s1=runs/fs_nc2c_s1 s2=runs/fs_nc2c_s2 w64=runs/fs_nc2c_w64 w32=runs/fs_nc2c_w32 \
             d3=runs/fs_nc2c_d3 c2k=runs/fs_nc2c_c2k \
  --single member1=runs/rfnc1_inv4_m0 member2=runs/rfnc1_inv4_m1 member3=runs/rfnc1_inv4_m2 member4=runs/rfnc1_inv4_m3 \
  --repeats seed1000=runs/rfnc1s1dc_it1 seed2000=runs/rfnc1s2dc_it1 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_sep23c
```

The two other sets (no forward part, no repeats):

```
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_may23_own --tag rfnm1 --final runs/rfnm1dc_it1 \
  --referee output/referee/nc2_own_may23_s0 output/referee/nc2_own_may23_s1 \
            output/referee/nc2_own_may23_s2 output/referee/nc2_own_may23_s3 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_may23
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_sep23a_own_cal3 --tag rfna1 --final runs/rfna1dc_it1 \
  --referee output/referee/nc2_own_sep23a_cal3_s0 output/referee/nc2_own_sep23a_cal3_s1 \
            output/referee/nc2_own_sep23a_cal3_s2 output/referee/nc2_own_sep23a_cal3_s3 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_sep23a
```

The same two windows on paths of 0.10 - 1.50 m (numerical inversion with `--set prior.clip_m=[0.10,1.5]`):

```
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_may23_own --tag rfnm1b --final runs/rfnm1bdc_it1 \
  --referee output/referee/nc2_own_may23_box010_s0 output/referee/nc2_own_may23_box010_s1 \
            output/referee/nc2_own_may23_box010_s2 output/referee/nc2_own_may23_box010_s3 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_may23_b010
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_sep23a_own_cal3 --tag rfna1b --final runs/rfna1bdc_it1 \
  --referee output/referee/nc2_own_sep23a_cal3_box010_s0 output/referee/nc2_own_sep23a_cal3_box010_s1 \
            output/referee/nc2_own_sep23a_cal3_box010_s2 output/referee/nc2_own_sep23a_cal3_box010_s3 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_sep23a_b010
```

The forward reference itself (10 hours on two CPU cores, 6.7 of them for the 1/1024 h level; its `.npz` is
not in the repository; `forward_reference.json` is the earlier file that ends at 1/256 h):

```
sbatch ... hpc/run_script_cpu.slurm scripts/forward_reference_check.py \
  --data output/real_nc2_sep23c_own \
  --members baseline=runs/fwd_nc2c s1=runs/fs_nc2c_s1 s2=runs/fs_nc2c_s2 w64=runs/fs_nc2c_w64 \
            w32=runs/fs_nc2c_w32 d3=runs/fs_nc2c_d3 c2k=runs/fs_nc2c_c2k \
  --tol 1e-8 --max-iter 400 --tol-factor 0 \
  --levels 1:101 0.25:101 0.0625:101 0.0625:201 0.015625:101 0.00390625:101 0.0009765625:101 \
  --out output/paper/nc2_sep23c/forward_reference_d
```
