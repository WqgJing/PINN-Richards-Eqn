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

State (15 Nov, 12:25). Five of the new windows have a finished PINN run: all four of the first selection and
one of the second. They are reported here as run and are not yet in the counts above (one SE-Nor check is
still going). All rows are in-sample (soil fitted on the same window with the well).

| station, window (11 d from) | selection | numerical inversion: from the well | PINN: from the well | PINN minus numerical | one estimate from every start |
|---|---|---|---|---|---|
| BE-Dor, 17 Mar 2024 | second; outside the 5 cm gate | 5.41 cm, r 0.913 | 5.41 cm, r 0.913 | 0.04 cm (largest 0.17 cm) | yes, both |
| SE-Nor, 30 Oct 2023 | first | 11.48 cm, r 0.92 | 9.30 cm, r 0.91 | 9.19 cm over the window; 1.07 cm on days 2 - 10 | yes, both |
| IT-SR2, 13 Nov 2019 | first | 40.23 cm, r 0.11 | 22.65 cm, r 0.44 | 31.61 cm over the window; 0.3 - 3.2 cm by day on days 0 - 3 and 8 - 10 | no (one of five PINN starts and one of four numerical starts end elsewhere) |
| DK-Sor, 8 Dec 2024 | first | 40.88 cm, r 0.32 | 42.71 cm, r 0.41 | 9.64 cm over the window; 0.0 - 1.2 cm by day on days 4 - 10; the PINN run did not converge | yes, both |
| FR-Bil, 29 Oct 2023 | first | 52.61 cm, r -0.52 (the other three starts 55.3 - 71.2 cm) | 43.03 cm, r -0.44 | 28.4 - 56.6 cm against the five numerical end points; training marginal, the PINN run did not converge | no (PINN starts 17.6 cm apart, numerical end points 4.9 - 53.4 cm apart) |

(RMS over the hourly record; r against the hourly well. Figures: `output/real_dor_mar24_own/result_rfdor1c.png`,
`output/real_nor_oct23_own/result_rfnor1c.png`, `output/real_sr2_nov19_own/result_rfsr21c.png`,
`output/real_sor_dec24_own/result_rfsor1c.png`, `output/real_bil_oct23_own/result_rfbil1c.png`.)

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
  Started on the PINN's path, the numerical inversion with 1 h steps returns to its own estimate, so at that step
  the difference on days 0 - 1 is the PINN's. 11 - 12 % of the solver's 1 h steps do not converge in its worst
  solve, so the numerical inversion was repeated with 1/4 h steps from the flat start: it ends 0.10 cm from the
  1 h estimate (largest 0.23 cm), 11.45 cm from the well, and the PINN is as far from it as from the 1 h one
  (9.18 cm over the window; 0.3 - 1.5 cm by day from day 2). The 1 h reference stands. The 1/4 h inversion
  started on the PINN's path is still running. Statement (i) is not shown over the whole window here.
- IT-SR2: the sensor does not determine the water table at this station in this window. The model run with the
  measured well fits the sensor to 0.0093, the numerical estimate to 0.0087, the PINN's to 0.0094; both
  estimates read each wetting of the sensor as a rise of the water table while the well rises smoothly. The two
  estimates differ on days 4 - 7, where the solver goes to the 1.50 m bound of the path range; the solver's own
  cost of the PINN's path is within 0.6 % of its best.
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
  window.
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
  criterion.
- Second selection, count: no window inside the 5 cm gate so far (BE-Dor 5.41 cm, FR-Bil 7.61 cm, IT-SR2 and
  DK-Sor far outside); SE-Nor is 4.35 cm from the well in three of four starts, the fourth still running.
- Found on these runs (`tasks/todo.md`, 06:57): the refinement rounds drew their local paths without keeping them
  inside the path range. Where days are poorly determined (SE-Nor, IT-SR2) the local distribution is far wider
  than the range, and most refinement paths lay outside what the inversion can evaluate. A switch
  (`sampling.local_box`, default off) projects the draws onto the range; a repeat of SE-Nor with it is running.
  All rows above were made without it and stand as run. DK-Sor shows the defect at its largest: after every
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

## Not yet in this set (running)

- ICOS, first selection: the 1/4 h repeats of the numerical inversion at DK-Sor and IT-SR2; the 1/4 h
  inversion started on the PINN's path at SE-Nor; the SE-Nor repeat with `sampling.local_box`.
- ICOS, second selection: the PINN run at FR-Bil (12 Dec 2020; outside the gate, see above) and a 1/4 h check
  of its numerical reference, both started 15 Nov, 10:29. Its first training stage ended with 17.4 - 18.8 % of
  the surface-flux error left (expected 15 - 25 %); refinement is running. The last start of the numerical
  inversion at SE-Nor is still running (the other four stations are done).
- At SE-Nor (6 Dec 2019) three of four starts end on one estimate 4.35 cm from the well, and a PINN run was
  started on those three (declared before launch; the 5 cm gate is evaluated when the fourth is in). Its
  first training stage was run in both network forms: the conductive form ended with 16.9 - 18.1 % of the
  surface-flux error left and continues; the additive form ended with 155 - 163 % left and was stopped, by a
  rule written before the two were compared. The refinement of this run draws most of its paths outside the
  range the inversion evaluates (the defect found at SE-Nor's first window; round 1 ends with a PDE loss of
  307 - 2417 against 11 at the end of the first stage). The switch that keeps the draws inside the range
  (`sampling.local_box`) held its loss expectation at SE-Nor's first window (PDE 8 - 106, flux error
  7.5 - 11.1 % in rounds 2 - 4, against 139 - 411829 and 18 - 47 % without it) and is the setting of every run
  started from now on. A second run with the switch was started here from the same first stage (15 Nov,
  12:20); it is the one this window will be reported by, the run without the switch beside it. No estimate
  exists yet.

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
