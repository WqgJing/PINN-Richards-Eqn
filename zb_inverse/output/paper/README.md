# Results set: water-table depth from surface flux and near-surface soil moisture

Built from one site window, US-NC2, 10 Sep 2023 12:00 + 11 days (hourly), PINN run `rfnc1`.
Figures and tables are written by `scripts/paper_figures.py`; every number below is in
[nc2_sep23c/metrics.json](nc2_sep23c/metrics.json) and laid out in [nc2_sep23c/tables.md](nc2_sep23c/tables.md).
The same figures and tables for the two further US-NC2 windows of "Other windows and sites" are in
[nc2_may23/](nc2_may23/tables.md) (`rfnm1`) and [nc2_sep23a/](nc2_sep23a/tables.md) (`rfna1`); those two sets
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
The numerical solution stored in the data set has a 1 h time step and is itself 4.2e-4 (sensor layer) from a
solve with a 1/256 h step ([forward_reference.json](nc2_sep23c/forward_reference.json),
`scripts/forward_reference_check.py`), so the network is given against both:

| quantity | PINN − numerical, 1/256 h step | PINN − numerical, 1 h step (the data set's) | numerical, 1 h step − 1/256 h step |
|---|---|---|---|
| all depths and times, RMS | 1.4e-5 (relative L2 0.0035 %, 0.022 % of the range) | 3.3e-4 (0.080 %, 0.51 %) | 3.3e-4 |
| all depths and times, max | 2.0e-4 | 5.8e-3 | 5.8e-3 |
| sensor layer, RMS | 1.7e-5 | 4.3e-4 | 4.2e-4 |
| sensor layer, max | 7.8e-5 | 4.5e-3 | 4.5e-3 |

- Surface flux of the network, RMS error / RMS of the prescribed flux, time mean on a 1-min grid: 1.65 %
  over the whole window, 1.46 % after the first hour (the network has zero surface flux at t = 0 by
  construction; point 3 below). The earlier version of this set gave 4.8 %: the same statistic on a
  half-hourly grid, on which every second point is a forcing node.
- The 1/256 h solve is not the limit either. It is 6.8e-6 (sensor layer; all depths 5.2e-6) from the 1/64 h
  solve, and the change from one step to the next falls by 3.0, 3.7 and 3.9 per refinement (first order in
  the step). So 1.7e-5 is the distance to a reference that is itself known to about 7e-6; it is not written
  as "the forward error is 1.7e-5". A 1/1024 h level is running.
- At a 1/16 h step, 201 nodes instead of 101 move the solution by 8e-7: the time step limits the reference,
  not the grid.
- The difference from the 1 h solution is the 1 h solver's step error, not the network's: in fig2 (lower
  panels) the curve "PINN − 1 h" lies on the curve "1 h − 1/256 h" in time, and over depth except close to
  the water table, where both are below 1e-5; the 5.8e-3 maximum is the solver's (5.8e-3), against 2.0e-4
  for the network.
- Other forward networks on the same problem ([tables.md](nc2_sep23c/tables.md), section 7), sensor layer,
  against the 1/256 h solve: two other seeds 1.75e-5 and 2.2e-5; 2048 instead of 8192 collocation points
  1.5e-5; width 64 5.5e-5; width 32 6.4e-5; 3 hidden layers instead of 6 5.4e-5. Against the 1 h solution
  all seven read 4.25e-4 to 4.32e-4, so that comparison cannot tell them apart.

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
   they are measured ([flux_grid.json](nc2_sep23c/flux_grid.json), `scripts/flux_grid_check.py`; the six
   networks behind the estimate: four operator members, two forward networks of the value correction).
   - The start. The network is hydrostatic with zero surface flux at t = 0 by construction, and the
     prescribed flux there is 11.3 mm/d. The residual of the row t = 0 (RMS over depth) is 19 - 33 sigma_pde
     for the members and 10 - 12 for the forward networks; it is about 1 after 10 min and 0.2 - 0.6 after
     20 min. The flux error is the full 11.3 mm/d at t = 0, below 0.9 mm/d from 2 min on and at most
     0.2 mm/d from 20 min on (sampled at 2, 5, 10, 20, 30 and 60 min).
   - After the first hour, on a 1-min grid: residual RMS 0.19 sigma_pde (members) and 0.18 (forward
     networks), maximum 7 - 12 sigma_pde, at the surface at t = 2.125 d. That is the forcing node at which
     the first rain pulse sets in (prescribed flux +1.5 mm/d at this node, -119 mm/d at the next).
     Surface flux, RMS error / RMS of the prescribed flux: 2.1 - 2.5 % (members), 2.0 and 1.7 %.
   - Forcing nodes. Both errors are larger at the hourly nodes of the prescribed flux than between them:
     residual 0.31 - 0.33 sigma_pde on the node rows against 0.17 on the half-hour rows, flux error
     7.5 - 7.6 % at the nodes against 1.6 - 1.8 % half-way between them (members). A half-hourly grid, on
     which every second row is a node, reads higher than a fine one: residual after the first hour
     0.25 - 0.26 on the half-hourly grid, 0.19 - 0.20 on a 5-min grid, 0.19 on the 1-min grid; flux
     5.2 - 5.3 % on the half-hourly grid, 2.3 - 2.6 % on the 1-min grid, 2.2 - 2.5 % on a 6-s grid.
   - Over the whole window the residual RMS is not a converged number: it is set by how many rows the grid
     puts into the first minutes (0.25 - 0.35 for the members on the 1-min grid). The tables therefore give
     the time after the first hour and the row t = 0 separately.
   - Correction of the earlier version of this set, which gave 0.5 - 1.5 sigma_pde on the half-hourly grid
     "with maxima of 45 - 130 sigma_pde at the three rain pulses and at t = 0". The maxima of the six
     networks over the whole window (47 - 130 sigma_pde) are at t = 0 for the members and at 1 - 2 min for
     the forward networks. "At the three rain pulses" had not been checked.
   - The held-out losses in fig4 are sampled as in training (10 % of the times log-uniform in the first
     day) and are not comparable with any of these figures.
4. The PINN is about 30 times more expensive than the numerical inversion with its 1 h step on this window
   (13.6 GPU hours against 26 CPU minutes for one start), 8 - 15 times more expensive than the numerical
   inversion with a 1/4 h step (55 - 102 CPU minutes for one start), and 5 times more expensive than one
   start with a 1/16 h step (160 CPU minutes), the step at which the two estimates agree to 0.014 cm.
5. A forward error quoted against the data set's own 1 h solution (3.3e-4) measures that solution's time
   step. The network's distance is 1.4e-5, against a 1/256 h solve that is itself 5e-6 from the level before.

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
| US-NC2 | 1 May 2023, 15 d | this window | rfnm1 | 0.03 | 0.89 / +0.04, 0.998 | 0.89 | in-sample; recession (3.6 mm of rain); flux on daily nodes, 5-day spline knots |
| US-NC2 | 1 Sep 2023, 6 d | this window | rfna1 | 0.09 | 0.64 / +0.11, 0.998 | 0.63 | in-sample; recession (0.3 mm of rain); flux on daily nodes, 5-day spline knots |
| US-NC2 | 27 Sep 2023, 9 d | May 2023 | rfnd | 0.06 | 1.59 / +0.55, 0.998 | 1.64 | held-out soil, but a diagnostic, not the declared test |
| US-NC2 | 27 Sep 2023, 9 d | 1 Sep 2023 | rfn1 | 0.07 | 9.48 / -9.25, 0.998 | 9.53 | the declared held-out test: fails against the well |
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
  Only `rfnc1` has been repeated at finer steps (0.07 cm at 1 h, 0.03 at 1/4 h, 0.014 at 1/16 h, see above).
- (i) holds to 0.5 cm or better in 10 of 14 runs, at four sites, and in those the starts end on one estimate
  (largest difference below 1e-5 cm). It fails in four: US-PAS Jul 2021 (2.9 cm), US-ONA Nov 2021 (2.7 cm),
  the US-ONA rain window (28 cm; starts 1.8 cm apart, value-correction loop not converged) and the window of
  this results set with the soil of 1 Sep 2023 instead of its own (`rfnc1t`, 3.9 cm; the starts end on one
  estimate, but the value-correction loop used its four iterations with steps of 1.3 - 4.4 posterior sd and
  did not reach the stop rule of 0.2; the held-out surface-flux loss of its operator networks ended every
  stage at 0.96 - 1.00 of its starting value, so these networks do not satisfy the flux condition).
  The last one matters for how the in-sample case is read: on the same 11 days of data, with the soil
  calibrated on the window that starts 9 days earlier, the PINN is 3.9 cm from the numerical inversion, and
  the numerical inversion is 3.05 cm from the well instead of 1.11.
- The two further US-NC2 windows (`rfnm1`, `rfna1`) are recessions with their own soils (Ks at its upper
  bound of 1e-3 m/s), a surface flux given on daily nodes and 5-day spline knots; they are further windows
  of the same site, in-sample, not new sites. Their rows show (i). How well their networks satisfy the
  equations, same statistics as point 3 above ([nc2_may23/tables.md](nc2_may23/tables.md),
  [nc2_sep23a/tables.md](nc2_sep23a/tables.md), section 6; six networks each):
  surface flux after the first hour 3.4 - 8.5 % of the RMS of the prescribed flux in `rfnm1` and
  1.3 - 2.2 % in `rfna1` (`rfnc1`: 1.7 - 2.5 %); PDE residual after the first hour 0.77 - 0.98 and
  0.88 - 1.20 sigma_pde, the same on half-hourly, 5-min and 1-min grids, with maxima of 10 - 16 at the water
  table (`rfnc1`: 0.18 - 0.19, maximum at the surface). There is no start corner in the residual of these
  two windows (row t = 0: 1.1 - 1.7 and 0.7 - 1.3). sigma_pde is derived from each data set (0.059 for
  `rfnc1`, 1.6e-4 and 3.9e-4 here), so the residual figures are not like-for-like between windows.
  The hydrostatic reading of the sensor with the same soil is 1.21 and 1.62 cm from the well (0.7 and
  1.5 cm too shallow on average). The numerical solution of these two data sets changes by 4e-6 and 3e-6
  at the sensor between a 1 h and a 1/4 h step (3e-4 on the window of this results set).
- (ii) holds over the whole window (3 cm RMSE, 2 cm bias, r 0.98) only at US-NC2: in-sample (rfnc1, rfnm1,
  rfna1) and with the May soil (rfnd, not the declared test). At US-ONA Nov 2022 with the Oct 2022 soil (rf22c, held-out) it
  holds for the 359 of 433 hours in which the sensor is below saturation and not over the whole window; on
  that recession the hydrostatic reading of the sensor does as well (1.93 cm below saturation), so the
  window does not show that the flow model adds skill.
- Where (ii) fails and (i) holds, the numerical inversion misses the well by the same amount: the limit there
  is the soil-column model and its soil, not the network.
- "Soil not usable" is the written rule: lowest-misfit calibration and an in-sample numerical inversion within
  2 cm of the well.

## Not yet in this set (running)

- A 1/1024 h level of the forward reference (`forward_reference_d`).
- A 12-s grid for the physics statistics of the tables (`flux_grid_fine`). So far: for the six networks
  behind the estimate the flux has been on a 6-s grid (within 5 % of the 1-min figure) and the residual on a
  5-min grid (within 4 %), not on a finer one; the seven forward networks of sections 2 and 7 of
  [tables.md](nc2_sep23c/tables.md) have been on the 1-min grid only.
- The numerical inversion of the two recession windows (`rfnm1`, `rfna1`) with a 1/4 h step.
- The same window with the soil of 1 Sep 2023 and a second network form (`rfnc1tc`; `rfnc1t` is in the table
  above). It is trained and inverted but not scored: `rfnc1t` was scored against a numerical inversion with a
  1 h step whose step error on that data set had not been measured, so a 1/4 h inversion of it runs first.

## Regenerate

```
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_sep23c_own --tag rfnc1 --final runs/rfnc1dc_it1 \
  --referee output/referee/nc2_own_sep23c_s0 output/referee/nc2_own_sep23c_s1 \
            output/referee/nc2_own_sep23c_s2 output/referee/nc2_own_sep23c_s3 \
  --forward runs/fwd_nc2c \
  --forward-reference output/paper/nc2_sep23c/forward_reference \
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

The forward reference itself (about 3.3 CPU hours; its `.npz` is not in the repository):

```
sbatch ... hpc/run_script_cpu.slurm scripts/forward_reference_check.py \
  --data output/real_nc2_sep23c_own \
  --members baseline=runs/fwd_nc2c s1=runs/fs_nc2c_s1 s2=runs/fs_nc2c_s2 w64=runs/fs_nc2c_w64 \
            w32=runs/fs_nc2c_w32 d3=runs/fs_nc2c_d3 c2k=runs/fs_nc2c_c2k \
  --tol 1e-8 --max-iter 400 --tol-factor 0 \
  --levels 1:101 0.25:101 0.0625:101 0.0625:201 0.015625:101 0.00390625:101 \
  --out output/paper/nc2_sep23c/forward_reference
```
