# Results set: water-table depth from surface flux and near-surface soil moisture

Built from one site window, US-NC2, 10 Sep 2023 12:00 + 11 days (hourly), PINN run `rfnc1`.
Figures and tables are written by `scripts/paper_figures.py`; every number below is in
[nc2_sep23c/metrics.json](nc2_sep23c/metrics.json) and laid out in [nc2_sep23c/tables.md](nc2_sep23c/tables.md).

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
| [fig6_physics](nc2_sep23c/fig6_physics.png) | physics at the estimate: prescribed against network surface flux; PDE residual over time and over depth |

## Main numbers

Forward problem (water table given; `runs/fwd_nc2c`, 47 min on one GPU), against the numerical solution:

| quantity | value |
|---|---|
| soil moisture over all depths and times, RMS | 3.3e-4 m3/m3 (relative L2 0.080 %, 0.51 % of the range) |
| largest error | 5.8e-3, at the surface during the rain pulses |
| sensor layer, RMS | 4.3e-4 |
| surface flux, RMS error / RMS of the prescribed flux | 4.8 % |

Inverse problem, hourly, against the well (265 hours):

| estimate | RMSE (cm) | bias (cm) | max (cm) | r |
|---|---|---|---|---|
| PINN | 1.10 | +0.11 | 3.16 | 0.990 |
| numerical inversion (same data and soil) | 1.11 | +0.11 | 3.13 | 0.990 |
| hydrostatic reading of the sensor (no flow) | 3.41 | +1.16 | 36.4 | 0.928 |
| spline fitted to the well (best a 14-coefficient path can do) | 0.81 | 0.00 | 3.46 | 0.995 |

- PINN minus numerical inversion: 0.07 cm RMS, 0.30 cm max.
- Five starts end on one estimate (largest difference 5e-7 cm). Posterior sd 1.1 cm (median).
- One network at a time instead of four (round 4): each estimate is within 0.01 - 0.02 cm RMS (max 0.09 cm)
  of the four-network estimate.
- Cost: 13.6 GPU hours of training (22 training runs) and 0.07 hours of inversions. The numerical inversion
  takes 26 min on a CPU for one start, 1.6 CPU hours for its four starts.

## What to say plainly in a text built on this

1. The PINN adds nothing to the distance from the well beyond what the numerical inversion has: 1.10 against
   1.11 cm. Of that, 0.81 cm is the spline's own distance from the hourly record.
2. On this window the wide stage is already at the estimate (0.10 cm from the numerical inversion); the
   refinement rounds and the value correction move it by 0.03 cm. This window does not show that those stages
   are needed. Other windows do (US-ONA Nov 2022: 1.28 cm after the wide stage, 0.52 after four rounds,
   0.10 after the value correction).
3. The PDE residual is not uniformly small. On a grid clustered at the surface its RMS is 0.5 - 0.7 sigma_pde
   for the forward networks and 0.9 - 1.5 for the operator members, with maxima of 45 - 130 sigma_pde at the
   three rain pulses and at t = 0. The held-out losses in fig4 are lower because they are sampled as in training.
4. The PINN is about 30 times more expensive than the numerical inversion on this window.

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
- (i) holds to 0.5 cm or better in 8 of 12 runs, at four sites, and in those the starts end on one estimate
  (largest difference below 1e-5 cm). It fails in four: US-PAS Jul 2021 (2.9 cm), US-ONA Nov 2021 (2.7 cm),
  the US-ONA rain window (28 cm; starts 1.8 cm apart, value-correction loop not converged) and the window of
  this results set with the soil of 1 Sep 2023 instead of its own (`rfnc1t`, 3.9 cm; the starts end on one
  estimate, but the value-correction loop used its four iterations with steps of 1.3 - 4.4 posterior sd and
  did not reach the stop rule of 0.2). The last one matters for how the in-sample case is read: on the same
  11 days of data, with the soil calibrated on the window that starts 9 days earlier, the PINN is 3.9 cm from
  the numerical inversion, and the numerical inversion is 3.05 cm from the well instead of 1.11.
- (ii) holds over the whole window (3 cm RMSE, 2 cm bias, r 0.98) only at US-NC2: in-sample (rfnc1) and with
  the May soil (rfnd, not the declared test). At US-ONA Nov 2022 with the Oct 2022 soil (rf22c, held-out) it
  holds for the 359 of 433 hours in which the sensor is below saturation and not over the whole window; on
  that recession the hydrostatic reading of the sensor does as well (1.93 cm below saturation), so the
  window does not show that the flow model adds skill.
- Where (ii) fails and (i) holds, the numerical inversion misses the well by the same amount: the limit there
  is the soil-column model and its soil, not the network.
- "Soil not usable" is the written rule: lowest-misfit calibration and an in-sample numerical inversion within
  2 cm of the well.

## Not yet in this set (running)

- Whole-pipeline repeats with other training seeds (`rfnc1s1`, `rfnc1s2`).
- Forward sensitivity to seed, width, depth and number of collocation points (`fs_nc2c_*`).
- Two further US-NC2 windows with their own soils (`rfnm1`, `rfna1`), both in-sample.
- The same window with the soil of 1 Sep 2023 and a second network form (`rfnc1tc`; `rfnc1t` is in the table above).

## Regenerate

```
sbatch ... hpc/run_script_cpu.slurm scripts/paper_figures.py \
  --data output/real_nc2_sep23c_own --tag rfnc1 --final runs/rfnc1dc_it1 \
  --referee output/referee/nc2_own_sep23c_s0 output/referee/nc2_own_sep23c_s1 \
            output/referee/nc2_own_sep23c_s2 output/referee/nc2_own_sep23c_s3 \
  --forward runs/fwd_nc2c \
  --single member1=runs/rfnc1_inv4_m0 member2=runs/rfnc1_inv4_m1 member3=runs/rfnc1_inv4_m2 member4=runs/rfnc1_inv4_m3 \
  --soil-note "calibrated on this window (in-sample)" --out output/paper/nc2_sep23c
```
