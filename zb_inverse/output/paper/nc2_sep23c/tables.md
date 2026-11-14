# Results tables: rfnc1 on output/real_nc2_sep23c_own

Written by scripts/paper_figures.py; every number is in metrics.json.

## 1. Data

| quantity | value |
|---|---|
| data set | output/real_nc2_sep23c_own |
| window | 2023-09-10T12:00, 11 days, 265 hourly values |
| soil-moisture sensor | 0 to 30 cm |
| soil (van Genuchten) | theta_s = 0.41935, theta_r = 5.6624e-09, alpha = 0.82231, n = 2.7469, Ks = 1.9228e-06; calibrated on this window (in-sample) |
| observed water table | 0.472 to 0.768 m |
| observed soil moisture | 0.379 to 0.428 |
| surface flux over the window | 20.7 mm downward, 12.6 mm upward |
| root uptake over the window | 23.7 mm |
| water-table path | cubic B-spline, 14 coefficients; the spline fitted to the well is 0.81 cm RMS from the hourly record |

## 2. Forward problem: PINN against the numerical solution (water table given)

| quantity | value |
|---|---|
| run | runs/fwd_nc2c |
| network | 38 inputs, 6 hidden layers of 128, tanh, 87681 parameters |
| training | 28000 steps, 47 min on one GPU |
| surface flux: RMS error / RMS of the prescribed flux | 4.82 % |
| PDE residual / sigma_pde: RMS (max) | 0.659 (62.40) |
| held-out losses at the end: PDE, surface flux | 0.0292, 0.0274 |


Soil-moisture difference (m3/m3). The numerical solution of the data set has a 1 h time step; the finest level of scripts/forward_reference_check.py has 1/256 h (101 nodes, Picard tolerance 1e-08).

| quantity | PINN − numerical, 1/256 h step | PINN − numerical, 1 h step (the data set's) | numerical, 1 h step − 1/256 h step |
|---|---|---|---|
| all depths and times: RMS | 1.42e-05 | 3.27e-04 | 3.25e-04 |
| all depths and times: relative L2 | 0.0035 % | 0.08 % | 0.08 % |
| all depths and times: RMS / range | 0.022 % | 0.51 % | 0.51 % |
| all depths and times: max | 2.01e-04 | 5.82e-03 | 5.83e-03 |
| sensor layer: RMS | 1.72e-05 | 4.26e-04 | 4.24e-04 |
| sensor layer: max | 7.77e-05 | 4.54e-03 | 4.54e-03 |


The numerical solution against itself, soil moisture, RMS (max). Picard tolerance 1e-08 at every level (the data set's: 1e-05); the first level differs from the solve stored in the data set by 1.8e-06 RMS at the sensor.

| time step | nodes | Picard iterations, max | seconds | minus the finest level: sensor layer | all depths | minus the level before: sensor layer | all depths | change of the refinement before / this change (sensor layer) |
|---|---|---|---|---|---|---|---|---|
| 1 h | 101 | 24 | 54 | 4.24e-04 (4.54e-03) | 3.25e-04 (5.83e-03) | - | - | - |
| 1/4 h | 101 | 22 | 191 | 1.31e-04 (1.39e-03) | 1.00e-04 (1.78e-03) | 2.95e-04 (3.14e-03) | 2.26e-04 (4.05e-03) | - |
| 1/16 h | 101 | 20 | 661 | 3.32e-05 (3.50e-04) | 2.56e-05 (4.45e-04) | 9.75e-05 (1.04e-03) | 7.49e-05 (1.33e-03) | 3.02 |
| 1/16 h | 201 | 20 | 1243 | 3.33e-05 (3.51e-04) | 2.56e-05 (4.48e-04) | 8.02e-07 (1.42e-06) | 1.21e-07 (3.56e-06) | - |
| 1/64 h | 101 | 18 | 2261 | 6.77e-06 (7.09e-05) | 5.21e-06 (9.00e-05) | 2.65e-05 (2.80e-04) | 2.04e-05 (3.58e-04) | 3.68 |
| 1/256 h | 101 | 17 | 7527 | - | - | 6.77e-06 (7.09e-05) | 5.21e-06 (9.00e-05) | 3.92 |

## 3. Inverse problem: water-table depth against the well (hourly)

| estimate | RMSE (cm) | bias (cm) | max (cm) | r |
|---|---|---|---|---|
| PINN | 1.10 | +0.11 | 3.16 | 0.9905 |
| numerical inversion (FEM, same data and soil) | 1.11 | +0.11 | 3.13 | 0.9904 |
| hydrostatic reading of the sensor (no flow) | 3.41 | +1.16 | 36.39 | 0.9280 |
| spline fitted to the well (best the path can do) | 0.81 | -0.00 | 3.46 | 0.9948 |

| quantity | value |
|---|---|
| PINN − numerical inversion | 0.07 cm RMS, 0.30 cm max |
| misfit of the soil moisture at the estimate (RMS) | 0.00134 |
| largest difference between the 5 starts | 4.9e-07 cm |
| posterior standard deviation (median) | 1.11 cm |
| numerical inversion: misfit of the soil moisture (RMS) | 0.00151 |
| numerical inversion: largest difference between its starts | 2.2e-03 cm |

## 4. The estimate stage by stage

| stage | from the well, RMS (cm) | from the numerical inversion, RMS (cm) | soil-moisture misfit | between starts, RMS (cm) | posterior sd, median (cm) |
|---|---|---|---|---|---|
| wide | 1.09 | 0.10 | 0.00135 | 1.8e-06 | 1.08 |
| round 1 | 1.11 | 0.06 | 0.00134 | 5.0e-11 | 1.11 |
| round 2 | 1.11 | 0.07 | 0.00134 | 2.6e-06 | 1.11 |
| round 3 | 1.11 | 0.06 | 0.00134 | 1.4e-06 | 1.11 |
| round 4 | 1.11 | 0.07 | 0.00134 | 8.1e-07 | 1.11 |
| value correction | 1.10 | 0.07 | 0.00134 | 1.6e-07 | 1.11 |

## 5. Training and cost

| stage | networks | steps each | GPU hours (all networks) | held-out PDE loss (median) | held-out flux loss (median) |
|---|---|---|---|---|---|
| wide | 4 | 42000 | 4.53 | 0.4113 | 0.3113 |
| round 1 | 4 | 18000 | 2.25 | 0.2144 | 0.1360 |
| round 2 | 4 | 18000 | 2.15 | 0.1601 | 0.1163 |
| round 3 | 4 | 18000 | 2.19 | 0.1305 | 0.0951 |
| round 4 | 4 | 18000 | 1.64 | 0.1343 | 0.0980 |
| forward networks of the value correction | 2 | 28000 | 0.85 | 0.0284 | 0.0337 |

| item | cost |
|---|---|
| training, all stages | 13.6 GPU hours |
| inversions of all stages | 0.07 hours |
| numerical inversion | 26 min for the start with the lowest cost (7 iterations); 1.59 CPU hours for all 4 starts |

## 6. Physics of the trained networks at the estimate

| network | surface flux: RMS error / RMS of the prescribed | PDE residual / sigma_pde: RMS | max |
|---|---|---|---|
| operator member 1 | 5.20 % | 1.477 | 129.45 |
| operator member 2 | 5.25 % | 1.150 | 102.27 |
| operator member 3 | 5.34 % | 1.207 | 114.54 |
| operator member 4 | 5.29 % | 0.874 | 77.10 |
| forward network 1 of the value correction | 4.99 % | 0.563 | 49.24 |
| forward network 2 of the value correction | 4.87 % | 0.496 | 44.49 |

## 7. Forward PINN: training seed, network size, collocation points

Soil moisture RMS against the numerical solution (m3/m3).

| run | width | hidden layers | collocation points | seed | parameters | all depths, 1/256 h step | sensor layer, 1/256 h step | all depths, 1 h step (the data set's) | sensor layer, 1 h step (the data set's) | sensor layer, minus the baseline network | surface flux | PDE residual RMS | minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 128 | 6 | 8192 | 0 | 87681 | 1.42e-05 | 1.72e-05 | 3.27e-04 | 4.26e-04 | - | 4.82 % | 0.659 | 47 |
| s1 | 128 | 6 | 8192 | 1 | 87681 | 1.40e-05 | 1.75e-05 | 3.27e-04 | 4.26e-04 | 1.90e-05 | 4.72 % | 0.728 | 47 |
| s2 | 128 | 6 | 8192 | 2 | 87681 | 1.78e-05 | 2.16e-05 | 3.28e-04 | 4.27e-04 | 2.08e-05 | 4.95 % | 0.549 | 47 |
| w64 | 64 | 6 | 8192 | 0 | 23361 | 4.26e-05 | 5.53e-05 | 3.31e-04 | 4.32e-04 | 5.47e-05 | 6.02 % | 0.727 | 48 |
| w32 | 32 | 6 | 8192 | 0 | 6561 | 4.89e-05 | 6.35e-05 | 3.30e-04 | 4.30e-04 | 6.30e-05 | 6.29 % | 1.193 | 44 |
| d3 | 128 | 3 | 8192 | 0 | 38145 | 4.28e-05 | 5.35e-05 | 3.29e-04 | 4.29e-04 | 5.35e-05 | 6.36 % | 1.132 | 40 |
| c2k | 128 | 6 | 2048 | 0 | 87681 | 1.32e-05 | 1.52e-05 | 3.27e-04 | 4.25e-04 | 1.72e-05 | 4.83 % | 0.686 | 45 |

## 8. Repeated runs with other training seeds (against the well)

| run | RMSE (cm) | bias (cm) | max (cm) | r | from the numerical inversion, RMS (cm) | from the first run, RMS (max) (cm) |
|---|---|---|---|---|---|---|
| first run | 1.10 | +0.11 | 3.16 | 0.9905 | - | - |
| seed1000 | 1.10 | +0.11 | 3.15 | 0.9905 | 0.06 | 0.00 (0.02) |
| seed2000 | 1.10 | +0.11 | 3.16 | 0.9905 | 0.06 | 0.01 (0.02) |

## 9. Inversion with one member of round 4 at a time (against the well)

| member | RMSE (cm) | bias (cm) | max (cm) | r | from the estimate with all members, round 4, RMS (max) (cm) |
|---|---|---|---|---|---|
| member1 | 1.11 | +0.11 | 3.18 | 0.9903 | 0.01 (0.05) |
| member2 | 1.11 | +0.10 | 3.19 | 0.9903 | 0.02 (0.06) |
| member3 | 1.11 | +0.11 | 3.16 | 0.9905 | 0.02 (0.06) |
| member4 | 1.10 | +0.11 | 3.16 | 0.9905 | 0.02 (0.09) |
