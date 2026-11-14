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
| soil moisture, all depths and times: RMS | 3.27e-04 |
| soil moisture, all depths and times: relative L2 | 0.080 % |
| soil moisture, all depths and times: RMS / range | 0.514 % |
| soil moisture, all depths and times: max | 5.82e-03 |
| soil moisture of the sensor layer: RMS | 4.26e-04 |
| soil moisture of the sensor layer: max | 4.54e-03 |
| surface flux: RMS error / RMS of the prescribed flux | 4.82 % |
| PDE residual / sigma_pde: RMS (max) | 0.659 (62.40) |
| held-out losses at the end: PDE, surface flux | 0.0292, 0.0274 |

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

## 9. Inversion with one member of round 4 at a time (against the well)

| member | RMSE (cm) | bias (cm) | max (cm) | r | from the estimate with all members, round 4, RMS (max) (cm) |
|---|---|---|---|---|---|
| member1 | 1.11 | +0.11 | 3.18 | 0.9903 | 0.01 (0.05) |
| member2 | 1.11 | +0.10 | 3.19 | 0.9903 | 0.02 (0.06) |
| member3 | 1.11 | +0.11 | 3.16 | 0.9905 | 0.02 (0.06) |
| member4 | 1.10 | +0.11 | 3.16 | 0.9905 | 0.02 (0.09) |
