# Results tables: rfnm1 on output/real_nc2_may23_own

Written by scripts/paper_figures.py; every number is in metrics.json.

## 1. Data

| quantity | value |
|---|---|
| data set | output/real_nc2_may23_own |
| window | 2023-05-01T12:00, 15 days, 361 hourly values |
| soil-moisture sensor | 0 to 30 cm |
| soil (van Genuchten) | theta_s = 0.53809, theta_r = 8.985e-09, alpha = 5.5845, n = 1.2182, Ks = 0.001; calibrated on this window (in-sample) |
| observed water table | 0.288 to 0.731 m |
| observed soil moisture | 0.398 to 0.495 |
| surface flux over the window | 1.9 mm downward, 10.9 mm upward |
| root uptake over the window | 28.8 mm |
| water-table path | cubic B-spline, 6 coefficients; the spline fitted to the well is 0.44 cm RMS from the hourly record |

## 3. Inverse problem: water-table depth against the well (hourly)

| estimate | RMSE (cm) | bias (cm) | max (cm) | r |
|---|---|---|---|---|
| PINN | 0.89 | +0.04 | 2.57 | 0.9975 |
| numerical inversion (FEM, same data and soil) | 0.89 | +0.03 | 2.49 | 0.9975 |
| hydrostatic reading of the sensor (no flow) | 1.21 | -0.72 | 2.79 | 0.9973 |
| spline fitted to the well (best the path can do) | 0.44 | -0.00 | 1.57 | 0.9994 |

| quantity | value |
|---|---|
| PINN − numerical inversion | 0.03 cm RMS, 0.08 cm max |
| misfit of the soil moisture at the estimate (RMS) | 0.00146 |
| largest difference between the 5 starts | 1.8e-07 cm |
| posterior standard deviation (median) | 0.31 cm |
| numerical inversion: misfit of the soil moisture (RMS) | 0.00149 |
| numerical inversion: largest difference between its starts | 4.9e-04 cm |

## 4. The estimate stage by stage

| stage | from the well, RMS (cm) | from the numerical inversion, RMS (cm) | soil-moisture misfit | between starts, RMS (cm) | posterior sd, median (cm) |
|---|---|---|---|---|---|
| wide | 0.89 | 0.04 | 0.00145 | 1.5e-08 | 0.31 |
| round 1 | 0.89 | 0.04 | 0.00146 | 2.0e-12 | 0.31 |
| round 2 | 0.88 | 0.04 | 0.00145 | 5.7e-07 | 0.31 |
| round 3 | 0.88 | 0.04 | 0.00145 | 1.5e-07 | 0.31 |
| round 4 | 0.89 | 0.04 | 0.00145 | 6.2e-07 | 0.31 |
| value correction | 0.89 | 0.03 | 0.00146 | 4.2e-08 | 0.31 |

## 5. Training and cost

| stage | networks | steps each | GPU hours (all networks) | held-out PDE loss (median) | held-out flux loss (median) |
|---|---|---|---|---|---|
| wide | 4 | 42000 | 3.02 | 5.0799 | 33.9326 |
| round 1 | 4 | 18000 | 2.32 | 3.0083 | 19.5849 |
| round 2 | 4 | 18000 | 1.88 | 2.0846 | 13.3203 |
| round 3 | 4 | 18000 | 1.61 | 1.5417 | 10.3508 |
| round 4 | 4 | 18000 | 1.77 | 1.1698 | 9.6405 |
| forward networks of the value correction | 2 | 28000 | 1.21 | 0.6889 | 4.8262 |

| item | cost |
|---|---|
| training, all stages | 11.8 GPU hours |
| inversions of all stages | 0.05 hours |
| numerical inversion | 79 min for the start with the lowest cost (9 iterations); 3.63 CPU hours for all 4 starts |

## 6. Physics of the trained networks at the estimate

| network | surface flux: RMS error / RMS of the prescribed (time mean, 1-min grid), whole window | surface flux, after the first hour | PDE residual / sigma_pde (1-min grid), after the first hour: RMS | max | PDE residual / sigma_pde, the row t = 0: RMS |
|---|---|---|---|---|---|
| operator member 1 | 6.91 % | 5.71 % | 0.977 | 12.11 | 1.06 |
| operator member 2 | 10.52 % | 8.46 % | 0.866 | 16.11 | 1.06 |
| operator member 3 | 4.11 % | 3.39 % | 0.787 | 9.56 | 1.71 |
| operator member 4 | 7.41 % | 6.48 % | 0.767 | 11.96 | 1.08 |
| forward network 1 of the value correction | 7.24 % | 6.56 % | 0.820 | 13.68 | 1.45 |
| forward network 2 of the value correction | 8.54 % | 7.74 % | 0.811 | 13.46 | 1.20 |
