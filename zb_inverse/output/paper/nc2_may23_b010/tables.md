# Results tables: rfnm1b on output/real_nc2_may23_own

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
| PINN | 0.89 | +0.03 | 3.23 | 0.9975 |
| numerical inversion (FEM, same data and soil) | 0.90 | +0.02 | 3.26 | 0.9975 |
| hydrostatic reading of the sensor (no flow) | 1.21 | -0.72 | 2.79 | 0.9973 |
| spline fitted to the well (best the path can do) | 0.44 | -0.00 | 1.57 | 0.9994 |

| quantity | value |
|---|---|
| PINN − numerical inversion | 0.03 cm RMS, 0.13 cm max |
| misfit of the soil moisture at the estimate (RMS) | 0.00105 |
| largest difference between the 5 starts | 7.2e-08 cm |
| posterior standard deviation (median) | 0.31 cm |
| numerical inversion: misfit of the soil moisture (RMS) | 0.00105 |
| numerical inversion: largest difference between its starts | 1.9e-04 cm |

## 4. The estimate stage by stage

| stage | from the well, RMS (cm) | from the numerical inversion, RMS (cm) | soil-moisture misfit | between starts, RMS (cm) | posterior sd, median (cm) |
|---|---|---|---|---|---|
| wide | 0.89 | 0.04 | 0.00104 | 3.4e-08 | 0.31 |
| round 1 | 0.89 | 0.03 | 0.00105 | 3.3e-13 | 0.31 |
| round 2 | 0.89 | 0.03 | 0.00104 | 4.9e-07 | 0.31 |
| round 3 | 0.89 | 0.04 | 0.00104 | 3.1e-08 | 0.31 |
| round 4 | 0.90 | 0.05 | 0.00104 | 1.1e-08 | 0.31 |
| value correction | 0.89 | 0.03 | 0.00105 | 1.8e-08 | 0.31 |

## 5. Training and cost

| stage | networks | steps each | GPU hours (all networks) | held-out PDE loss (median) | held-out flux loss (median) |
|---|---|---|---|---|---|
| wide | 4 | 42000 | 3.07 | 4.9039 | 54.3816 |
| round 1 | 4 | 18000 | 1.66 | 4.6371 | 29.6497 |
| round 2 | 4 | 18000 | 1.39 | 3.5227 | 25.7007 |
| round 3 | 4 | 18000 | 1.40 | 3.2229 | 22.2715 |
| round 4 | 4 | 18000 | 1.38 | 2.4982 | 19.4075 |
| forward networks of the value correction | 2 | 28000 | 1.58 | 0.9794 | 4.9158 |

| item | cost |
|---|---|
| training, all stages | 10.5 GPU hours |
| inversions of all stages | 0.05 hours |
| numerical inversion | 60 min for the start with the lowest cost (6 iterations); 4.16 CPU hours for all 4 starts |

## 6. Physics of the trained networks at the estimate

| network | surface flux: RMS error / RMS of the prescribed (time mean, 1-min grid), whole window | surface flux, after the first hour | PDE residual / sigma_pde (1-min grid), after the first hour: RMS | max | PDE residual / sigma_pde, the row t = 0: RMS |
|---|---|---|---|---|---|
| operator member 1 | 12.62 % | 10.83 % | 1.170 | 21.63 | 1.82 |
| operator member 2 | 13.07 % | 11.17 % | 1.122 | 20.34 | 1.81 |
| operator member 3 | 14.07 % | 12.23 % | 1.028 | 29.41 | 1.77 |
| operator member 4 | 6.45 % | 5.66 % | 0.834 | 11.22 | 1.87 |
| forward network 1 of the value correction | 8.43 % | 7.63 % | 1.105 | 19.48 | 2.04 |
| forward network 2 of the value correction | 7.90 % | 7.31 % | 0.810 | 11.66 | 2.12 |
