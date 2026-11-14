# Results tables: rfna1b on output/real_nc2_sep23a_own_cal3

Written by scripts/paper_figures.py; every number is in metrics.json.

## 1. Data

| quantity | value |
|---|---|
| data set | output/real_nc2_sep23a_own_cal3 |
| window | 2023-09-01T12:00, 6 days, 145 hourly values |
| soil-moisture sensor | 0 to 30 cm |
| soil (van Genuchten) | theta_s = 0.54931, theta_r = 0.19853, alpha = 14.803, n = 1.2896, Ks = 0.00099968; calibrated on this window (in-sample) |
| observed water table | 0.218 to 0.533 m |
| observed soil moisture | 0.404 to 0.499 |
| surface flux over the window | 0.0 mm downward, 5.6 mm upward |
| root uptake over the window | 18.6 mm |
| water-table path | cubic B-spline, 5 coefficients; the spline fitted to the well is 0.47 cm RMS from the hourly record |

## 3. Inverse problem: water-table depth against the well (hourly)

| estimate | RMSE (cm) | bias (cm) | max (cm) | r |
|---|---|---|---|---|
| PINN | 0.56 | +0.10 | 1.52 | 0.9983 |
| numerical inversion (FEM, same data and soil) | 0.53 | +0.00 | 1.27 | 0.9984 |
| hydrostatic reading of the sensor (no flow) | 1.62 | -1.50 | 2.34 | 0.9983 |
| spline fitted to the well (best the path can do) | 0.47 | -0.00 | 0.89 | 0.9988 |

| quantity | value |
|---|---|
| PINN − numerical inversion | 0.11 cm RMS, 0.31 cm max |
| misfit of the soil moisture at the estimate (RMS) | 0.00184 |
| largest difference between the 5 starts | 6.7e-07 cm |
| posterior standard deviation (median) | 0.28 cm |
| numerical inversion: misfit of the soil moisture (RMS) | 0.00181 |
| numerical inversion: largest difference between its starts | 1.9e-03 cm |

## 4. The estimate stage by stage

| stage | from the well, RMS (cm) | from the numerical inversion, RMS (cm) | soil-moisture misfit | between starts, RMS (cm) | posterior sd, median (cm) |
|---|---|---|---|---|---|
| wide | 0.56 | 0.11 | 0.00184 | 7.1e-07 | 0.28 |
| round 1 | 0.57 | 0.12 | 0.00184 | 1.2e-10 | 0.28 |
| round 2 | 0.57 | 0.13 | 0.00184 | 9.5e-07 | 0.28 |
| round 3 | 0.57 | 0.13 | 0.00184 | 1.6e-06 | 0.28 |
| round 4 | 0.57 | 0.13 | 0.00184 | 3.3e-07 | 0.28 |
| value correction | 0.56 | 0.11 | 0.00184 | 2.4e-07 | 0.28 |

## 5. Training and cost

| stage | networks | steps each | GPU hours (all networks) | held-out PDE loss (median) | held-out flux loss (median) |
|---|---|---|---|---|---|
| wide | 4 | 42000 | 3.11 | 0.7383 | 3.0733 |
| round 1 | 4 | 18000 | 1.94 | 0.9056 | 2.9925 |
| round 2 | 4 | 18000 | 1.68 | 0.8387 | 2.1356 |
| round 3 | 4 | 18000 | 1.69 | 0.8890 | 2.1969 |
| round 4 | 4 | 18000 | 1.12 | 0.8424 | 2.0204 |
| forward networks of the value correction | 2 | 28000 | 1.60 | 0.4341 | 1.2181 |

| item | cost |
|---|---|
| training, all stages | 11.1 GPU hours |
| inversions of all stages | 0.03 hours |
| numerical inversion | 78 min for the start with the lowest cost (11 iterations); 2.96 CPU hours for all 4 starts |

## 6. Physics of the trained networks at the estimate

| network | surface flux: RMS error / RMS of the prescribed (time mean, 1-min grid), whole window | surface flux, after the first hour | PDE residual / sigma_pde (1-min grid), after the first hour: RMS | max | PDE residual / sigma_pde, the row t = 0: RMS |
|---|---|---|---|---|---|
| operator member 1 | 2.08 % | 1.31 % | 0.959 | 8.23 | 1.17 |
| operator member 2 | 2.17 % | 1.57 % | 1.122 | 9.04 | 1.07 |
| operator member 3 | 3.07 % | 2.20 % | 0.882 | 8.23 | 1.35 |
| operator member 4 | 2.56 % | 2.02 % | 1.380 | 11.86 | 1.07 |
| forward network 1 of the value correction | 3.40 % | 2.30 % | 0.830 | 8.86 | 1.31 |
| forward network 2 of the value correction | 2.66 % | 1.91 % | 0.978 | 14.09 | 1.35 |
