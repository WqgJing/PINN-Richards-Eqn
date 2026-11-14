# Results tables: rfna1 on output/real_nc2_sep23a_own_cal3

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
| PINN | 0.64 | +0.11 | 1.94 | 0.9977 |
| numerical inversion (FEM, same data and soil) | 0.63 | +0.02 | 1.82 | 0.9978 |
| hydrostatic reading of the sensor (no flow) | 1.62 | -1.50 | 2.34 | 0.9983 |
| spline fitted to the well (best the path can do) | 0.47 | -0.00 | 0.89 | 0.9988 |

| quantity | value |
|---|---|
| PINN − numerical inversion | 0.09 cm RMS, 0.13 cm max |
| misfit of the soil moisture at the estimate (RMS) | 0.00195 |
| largest difference between the 5 starts | 3.5e-09 cm |
| posterior standard deviation (median) | 0.28 cm |
| numerical inversion: misfit of the soil moisture (RMS) | 0.00194 |
| numerical inversion: largest difference between its starts | 5.4e-04 cm |

## 4. The estimate stage by stage

| stage | from the well, RMS (cm) | from the numerical inversion, RMS (cm) | soil-moisture misfit | between starts, RMS (cm) | posterior sd, median (cm) |
|---|---|---|---|---|---|
| wide | 0.64 | 0.10 | 0.00195 | 1.7e-07 | 0.28 |
| round 1 | 0.65 | 0.10 | 0.00195 | 2.8e-12 | 0.28 |
| round 2 | 0.65 | 0.10 | 0.00195 | 5.4e-07 | 0.28 |
| round 3 | 0.65 | 0.11 | 0.00195 | 4.7e-07 | 0.28 |
| round 4 | 0.65 | 0.11 | 0.00195 | 4.2e-08 | 0.28 |
| value correction | 0.64 | 0.09 | 0.00195 | 7.5e-10 | 0.28 |

## 5. Training and cost

| stage | networks | steps each | GPU hours (all networks) | held-out PDE loss (median) | held-out flux loss (median) |
|---|---|---|---|---|---|
| wide | 4 | 42000 | 2.54 | 0.8844 | 2.2796 |
| round 1 | 4 | 18000 | 2.12 | 0.8852 | 2.2993 |
| round 2 | 4 | 18000 | 1.11 | 0.8369 | 1.8846 |
| round 3 | 4 | 18000 | 1.38 | 0.8615 | 2.2099 |
| round 4 | 4 | 18000 | 1.90 | 0.8005 | 2.0012 |
| forward networks of the value correction | 2 | 28000 | 1.23 | 0.4023 | 0.9934 |

| item | cost |
|---|---|
| training, all stages | 10.3 GPU hours |
| inversions of all stages | 0.02 hours |
| numerical inversion | 38 min for the start with the lowest cost (5 iterations); 2.73 CPU hours for all 4 starts |

## 6. Physics of the trained networks at the estimate

| network | surface flux: RMS error / RMS of the prescribed (time mean, 1-min grid), whole window | surface flux, after the first hour | PDE residual / sigma_pde (1-min grid), after the first hour: RMS | max | PDE residual / sigma_pde, the row t = 0: RMS |
|---|---|---|---|---|---|
| operator member 1 | 2.04 % | 1.35 % | 1.202 | 10.57 | 0.71 |
| operator member 2 | 2.28 % | 1.59 % | 1.059 | 9.54 | 0.79 |
| operator member 3 | 2.89 % | 2.19 % | 1.096 | 10.54 | 0.82 |
| operator member 4 | 2.04 % | 1.41 % | 0.982 | 10.29 | 0.89 |
| forward network 1 of the value correction | 2.30 % | 1.28 % | 0.882 | 12.40 | 1.33 |
| forward network 2 of the value correction | 3.13 % | 2.01 % | 0.973 | 13.40 | 0.90 |
