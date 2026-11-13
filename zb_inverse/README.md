# zb_inverse: water-table depth from surface flux and near-surface soil moisture

Inverse problem for the 1-D Richards equation with the water table as lower boundary (h = 0 at z = zb(t)):
given the surface flux q_surf(t) and the soil moisture θ(t) of one near-surface sensor, estimate zb(t).
The estimate comes from a physics-informed neural network. The finite-element solver is used only to make
synthetic twins, to calibrate the soil on a window with a known water table, and to score results.

## Method

- zb(t; c) = zb_min + softplus(cubic B-spline with coefficients c, 5-day knots); the equation is solved in the
  Landau coordinate ξ = z / zb(t).
- The network learns the solution operator c ↦ h(ξ, t; c) from the PDE residual and the surface-flux condition
  alone. It never sees soil-moisture data, so there is no data-versus-physics weight.
- The coefficients are fitted to θ at the sensor through an ensemble of 4 networks (Levenberg-Marquardt, exact
  autograd sensitivities, several starts, far starts routed through the wide-prior networks first).
- The networks are retrained around the current estimate and the inversion is repeated (`hpc/param_pipeline.sh`).
- A forward PINN trained at the current estimate corrects the operator's value there (`hpc/dc_loop.sh`).

## Layout

| path | content |
|---|---|
| `src/zbinv/` | model, physics, spline, parametric training and inversion, site-data reader |
| `scripts/` | entry points: `param_train.py`, `param_invert.py`, `make_real.py`, `calibrate_soil.py`, scoring and figures |
| `hpc/` | SLURM scripts and the drivers (every Python step runs as a batch job) |
| `configs/` | YAML configurations of twins, site windows and networks |
| `tests/` | unit tests (`hpc/run_tests.slurm`) |
| `tasks/todo.md` | plan, working notes and the numbered findings of every try |
| `output/` | scored summaries and figures of each try (checkpoints, logs and data are not tracked) |

Sign convention: q_surf = E_soil + Q_s − P (m/s), positive upward; transpiration is a root-zone sink.

## Results on site data (AmeriFlux, 12 Nov 2026)

Protocol: the soil (van Genuchten, 5 parameters) is calibrated with the solver on an earlier recession of the
same site where the water table is known; nothing is fitted on the test window. The PINN then estimates zb(t)
on the test window from q_surf and the sensor alone (`hpc/real_full_obs.sh`, same settings at every site).
"Solver" is the same inversion with the finite-element model in place of the network (`scripts/fem_referee.py`,
scoring only). "Hydrostatic" reads the sensor through the retention curve hour by hour, without flow.
Errors against the well: RMSE / bias in cm, r. US-ONA is scored on the hours the sensor is below saturation.

| site, test window (soil from) | well (m) | PINN − solver, rms (cm) | PINN | solver | hydrostatic | figure |
|---|---|---|---|---|---|---|
| US-ONA, 26 Nov 2022 + 18 d (Oct 2022) | 0.34 – 0.88 | 0.35 | 2.37 / +1.47 / 0.983 | 2.44 / +1.69 / 0.983 | 1.93 / +0.84 / 0.983 | [png](output/real_ona_nov22_caloct22/result_rf22c.png) |
| US-ONA, 17 Oct 2022 + 15 d (Nov 2021) | 0.39 – 0.91 | 0.45 | 3.15 / +2.45 / 0.969 | 3.23 / +2.69 / 0.975 | 2.65 / +1.87 / 0.974 | [png](output/real_ona_oct22_soilnov21/result_rf22ob.png) |
| US-PAS, 15 Oct 2021 + 7 d (Sep 2021) | 0.39 – 1.01 | 0.03 | 4.52 / +0.17 / 0.995 | 4.50 / +0.19 / 0.994 | 5.28 / −2.69 / 0.998 | [png](output/real_pas_oct21_soilsep21/result_rfp2.png) |
| US-NC2, 27 Sep 2023 + 9 d (1 Sep 2023) | 0.29 – 0.62 | 0.07 | 9.48 / −9.25 / 0.998 | 9.53 / −9.31 / 0.998 | 11.1 / −11.0 / 1.000 | [png](output/real_nc2_sep23b_soilsep23a_cal3/result_rfn1.png) |
| US-NC2, same window (May 2023; diagnostic, soil not chosen by the rule) | 0.29 – 0.62 | 0.06 | 1.59 / +0.55 / 0.998 | 1.64 / +0.55 / 0.998 | 1.64 / −0.33 / 0.998 | [png](output/real_nc2_sep23b_soilmay23/result_rfnd.png) |
| US-HB2, 7 Mar 2020 + 15 d (Sep 2019; dry run of the chain) | 0.26 – 0.57 | 0.22 | 2.57 / +1.15 / 0.965 | 2.50 / +1.23 / 0.968 | 1.70 / +0.62 / 0.987 | [png](output/real_hb2_mar20_soilsep19_pre/result_rfh1p.png) |

What the table says:

- The PINN reproduces the solver's inversion to 0.03 – 0.45 cm at four sites (sand, Spodosol, organic soil;
  point and layer sensors), and every start (hydrostatic, flat, 0.3 m, 1.0 m) ends at the same estimate.
- The distance to the well is that of the forward model, not of the network: one retention curve calibrated
  on one recession does not always hold on the next (US-NC2: the recession after a storm reads 0.02 drier at
  the same water table, which is 9 cm of depth). The target set beforehand (RMSE ≤ 3 cm, |bias| ≤ 2 cm,
  r ≥ 0.98) is met on US-ONA Nov 2022 only; it has not yet been met on a second held-out window.
- On these recessions the profile is close to equilibrium: the hydrostatic reading with the same soil is as
  close to the well as either inversion.

Every try, including the ones that failed, is in `tasks/todo.md` (finding 73).
