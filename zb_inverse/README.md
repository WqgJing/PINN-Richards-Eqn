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
