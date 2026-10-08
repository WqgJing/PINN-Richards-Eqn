# PINN for the Richards Equation (inverse)

Recovers water table depth z_b(t) from surface soil moisture and surface flux.
A network predicts pressure head h(z, t), and z_b(t) is a spline learned jointly
with it. The loss combines the 1D Richards equation residual (van Genuchten
soil), a flux boundary at the surface, h = 0 at the water table, and a misfit to
observed surface moisture.

Tested on AmeriFlux US-PAS (Apr-Nov 2021): surface theta r = 0.97, z_b r = 0.91.

## Layout

    src/        model, training loop, config loader, plotting
    configs/    run configs (inverse_US-PAS_2021_BEST.yaml is the reference)
    hpc/        SLURM scripts
    examples/   synthetic test case

## Running

    pip install -r requirements.txt
    python -m src.run_solver_inverse --config configs/inverse_US-PAS_2021_BEST.yaml

On the cluster (from `hpc/`): `sbatch run_inverse_US-PAS.slurm configs/<name>.yaml`.
`--synthetic` runs on synthetic data for testing.

Input is one NPZ per site (`data/`, not tracked) with `times`, `qsurf` (m/s,
positive upward), `smap_sm`, and optionally `zb` for evaluation.
