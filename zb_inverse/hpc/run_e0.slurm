#!/bin/bash
#SBATCH --job-name=zbinv_e0
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/e0_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/e0_%j.out

# E0: tests, twin truth (3 windows), FEM cross-check, identifiability bound.
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python   # env python by absolute path ('module' is not available in batch shells here)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"

run () { echo; echo "=== $* ==="; "$@"; echo "exit=$? ($*)"; }

run $PY -u tests/run_tests.py
run $PY -u scripts/make_twin.py --config configs/twin_pas_apr.yaml --refine
run $PY -u scripts/make_twin.py --config configs/twin_pas_d80.yaml
run $PY -u scripts/crosscheck_fem.py --config configs/twin_pas_apr.yaml
run $PY -u scripts/crosscheck_fem.py --config configs/twin_pas_d80.yaml
run $PY -u scripts/identifiability.py --config configs/twin_pas_apr.yaml
run $PY -u scripts/identifiability.py --config configs/twin_pas_d80.yaml
run $PY -u scripts/make_twin.py --config configs/twin_pas_season.yaml
run $PY -u scripts/identifiability.py --config configs/twin_pas_season.yaml
echo "done $(date)"
