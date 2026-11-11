#!/bin/bash
#SBATCH --job-name=zbinv_tests
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/tests_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/tests_%j.out

# Usage: sbatch hpc/run_tests.slurm [name filter ...]
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python   # env python by absolute path ('module' is not available in batch shells here)
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
$PY -u tests/run_tests.py "$@"
