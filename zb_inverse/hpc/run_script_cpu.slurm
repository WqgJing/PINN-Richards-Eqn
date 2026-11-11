#!/bin/bash
#SBATCH --job-name=zbinv_cpu
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/cpu_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/cpu_%j.out

# Usage: sbatch hpc/run_script_cpu.slurm scripts/<name>.py [args...]
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python
T=${ZB_THREADS:-1}   # threads per process (sbatch --export=ALL,ZB_THREADS=<n> for one multi-threaded training)
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date); args: $*"
$PY -u "$@"
echo "exit=$? at $(date)"
