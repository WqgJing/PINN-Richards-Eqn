#!/bin/bash
#SBATCH --job-name=zbinv_gpu_script
#SBATCH --account=gts-rbras6-coda20
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/gpu_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/gpu_%j.out

# Usage: sbatch hpc/run_script_gpu.slurm scripts/<name>.py [args...]   (any script on one GPU)
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python   # env python by absolute path ('module' is not available in batch shells here)
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date); args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -u "$@"
echo "exit=$? at $(date)"
