#!/bin/bash
#SBATCH --job-name=inv_lr5e4
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_lr5e4_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_lr5e4_%j.err
#SBATCH --account=gts-rbras6-coda20
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --ntasks=1

module load anaconda3/2023.03
source activate /storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch

PROJECT_ROOT="/storage/project/r-rbras6-0/jjason3/PINN_inverse"
cd "$PROJECT_ROOT"

mkdir -p hpc/logs

echo "========================================"
echo "Inverse PINN - LR 5e-4 Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

nvidia-smi

CONFIG="configs/inverse_US-PAS_2021_lr5e4.yaml"
python3 -u -m src.run_solver_inverse --config "$CONFIG"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
