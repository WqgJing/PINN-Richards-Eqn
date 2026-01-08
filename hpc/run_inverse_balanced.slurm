#!/bin/bash
#SBATCH --job-name=inv_bal
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_balanced_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_balanced_%j.err
#SBATCH --account=gts-rbras6-coda20
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

module load anaconda3/2023.03
source activate /storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch

PROJECT_ROOT="/storage/project/r-rbras6-0/jjason3/PINN_inverse"
cd "$PROJECT_ROOT"

mkdir -p hpc/logs

echo "========================================"
echo "BALANCED WEIGHTS Inverse PINN"
echo "- wt_head_weight: 200 (20x baseline)"
echo "- flux_bc_weight: 100 (10x baseline)"
echo "- data_weight: 100 (0.5x baseline)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

nvidia-smi

CONFIG="configs/inverse_US-PAS_2021_balanced_weights.yaml"
python3 -u -m src.run_solver_inverse --config "$CONFIG"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
