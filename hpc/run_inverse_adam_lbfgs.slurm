#!/bin/bash
#SBATCH --job-name=inv_lbfgs
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_adam_lbfgs_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc/logs/inverse_adam_lbfgs_%j.err
#SBATCH --account=gts-rbras6-coda20
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00       # Expect faster with fewer epochs
#SBATCH --ntasks=1

module load anaconda3/2023.03
source activate /storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch

PROJECT_ROOT="/storage/project/r-rbras6-0/jjason3/PINN_inverse"
cd "$PROJECT_ROOT"

mkdir -p hpc/logs

echo "========================================"
echo "ADAM + L-BFGS HYBRID Training"
echo "========================================"
echo "Strategy:"
echo "  - 10k Adam epochs (instead of 20k)"
echo "  - 500 L-BFGS refinement iterations"
echo "  - Expected total time: ~3-4 minutes"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

nvidia-smi

CONFIG="configs/inverse_US-PAS_2021_adam_lbfgs.yaml"
echo "Config file: $CONFIG"

python3 -u -m src.run_solver_inverse --config "$CONFIG"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
