#!/bin/bash
#SBATCH --job-name=inv_US-PAS              # Job name
#SBATCH --output=logs/inverse_US-PAS_%j.out  # Output file (%j = job ID)
#SBATCH --error=logs/inverse_US-PAS_%j.err   # Error file
#SBATCH --account=gts-rbras6-coda20        # Account
#SBATCH --partition=gpu-h200               # GPU partition (H200)
#SBATCH --gres=gpu:h200:1                  # Request 1 H200 GPU
#SBATCH --cpus-per-task=8                  # Number of CPU cores
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=48:00:00                    # Time limit (48 hours for 100k epochs)
#SBATCH --ntasks=1                         # Number of tasks

# Load anaconda module and activate environment
module load anaconda3/2023.03
source activate /storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "========================================"
echo "INVERSE PINN - US-PAS (Florida Pasture)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Configuration
CONFIG="${1:-configs/inverse_US-PAS_2021.yaml}"
echo "Config file: $CONFIG"

# Change to project root
cd ..
echo "Project root: $(pwd)"

# Run inverse PINN training
python3 -u -m src.run_solver_inverse --config "$CONFIG"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
