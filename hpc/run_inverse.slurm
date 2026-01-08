#!/bin/bash
#SBATCH --job-name=inverse_pinn            # Job name
#SBATCH --output=logs/inverse_%j.out       # Output file (%j = job ID)
#SBATCH --error=logs/inverse_%j.err        # Error file
#SBATCH --account=gts-rbras6-coda20        # Account
#SBATCH --partition=gpu-h200               # GPU partition (H200)
#SBATCH --gres=gpu:h200:1                  # Request 1 H200 GPU
#SBATCH --cpus-per-task=8                  # Number of CPU cores
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=12:00:00                    # Time limit (12 hours)
#SBATCH --ntasks=1                         # Number of tasks

# Initialize conda (required for conda activate to work)
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate /storage/scratch1/9/jjason3/conda_envs/Recharge_PINN

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "========================================"
echo "INVERSE PINN TRAINING"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Run the training script
CONFIG="${1:-configs/inverse_baseline.yaml}"
SYNTHETIC="${2:---synthetic}"  # Default to synthetic data for testing
echo "Config file: $CONFIG"
echo "Data mode: $SYNTHETIC"

# Change to project root
cd ..
echo "Project root: $(pwd)"

# Run inverse PINN training using the run_solver_inverse.py script
python3 -u -m src.run_solver_inverse --config "$CONFIG" $SYNTHETIC

echo "========================================"
echo "End time: $(date)"
echo "========================================"
