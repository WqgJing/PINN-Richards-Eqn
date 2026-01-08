#!/bin/bash
#SBATCH --job-name=regen_viz
#SBATCH --account=gts-rbras6-coda20
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/regen_viz_%j.out
#SBATCH --error=logs/regen_viz_%j.err

source ~/.bashrc
conda activate /storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch

cd /storage/project/r-rbras6-0/jjason3/PINN_inverse/hpc
python regen_viz.py
