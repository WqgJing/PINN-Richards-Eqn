#!/bin/bash
#SBATCH --job-name=zbinv_traincpu
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/train_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/train_%j.out

# CPU fallback for scripts/train.py when the GPU queue is long (same arguments as train_gpu.slurm).
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date); cpus=$SLURM_CPUS_PER_TASK; args: $*"
$PY -u scripts/train.py "$@"
echo "exit=$? at $(date)"
