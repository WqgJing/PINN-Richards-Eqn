#!/bin/bash
#SBATCH --job-name=zbinv_cpu_mt
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/cpu_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/cpu_%j.out

# Usage: sbatch hpc/run_script_cpu_mt.slurm scripts/<name>.py [args...]
# Like run_script_cpu.slurm but multi-threaded (threads = cpus-per-task): PINN training on a CPU node when the GPU
# queues are full. scripts/param_train.py picks the device itself (cuda if available, else cpu).
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date), $SLURM_CPUS_PER_TASK threads; args: $*"
$PY -u "$@"
echo "exit=$? at $(date)"
