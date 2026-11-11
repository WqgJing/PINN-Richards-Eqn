#!/bin/bash
#SBATCH --job-name=zbinv_e0b
#SBATCH --account=gts-rbras6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/e0b_%j.out
#SBATCH --error=/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/logs/e0b_%j.out

# E0b: finer twin truth (801 nodes, refinement 401/1601), matched-resolution FEM check,
# identifiability with coarser knots (10 and 15 days), identifiability on 201 nodes.
PY=/storage/home/hcoda1/9/jjason3/project/PINN_inverse/conda_envs/pinn_torch/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
run () { echo; echo "=== $* ==="; "$@"; echo "exit=$? ($*)"; }

for w in apr d80; do
  run $PY -u scripts/make_twin.py --config configs/twin_pas_$w.yaml --refine
  run $PY -u scripts/crosscheck_fem.py --config configs/twin_pas_$w.yaml --set reference.n_nodes=51 --set reference.grid_gamma=1.0 --tag _ref51uniform
  for k in 10 15; do
    run $PY -u scripts/identifiability.py --config configs/twin_pas_$w.yaml --set reference.n_nodes=201 --set zb_spline.knot_days=$k --tag _knot$k
  done
done
echo "done $(date)"
