#!/bin/bash
# The adopted method on one site data set from start to end (todo.md findings 72-73), one driver on the login node
# (bash + sbatch only): hpc/param_pipeline.sh (wide stage, 4 members -> inversion -> 4 refinement rounds, far starts
# through the wide members), then hpc/dc_loop.sh (value correction with forward PINNs at the estimate, up to 4
# iterations). Flags as in the runs rf22 / rf21b and their loops rf22dc / rf21bdc.
# Usage: setsid nohup hpc/real_full.sh <tag> <data set dir of scripts/make_real.py> > logs/full_<tag>.log 2>&1 < /dev/null &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; D=$(realpath $2)
R=$PWD/runs
TS="--set twin_npz=$D/real.npz --set obs.depth_m=[0.1,0.1] --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 --set model.forcing_memory_taus_days=[0.015625,0.0625,0.25,1,4,16] --set model.forcing_split_sign=true --set train.n_box=128 --set train.box_n_t=16"
echo "full $TAG on $D at $(date)"
hpc/param_pipeline.sh $TAG 4 "$TS" "--box" > logs/pipe_$TAG.log 2>&1
grep -q "^pipeline $TAG done" logs/pipe_$TAG.log || { echo "pipeline $TAG did not finish -> stop at $(date)"; exit 1; }
M=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_r4_m$k "; done)
W=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_w$k "; done)
COARSE="$W" WARM=1 hpc/dc_loop.sh ${TAG}dc "$M" $R/${TAG}_inv4/local_center.npz 1 4 "$TS" "--box" > logs/dc_${TAG}dc.log 2>&1
echo "full $TAG done at $(date)"
