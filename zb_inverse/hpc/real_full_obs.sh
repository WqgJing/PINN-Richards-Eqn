#!/bin/bash
# hpc/real_full.sh for a sensor other than the point sensor at 0.1 m: the observation depth is the third argument,
# written as obs.depth_m of the data set's config (e.g. "[0.0,0.30]": mean over the layer 0 - 0.30 m). Everything
# else is hpc/real_full.sh unchanged (that file is kept as it is: runs in progress read it).
# Usage: setsid nohup hpc/real_full_obs.sh <tag> <data set dir of scripts/make_real.py> "<obs.depth_m>" \
#            > logs/full_<tag>.log 2>&1 < /dev/null &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; D=$(realpath $2); DEPTH=$3
R=$PWD/runs
TS="--set twin_npz=$D/real.npz --set obs.depth_m=$DEPTH --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 --set model.forcing_memory_taus_days=[0.015625,0.0625,0.25,1,4,16] --set model.forcing_split_sign=true --set train.n_box=128 --set train.box_n_t=16"
echo "full $TAG on $D, sensor $DEPTH, at $(date)"
hpc/param_pipeline.sh $TAG 4 "$TS" "--box" > logs/pipe_$TAG.log 2>&1
grep -q "^pipeline $TAG done" logs/pipe_$TAG.log || { echo "pipeline $TAG did not finish -> stop at $(date)"; exit 1; }
M=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_r4_m$k "; done)
W=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_w$k "; done)
COARSE="$W" WARM=1 hpc/dc_loop.sh ${TAG}dc "$M" $R/${TAG}_inv4/local_center.npz 1 4 "$TS" "--box" > logs/dc_${TAG}dc.log 2>&1
echo "full $TAG done at $(date)"
