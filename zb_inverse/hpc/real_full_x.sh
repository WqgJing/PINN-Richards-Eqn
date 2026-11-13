#!/bin/bash
# hpc/real_full_obs.sh with a fourth argument: further training options, appended to those of every training stage
# (wide members, refinement rounds, forward PINNs of the value-correction loop), e.g. "--set model.departure=conductive".
# Everything else is hpc/real_full_obs.sh unchanged (that file is kept as it is: runs in progress read it).
# Usage: setsid nohup hpc/real_full_x.sh <tag> <data set dir of scripts/make_real.py> "<obs.depth_m>" "<options>" \
#            > logs/full_<tag>.log 2>&1 < /dev/null &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; D=$(realpath $2); DEPTH=$3; XSET=$4
R=$PWD/runs
TS="--set twin_npz=$D/real.npz --set obs.depth_m=$DEPTH --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 --set model.forcing_memory_taus_days=[0.015625,0.0625,0.25,1,4,16] --set model.forcing_split_sign=true --set train.n_box=128 --set train.box_n_t=16 $XSET"
echo "full $TAG on $D, sensor $DEPTH, options '$XSET', at $(date)"
hpc/param_pipeline.sh $TAG 4 "$TS" "--box" > logs/pipe_$TAG.log 2>&1
grep -q "^pipeline $TAG done" logs/pipe_$TAG.log || { echo "pipeline $TAG did not finish -> stop at $(date)"; exit 1; }
M=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_r4_m$k "; done)
W=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_w$k "; done)
COARSE="$W" WARM=1 hpc/dc_loop.sh ${TAG}dc "$M" $R/${TAG}_inv4/local_center.npz 1 4 "$TS" "--box" > logs/dc_${TAG}dc.log 2>&1
echo "full $TAG done at $(date)"
