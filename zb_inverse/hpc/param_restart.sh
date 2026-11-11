#!/bin/bash
# Re-invert the wide (round-0) members with the current param_invert.py, then refinement rounds 1..E.
# Usage: nohup hpc/param_restart.sh <tag> <E> "<train --set>" "<invert --set>" > logs/pipe_<tag>.log &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; E=$2; TSET=$3; ISET=$4; R=$PWD/runs
m=$(for k in 0 1 2 3; do echo -n "$R/${TAG}_r0_m$k "; done)
j=$(sbatch --parsable -J ${TAG}_inv0 hpc/run_script_cpu.slurm scripts/param_invert.py --members $m --out $R/${TAG}_inv0 $ISET)
while true; do st=$(sacct -j $j -X -n -P -o State 2>/dev/null | head -1); case "$st" in COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) break;; esac; sleep 30; done
echo "${TAG}_inv0: job $j $st at $(date)"
hpc/param_rounds.sh $TAG 1 $E "$TSET" "$ISET"
