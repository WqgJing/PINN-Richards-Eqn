#!/bin/bash
# Solver side of one window of site data from its config to the in-sample check (SCORING / CALIBRATION ONLY: no PINN
# run reads its output except the calibrated soil, as on every site window):
#   data set  scripts/make_real.py with the starting soil of configs/real_<site>_<window>.yaml
#   soil      four starts of scripts/calibrate_soil.py on that data set -> output/soil_<site>_<window>/start?.json
#   check     hpc/transfer_solver.sh <site> <window> "<obs.depth_m>" [<test> ...]: data set with the lowest-misfit
#             soil and solver inversions from four starts (the window itself, and every test window given)
# Runs on the login node as a driver only (bash + sbatch; every Python step is a batch job).
# Usage: setsid nohup hpc/cycle_solver.sh <site> <window> "<obs.depth_m>" [<test> ...] \
#            > logs/cycle_<site>_<window>.log 2>&1 < /dev/null &
#   e.g. hpc/cycle_solver.sh nc2 sep23c "[0.0,0.30]"
#   CT=06:00:00 : time limit of one calibration start (default)        RT : as hpc/transfer_solver.sh
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse || exit 1
SITE=$1; W=$2; DEPTH=$3; shift 3
O=$PWD/output
CT=${CT:-06:00:00}

submit() {   # name, cpus, mem, time, script args... ; as hpc/transfer_solver.sh: retried until the controller answers
  local name=$1 cpus=$2 mem=$3 tl=$4 j i; shift 4
  while true; do
    j=$(timeout 180 sbatch --parsable -J $name --account=gts-rbras6 --partition=cpu-small --qos=inferno -n1 -c$cpus --mem=$mem -t $tl hpc/run_script_cpu.slurm "$@" 2>/dev/null) && [ -n "$j" ] && break
    for i in 1 2 3; do timeout 180 scancel -u $USER --name=$name --partition=cpu-small 2>/dev/null && break; sleep 20; done
    sleep 30
  done
  echo ${j%%;*}
}

wait_job() {   # job id: blocks until finished, echoes the final state
  local st
  while true; do
    st=$(timeout 120 sacct -j $1 -X -n -P -o State 2>/dev/null | head -1)
    case "$st" in COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) break;; esac
    sleep 30
  done
  echo $st
}

D=$O/real_${SITE}_$W
echo "cycle $SITE $W at $(date)"
j=$(submit mr_${SITE}_$W 4 16G 00:40:00 scripts/make_real.py --config configs/real_${SITE}_$W.yaml)
echo "make_real job $j $(wait_job $j)"
for i in $(seq 1 24); do [ -f $D/real.npz ] && break; sleep 5; done   # files of a compute node can appear late
[ -f $D/real.npz ] || { echo "no data set $D (log logs/cpu_$j.out) -> stop at $(date)"; exit 1; }
jq -c '{n_coef, zb_range_m, theta_obs_range, zb_spline_vs_record, admissible}' $D/real_summary.json

ids=""
for k in 0 1 2 3; do
  ids="$ids $(submit cal_${SITE}${W}_s$k 2 8G $CT scripts/calibrate_soil.py --data $D --out $O/soil_${SITE}_$W --start $k)"
done
echo "calibration jobs:$ids"
k=0
for j in $ids; do echo "calibration start $k: job $j $(wait_job $j)"; k=$((k + 1)); done
for i in $(seq 1 24); do [ "$(ls $O/soil_${SITE}_$W/start?.json 2>/dev/null | wc -l)" -eq 4 ] && break; sleep 5; done
n=$(ls $O/soil_${SITE}_$W/start?.json 2>/dev/null | wc -l)
[ "$n" -eq 4 ] || { echo "$n of the four calibration starts wrote a soil -> stop at $(date) (calibrate_soil.py --resume continues one)"; exit 1; }

hpc/transfer_solver.sh $SITE $W "$DEPTH" "$@"
echo "cycle $SITE $W done at $(date)"
