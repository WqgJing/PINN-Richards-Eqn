#!/bin/bash
# Forward PINNs on a finite-difference design of zb paths around one estimate (scripts/fd_design.py):
# 2 x n_coef ordinary forward PINNs (sampling.mode=point, physics only), each continued from the SAME forward PINN
# trained at the centre path, so their errors are largely common and cancel in the differences.
# Their theta_5cm values give the slope correction of scripts/param_invert.py --fwd-fd (hpc/dc_loop.sh FD="<dirs>").
# Usage: setsid nohup hpc/fd_set.sh <name> <center local_center.npz> <parent model.pt> "<train --set>" \
#            > logs/<name>.log 2>&1 < /dev/null &
# Result: runs/<name>_design/{p,m}<i>.npz (the paths) and runs/<name>_{p,m}<i> (the forward PINNs).
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
NAME=$1; CEN=$2; PARENT=$3; TSET=$4
SCALE=${SCALE:-3.0}; WLR=${WLR:-2.0e-4}; GT=${GT:-04:00:00}
R=$PWD/runs; C=configs/p1_param_synth_v3.yaml
# same schedule as a continued forward member of hpc/dc_loop.sh (WARM=1)
FWD_SET="--set sampling.mode=point --set train.adam_steps=25000 --set train.lr=$WLR --set train.lr_final=1.0e-5 --set train.lbfgs_blocks=15"

wait_jobs() {   # one sacct call per minute for the whole list; a failed or incomplete answer is not "finished"
  local ids=$1 st n
  n=$(echo "$ids" | tr ',' '\n' | wc -l)
  while true; do
    sleep 60
    st=$(timeout 120 sacct -j "$ids" -X -n -P -o State 2>/dev/null) || continue
    [ "$(echo "$st" | grep -c .)" -ge "$n" ] || continue
    echo "$st" | grep -qvE "^(COMPLETED|FAILED|CANCELLED.*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$" || break
  done
}

submit() {   # sbatch --parsable with a time limit, retried until the controller answers with a job id; a submission that
             # timed out may still have been accepted, so any job of that name on that partition is cancelled first
  local name=$1 part=$2 j i; shift 2
  while true; do
    j=$(timeout 180 sbatch --parsable -J $name -p $part "$@" 2>/dev/null) && [ -n "$j" ] && break
    for i in 1 2 3 4 5; do timeout 180 scancel -u $USER --name=$name --partition=$part 2>/dev/null && break; sleep 30; done
    sleep 30
  done
  echo ${j%%;*}
}

echo "fd set $NAME around $CEN, parent $PARENT, +- $SCALE posterior sd at $(date)"
[ -f "$CEN" ] && [ -f "$PARENT" ] || { echo "MISSING centre or parent -> stop"; exit 1; }
j=$(submit ${NAME}_design cpu-small --account=gts-rbras6 --qos=inferno -n1 -c2 --mem=8G -t 00:15:00 \
      hpc/run_script_cpu.slurm scripts/fd_design.py --center $CEN --out $R/${NAME}_design --scale $SCALE)
wait_jobs $j
for i in $(seq 1 24); do [ -f $R/${NAME}_design/design.npz ] && break; sleep 5; done
[ -f $R/${NAME}_design/design.npz ] || { echo "MISSING $R/${NAME}_design/design.npz (job $j) -> stop"; exit 1; }
grep -A 12 "^design of" logs/cpu_$j.out

ids=""; k=0
for f in $R/${NAME}_design/[pm][0-9]*.npz; do
  b=$(basename $f .npz)
  j=$(submit ${NAME}_$b gpu-v100 --gres=gpu:V100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm \
        scripts/param_train.py --config $C $FWD_SET --set sampling.local_file=$f --set train.init_from=$PARENT \
        --set train.seed=$((7000 + k)) $TSET --set output.dir=$R/${NAME}_$b)
  echo "${NAME}_$b: job $j"
  ids="$ids,$j"; k=$((k + 1))
done
wait_jobs ${ids#,}
miss=0
for f in $R/${NAME}_design/[pm][0-9]*.npz; do
  b=$(basename $f .npz)
  for i in $(seq 1 24); do [ -f $R/${NAME}_$b/model.pt ] && break; sleep 5; done
  [ -f $R/${NAME}_$b/model.pt ] || { echo "MISSING $R/${NAME}_$b/model.pt"; miss=$((miss + 1)); }
done
echo "fd set $NAME done, $k trained, $miss missing at $(date)"
