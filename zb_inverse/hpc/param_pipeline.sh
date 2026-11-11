#!/bin/bash
# Parametric PINN, full method with no truth anywhere (gates G2/G3):
#   wide physics-only training of K members (prior over zb paths)  ->  multi-start LM inversion (ensemble)
#   -> R rounds of [physics-only refinement of every member around the current estimate -> inversion].
# Runs on the login node as a driver only (bash + sbatch; every Python step is a batch job).
# Usage: nohup hpc/param_pipeline.sh <tag> <rounds> "<train --set ...>" "<invert --set ...>" [wide_dir_tag] > logs/pipe_<tag>.log &
#   train sets: forcing only (q noise); invert sets: theta noise level of the data to invert.
#   wide_dir_tag: reuse the wide members runs/<wide_dir_tag>_w<k> (shared by datasets with the same forcing).
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; ROUNDS=$2; TSET=$3; ISET=$4; WIDE=${5:-$TAG}
K=${K:-4}
R=$PWD/runs; C=configs/p1_param_synth_v3.yaml
WIDE_SET=${WIDE_SET:-"--set sampling.mode=prior --set train.adam_steps=40000 --set train.lbfgs_blocks=10"}
LOCAL_SET=${LOCAL_SET:-"--set sampling.frac_local=0.7 --set train.adam_steps=15000 --set train.lr=3.0e-4 --set train.lbfgs_blocks=15"}

GT=${GT:-04:00:00}   # GPU time limit (a member trains in about 35 min on a V100; a short limit backfills sooner)

submit() {   # as hpc/dc_loop.sh: sbatch --parsable with a time limit, retried until the controller answers with a job
             # id; a submission that timed out may still have been accepted, so any job of that name on that
             # partition is cancelled first
  local name=$1 part=$2 j i; shift 2
  while true; do
    j=$(timeout 180 sbatch --parsable -J $name -p $part "$@" 2>/dev/null) && [ -n "$j" ] && break
    for i in 1 2 3 4 5; do timeout 180 scancel -u $USER --name=$name --partition=$part 2>/dev/null && break; sleep 30; done
    sleep 30
  done
  echo ${j%%;*}
}

gpu_job() {   # name, args... ; races H200/A100/V100, blocks until finished; echoes final state line
  local name=$1; shift
  local a b c
  a=$(submit $name gpu-h200 --gres=gpu:h200:1 -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
  b=$(submit $name gpu-a100 --gres=gpu:A100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
  c=$(submit $name gpu-v100 --gres=gpu:V100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
  hpc/race.sh $a,$b,$c > logs/race_$name.log 2>&1
  echo "$name: $(tail -1 logs/race_$name.log)"
}

cpu_job() {   # name, script args... ; blocks until finished
  local name=$1; shift
  local j st
  j=$(submit $name cpu-small --account=gts-rbras6 --qos=inferno -n1 -c8 --mem=24G -t 01:30:00 hpc/run_script_cpu.slurm "$@")
  while true; do
    st=$(timeout 120 sacct -j $j -X -n -P -o State 2>/dev/null | head -1)
    case "$st" in COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) break;; esac
    sleep 30
  done
  echo "$name: job $j $st (log logs/cpu_$j.out)"
}

ok() {   # files written on a compute node can take a while to appear on the login node: retry up to 2 min
  local i
  for i in $(seq 1 24); do [ -f "$1" ] && return 0; sleep 5; done
  echo "MISSING $1 -> stop at $(date)"; exit 1
}

START_ROUND=${START_ROUND:-1}   # > 1: continue a finished pipeline from that round (members and inversion of the round before must exist)
echo "pipeline $TAG: rounds $START_ROUND-$ROUNDS, K $K, train [$TSET], invert [$ISET], wide $WIDE at $(date)"
wide_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${WIDE}_w$k "; done)
if [ "$START_ROUND" -le 1 ]; then
  # 1. wide stage (skipped if the members exist)
  for k in $(seq 0 $((K - 1))); do
    if [ ! -f $R/${WIDE}_w$k/model.pt ]; then
      gpu_job ${WIDE}_w$k scripts/param_train.py --config $C $WIDE_SET --set train.seed=$k $TSET --set output.dir=$R/${WIDE}_w$k &
    fi
  done
  wait
  for k in $(seq 0 $((K - 1))); do ok $R/${WIDE}_w$k/model.pt; done
  prev_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${WIDE}_w$k "; done)
  cpu_job ${TAG}_inv0 scripts/param_invert.py --members $prev_members --out $R/${TAG}_inv0 $ISET
  ok $R/${TAG}_inv0/local_center.npz
else
  for k in $(seq 0 $((K - 1))); do ok $R/${TAG}_r$((START_ROUND - 1))_m$k/model.pt; done
  ok $R/${TAG}_inv$((START_ROUND - 1))/local_center.npz
  prev_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${TAG}_r$((START_ROUND - 1))_m$k "; done)
fi
# 2. refinement rounds
for r in $(seq $START_ROUND $ROUNDS); do
  for k in $(seq 0 $((K - 1))); do
    pm=$(echo $prev_members | cut -d' ' -f$((k + 1)))
    gpu_job ${TAG}_r${r}_m$k scripts/param_train.py --config $C $LOCAL_SET --set sampling.mode=local_file \
      --set sampling.local_file=$R/${TAG}_inv$((r - 1))/local_center.npz --set train.init_from=$pm/model.pt \
      --set train.seed=$((100 * r + k)) $TSET --set output.dir=$R/${TAG}_r${r}_m$k &
  done
  wait
  for k in $(seq 0 $((K - 1))); do ok $R/${TAG}_r${r}_m$k/model.pt; done
  prev_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${TAG}_r${r}_m$k "; done)
  # far starts go through the wide members first (--coarse, todo.md findings 42 and 72q): the refined members are
  # only accurate near their centre, and a far start can stop in a spurious minimum of theirs
  cpu_job ${TAG}_inv$r scripts/param_invert.py --members $prev_members --out $R/${TAG}_inv$r $ISET \
    --prev-center $R/${TAG}_inv$((r - 1))/local_center.npz --coarse $wide_members
  ok $R/${TAG}_inv$r/local_center.npz
done
echo "pipeline $TAG done at $(date)"
