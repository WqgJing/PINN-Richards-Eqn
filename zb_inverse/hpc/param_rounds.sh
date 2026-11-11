#!/bin/bash
# Continue the parametric-PINN refinement loop (same steps as hpc/param_pipeline.sh rounds) from round S to E.
# Needs runs/<tag>_inv<S-1>/local_center.npz and the members of round S-1 (runs/<tag>_r<S-1>_m<k>).
# Usage: nohup hpc/param_rounds.sh <tag> <S> <E> "<train --set ...>" "<invert --set ...>" > logs/pipe_<tag>_more.log &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; S=$2; E=$3; TSET=$4; ISET=$5
K=${K:-4}
R=$PWD/runs; C=configs/p1_param_synth_v3.yaml
LOCAL_SET=${LOCAL_SET:-"--set sampling.frac_local=0.7 --set train.adam_steps=15000 --set train.lr=3.0e-4 --set train.lbfgs_blocks=15"}

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

gpu_job() {
  local name=$1; shift
  local a b c
  a=$(submit $name gpu-h200 --gres=gpu:h200:1 -A gts-rbras6-coda20 --time=04:00:00 hpc/run_script_gpu.slurm "$@")
  b=$(submit $name gpu-a100 --gres=gpu:A100:1 -q inferno -A gts-rbras6-coda20 --time=04:00:00 hpc/run_script_gpu.slurm "$@")
  c=$(submit $name gpu-v100 --gres=gpu:V100:1 -q inferno -A gts-rbras6-coda20 --time=04:00:00 hpc/run_script_gpu.slurm "$@")
  hpc/race.sh $a,$b,$c > logs/race_$name.log 2>&1
  echo "$name: $(tail -1 logs/race_$name.log)"
}

cpu_job() {
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

echo "rounds $TAG $S..$E at $(date)"
ok $R/${TAG}_inv$((S - 1))/local_center.npz
prev_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${TAG}_r$((S - 1))_m$k "; done)
for r in $(seq $S $E); do
  for k in $(seq 0 $((K - 1))); do
    pm=$(echo $prev_members | cut -d' ' -f$((k + 1)))
    ok $pm/model.pt
    gpu_job ${TAG}_r${r}_m$k scripts/param_train.py --config $C $LOCAL_SET --set sampling.mode=local_file \
      --set sampling.local_file=$R/${TAG}_inv$((r - 1))/local_center.npz --set train.init_from=$pm/model.pt \
      --set train.seed=$((100 * r + k)) $TSET --set output.dir=$R/${TAG}_r${r}_m$k &
  done
  wait
  for k in $(seq 0 $((K - 1))); do ok $R/${TAG}_r${r}_m$k/model.pt; done
  prev_members=$(for k in $(seq 0 $((K - 1))); do echo -n "$R/${TAG}_r${r}_m$k "; done)
  cpu_job ${TAG}_inv$r scripts/param_invert.py --members $prev_members --out $R/${TAG}_inv$r $ISET \
    --prev-center $R/${TAG}_inv$((r - 1))/local_center.npz
  ok $R/${TAG}_inv$r/local_center.npz
done
echo "rounds $TAG done at $(date)"
