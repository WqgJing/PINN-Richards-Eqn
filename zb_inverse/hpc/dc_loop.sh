#!/bin/bash
# Value-corrected refinement with two physics-only networks (no solver):
#   - forward PINN(s) at the current estimate c_k (sampling.mode=point: an ordinary forward PINN for that one zb path)
#     give an accurate theta_5cm(c_k);
#   - the operator ensemble gives d theta / d c (and the model between iterates);
#   - scripts/param_invert.py --fwd re-inverts theta_op(c) + [theta_fwd(c_k) - theta_op(c_k)] -> c_{k+1}.
# Repeat until the zb move per 2.5-day segment (days 0-25) is below STOP x posterior sd.
# Usage: setsid nohup hpc/dc_loop.sh <tag> "<operator member dirs>" <start local_center.npz> <it0> <it_max> \
#            "<train --set>" "<invert --set>" > logs/dc_<tag>.log 2>&1 < /dev/null &
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
TAG=$1; OPS=$2; CEN=$3; IT0=$4; ITMAX=$5; TSET=$6; ISET=$7
STOP=${STOP:-0.2}; KF=${KF:-2}; FJ=${FJ:-0}; SEC=${SEC:-0}; PREV=${PREV:-}; COARSE=${COARSE:-}; NTAN=${NTAN:-4096}; GT=${GT:-04:00:00}
# WARM=1: member k of round it starts from member k of round it-1 (the path moved little, so training continues);
# INIT="<model.pt of member 0> <model.pt of member 1> ...": the same for the first round (from an earlier loop)
# a continued member starts at the lower learning rate WLR (lr 1e-3 from a trained state blows the loss up, finding 39)
WARM=${WARM:-0}; INIT=(${INIT:-}); WLR=${WLR:-2.0e-4}
R=$PWD/runs; C=configs/p1_param_synth_v3.yaml
FWD_SET="--set sampling.mode=point --set train.adam_steps=25000 --set train.lr=1.0e-3 --set train.lr_final=1.0e-5 --set train.lbfgs_blocks=15"
IX=""
# FJ=1: the forward PINN is also trained on the sensitivity (linearised) equations at c_k (train.n_tan), and the
# inversion uses its d theta / d c there (--fwd-jac): every round is a Gauss-Newton step of the forward PINN itself.
if [ "$FJ" = 1 ]; then FWD_SET="$FWD_SET --set train.n_tan=$NTAN"; IX="--fwd-jac"; fi
# COARSE="<wide-stage member dirs>": far starts go through the wide stage first (coarse-to-fine)
if [ -n "$COARSE" ]; then IX="$IX --coarse $COARSE"; fi
# FD="<dirs of forward PINNs on a finite-difference design, hpc/fd_set.sh>" (and FJ=0): the slope of theta_fwd - theta_op
# fitted over the design corrects the operator Jacobian in every direction (--fwd-fd); the design is kept between
# rounds (chord iteration), only the forward PINNs at the new estimate are trained
FD=${FD:-}
if [ -n "$FD" ] && [ "$FJ" != 1 ]; then IX="$IX --fwd-fd $FD"; fi
# SEC=1 (and FJ=0): multi-secant slope correction from the forward PINNs of all earlier rounds (and of PREV, dirs of
# forward PINNs from an earlier loop), no extra training

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
  a=$(submit $name gpu-h200 --gres=gpu:h200:1 -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
  b=$(submit $name gpu-a100 --gres=gpu:A100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
  c=$(submit $name gpu-v100 --gres=gpu:V100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
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

echo "dc loop $TAG it $IT0..$ITMAX from $CEN, operator: $OPS, stop < $STOP sigma, FJ=$FJ, SEC=$SEC, WARM=$WARM, FD: $(echo $FD | wc -w) paths, coarse: ${COARSE:-none} at $(date)"
ok $CEN
for it in $(seq $IT0 $ITMAX); do
  for k in $(seq 0 $((KF - 1))); do
    # restart after an interruption: a member of this round that is already trained is kept
    if [ -f $R/${TAG}_f${it}_m$k/model.pt ]; then echo "${TAG}_f${it}_m$k: already trained, reused"; continue; fi
    ws=""
    if [ "$WARM" = 1 ] && [ -f $R/${TAG}_f$((it - 1))_m$k/model.pt ]; then ws="--set train.init_from=$R/${TAG}_f$((it - 1))_m$k/model.pt --set train.lr=$WLR"
    elif [ "$it" = "$IT0" ] && [ -n "${INIT[$k]:-}" ]; then ws="--set train.init_from=${INIT[$k]} --set train.lr=$WLR"; fi
    gpu_job ${TAG}_f${it}_m$k scripts/param_train.py --config $C $FWD_SET --set sampling.local_file=$CEN $ws \
      --set train.seed=$((2000 + 10 * it + k)) $TSET --set output.dir=$R/${TAG}_f${it}_m$k &
  done
  wait
  fwd=""
  for k in $(seq 0 $((KF - 1))); do ok $R/${TAG}_f${it}_m$k/model.pt; fwd="$fwd $R/${TAG}_f${it}_m$k"; done
  hist=""
  if [ "$SEC" = 1 ] && [ "$FJ" != 1 ]; then
    hist="$PREV"
    for d in $R/${TAG}_f[0-9]*_m[0-9]*; do
      j=${d##*_f}; j=${j%%_m*}
      [ -f $d/model.pt ] && [ "$j" -lt "$it" ] && hist="$hist $d"
    done
    [ -n "$hist" ] && hist="--fwd-prev $hist"
  fi
  cpu_job ${TAG}_it$it scripts/param_invert.py --members $OPS --fwd $fwd --prev-center $CEN --out $R/${TAG}_it$it $IX $hist $ISET
  ok $R/${TAG}_it$it/summary.json
  s=$(jq '.step_over_sigma_max' $R/${TAG}_it$it/summary.json)
  echo "it $it: step/sigma max $s, by segment $(jq -c '[.step_over_sigma_by_segment[]*100|round/100]' $R/${TAG}_it$it/summary.json)," \
       "fwd offset rms $(jq '.fwd.offset_rms' $R/${TAG}_it$it/summary.json), [scoring only] zb rmse days 0-25" \
       "$(jq '.zb_rmse_inner*10000|round/100' $R/${TAG}_it$it/summary.json) cm, starts spread" \
       "$(jq '.start_spread_rms_inner_m*10000|round/100' $R/${TAG}_it$it/summary.json) cm at $(date)"
  CEN=$R/${TAG}_it$it/local_center.npz
  if awk "BEGIN{exit !($s < $STOP)}"; then echo "converged at it $it"; break; fi
done
echo "dc loop $TAG done at $(date)"
