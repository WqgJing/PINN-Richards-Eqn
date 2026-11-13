#!/bin/bash
# One forward PINN at the path of the well's spline of a site data set, scored against the solver at the same path.
# DIAGNOSTIC of the operator only: it uses the observed water table, and no inversion reads its output.
# Settings and submission pattern of the forward stage of hpc/dc_loop.sh, data settings of hpc/real_full_obs.sh.
# Usage: setsid nohup hpc/fwd_truth.sh <run name> <data set dir of scripts/make_real.py> "<obs.depth_m>" \
#            [extra --set options] > logs/fwd_<run name>.log 2>&1 < /dev/null &
#   writes runs/<run name>_local.npz  centre = truth_coef of the data set; diagonal covariance (point mode never
#                                     samples it)
#          runs/<run name>            the forward PINN, validation.json (theta at the sensor against the solver),
#                                     truth_profile.npz (hourly error series, --sensor fem)
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse || exit 1
name=$1; D=$(realpath $2); DEPTH=$3; shift 3
C=configs/p1_param_synth_v3.yaml
GT=04:00:00
TS="--set twin_npz=$D/real.npz --set obs.depth_m=$DEPTH --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 --set model.forcing_memory_taus_days=[0.015625,0.0625,0.25,1,4,16] --set model.forcing_split_sign=true --set train.n_box=128 --set train.box_n_t=16"
FWD_SET="--set sampling.mode=point --set train.adam_steps=25000 --set train.lr=1.0e-3 --set train.lr_final=1.0e-5 --set train.lbfgs_blocks=15"

submit() {   # job name, partition, sbatch args...: retried until the controller answers
  local name=$1 part=$2 j i; shift 2
  while true; do
    j=$(timeout 180 sbatch --parsable -J $name -p $part "$@" 2>/dev/null) && [ -n "$j" ] && break
    for i in 1 2 3 4 5; do timeout 180 scancel -u $USER --name=$name --partition=$part 2>/dev/null && break; sleep 30; done
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

# local file of point mode, assembled without Python (login node): an .npz is a zip of .npy files
L=$PWD/runs/${name}_local.npz
T=$(mktemp -d)
unzip -p $D/real.npz truth_coef.npy > $T/center.npy
n=$(( ($(stat -c %s $T/center.npy) - 128) / 8 ))
[ "$n" -ge 4 ] || { echo "truth_coef of $D not read ($n coefficients) -> stop"; exit 1; }
perl -e 'my $n = shift; my $d = "{\x27descr\x27: \x27<f8\x27, \x27fortran_order\x27: False, \x27shape\x27: ($n, $n), }";
         print "\x93NUMPY\x01\x00", pack("v", 118), $d, " " x (117 - length($d)), "\n";
         for my $i (0 .. $n - 1) { for my $j (0 .. $n - 1) { print pack("d<", $i == $j ? 0.01 : 0.0) } }' $n > $T/cov.npy
rm -f $L; /usr/bin/zip -q -j $L $T/center.npy $T/cov.npy; rm -r $T
echo "$name: $n coefficients, local file $L ($(unzip -l $L | tail -1))"

set -- scripts/param_train.py --config $C $FWD_SET --set sampling.local_file=$L --set train.seed=0 $TS "$@" \
  --set output.dir=runs/$name
a=$(submit $name gpu-h200 --gres=gpu:h200:1 -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
b=$(submit $name gpu-a100 --gres=gpu:A100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
c=$(submit $name gpu-v100 --gres=gpu:V100:1 -q inferno -A gts-rbras6-coda20 --time=$GT hpc/run_script_gpu.slurm "$@")
echo "$name: submitted $a $b $c at $(date)"
hpc/race.sh $a,$b,$c > logs/race_$name.log 2>&1
echo "$name: $(tail -1 logs/race_$name.log)"
[ -f runs/$name/model.pt ] || { sleep 60; [ -f runs/$name/model.pt ]; } || { echo "$name: no model -> stop at $(date)"; exit 1; }

v=$(submit val_$name cpu-small --account=gts-rbras6 --qos=inferno -n1 -c8 --mem=16G -t 02:00:00 hpc/run_script_cpu.slurm \
    scripts/param_validate.py --members runs/$name --out runs/$name/validation.json --n-local 0 --n-prior 0)
p=$(submit tep_$name cpu-small --account=gts-rbras6 --qos=inferno -n1 -c4 --mem=16G -t 00:40:00 hpc/run_script_cpu.slurm \
    scripts/truth_error_profile.py --members runs/$name --sensor fem --out runs/$name/truth_profile.npz)
echo "$name: validation job $v $(wait_job $v) (log logs/cpu_$v.out), error profile job $p $(wait_job $p) (log logs/cpu_$p.out)"
grep "^\[val\] truth" logs/cpu_$v.out | cut -c1-300
echo "forward $name done at $(date)"
