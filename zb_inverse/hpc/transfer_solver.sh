#!/bin/bash
# Solver side of one soil transfer on site data (SCORING / CALIBRATION ONLY: no PINN run reads its output).
#   soil      the start of scripts/calibrate_soil.py with the lowest misfit in output/soil_<site>_<cal>
#   data sets with that soil (scripts/make_real.py): the calibration window itself (in-sample check of the
#             usable-soil rule) and every test window
#   inversion scripts/fem_referee.py of each data set from the four starts, --box
# Runs on the login node as a driver only (bash + sbatch; every Python step is a batch job).
# Usage: nohup hpc/transfer_solver.sh <site> <cal> "<obs.depth_m>" <test> [<test> ...] > logs/transfer_<site>_<cal>.log 2>&1 &
#   e.g. hpc/transfer_solver.sh hb2 sep19 "[0.1524,0.1524]" mar20 jul20 aug21
#   reads  configs/real_<site>_<window>.yaml, output/soil_<site>_<cal>/start?.json
#   writes output/real_<site>_<cal>_own, output/real_<site>_<test>_soil<cal>,
#          output/referee/<site>_own_<cal>_s<k>, output/referee/<site>_<cal>on_<test>_s<k>   (k = 0..3)
#   RT=08:00:00 : time limit of the solver inversions (default 03:00:00)
#   SOIL_SET="--set soil.theta_s=.. ..." : dry run of the chain with that soil; every name then ends in _pre
#   OWN=0 : test windows only; the data set and inversions of the calibration window exist and are not rewritten
set -u
cd /storage/project/r-rbras6-0/jjason3/pinn/zb_inverse
SITE=$1; CAL=$2; DEPTH=$3; shift 3
O=$PWD/output
RT=${RT:-03:00:00}   # time limit of one solver inversion (US-HB2: 10 - 25 min per LM iteration)

submit() {   # name, cpus, mem, time, script args... ; as hpc/param_pipeline.sh: retried until the controller answers
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

ok() {   # files written on a compute node can take a while to appear on the login node: retry up to 2 min
  local i
  for i in $(seq 1 24); do [ -f "$1" ] && return 0; sleep 5; done
  echo "MISSING $1 -> stop at $(date)"; exit 1
}

X=""
echo "transfer $SITE $CAL -> $* at $(date)"
if [ -n "${SOIL_SET:-}" ]; then   # dry run of the chain with a soil given by hand (not the rule's): every name ends in _pre
  S=$SOIL_SET; X=_pre
  echo "DRY RUN, soil from SOIL_SET (the results *_pre do not count): $S"
else
  n=$(ls $O/soil_${SITE}_$CAL/start?.json 2>/dev/null | wc -l)
  [ "$n" -eq 4 ] || { echo "output/soil_${SITE}_$CAL holds $n of the four starts -> stop"; exit 1; }
  best=$(for f in $O/soil_${SITE}_$CAL/start?.json; do echo "$(jq -r '.rmse' $f) $f"; done | sort -g | head -1 | cut -d' ' -f2)
  S=$(jq -r '.soil | "--set soil.theta_s=\(.theta_s) --set soil.theta_r=\(.theta_r) --set soil.alpha=\(.alpha) --set soil.n=\(.n) --set soil.Ks=\(.Ks)"' $best)
  for f in $O/soil_${SITE}_$CAL/start?.json; do echo "  $(basename $f): $(jq -c '[.rmse, .on_bound, .soil.theta_s, .soil.theta_r, .soil.alpha, .soil.n, .soil.Ks]' $f)"; done
  echo "soil: $best"; echo "  $S"
fi

one() {   # window, data-set dir, referee name
  local w=$1 d=$2 r=$3 j st k s ids=""
  j=$(submit mr_${r} 4 16G 00:40:00 scripts/make_real.py --config configs/real_${SITE}_$w.yaml $S --set output.dir=$d)
  st=$(wait_job $j); echo "$r: make_real job $j $st"
  ok $d/real.npz
  echo "$r: FEM over the record vs sensor $(jq -c '.fem_vs_obs["fT=gldas, record water table"]' $d/real_summary.json)"
  k=0
  for s in hydrostatic hydrostatic_flat constant:1.0 constant:0.3; do
    j=$(submit rf_${r}_s$k 2 8G $RT scripts/fem_referee.py --config configs/p1_param_synth_v3.yaml \
        --set twin_npz=$d/real.npz --set obs.depth_m=$DEPTH --set obs.noise_sigma=0.0 --set obs.q_noise_frac=0.0 \
        --box --start $s --out $O/referee/${r}_s$k)
    ids="$ids $j"; k=$((k + 1))
  done
  k=0
  for j in $ids; do
    st=$(wait_job $j)
    if [ -f $O/referee/${r}_s$k/summary.json ] || { sleep 60; [ -f $O/referee/${r}_s$k/summary.json ]; }; then
      echo "${r}_s$k: job $j $st $(jq -c '[.start, .cost, (.zb_rmse*1e4|round/100), (.zb_bias*1e4|round/100), (.zb_maxabs*1e4|round/100), .coef_on_bound, .iterations]' $O/referee/${r}_s$k/summary.json)"
    else
      echo "${r}_s$k: job $j $st, no summary (log logs/cpu_$j.out)"
    fi
    k=$((k + 1))
  done
}

if [ "${OWN:-1}" = 1 ]; then one $CAL $O/real_${SITE}_${CAL}_own$X ${SITE}_own_$CAL$X & fi
for t in "$@"; do one $t $O/real_${SITE}_${t}_soil$CAL$X ${SITE}_${CAL}on_$t$X & done
wait
echo "transfer $SITE $CAL done at $(date)"
