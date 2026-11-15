#!/bin/bash
# early_flux.sh <tag> ...: held-out PDE and flux loss of the running wide networks (losses only, from the live job logs)
cd "$(dirname "$0")/.." || exit 1
for tag in "$@"; do for k in 0 1 2 3; do
  j=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" | awk -v n="${tag}_w$k" '$2==n{print $1}' | head -1)
  [ -z "$j" ] && j=$(grep -E "^${tag}_w$k: [0-9]+ " logs/pipe_$tag.log 2>/dev/null | awk '{print $2}' | head -1)
  f=logs/gpu_$j.out; [ -n "$j" ] && [ -f "$f" ] || { echo "${tag}_w$k: no log yet"; continue; }
  grep -E "^\[(init|adam)\] step=" "$f" | sed -E 's/.*step=([0-9]+) .*val_pde=([^ ]+) val_flux=([^ ]+) .*/\1 \2 \3/' | awk -v r="${tag}_w$k" 'NR==1{p0=$2;f0=$3} {s=$1;p=$2;f=$3} $1==5000{f5=$3} END{if(NR==0){print r": no step line yet"; exit} printf "%s: step 0 PDE %.2f flux %.1f | flux factor at step 5000: %s | latest step %d: PDE %.3f flux %.3f, factor %.1f (%.0f %% of node rms)\n", r,p0,f0,(f5>0?sprintf("%.1f",f0/f5):"not reached"),s,p,f,f0/f,10*sqrt(f)}'
done; done
