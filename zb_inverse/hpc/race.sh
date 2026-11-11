#!/bin/bash
# Usage: hpc/race.sh <id1,id2,...>   Duplicate submissions of one run on different partitions:
# as soon as one starts running, cancel the others; then wait for it to finish and print its state.
# Every scheduler call has a time limit and is retried (the controller can hang for minutes): a hung squeue must not
# leave the duplicates queued, since a second copy that starts would write into the same output directory.
ids="$1"
while true; do
  out=$(timeout 120 squeue -j "$ids" -h -o "%i %T" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then sleep 30; continue; fi
  run=$(echo "$out" | awk '$2=="RUNNING"{print $1; exit}')
  if [ -n "$run" ]; then
    others=$(echo "$ids" | tr ',' '\n' | grep -v "^$run$" | tr '\n' ' ')
    for i in $(seq 1 20); do timeout 150 scancel $others 2>/dev/null && break; sleep 20; done
    echo "started $run; cancelled $others at $(date)"
    break
  fi
  if [ -z "$out" ]; then echo "none left in queue: $(timeout 120 sacct -j "$ids" -X -n -P -o JobID,State | tr '\n' ' ')"; exit 0; fi
  sleep 30
done
while true; do
  st=$(timeout 120 sacct -j "$run" -X -n -P -o State 2>/dev/null) || { sleep 30; continue; }
  case "$st" in COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) echo "$run $st at $(date)"; exit 0;; esac
  sleep 60
done
