#!/bin/bash
# Usage: hpc/wait_jobs.sh <jobid>[,<jobid>...]
# Blocks until every job has reached a terminal state according to sacct (a failing sacct call is
# retried, never read as "done"), then prints the final states.
ids="$1"
n=$(echo "$ids" | tr ',' '\n' | wc -l)
while true; do
  states=$(sacct -j "$ids" -X -n -P -o State 2>/dev/null); rc=$?
  if [ $rc -eq 0 ]; then
    nterm=$(echo "$states" | grep -cE '^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)')
    [ "$nterm" -ge "$n" ] && break
  fi
  sleep 60
done
sacct -j "$ids" -X -o JobID,JobName%14,State,Elapsed,ExitCode
