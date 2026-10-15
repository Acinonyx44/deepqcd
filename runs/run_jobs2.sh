#!/bin/bash
# Waits for runs/run_jobs.sh to finish, then runs runs/jobs2.txt the same way.
cd "$(dirname "$0")/.."
until grep -q "ALL DONE" runs/logs/_progress 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cat runs/jobs2.txt | xargs -P 4 -I{} bash -c 'log=runs/logs/$(echo "{}" | tr " /" "__").log; echo "start {}" >> runs/logs/_progress; .venv/bin/python {} > "$log" 2>&1; echo "end($?) {}" >> runs/logs/_progress'
echo ALL2 DONE >> runs/logs/_progress
