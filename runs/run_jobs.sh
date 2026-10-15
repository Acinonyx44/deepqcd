#!/bin/bash
# Runs every line of runs/jobs.txt with 4 workers; one log per job in runs/logs.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cat runs/jobs.txt | xargs -P 4 -I{} bash -c 'log=runs/logs/$(echo "{}" | tr " /" "__").log; echo "start {}" >> runs/logs/_progress; .venv/bin/python {} > "$log" 2>&1; echo "end($?) {}" >> runs/logs/_progress'
echo ALL DONE >> runs/logs/_progress
