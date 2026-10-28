#!/usr/bin/env bash
# Second pass of the bank studies after the streaming curves were given a fine grid of alert rates (and seven targets).
# Waits for runs/run_final.sh to finish; keeps the first-pass results in runs/bank/v1.
cd "$(dirname "$0")/.."
while ! grep -q "^=== all done" runs/final_queue.log 2>/dev/null; do sleep 60; done
export QCD_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3     # as in the first pass, so the window results repeat
mkdir -p runs/bank/v1
for d in aml-hi-small aml-li-small sparkov tabformer aml-hi-medium; do
    [ -f "runs/bank/v1/$d.json" ] || cp "runs/bank/$d.json" "runs/bank/v1/$d.json"
    if [ -s "runs/bank/${d}_stream.npz" ]; then echo "=== $d done before"; continue; fi
    echo "=== bank $d $(date +%T)"; .venv/bin/python bank_final.py "$d" > "runs/bank/$d.log" 2>&1; tail -n 2 "runs/bank/$d.log"
done
echo "=== second pass done $(date +%T)"
