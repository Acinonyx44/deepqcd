#!/usr/bin/env bash
# The final bank and prediction-market studies, one after another (three threads each, so a one-thread check can run
# beside them). Logs and results in runs/bank and runs/pm. A job whose results file exists is skipped, so the queue can
# be restarted after an interruption.
cd "$(dirname "$0")/.."
mkdir -p runs/bank runs/pm
export QCD_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3     # one core stays free for checks
bank() {
    local out="runs/bank/$1$([ "$2" = --curve ] && echo _curve)"
    [ -s "$out.json" ] && { echo "=== bank $* done before"; return; }
    echo "=== bank $* $(date +%T)"; .venv/bin/python bank_final.py "$@" > "$out.log" 2>&1; tail -n 3 "$out.log"
}
pm() {
    local out="runs/pm/$1$([ "$2" = --wallets ] && echo _wallets)"
    [ -s "$out.json" ] && { echo "=== pm $* done before"; return; }
    echo "=== pm $* $(date +%T)"; .venv/bin/python pm_final.py --target "$@" > "$out.log" 2>&1; tail -n 3 "$out.log"
}
bank aml-hi-small
bank sparkov
bank aml-li-small
pm metaorder
pm churn
pm informed
pm informed --wallets
pm jump
if [ -s runs/pm/replay.json ]; then echo "=== pm replay done before"; else
    echo "=== pm replay $(date +%T)"; .venv/bin/python pm_replay.py > runs/pm/replay.log 2>&1; tail -n 3 runs/pm/replay.log; fi
bank tabformer
bank aml-hi-medium
bank aml-hi-medium --curve
bank tabformer --curve
echo "=== all done $(date +%T)"
