#!/usr/bin/env bash
# The final bank and prediction-market studies, one after another (three threads each, so a one-thread check can run beside them). Logs in runs/bank and runs/pm.
cd "$(dirname "$0")/.."
mkdir -p runs/bank runs/pm
export QCD_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3     # one core stays free for checks
bank() { echo "=== bank $* $(date +%T)"; .venv/bin/python bank_final.py "$@" > "runs/bank/$1$([ "$2" = --curve ] && echo _curve).log" 2>&1; tail -n 3 "runs/bank/$1$([ "$2" = --curve ] && echo _curve).log"; }
pm() { echo "=== pm $* $(date +%T)"; .venv/bin/python pm_final.py --target "$@" > "runs/pm/$1$([ "$2" = --wallets ] && echo _wallets).log" 2>&1; tail -n 3 "runs/pm/$1$([ "$2" = --wallets ] && echo _wallets).log"; }
bank sparkov
bank aml-li-small
pm metaorder
pm churn
pm informed
pm informed --wallets
pm jump
echo "=== pm replay $(date +%T)"; .venv/bin/python pm_replay.py > runs/pm/replay.log 2>&1; tail -n 3 runs/pm/replay.log
bank tabformer
bank aml-hi-medium
bank aml-hi-medium --curve
bank tabformer --curve
echo "=== all done $(date +%T)"
