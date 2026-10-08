#!/usr/bin/env bash
# Rebuild every number in notes/results.md and every figure in figures/ from scratch.
# Logs go to runs/<name>.log (git-ignored). Takes about an hour on a laptop CPU.
#
#   ./regen.sh            run everything
#   ./regen.sh --retrain  also refit the detect.py networks instead of reusing runs/models/
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p runs

run() {  # run <log name> <script> [args...]
    local name=$1; shift
    echo "== $name"
    "$PY" "$@" 2>&1 | tee "runs/$name.log"
}

run tests     tests/test_qcd.py
run iid       deepqcd_iid.py
run ar        deepqcd_ar.py
run transient deepqcd_transient.py
run vol       deepqcd_vol.py
run vol_sq    deepqcd_vol.py --sq
if [ -d data/SKAB ]; then  # real datasets: ./fetch_data.sh first
    run real deepqcd_real.py
    run real_hybrid deepqcd_real.py --hybrid
    run real_tau1 deepqcd_tau1.py
fi
for src in iid ar garch; do
    run "detect_${src}_tau500" detect.py --source "$src" --tau 500 --fap 1000 "$@"
done
