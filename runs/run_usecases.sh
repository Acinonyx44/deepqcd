#!/usr/bin/env bash
# The six finance use cases, one after another (each uses 4 threads). Logs in runs/logs/usecase_<name>.log.
cd "$(dirname "$0")/.."
for job in fraud amlworld toxic credit mm_pnl; do
  .venv/bin/python usecase_$job.py > runs/logs/usecase_$job.log 2>&1
done
# the squeeze case needs the Yahoo download (squeeze_fetch.py) to be complete
until grep -q "^yahoo:" runs/logs/squeeze_fetch.log; do sleep 30; done
.venv/bin/python usecase_squeeze.py > runs/logs/usecase_squeeze.log 2>&1
echo all done > runs/logs/usecases_done.txt
