#!/usr/bin/env bash
# Downloads for the six finance use cases (usecases_*.py). Everything goes to data/ (not tracked).
#   binance  ETHUSDT perpetual aggregated trades, 2026-07-01..2026-10-20 (data.binance.vision, public)
#   amlworld IBM AMLworld HI-Small (Altman et al. 2023, CDLA-Sharing-1.0), Hugging Face mirror of the Kaggle release
#   ellipticpp  Elliptic++ wallets (Elmougy & Liu, KDD 2023), Hugging Face mirror (AI4FinTech/ellipticpp)
#   tabformer IBM TabFormer synthetic credit-card transactions (Padhi et al. 2021), GitHub LFS
#   uci      Default of credit card clients (Yeh & Lien 2009), UCI repository
# Short-squeeze data (Yahoo daily prices, FINRA daily short volume) is fetched by usecase_squeeze.py.
set -euo pipefail
cd "$(dirname "$0")/data"
what=${1:-all}
get() { [ -s "$2" ] || curl -sS -L --retry 4 --retry-delay 3 -o "$2.part" "$1" && { [ -s "$2" ] || mv "$2.part" "$2"; }; }
if [[ $what == all || $what == binance ]]; then
  mkdir -p binance && d=2026-07-01
  while [[ $d < 2026-10-21 ]]; do
    get "https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-$d.zip" "binance/ETHUSDT-aggTrades-$d.zip"
    d=$(date -I -d "$d + 1 day")
  done
fi
if [[ $what == all || $what == amlworld ]]; then
  mkdir -p amlworld
  for f in HI-Small_Trans.csv HI-Small_accounts.csv HI-Small_Patterns.txt; do
    get "https://huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small/resolve/main/$f" "amlworld/$f"
  done
fi
if [[ $what == all || $what == ellipticpp ]]; then
  mkdir -p ellipticpp
  for f in wallets_features_classes_combined.csv AddrTx_edgelist.csv TxAddr_edgelist.csv txs_classes.csv; do
    get "https://huggingface.co/datasets/AI4FinTech/ellipticpp/resolve/main/$f" "ellipticpp/$f"
  done
fi
if [[ $what == all || $what == tabformer ]]; then
  mkdir -p tabformer
  get "https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz" tabformer/transactions.tgz
  [ -f tabformer/card_transaction.v1.csv ] || tar xzf tabformer/transactions.tgz -C tabformer
fi
if [[ $what == all || $what == uci ]]; then
  mkdir -p uci
  get "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip" uci/default.zip
  (cd uci && unzip -o -q default.zip)
  # the UCI file is .xls; convert it once to CSV with xlrd in a throwaway environment (keeps the project's deps as they are)
  if [ ! -s uci/default.csv ]; then
    python3 -m venv uci/xlsenv && uci/xlsenv/bin/pip install -q xlrd
    uci/xlsenv/bin/python -c "
import xlrd, csv
sh = xlrd.open_workbook('uci/default of credit card clients.xls').sheet_by_index(0)
w = csv.writer(open('uci/default.csv', 'w', newline=''))
for i in range(1, sh.nrows):
    w.writerow([int(v) if isinstance(v, float) and v.is_integer() else v for v in sh.row_values(i)])
"
  fi
fi
echo "done $what"
