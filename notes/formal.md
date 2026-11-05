# Formal results page: five use cases and prediction markets

The page for the professor, in the form of the study summary: one section per problem with its data diagram, results
table, PD-PFA curves and a reading. It collects the best evidence for each use case.
- **Calibrated protocol** (`bank_final.py`, `pm_final.py`, `pm_replay.py`; `notes/final.md`): laundering, card fraud
  and Polymarket.
- **The paper's Eq. 11 on the test streams, three seeds** (`usecase_*.py`; `notes/usecases.md`): the toxic-flow
  alarm, short squeezes and card default.

## Building it

- `report/build_formal.py` builds `report/formal.html` from `report/formal_template.html` (structure and prose) and
  `report/formal_src/` (figures and tables).
- Run it with Python 3.12 or later: `python3.13 report/build_formal.py`. It needs no packages.
- **Where `report/formal_src/` comes from.** The run outputs in `runs/` were lost when the container was reset; they are
  not in git. So `report/extract_published.py` took the figures and tables from the two published pages: the Study
  Summary and the Final Studies.
- **New in this page:**
  - the summary dot plot with paired differences;
  - data diagrams updated to the scaled data;
  - the five-panel bank figures split into a laundering part and a card part.

## Sections

| Section | Content | Protocol |
|---|---|---|
| Summary | at-a-glance table; DeepQCD against the best other detector | both |
| 1 | transient model, data pipeline, detectors, the two protocols, how to read the numbers | - |
| 2 Laundering | AMLworld HI-Small, LI-Small, HI-Medium: window test, streaming, patterns, learning curves; AMLSim in brief | calibrated |
| 3 Toxic-flow alarm | Binance ETHUSDT perpetual, one-second steps, fast runs | Eq. 11 |
| 4 Card fraud | TabFormer and Sparkov: window test, streaming | calibrated |
| 5 Short squeezes | 1,398 heavily shorted stocks, daily | Eq. 11 |
| 6 Card default | UCI, six months | Eq. 11 |
| 7 Polymarket | metaorders, churn, informed flow (with and without wallet features), price jumps | calibrated |
| 8 Market making | replay of the alarms on 748 test markets | calibrated |
| 9-10 | conditions for a win, deployment advice, limits and next steps | - |
| A | features and rules of every study | - |

## Headline

| Use case | DeepQCD vs best other | Verdict |
|---|---|---|
| Laundering | PFA 0.01: 0.55 vs 0.47, 0.34 vs 0.30, 0.38 vs 0.26 (GB), every interval above zero | win at 0.01, tie at 0.05 |
| Toxic-flow alarm | PFA 0.05: 0.46 vs 0.29 (GB) | win |
| Card fraud | PFA 0.01: 0.69 vs 0.73 (GB) in the window test; streaming 0.85 vs 0.82 at 5 per 1,000 | tie with GB |
| Short squeezes | PFA 0.05: 0.76 vs 0.75 (CUSUM) | tie |
| Card default | PFA 0.05: 0.31 vs 0.30 (CUSUM) | tie |
| Polymarket metaorders | PFA 0.05: 0.43 vs 0.38 (GB) | win |
| Polymarket churn | PFA 0.01: 0.44 vs 0.37 (GB) | win at 0.01 |
| Informed flow, price jumps | at most 0.2; jumps only in sports (0.31) | no detector works |
| Market making | no alarm changes P&L measurably | no edge yet |

## Open

- **Rerun three studies with calibration.** The toxic-flow alarm, squeezes and card default under the calibrated
  protocol, with gradient boosting on the last two. Their data are public and can be fetched again: `fetch_usecases.sh`
  and `usecase_squeeze.py`. Yahoo rate-limits the squeeze download.
