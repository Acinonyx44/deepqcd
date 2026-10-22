# Six finance use cases: where DeepQCD earns its keep

All six use the same pieces (`usecases_common.py`):
- **DeepQCD:** the original network and training from `original_qcd.py`, with three seeds.
- **A per-step logistic model:** the same features, learned, but with no memory.
- **The window-limited CUSUM:** `window_dec_stat` on the Gaussian-LLR projection.
- **Industry rules:** the rules each business already runs.

The test is always the paper's transient protocol on native streams:
- **PFA** is an alarm before the event.
- **PD** is a first alarm during the event.
- **Threshold:** Eq. 11, the threshold with the largest PD subject to PFA <= alpha.

Each `usecase_*.py` writes `runs/usecases/<name>.json`. The page is built with `report/build_usecases.py`, which reads
the results and `report/usecases_text.json`. Data comes from `fetch_usecases.sh` and `squeeze_fetch.py`; all of it is
public. `runs/run_usecases.sh` runs all six one after another.

## Results

| Use case | Data | Test events | DeepQCD | Best other | Verdict |
|---|---|---|---|---|---|
| Crypto toxic flow (market makers) | Binance ETHUSDT perp trades, 112 days | 360 fast runs | 0.46 at PFA 0.05 (0.44-0.48) | 0.21 (price move, 5 s) | win |
| Card fraud (issuers) | IBM TabFormer, 24.4M transactions | 1,680 bursts | 0.54 at PFA 0.01 (0.50-0.61) | 0.33 (logistic) | win |
| Laundering (banks) | IBM AMLworld HI-Small, 5.1M transfers | 1,676 accounts | 0.41 at 0.01; 0.62 at 0.05 | 0.27 at 0.01 (logistic); 0.64 at 0.05 (ACH + new counterparty rule) | win at 1%, tie at 5% |
| Short squeezes (hedge funds) | Yahoo + FINRA short volume, 1,398 stocks | 260 rallies | 0.76 at 0.05 | 0.75 (CUSUM) | tie |
| Card default (banks) | UCI, 30,000 clients x 6 months | 2,019 defaulters | 0.31 at 0.05; 0.63 at 0.2 | 0.30 (CUSUM); 0.64 at 0.2 (months late) | tie |
| Polymarket market making (P&L) | 298 held-out markets | - | either alarm: +0.04 c per $100 | busy rule: -0.16; always quoting: -12.66 | alarms help; DeepQCD adds little |

### Crypto toxic flow

- **Events.** Swings of at least 15 bps completed within 60 s, found by a zigzag (reversal 5 bps). There are about 80
  a day.
- **Features.** Ten per second.
- **Split.** Train 2026-07-01..09-30, test 2026-10-01..20. Test runs follow 5 quiet minutes.
- **Detection.** PD at PFA 0.05 is 0.46 for DeepQCD, against 0.21 for the price rule, 0.16 for the CUSUM and 0.14 for
  the logistic model. At PFA 0.01 it is 0.34 against 0.12.
- **Smaller runs.** The gain is on 15-20 bps runs: 0.43 against at most 0.13.
- **Value.** At PFA 0.05, DeepQCD avoids 2.6-2.7 bps of adverse move per run. The best rule avoids 1.6 and the CUSUM
  1.5. DeepQCD alarms later (median 22.5 s, with 5.7 of 20.5 bps still to come) but catches far more runs.
- **More data helped.** With 40 days (30 for training) and 10 quiet minutes before each test run, DeepQCD caught 0.35
  of 102 runs, against 0.20.

### Card fraud (TabFormer)

- **Events.** A burst is fraud transactions at most 20 transactions apart.
- **Features.** Fifteen per transaction.
- **Split.** Bursts before 2016 train; bursts from 2016 test, each after 30 clean transactions.
- **Detection.** At PFA 0.01: DeepQCD 0.54, logistic 0.33, CUSUM 0.28, rules at most 0.09. At PFA 0.05: 0.69, 0.66,
  0.64 and 0.23.
- **By how the fraud starts.**
  - Online: DeepQCD 0.76, against 0.60 for the logistic model.
  - In person: DeepQCD 0.61, against 0.73 for the logistic model and 0.77 for the CUSUM.
- **Speed.** The median number of fraudulent transactions before a caught burst's alarm is 1, for every detector.

### Laundering (AMLworld)

- **Stream.** Per account, one step per transaction, with 15 features. The change runs from the first laundering
  transfer to one transfer after the last.
- **Split.** Schemes before 2022-09-06 train; schemes after 09-06 12:00 test. False alarms are counted on 15
  transactions for every stream.
- **One rule carries most of it.** 87% of laundering transfers are ACH (12% of all transfers). The rule "ACH with a
  new counterparty" reaches 0.64 at PFA 0.05, but being binary it gets 0 at PFA 0.01.
- **DeepQCD.** 0.41 at PFA 0.01 (logistic 0.27, CUSUM 0.02) and 0.62 at 0.05.
- **Misses.** Laundering that doesn't start with ACH (15% of accounts) is missed by every detector.
- **Elliptic++ was dropped.** Its wallets live a median of 1 time step, its labels are static, and only 85 illicit
  wallets have 3 or more steps. That makes it a classification problem, not change detection.

### Polymarket market-making replay

- **Setup.** We take the other side of every taker trade while quoting, scored by the 5-minute markout. Thresholds
  are set on training markets so each alarm is on in 2% of quiet minutes. Decisions use the previous minute.

| Strategy | Volume filled | P&L (cents per $100 of all volume) |
|---|---|---|
| always quote | 100% | -12.66 |
| step back when busy | 32% | -0.16 |
| step back on metaorder alarm (CUSUM, side being hit) | 59% | -3.81 |
| step back on alarm unless churn | 99% | -12.55 |
| step back on churn alarm (DeepQCD) | 32% | -0.07 |
| step back on either alarm | 31% | +0.04 |
| oracle (true metaorders, side being hit) | 71% | +1.86 |

- **Markouts by regime** (always quoting, cents per share): -0.23 during metaorders, -0.37 during churn, -0.06
  otherwise.
- **Churn is the most toxic flow.** The churners are mostly skilled scalpers, not wash traders, so keeping quotes up
  through churn loses money.
- **Not modelled.** The replay ignores queue position and Polymarket's liquidity rewards.

### Short squeezes

- **Universe.** 1,398 stocks with a median FINRA short-volume share of at least 40%, from 2021 to 2026.
- **Events.** Rallies of at least 50% within 10 days on 3x volume. Rallies before 2025 train; 260 from 2025 on test.
- **Detection.** At PFA 0.05: DeepQCD 0.76, CUSUM 0.75, 2-day return rule 0.69, logistic 0.69. At PFA 0.01: 0.70
  against 0.67.
- **Value.** Alarms come about 5 days in, with about 17% of a 93% rally left. DeepQCD and the CUSUM each avoid about
  13% per rally.
- **The best value is the volume-spike rule.** It catches only 0.41, but early (43% of the rally still ahead), so it
  avoids 17.6% per rally.

### Card default (UCI)

- **Setup.** Six monthly steps per client, with 8 features (repayment status included, as the bank sees it). Labels
  are 1 in every month of a defaulter (the Bayesian setting).
- **Detection.**
  - At PFA 0.05: DeepQCD 0.31, CUSUM 0.30, logistic 0.25, "months late now" 0.10.
  - At PFA 0.2, the usual credit operating point, everyone is at 0.62-0.64.
- **Stronger data needs registration.** Freddie Mac and Fannie Mae loan histories would be the better test.

## Reading

- **DeepQCD wins** when three things hold:
  - thousands of labelled events;
  - a fast change that shows as several cues judged against the entity's own history;
  - a strict false-alarm level.

  That is crypto runs, card fraud, and AML at 1%. The per-step logistic model, with the same features, loses by
  0.1-0.3 in PD, so the gain comes from reading the sequence.
- **It ties** when the signal is price-led and slow (squeezes) or the history is short and the current status says it
  all (credit).
- **For market makers, alarms that pull quotes in busy or toxic periods turn losses into break-even.** But a busy
  rule does nearly as much, and only a better metaorder detector closes the gap to the oracle.
