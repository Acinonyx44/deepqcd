# Money laundering on IBM AMLSim: where DeepQCD wins

**Question.** When an account is drawn into a laundering scheme, its behaviour changes for the days the scheme runs
and then returns to normal: a transient change. With exact labels in quantity, does DeepQCD beat the window-limited
CUSUM and the rules banks use?

## Files

- `fetch_amlsim.sh`: builds IBM AMLSim from source (commit 7338a4b) and simulates six banks into
  `data/amlsim/qcd10K_s<seed>/` (not tracked). It takes about 5 minutes.
  - **MASON.** The MASON 20 jar can no longer be downloaded, so the two classes AMLSim uses are compiled from MASON's
    GitHub source.
  - **networkx.** networkx 1.11 is patched for Python 3.11 (`fractions.gcd` becomes `math.gcd`).
  - **Generator.** The generator gets a `random.sample` shim for sets, plus a fix for alert members with no edge.
  - The simulation logic is unchanged.
- `aml_qcd.py`: the experiment. Run `.venv/bin/python aml_qcd.py`, which takes about 4 minutes and writes
  `runs/aml/results.json`.

## Setup

- **Banks.** Each bank is the 10K parameter set (about 10,000 accounts, 5,300 of them active) over 720 days, with
  300 schemes in eight typologies: fan-in, fan-out, cycle, bipartite, stack, random, scatter-gather and
  gather-scatter. A scheme has 5-10 accounts, lasts 5-20 days and moves $100-1,000 per transfer (the normal range).
  That makes about 1,600 laundering accounts per bank. Banks 1-4 train; banks 5 and 6 test.
- **Features.** Nine daily features per account: transfers in and out, amounts in and out, new counterparties in and
  out, distinct counterparties over 7 days, pass-through (min of in and out over 3 days), and hub contact (the largest
  7-day counterparty count among today's counterparties).
- **Change.** From the account's first laundering transfer to 3 days after its last.
- **Test streams.** The 120 days before the change and then the change, for 2,394 laundering accounts (268 hubs, 2,126
  members), plus 4,000 windows of accounts never involved. Alarms count from day 14.
- **Detectors.**
  - **DeepQCD**: the original network and training on about 24,000 windows, three seeds.
  - **WL-CUSUM**: `window_dec_stat` over 7 days on the Gaussian-LLR projection.
  - **Seven rules**: pass-through, counterparties, velocity, hub contact, new counterparties, dormant account
    reactivated, and dormant plus new counterparty.
  - **Rules combined**: each rule scaled by its 99th percentile on normal days, then the largest.

## Results

The first three columns are PD at the given PFA. The last three split PD at PFA 0.05 by group: hub accounts, member
accounts, and accounts active in the 120 days before their scheme (769).

| Detector | PFA 0.01 | 0.05 | 0.1 | hubs | members | active before |
|---|---|---|---|---|---|---|
| DeepQCD (3 seeds) | 0.69 | **0.82** (0.87, 0.80, 0.80) | 0.88 | 0.68 | 0.84 | **0.50** |
| WL-CUSUM | 0.06 | 0.12 | 0.63 | 0.59 | 0.06 | 0.12 |
| Rules combined | 0.68 | 0.70 | 0.73 | 0.43 | 0.73 | 0.06 |
| Rule: dormant + new counterparty | 0.68 | 0.70 | 0.73 | 0.43 | 0.73 | 0.06 |
| Rule: dormant account reactivated | 0.68 | 0.68 | 0.70 | 0.41 | 0.71 | 0.01 |
| Rule: new counterparties (3 days) | 0.02 | 0.07 | 0.17 | 0.40 | 0.03 | 0.06 |
| Rule: counterparties (7 days) | 0.00 | 0.06 | 0.09 | 0.30 | 0.03 | 0.06 |
| Rule: velocity (7 days) | 0.00 | 0.04 | 0.09 | 0.20 | 0.02 | 0.04 |
| Rule: pass-through (3 days) | 0.02 | 0.02 | 0.04 | 0.06 | 0.02 | 0.07 |
| Rule: hub contact | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |

The median delay of DeepQCD is 0 days: most laundering accounts are flagged on their first laundering day.

## What the network keys on

- **Dormancy plus a new counterparty.** 67% of laundering accounts made no transfer in the 120 days before their scheme
  (39% of normal windows). 93% deal with a new counterparty on their first laundering day (14.5% of normal active
  days). DeepQCD learned the combination. The dormant-account rule captures most of it, which is why it is the only
  rule above 0.08.
- **A new partner for an account with one partner.** For accounts active before their scheme, 76-79% have a new
  counterparty on day one (14% of normal active days). These accounts had typically dealt with one partner in the
  previous 120 days. The network judges the new counterparty against the account's own history; a fixed threshold
  cannot.

## Limits

- **AMLSim is a simulator.** Its normal accounts have fixed partners and many are dormant, so these cues are stronger
  than in a real bank.
- **The banks are enriched.** About 30% of active accounts launder. At one in a thousand, PFA 0.01 with PD 0.7 gives
  about 14 false alerts per laundering account found. Thresholds there need the rare-event recipe
  (`notes/rare-events.md`).
- **Next test.** IBM's AMLworld (Altman et al. 2023, arXiv 2306.16424) has richer normal behaviour. It is on Kaggle,
  which needs an API key.
