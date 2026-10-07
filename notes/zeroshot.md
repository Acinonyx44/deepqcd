# Trial: zero-shot change detection with pretrained models (Jev and alternatives)

## Idea

DeepQCD trains a network to output **d_t = P(change has happened | x_1..x_t)** (Eq. 6 of the paper) from
labelled episodes. A pretrained model can be *asked* for the same quantity with no training at all. If it is
good enough, it drops straight into the generic QCD procedure: s_t is the history, d_t is its answer, and an
alarm is raised when d_t ≥ h.

`zeroshot_qcd.py` tests four ways of doing that, scored exactly like everything in `deepqcd_real.py`: same
test windows, PFA-matched operating points, and DeepQCD (one seed) plus the classical rivals refit on the same
training windows.

| backend | what it is | key needed | cost |
|---|---|---|---|
| `jev` | TypeSafe AI's **Jev** "System One" decision model (Sept. 2026): JSON state plus a typed yes/no question, answered with a probability, no text generated | `TYPESAFE_API_KEY` | ~$1.60 for all 5 datasets |
| `hf` | **the same interface on an open-weight model** (default Qwen2.5-0.5B-Instruct): one forward pass on the same prompt, P(yes) read from the next-token logits over "Yes" vs "No", nothing generated | none (downloads from Hugging Face) | free; slow on CPU (default 60 windows), fast on a GPU |
| `openai` | **any OpenAI-compatible endpoint with logprobs** (OpenAI, Groq, Together, Fireworks, vLLM, Ollama): `max_tokens=1`, P(yes) from the top logprobs | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for non-OpenAI) | gpt-4o-mini ~$6 for all 5 datasets |
| `chronos` | **Amazon Chronos-Bolt**, a zero-shot time-series foundation model: forecasts each block of every feature from the history; a CUSUM accumulates how far the observations fall outside its predicted quantiles | none (downloads from Hugging Face) | free; runs on **all 21 datasets** |

The three text backends (`jev`, `hf`, `openai`) run on the 5 datasets whose features have plain-language
meanings: room occupancy, crypto pump-and-dump, S&P 500 stress, SKAB water pump, keystroke takeover. Each is
told what the stream is, what the features are, and what the change looks like (`SPEC` in the script).
Chronos is purely numeric, so it runs on every dataset in `realdata.py`.

Every backend is asked every few steps (stride per dataset). Every other detector is also reported
"@ schedule": it sees every sample but may alarm only at the same decision times. That is the like-for-like
comparison. Text answers are cached in `runs/zeroshot/`, so reruns cost nothing.

## Results

Run on 2026-10-07 in a CPU-only cloud session (4 cores, no GPU). Every cell is **detection rate / mean
detection delay** (delay in the dataset's own sample unit) at the operating point with PFA ≤ 0.1, on the same
test windows for every detector. "@ sched." means the detector may alarm only at the zero-shot model's decision
times, which is the like-for-like comparison. "Best classical" is the best of CUSUM, MEWMA, Shewhart and (where
present) the self-calibrating chart, all @ schedule. Full rows at PFA 0.05 / 0.1 / 0.25 are in
`runs/zeroshot/<backend>-<dataset>.json` (git-ignored).

### Chronos-Bolt (`amazon/chronos-bolt-small`), all 21 datasets

| dataset | windows | asked every | zero-shot DR / ADD | DeepQCD @ sched. DR / ADD | best classical @ sched. | DR / ADD |
|---|---|---|---|---|---|---|
| skab | 200 | 10 × s | 0.34 / 88.4 | 0.22 / 98.0 | MEWMA chart | 0.30 / 93.3 |
| tep | 200 | 20 × 3 min | 0.61 / 102.3 | 0.68 / 106.8 | Shewhart chart | 0.91 / 44.8 |
| occupancy | 200 | 5 × min | 0.84 / 14.8 | 1.00 / 2.8 | CUSUM (fitted Gaussians) | 1.00 / 2.0 |
| occupancy-nolight | 200 | 6 × min | 0.54 / 30.7 | 0.08 / 47.9 | CUSUM (fitted Gaussians) | 0.38 / 42.1 |
| cmapss | 200 | 12 × cycle | 0.31 / 111.5 | 1.00 / 36.7 | MEWMA chart | 1.00 / 53.2 |
| smd | 200 | 10 × min | 0.17 / 23.2 | 0.13 / 25.7 | Shewhart chart | 0.09 / 27.0 |
| hai | 200 | 12 × s | 0.31 / 79.1 | 0.85 / 33.2 | MEWMA chart | 0.94 / 28.5 |
| nab | 200 | 10 × step | 0.09 / 92.6 | 0.14 / 87.0 | CUSUM (fitted Gaussians) | 0.32 / 70.3 |
| tcpd | 200 | 3 × step | 0.09 / 21.4 | 0.25 / 18.1 | MEWMA chart | 0.30 / 17.2 |
| beedance | 200 | 3 × frame | 0.47 / 19.6 | 0.88 / 14.6 | MEWMA chart | 0.30 / 22.0 |
| hasc | 200 | 10 × sample | 0.15 / 82.7 | 0.11 / 88.5 | CUSUM (fitted Gaussians) | 0.14 / 84.1 |
| fishkiller | 200 | 5 × step | 0.06 / 15.4 | 0.00 / 16.2 | CUSUM (fitted Gaussians) | 0.01 / 15.2 |
| yahoo | 200 | 2 × hour | 0.14 / 2.0 | 0.13 / 2.5 | CUSUM (fitted Gaussians) | 0.13 / 2.5 |
| pmubage | 200 | 10 × 1/30 s | 0.97 / 10.5 | 0.89 / 24.2 | Shewhart chart | 0.97 / 8.9 |
| sp500 | 200 | 5 × day | 0.47 / 37.1 | 0.87 / 17.1 | CUSUM (fitted Gaussians) | 0.93 / 11.0 |
| seismic | 200 | 30 × 10 ms | 0.84 / 75.0 | 0.25 / 242.7 | Shewhart chart | 0.94 / 57.4 |
| fog | 200 | 16 × 1/32 s | 0.02 / 114.4 | 0.00 / 113.8 | CUSUM (fitted Gaussians) | 0.29 / 97.5 |
| pumpdump | 200 | 2 × 5 s chunk | 0.04 / 11.5 | 0.91 / 1.9 | CUSUM (fitted Gaussians) | 0.60 / 5.1 |
| keystroke | 200 | 2 × entry | 0.71 / 13.4 | 0.31 / 31.5 | Self-calibrating chart | 0.98 / 4.8 |
| iot-mirai | 200 | 20 × packet | 0.60 / 139.1 | 1.00 / 10.4 | CUSUM (fitted Gaussians) | 1.00 / 10.6 |
| iot-mirai-iid | 100 | 20 × packet | 0.86 / 59.6 | 1.00 / 9.6 | CUSUM (fitted Gaussians) | 1.00 / 10.5 |

`iot-mirai-iid` used 100 windows instead of 200: at 200 its many features (115 per packet) need about 2 hours of
CPU forecasting, past the session's job limit. `iot-mirai` at 200 windows took 1 h 45 min; `iot-mirai-iid` at 100 took 54 min.

**What it shows.**

- **Zero-shot forecast surprise wins on a few hard datasets**: SKAB (0.34 vs 0.30 best classical, 0.22 DeepQCD),
  occupancy without the light sensor (0.54 vs 0.38 / 0.08) and SMD (0.17 vs 0.09 / 0.13). These are the cases
  where the change is a shift in how the series *moves* rather than in its level, and no detector does well.
- **It does not beat the simple charts on step changes.** Seismic 0.84 vs Shewhart 0.94, Tennessee Eastman 0.61
  vs 0.91, grid events 0.97 vs 0.97 (Shewhart slightly faster). It does beat DeepQCD on seismic (0.25) and grid
  events (0.89), where DeepQCD lost before too.
- **Keystroke takeover**: 0.71, more than twice DeepQCD's 0.31, but still well below the self-calibrating chart
  (0.98).
- **It fails where the change is in level and slow**: C-MAPSS (0.31 vs 1.00), HAI (0.31 vs 0.94), S&P 500 stress
  (0.47 vs 0.93), pump-and-dump (0.04 vs 0.91 DeepQCD) and IoT Mirai (0.60, ADD 139 vs 1.00, ADD 10). A
  forecaster conditioned on the whole history adapts to a drifting level, so the surprise CUSUM never fills.
- **Cost**: from 4 s (TCPD) to 1 h 45 min (IoT, 115 features) of CPU per dataset; about 5 hours for all 21 (IoT and HAI/SMD dominate).

### Qwen2.5-0.5B-Instruct (`hf` backend), 5 text datasets

Skipped on CPU. This machine manages about 700 prompts an hour, and the full run is about 17,000 prompts
(5 datasets × 60 windows), or roughly 30 hours. It needs a GPU, where `zeroshot_qcd.py --backend hf` takes
minutes. Answers are cached in `runs/zeroshot/hf-<dataset>.jsonl`, so a later run picks up any partial cache.

One indicative point comes from a 15-window CPU pass on room occupancy. Qwen detected 0.07 at ADD 52.7 min,
against 1.00 at ADD 2.1 min for CUSUM and 1.00 at ADD 2.8 min for DeepQCD on the same 15 windows. Fifteen windows
is too few to draw conclusions, but a 0.5B model reading the raw readings as JSON did not pick up the obvious
occupancy signal.

### Code changes made for this run

- Chronos forecasts are now sent in chunks of 2,048 series. One call with every window × feature ran out of
  memory on the IoT datasets (the process was killed).
- Text-backend answers are now cached in chunks of 256 as they arrive instead of all at the end, so a run that
  is stopped part-way resumes from where it was. The first Qwen pass lost about 1,450 answers that way.

## Verification before the first real run

The session that built this could not reach any model host: `api.typesafe.ai`, Hugging Face, OpenAI and Chronos's S3/CloudFront
mirror were all denied, no provider key was set, and gcloud has no project. Network changes apply only to new
sessions.

What is verified:

- **All four code paths** run end to end with `--mock`. The text backends get a labelled heuristic stand-in;
  Chronos gets a naive last-value forecaster in place of the model. Mock rows are labelled
  "MOCK stand-in (not a model)" and are not saved as results.
- **The `hf` path** also ran through a real `transformers` forward pass on a tiny randomly initialised local
  model: chat template, left padding, batching, and Yes/No logits.
- **Jev's wire format** follows the official `typesafe-sdk` 0.7.2. The client uses only the standard library;
  installing TypeSafe's SDK into the project was refused as untrusted third-party code. `transformers` and
  `chronos-forecasting` (Amazon's official package) are in the optional `zeroshot` dependency group.

## To run it

Use a session whose environment allows the hosts (or "all domains"), or a laptop.

```bash
uv sync --group zeroshot --group realdata
./fetch_data.sh                                               # if data/ is empty

.venv/bin/python zeroshot_qcd.py --backend chronos            # no key; all 21 datasets
.venv/bin/python zeroshot_qcd.py --backend hf                 # no key; 5 text datasets, 60 windows each
.venv/bin/python zeroshot_qcd.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct occupancy   # bigger model
.venv/bin/python zeroshot_qcd.py --backend jev --estimate     # then without --estimate, with TYPESAFE_API_KEY
.venv/bin/python zeroshot_qcd.py --backend openai --estimate  # with OPENAI_API_KEY (and OPENAI_BASE_URL)
```

Keys go in the environment's settings (API credentials or environment variables), never in the repo or the
chat. Options: `--windows N`, `--workers N` (parallel API calls), `--batch N` (hf), `--model NAME`. Results
are written to `runs/zeroshot/<backend>-<dataset>.json`.

## What to look for

- **Zero-shot vs trained.** Does a model that has never seen a labelled episode match DeepQCD on its
  schedule? Keystroke takeover is the most interesting case: the change is defined relative to the session's
  own start, which plain-language reasoning about "the same person" might handle. DeepQCD only reached 43 %
  there, while a self-calibrating chart reached 97 %.
- **Chronos on step changes.** Forecast-surprise detection is the natural zero-shot rival to Shewhart and
  STA/LTA on the seismic, Tennessee Eastman and grid-event data, where DeepQCD lost.
- **Interface vs model.** Comparing `jev` with `hf` on the same prompts separates what Jev's "System One"
  training buys from what any logprob-read model gives.
- **Cost of a decision.**
  - Jev quotes ~0.1-0.5 s and ~$0.00002 per call.
  - A local 0.5B model takes ~0.5-2 s per call on CPU.
  - Either suits streams sampled every few seconds or slower (occupancy, markets, plant monitoring), not
    100 Hz seismic or packet-level IoT. There Chronos, which batches thousands of series per call, is the
    zero-shot option.
