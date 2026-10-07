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

## Status: built and verified, not yet run against real models

This session could not reach any model host: `api.typesafe.ai`, Hugging Face, OpenAI and Chronos's S3/CloudFront
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
