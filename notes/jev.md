# Trial: Jev (TypeSafe AI) as a zero-shot change detector

## Idea

Jev is TypeSafe AI's "System One" decision model, released September 2026. It takes a *state* (text or
JSON) and typed questions, and returns probabilities over the declared answers instead of generating text.
TypeSafe quotes 70-500 ms per call and $0.042 per million input tokens.

Ask it the yes/no question "has the process already left its normal regime?" about the stream seen so
far, and it returns an estimate of **P(change has happened | x_1..x_t)**. That is precisely DeepQCD's
decision statistic d_t (Eq. 6 of the paper). So Jev drops into the generic QCD procedure with **no
training**:

- s_t is the recent history, serialized as JSON: context, feature names, a summary of the start of the
  stream, and the last 20 observations;
- d_t is Jev's probability;
- an alarm is raised when d_t ≥ h.

DeepQCD learns d_t from labelled episodes; Jev is asked for it zero-shot, from a plain-language
description of the change.

`jev_qcd.py` tests this on the 5 datasets where the features have meanings a language-trained model can
use:

- room occupancy;
- crypto pump-and-dump;
- S&P 500 stress;
- SKAB water pump;
- keystroke takeover.

It uses the same test windows and the same PFA-matched scoring as `deepqcd_real.py`. DeepQCD (1 seed)
and the classical rivals are refit on the same training windows. Jev is asked every few steps; every
other detector is also reported "@ Jev schedule" (sees every sample, may alarm only at Jev's decision
times), which is the like-for-like comparison.

## Status: built and verified offline, not yet run against Jev

This cloud session cannot reach `api.typesafe.ai` (blocked by the environment's network policy) and has no
API key.

- The client follows the official `typesafe-sdk` 0.7.2 request format: `POST /v1/systemone`, Bearer
  auth, `noul` questions. It uses only the standard library, so the project needs no new dependency.
- `--mock` runs the whole pipeline with a local stand-in, labelled as such; its numbers say nothing about
  Jev.
- `--estimate` prints the call count and token count without calling anything.

| dataset | calls (200 windows) | ~input tokens / call | ~cost |
|---|---|---|---|
| occupancy (every 5 min) | 5,800 | 400 | $0.10 |
| pump-and-dump (every 2 chunks) | 28,177 | 550 | $0.65 |
| S&P 500 (every 5 days) | 7,909 | 280 | $0.09 |
| SKAB (every 10 s) | 6,678 | 525 | $0.15 |
| keystrokes (every 2 entries) | 10,124 | 1,400 | $0.60 |

That is about **$1.60 for the full trial**. At 16 parallel calls and ~0.3 s each, it takes ~1 min per 3,000
calls. Answers are cached in `runs/jev/`, so reruns are free.

## To run it

1. Allow `api.typesafe.ai` in the environment's network settings, or run on a laptop.
2. Set `TYPESAFE_API_KEY` in the environment (environment variable or secret, never in the repo).
3. Run:

   ```bash
   .venv/bin/python jev_qcd.py --estimate     # check volume first
   .venv/bin/python jev_qcd.py                # all five; or name some: jev_qcd.py occupancy pumpdump
   ```

   Options: `--windows N` (default 200), `--workers N` (default 16 parallel requests). Results go to
   `runs/jev/<dataset>.json`.

## What to look for

- **Zero-shot vs trained.** Does a model that has never seen a labelled episode match DeepQCD on its
  schedule? The most interesting cases are keystrokes, where the change is defined relative to the
  session's own start (a strength of language-level reasoning about "the same person"), and pump-and-dump,
  where the description of a pump is very specific.
- **Calibration.** Jev returns a probability, as DeepQCD does. The PFA-matched comparison does not need
  calibrated outputs, but a well-calibrated d_t lets you set h = 0.9 and mean it.
- **Cost of a decision.** At ~$0.00002 and ~0.1-0.5 s per call, Jev suits streams sampled at seconds to
  days (occupancy, markets, plant monitoring), not 100 Hz seismic or packet-level IoT.
