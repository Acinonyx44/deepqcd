# The DeepQCD workflow: how the detector is trained and how it is tested

Paper Secs. 4.3-4.4 and 5, with every step mapped to the code in this repo.

In one breath: simulate a lot of streams where you *know* when the change happened, label every time step
0 before it and 1 after, train a recurrent network to output that label online, and then at test time run
it one observation at a time and stop the first time its output crosses a threshold. The network is never
told the change-point at test time; the threshold is the only knob.

---

## Phase 1 — offline training (Alg. 1)

### What the training set is

`D = {(x_t^i, d̂_t^i) : t = 1..T, i = 1..I}` — `I` independent streams of length `T`, each with its own
change-point `τ_i`, and a label at **every time step**:

    d̂_t = 0   for t < τ        (Eq. 7)
    d̂_t = 1   for t >= τ       (Eq. 8)

Sec. 5 uses `I = 3200`, `T = 2000`, and `τ ~ geo(0.001)` drawn independently per stream, so the change
lands anywhere from step 1 to beyond the end of the stream (mean 1000, and a fair fraction of streams
never change within 2000 steps — those are all-zero labels, and they matter: they are what teaches the
network to stay quiet).

Shapes: observations `(I, T, p)`, labels `(I, T)`. One stream is a `(T, p)` matrix, not a single example.

### What the loss actually teaches

Binary cross-entropy between `d_t` and `d̂_t`, averaged over **all streams and all time steps**:

    L(φ) = −E[ d̂_t log d_t + (1 − d̂_t) log(1 − d_t) ]

This is the whole trick, and it is worth being precise about what it does and does not do:

- It is a *per-time-step* classification loss, not a sequence-level one. The question at each `t` is
  "has the change already happened?", answered from `x_1..x_t` only.
- Minimising BCE against a 0/1 label makes the network's output approach the **posterior probability**
  `P(t >= τ | x_1..x_t)` — that is the standard property of a proper scoring rule, and it is why Eq. (6)
  can define `d_t` as that posterior. It is the same quantity the Shiryaev statistic computes exactly
  from a known model; DeepQCD estimates it from data instead.
- It contains **no notion of delay or false alarm**. Nothing in the loss says an early alarm is worse
  than a late one. Those only appear at test time, through the threshold. A network can have excellent
  BCE and still be a poor detector, so the loss value is a training diagnostic, never a result.

### The training loop itself

Adam, learning rate 0.001, batch size 32, and early stopping on the validation BCE: keep the best
validation loss seen so far, count epochs without improvement, stop after `C` of them.

Two honest notes about the code:

- The authors' notebooks do **not** implement early stopping — they run a fixed 20 epochs, despite
  Alg. 1. Our `train()` implements it (`patience=3`), which is why some runs stop before 20.
- Validation streams use a **fixed** change-point at `T/2` rather than a geometric one, following the
  notebooks. So the validation loss measures something slightly different from the training objective.

## Phase 2 — real-time detection (Alg. 2)

    t ← 0, d_0 ← 0
    while d_t < h:
        t ← t + 1
        feed x_t to the network, read off d_t
    Γ ← t          # declare the change

Two things change relative to training:

1. **One observation at a time.** The recurrent state carries the history; nothing else is stored. In
   Keras the authors rebuild the network with `stateful=True, batch_input_shape=(1,1,p)` and copy the
   trained weights across; `model2.reset_states()` between independent trials is essential, and forgetting
   it is the classic way to get nonsense.
2. **No fixed horizon.** Training streams are 2000 steps; at test time the loop runs until it alarms,
   which under a stringent threshold can be hundreds of thousands of steps. The network is extrapolating
   far past any sequence length it ever saw. It holds up here, but it is an assumption, not a guarantee.

## The four train/test mismatches worth knowing

These are the places where "it trained fine" and "it detects well" come apart.

| | training | testing |
|---|---|---|
| change-point | always `τ ~ geo(0.001)` | `τ ~ geo(ρ)` (Bayesian), `τ = ∞` (FAP), `τ = 1` or 200 (ADD) |
| sequence length | 2000 steps | unbounded, until an alarm |
| input | whole stream at once | one observation at a time, state carried |
| objective | BCE against 0/1 labels | delay vs false alarms at a threshold |

The change-point mismatch is deliberate and the paper leans on it (Sec. 5.1.2: the minimax results use
the *same* network trained in the Bayesian setting). The sequence-length one is silent. The input one is
only safe if the network is genuinely causal and stateful — `check_causality()` in `qcd.py` asserts that
feeding a whole stream gives exactly the same `d_t` as feeding it step by step, to 1e-5. Every experiment
script runs that check before trusting its chunked evaluation.

## Phase 3 — what gets measured

`Γ = inf{t : d_t >= h}` is the only output. Everything else is a statistic of `Γ` against the true `τ`,
computed over many independent streams, **for every `h` on a grid**. Three protocols:

| protocol | streams generated with | measured | in the paper |
|---|---|---|---|
| Bayesian | `τ ~ geo(ρ)` | `PFA = P(Γ < τ)`, `ADD = E[(Γ − τ)+]` | Figs. 5, 7 |
| minimax, false alarms | `τ = ∞` (never changes) | `FAP = E[Γ]`, bigger is better | Figs. 6, 8 |
| minimax, delay | `τ = 1` (changed from the start) | `ADD = E[Γ − 1]` | Figs. 6, 8 |
| transient | `τ1 = 1000`, `τ2 = τ1 + 25` | `PD = P(τ1 <= Γ < τ2)` against PFA | Fig. 9 |

Then each detector's `(false-alarm measure, delay measure)` pairs are plotted as a curve, one point per
threshold. **Detectors are only comparable at the same false-alarm level**, never at the same numeric
`h` — CUSUM's statistic is an accumulated log-likelihood ratio, Shiryaev's is a probability in (0,1),
the rolling variance is in units of variance. Reading two curves at the same `h` is meaningless.

Two measurement choices that change what the numbers mean:

- `ADD = E[(Γ − τ)+]` averages over **all** streams, counting a false alarm as zero delay. It is not the
  delay among successful detections. A detector that false-alarms constantly scores a flatteringly low
  ADD, which is fine only because PFA is reported alongside it.
- `τ = 1` measures a change at the very first observation. A recurrent network can learn to be
  hyper-sensitive during start-up at almost no cost in FAP (a steady-state quantity), which the
  time-invariant CUSUM/SR recursions cannot do. This is why `deepqcd_iid.py` and `deepqcd_ar.py` also
  report the conditional ADD at `τ = 200`, and why the apparent win in Figs. 6 and 8 disappears there.
  See `results.md`.

## Where the threshold comes from

The paper sweeps `h` and plots the curve; it never picks a value. In practice you have a false-alarm
budget instead, so `detect.py` adds a **calibration phase**: measure `FAP(h)` on no-change streams, pick
the `h` that meets the budget, then measure delays at that `h`. That is the only way to put a learned
statistic and a log-likelihood-ratio statistic on the same footing.

## Map to this repo

| paper | code |
|---|---|
| network, Fig. 3/4 (φ = recurrent, ω = dense) | `qcd.DeepQCD` |
| labels, Eqs. (7)-(8) | each source's `training_set`, in `sources.py` |
| Alg. 1, offline training | `qcd.train` (+ `qcd.load_or_train` for weight caching) |
| Alg. 2, real-time detection | `qcd.NetDetector` + `qcd.stopping_times` |
| generic procedure, Fig. 2 / Eqs. (3)-(5) | the detector interface: `reset(n)`, `__call__(chunk) -> d` |
| Shiryaev, Eq. (A.6) | `qcd.shiryaev` |
| CUSUM, Eq. (A.2) | `qcd.cusum` |
| Shiryaev-Roberts, Eq. (A.9) | `qcd.shiryaev_roberts` |
| window-limited CUSUM, Sec. 5.3 | `qcd.WindowCUSUM` |
| AR(1) conditional LR, Eq. (10) | `sources.AR.llr` |
| Sec. 5.1 / 5.2 / 5.3 experiments | `deepqcd_iid.py` / `deepqcd_ar.py` / `deepqcd_transient.py` |

One implementation difference from both the paper and the notebooks: the notebooks call `predict` once
per time step per stream, which is correct but takes days. Here a detector holds state for `n` streams
at once and consumes a chunk of time steps per call, so `stopping_times` evaluates thousands of streams
against a whole threshold grid in one pass. `check_causality` is what licenses that: the chunked
computation is bit-for-bit the online one.

## Running it

```bash
.venv/bin/python detect.py --source garch --tau 500 --fap 1000
```

1. builds the source (`sources.py`)
2. trains the network on simulated labelled streams, or reloads `runs/models/<source>.pt`
3. calibrates every detector to one false alarm per 1000 steps, on no-change streams
4. runs all of them on fresh streams whose change is at `t = 500`
5. prints how often each cried wolf early, how often and how late it caught the change
6. saves `figures/detect_<source>_tau<τ>.png`: the statistic crossing its threshold on one stream, the
   delay distribution, the tradeoff curve with the operating points marked, and how fast the change is
   caught
