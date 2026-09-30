# Section 5 notebook teaching and audit notes

Read-only source inspection on 2026-09-19. Cell numbers below count **all cells, including Markdown, starting at 1**, not execution counts. Existing source files were not changed. All saved execution counts are null; outputs exist but are historical, not fresh verification.

## Common mental model

All four notebooks follow: make labeled synthetic streams → learn a recurrent classifier → copy learned weights into a one-observation-at-a-time model → reset memory between independent streams → compare first threshold crossing with truth → sweep thresholds to draw speed/error tradeoff curves.

They train a score at **each time**, not a single yes/no label for an entire stream, not a future-value forecast, and not a regressor for the change time. `return_sequences=True` retains all intermediate recurrent outputs so the loss can supervise every time step. One stream has shape `(time, features)`; a batch has shape `(streams, time, features)`; labels have shape `(streams, time, 1)`.

Network: `LSTM(16, return_sequences=True)` → `Dense(10, activation='relu')` → `Dense(1, activation='sigmoid')`. The first Dense applies the same learned transformation separately at every time step. LSTM recurrent connections encode the history. In an LSTM the carried state includes both a 16-value hidden vector and a 16-value cell vector, not literally only 16 stored values. The network has 1,717 parameters for seven inputs and 1,333 for one input.

Loss is binary cross-entropy; Adam optimizer is selected by name; batch size 32; 100 batches = 3,200 streams; 20 epochs. No direct false-alarm/delay loss, early stopping, random seed, confidence interval, or checkpoint saving appears. A low BCE or high stepwise accuracy alone does not establish a useful sequential alarm system. A network can classify almost every frame correctly yet have costly first false alarms or slow transitions.

`model` consumes full sequences for training. `model2` has the same trainable architecture and copied weights, but input shape `(1,1,p)`, fixed batch size 1, and `stateful=True`, retaining memory across calls. `model2.reset_states()` is essential between independent trials; resetting at each observation would remove the detector's temporal memory. Copying weights does not mean retraining. The sigmoid score is trained toward a posterior-like interpretation, but code does not establish exact calibration.

`h_vec` is a collection of operating thresholds, not neural parameters. One trajectory can be evaluated at many thresholds: `alarm_flag_vec` remembers which thresholds have already fired so each trial contributes only its first alarm. When thresholds are ordered, crossing the highest threshold implies every lower one has fired at some point; this explains `while alarm_flag_vec[-1] == 0`. No maximum loop length is imposed; stringent false-alarm thresholds may make simulation very slow. Calling `predict` once per time step adds considerable overhead.

## IID Bayesian notebook

Path: `DeepQCD_codes 2/Sec. 5.1 -- IID/Justification_iidData_BayesianSetting.ipynb` (23 cells).

| Cells | Purpose |
|---|---|
| 2–5 | Legacy installation/imports; do not execute installation cells casually |
| 6 | Seven independent Gaussian features, mean 0 before and mean 1 after change, unit variance each |
| 7–8 | 3,200 streams, 2,000 observations each, geometric change time with rho=.001; labels 0 before tau, 1 from tau onward |
| 9 | 500 validation streams, 1,000 pre-change and 1,000 post-change observations |
| 10–13 | Architecture, BCE/Adam compile, fit for 20 epochs, training/validation loss plot |
| 14–15 | One-step stateful model, copy weights |
| 18 | 10,000 Bayesian trials, 51 thresholds, DeepQCD and Shiryaev on the **same observations** |
| 19–22 | Historical PFA and ADD arrays |
| 23 | ADD versus PFA curve, paper Figure 5 |

Input shape `(3200,2000,7)`, labels `(3200,2000,1)`. Geometric rho=.001 means an independent probability .001 of beginning change at each eligible time, expected change time 1,000; some generated streams contain no change within 2,000 steps. `tau-1` pre-observations correctly implements the paper's 1-based indexing. Validation comment calls change time 1000, but 1,000 pre-observations make the first changed observation 1001 under this convention.

The baseline knows the distributions and prior. Its LR expression simplifies to `exp(sum(x_t) - 3.5)` for this seven-feature Gaussian shift. It first predicts posterior change probability before seeing the next observation, `p_til = p + (1-p)*rho`, then Bayes-updates with LR. DeepQCD receives observations only in online prediction.

PFA is fraction of trials alarming before tau. ADD is the mean of `max(T-tau,0)` across **all** trials; false alarms contribute zero. It is **not** conditional delay among successful detections. Each plot point is a different threshold; compare algorithms at the same PFA, not necessarily the same numeric threshold. The plot reverses the log PFA x-axis, so toward the right means stricter false-alarm control.

Paper claim to understand: approximate the statistically optimal benchmark under this known synthetic example; not a proof that finite-trained DeepQCD is optimal. Historical epoch-20 output reports training BCE .0037 and validation BCE .0041, but these are saved outputs.

## IID minimax notebook

Path: `DeepQCD_codes 2/Sec. 5.1 -- IID/Justification_iidData_MinimaxSetting.ipynb` (32 cells).

Cells 2–15 repeat the Bayesian notebook's generation and training, independently. Paper says it uses the same trained model from Bayesian setting; the provided minimax notebook retrains rather than loading that exact learned weight realization.

| Cells | Purpose |
|---|---|
| 19–24 | Estimate average false-alarm period under never-changing data for DeepQCD, CUSUM, SR; 1,000 trials each |
| 26–31 | Estimate ADD when change occurs immediately (tau=1) for the same methods and each method's same threshold grid; 10,000 trials each |
| 32 | ADD versus FAP, paper Figure 6 |

CUSUM adds log-likelihood evidence but resets a negative accumulated total to zero: `g=max(0,g+LLR)`. SR uses `g=(1+g)*LR`. DeepQCD uses the learned recurrence and sigmoid score. Their score units differ, so each needs its own threshold grid.

FAP is an expected waiting time to a false alarm with no change ever: larger is better. ADD is a waiting time after a true change: smaller is better. Compare vertical heights at matched horizontal FAP. Measuring only tau=1 is not evaluating the supremum over change times and histories needed for a worst-case minimax claim.

**Critical source issue:** baseline data in cells 21,23,28,30 use `mu_pre + np.random.randn()` or `mu_post + np.random.randn()`, broadcasting one scalar noise across all seven features. DeepQCD uses `randn(1,1,input_dim)` (seven independent noises). Thus baselines use covariance all-ones, not identity; yet the baseline LR formula still assumes identity covariance. At pre-change, the intended log-LR has mean -3.5 and variance 7; the displayed baseline generator produces mean -3.5 and variance 49. Correct comparisons require identical data distributions. Relevant raw notebook lines: 599, 676, 835, 906; compare learned detector line 523 / 762. This issue cannot be inferred to have caused paper figures because saved outputs may not correspond to current source.

## AR notebook

Path: `DeepQCD_codes 2/Sec. 5.2 -- AR/Justification_AR.ipynb` (40 cells).

| Cells | Purpose |
|---|---|
| 6 | Before: `x_t = -.3*x_previous + noise`; after: `x_t = 1 + .2*x_previous + noise` |
| 7–10 | 3,200 training and 500 validation streams, each 2,000 steps and one feature |
| 11–14 | Same neural architecture/optimizer/loss and 20 training epochs |
| 15–16 | Stateful online model and weight copy |
| 19–24 | 5,000 Bayesian trials, DeepQCD versus modified Shiryaev, plot Figure 7 |
| 27–32 | FAP for DeepQCD, modified CUSUM, modified SR, 1,000 trials each |
| 34–39 | tau=1 ADD for all three, 10,000 trials each |
| 40 | ADD versus FAP, Figure 8 |

Input shape `(3200,2000,1)`. AR(1) means today's observation partly depends on yesterday's. `lambda_pre=-.3` encourages alternating fluctuations; `lambda_post=.2` carries some same-direction persistence, while the drift also changes. Note mu=1 is an intercept, not stationary mean: after-change stationary mean is 1/(1-.2)=1.25.

The baseline LR uses the conditional Gaussian means, `mu_pre+lambda_pre*x_previous` versus `mu_post+lambda_post*x_previous`, rather than pretending observations are temporally independent. Its formula matches paper Eq. (10). The LSTM must learn useful historical dependence from labeled examples.

Indexing issue: training/validation index starts at t=0 and explicitly leaves first observation zero (the `t>0` condition), using t=tau for first post label. Online tests start at t=1, initial previous observation zero, and draw actual random first observation. This yields a minor training/test initialization and indexing mismatch. No analogous scalar-broadcast covariance issue here because dimension is one. A possible modern NumPy issue is `np.max([0, g+LLR])` when g+LLR is shape-(1,) rather than scalar; it should be tested during porting.

Paper claims: roughly match modified Shiryaev in Bayesian test; favorable ADD at tested FAP in minimax-style tau=1 tests. These remain empirical findings for the stated setup.

## Transient notebook

Path: `DeepQCD_codes 2/Sec. 5.3 -- TransientQCD/Justification_TransientQCD.ipynb` (38 cells).

| Cells | Purpose |
|---|---|
| 6–7 | Gaussian means 0 and 1; helper for window-limited CUSUM |
| 8–11 | 3,200 training and 500 validation streams, 3,000 steps; labels 0→1→0 |
| 12–15 | Train same architecture for 20 epochs |
| 16–17 | Online stateful clone |
| 20–26 | 10,000 trials each for DeepQCD and window CUSUM, tau1=1000, transient duration K=25; PD versus PFA, paper Figure 9 |
| 27–38 | Additional experiment detecting return from changed regime to normal, absent from paper Section 5.3 |

Training: tau1~Geometric(.001), duration~Geometric(.002): average onset 1,000 and average duration 500. Validation swaps rates to .002/.001 (average onset 500 and duration 1,000); this is a different validation time distribution. Test duration is fixed at only 25; training longer transients and testing short ones exercises generalization. Labels mean **currently in the changed regime**, not **a change has ever occurred**: after return, label is 0.

`window_dec_stat` processes the last K observations by repeated `max(0,g+x-.5)`. For mean shift 0→1 and variance 1, x-.5 is the scalar log-LR. It computes the largest nonnegative suffix sum; the empty suffix (score 0) is effectively included. Paper's written max over nonempty suffixes can be negative, but for positive upper alarm thresholds this difference does not affect crossings.

Three exclusive first-alarm outcomes: false alarm T<tau1; successful detection tau1<=T<tau2; miss T>=tau2. Consequently PD=1-PFA-PMD. `mdr_vec` is actually unconditional PMD, not a miss fraction conditioned on no earlier false alarm. Compare higher PD at equal PFA; curve is not necessarily monotonic across all PFA because very low thresholds cause premature false alarms. Window baseline is initialized with K prechange samples, whereas recurrent model starts with zero state (both then run 999 prechange observations).

Additional return-to-normal experiment warms neural state with 50 changed observations and detects `score <= threshold`. Its extra ADD comparison uses `(t-1)` for DeepQCD (cell 34) versus `t` for window baseline (cell 36), an inconsistent one-step delay convention (raw notebook line 1169). Baseline warms with 25 changed samples, not 50. Do not present this extra experiment as paper Figure 9 or as a fair verified result. Onset simulations loop one redundant step beyond tau2; labels are indexed from zero in training versus one in testing, another one-step convention difference.

## Reproducibility status and teaching cautions

- All four notebooks contain Colab/GPU metadata and old saved install logs showing Python 3.6, NumPy 1.17.5, TensorFlow GPU 2.0.0-alpha0. These install cells uninstall TensorFlow then install a 2019 alpha. No dependency manifest, checkpoint files, tests, or seeds for these notebooks were found.
- Current shell `python3` is 3.14.6 and `importlib.util.find_spec` reports tensorflow, numpy, matplotlib, keras, jupyter missing **in that interpreter**. This is not a claim about every other environment available on the computer.
- Stored outputs already warn that `if x_train == []` / `if y_train == []` compare arrays with empty lists and will fail in future NumPy. A safe port should accumulate arrays in a list then concatenate once, avoiding both ambiguous comparisons and repeated copying.
- Current Keras may require layer-level recurrent-state reset; old model-level `reset_states()` must be checked in the chosen environment. TensorFlow compatibility should be established before training. Do not auto-install new dependencies in this workspace under the user's global instructions.
- Historical output consistency issue: IID Bayesian source says 10,000 trials, but stored PFA includes 0.000150988978, not an integer multiple of 1/10,000 (reciprocal approximately 6,623). Thus saved output and visible source are not a reliable exact reproduction pair.
- No notebooks were executed, no training or evaluation was launched, and no packages installed in this review. Reported losses/curves belong to existing saved outputs.
- Strong learning outcome: user can explain each axis and shape, locate data vs training vs online evaluation, distinguish label from score from threshold from alarm time, distinguish training history from runtime state, and identify assumptions needed for fair baseline comparisons. Reproducing every plotted number is a later engineering task.

## Commands/checks performed

`rg --files` inventoried notebooks/files; Python standard-library JSON parsing read all four notebook sources/metadata/outputs; `sed`/`rg` inspected the extracted paper's Sections 5.1–5.3 and relevant raw notebook lines; `importlib.util.find_spec` checked module availability without imports or installations. All read commands completed successfully. No behavior-changing edit occurred and no training/test suite was run.
