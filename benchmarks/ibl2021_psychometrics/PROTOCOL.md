# Frozen protocol: IBL 2021 psychometric and training-duration replication

**Frozen: 2026-07-28.** Everything below was written before any of the quantities it
declares had been computed. The single, disclosed exception is recorded under
[Prior exposure](#prior-exposure).

This protocol governs an attempted **replication of numbers published by other people**.
It is not an application of Unspool's pipeline to IBL data — six of those already exist in
this repository and are graded gestural because they report their own numbers. The object
here is narrower and harder: take a number the International Brain Laboratory printed,
recompute it from the public data, and compare it under a tolerance fixed in advance.

## Source

International Brain Laboratory, Aguillon-Rodriguez V., Angelaki D., Bayer H., Bonacchi N.,
Carandini M., et al. (2021). *Standardized and reproducible measurement of decision-making
in mice.* **eLife** 10:e63711. <https://doi.org/10.7554/eLife.63711>

Correction: eLife 2022;11:e84310 revises only the quiescent-period duration
(0.2–0.5 s to 0.4–0.7 s). It changes no value replicated here.

Data: the public ONE release tagged `2021_Q1_IBL_et_al_Behaviour` served by
`https://openalyx.internationalbrainlab.org`, licensed CC-BY 4.0. Every trial table is
addressed by its exact dataset UUID and verified against its released MD5 by
`unspool.adapters.ibl_one`, which preserves IBL's native `-1/0/+1` choice coding.

## What the paper actually says

The brief that commissioned this work contained three errors of attribution. They were
found by reading the article, and they change the analysis. They are recorded here because
a protocol that silently corrected them would hide the most important methodological
finding of the exercise.

| Value | What the brief called it | What the paper says |
| --- | --- | --- |
| 17.8 ± 11.7% | "psychometric threshold" | contrast threshold **averaged over training sessions** (Figure 2b), *not* the trained-state threshold |
| 9.5 ± 3.6% | "lapse rate" | **empirical error rate on easy (50%, 100%) contrast trials** (Figure 3c). The paper defines it in-line: "the errors made in response to easy contrasts". It is *not* the fitted γ/λ |
| 14.3 ± 3.8 | "psychometric slope" | the **contrast threshold σ** at the three proficiency sessions (Figure 3d, whose legend reads "Same, for contrast threshold and bias") |

**17.8 and 14.3 are the same parameter.** The fitted psychometric has no slope parameter
distinct from σ; the Results prose calls σ "the slope" while its own figure legend and
Methods call it the threshold. The two numbers differ because they are measured over
different session sets, not because they are different quantities. Any replication that
fits a "slope" and a "threshold" separately has misread the paper.

Accordingly this protocol declares **six** claims, not five, and describes each in the
paper's own terms rather than the brief's.

## Psychometric parameterisation

The paper's Methods print:

```
P(rightward) = γ + (1 − γ − λ) · [ erf( (c − µ) / σ ) + 1 ] / 2
```

with `c` the **signed contrast in percent** (−100 … +100, negative = left stimulus),
`µ` the response bias in percent contrast, `σ` the contrast threshold in percent contrast,
`γ` the lapse rate for left stimuli and `λ` the lapse rate for right stimuli.

This is `erf_psycho_2gammas` in the IBL's own `psychofit` package, parameter order
`[bias, threshold, lapse_low, lapse_high]`. This benchmark reimplements it directly rather
than depending on `psychofit`, and fits by maximum likelihood on Bernoulli rightward-choice
counts per signed contrast level, using the IBL's published bounds:

- `parstart = [mean(contrasts), 20.0, 0.05, 0.05]`
- `parmin   = [min(contrasts),   0.0, 0.0,  0.0 ]`
- `parmax   = [max(contrasts), 100.0, 1.0,  1.0 ]`

σ is reported **directly, in percent contrast**. It is not inverted, and no separate slope
is fitted. This parameterisation is chosen because it is the one the paper used; a logistic
with a single symmetric lapse (Unspool's `LapsePsychometric`) would not be comparable, and
is deliberately not used here.

## Cohort

The paper's cohort for the training-duration and Figure 3 numbers is the **n = 140 mice
that reached basic-task ("Level 1") proficiency**, out of 206 that began training, pooled
across **7 institutions**. The public release exposes 9 *laboratory* names; the paper's 7
institutions are recovered by the mapping the IBL uses throughout, which merges
`hoferlab` + `mrsicflogellab` into SWC and `churchlandlab` + `zadorlab` into CSHL.

Cohort criteria, fixed here:

1. Every subject in release `2021_Q1_IBL_et_al_Behaviour` **whose earliest session in the
   release runs `_iblrig_tasks_trainingChoiceWorld`**. A subject whose earliest release
   session is already `biasedChoiceWorld` has had its training history truncated by the
   release and cannot contribute a training duration. This criterion reads only protocol
   names and dates — never choices, feedback, or accuracy.
2. Of those, every subject that **meets the paper's `trained_1a` or `trained_1b`
   criterion** on three consecutive training sessions. Proficiency is determined by the
   criterion, *not* by observing a protocol transition, because the release does not
   contain the subsequent biased sessions for every mouse.

The `trained_1a` / `trained_1b` criteria, transcribed from the paper's Methods:

| Requirement | `trained_1a` | `trained_1b` |
| --- | --- | --- |
| Completed trials per session | ≥ 200 | ≥ 400 |
| Performance on easy (50%, 100%) trials | > 80% | > 90% |
| Fitted \|bias\| µ | < 16% | < 10% |
| Fitted threshold σ | < 19% | < 20% |
| Fitted lapse γ **and** λ, each | < 0.2 | < 0.1 |
| Median reaction time at 0% contrast | — | < 2 s |
| Consecutive sessions | 3 | 3 |

Two ambiguities are resolved in advance:

- The Results text says the lapse bound applies "for their **sum**"; the Methods and the
  IBL's own `brainbox.behavior.training` apply it to **each lapse separately**. The
  per-lapse form is used, because it is the one the reference implementation uses.
- `trained_1b`'s threshold bound (20%) is looser than `trained_1a`'s (19%). This is as
  printed and is not corrected.

The psychometric fit inside the criterion is performed on **all trials of the three
sessions pooled**, following the paper ("using a combination of all the trials of the three
sessions"). A subject is deemed proficient at the **earliest** three-session window
satisfying **either** criterion; the paper applied 1a to 80 mice and 1b to 60, but does not
publish which, so the disjunction is the only implementable form and this is declared a
known deviation.

Basic-task training sessions contain no blocks — the paper states stimuli "appeared with
equal probability". No filter on `probabilityLeft` is therefore applied to training
sessions, matching the paper's own construction.

## The six claims and their pre-declared acceptance tolerances

### Tolerance principle

Every published value here is a **cohort mean with a published dispersion**. A reproduced
mean over a cohort assembled from a public release cannot be expected to match to printed
precision, so a rounding-width tolerance would be dishonest — it would guarantee failure
for reasons that have nothing to do with whether the analysis was reproduced.

The rule adopted, fixed before computation, is:

> **A claim passes if the reproduced cohort mean falls inside the 95% confidence interval
> that the paper's own reported dispersion places around its own mean**, that is
> `|mean_reproduced − mean_published| ≤ 1.96 × SD_published / √n_published`.

The justification is that this is the paper's *own* statement of how precisely it knows its
mean. If the reproduction lands outside that interval, the two cohorts are not consistent
with being the same population under the paper's own uncertainty, and that is a genuine
disagreement worth recording. If it lands inside, the reproduction has recovered the
published quantity to the precision the paper itself claims.

Three properties make this rule non-gameable:

- It depends only on published quantities (`SD_published`, `n_published`). It cannot be
  widened by narrowing the reproduced cohort, which is the standard way this kind of
  tolerance is abused.
- It is *tighter* than the two-sample band that would be defensible if the two cohorts were
  treated as independent samples (`1.96 × SD × √(1/n₁ + 1/n₂)`). The stricter of the two
  available principled rules is chosen deliberately.
- For the largest claims it is narrow in relative terms: ±2.15 days on 18.4 is ±11.7%,
  and a mis-implemented proficiency criterion moves training duration by far more.

Cohort size is compared under a relative tolerance instead, because it is not a mean with a
dispersion.

### Claims

| id | Published | Figure | n | SD | Tolerance (absolute unless stated) |
| --- | --- | --- | --- | --- | --- |
| `threshold_during_training_pct` | 17.8 | 2b | 140 | 11.7 | **1.938** |
| `easy_trial_error_pct_at_proficiency` | 9.5 | 3c | 7 (see below) | 3.6 | **2.667** |
| `threshold_pct_at_proficiency` | 14.3 | 3d | 7 | 3.8 | **2.815** |
| `training_days_to_proficiency` | 18.4 | 2g | 140 | 13.0 | **2.153** |
| `training_kilotrials_to_proficiency` | 10.8 | 2 suppl. 1 | 140 | 8.6 | **1.425** |
| `n_proficient_subjects` | 140 | 2 legend | — | — | **relative 0.05** (±7 mice) |

Claim definitions:

- **`threshold_during_training_pct`** — for each subject, fit the psychometric per session
  over training sessions from the **first session containing a 12.5% contrast** (the
  paper's "12% contrast … six or more different trial types") up to and including the
  proficiency session; take the subject's mean σ; report the mean over subjects. The paper
  qualifies this value as measured after threshold "stabilized after the first ~10
  sessions", which is a description of the trace rather than an exclusion rule, so no
  session-index exclusion is applied and this is declared a known interpretive risk.
- **`easy_trial_error_pct_at_proficiency`** — `100 × (1 − proportion correct on |contrast|
  ∈ {50%, 100%} trials)`, pooled over each subject's three proficiency sessions, averaged
  over subjects. It is computed **empirically**, not from γ and λ, because that is the
  paper's stated definition.
- **`threshold_pct_at_proficiency`** — σ from a single fit to all trials of the three
  proficiency sessions pooled per subject, averaged over subjects.
- **`training_days_to_proficiency`** — number of distinct training **dates** from the
  subject's first training session in the release up to and including the last of its three
  proficiency sessions.
- **`training_kilotrials_to_proficiency`** — total completed trials over that same span,
  divided by 1000.
- **`n_proficient_subjects`** — the size of cohort criterion 2.

### A declared ambiguity in the paper, resolved against my own interest

For `easy_trial_error_pct_at_proficiency` and `threshold_pct_at_proficiency`, the paper
prints "s.d., **n = 7 laboratories**" while the corresponding figure panels plot one point
**per mouse**. The two readings give very different tolerances:

| Claim | tolerance under printed n = 7 | tolerance under n = 140 mice |
| --- | --- | --- |
| `easy_trial_error_pct_at_proficiency` | 2.667 | 0.596 |
| `threshold_pct_at_proficiency` | 2.815 | 0.630 |

**The gating tolerance uses the printed n = 7**, because taking a paper at its printed word
is the only defensible default and inventing a stricter n the paper never stated would be
substituting my judgement for theirs. This is the *more permissive* choice, so to prevent
it from being a free pass, the benchmark **additionally computes and reports whether the
strict n = 140 band is met**, in `result.json` and the README, as non-gating evidence. A
claim that passes only the wide band is explicitly weaker evidence and must be read that
way.

### Deferred claim

The brief also lists the **28.5% shift in rightward choices at 0% contrast between 20:80
and 80:20 blocks** (Figure 4d,e). That value is verified as correctly quoted, but it is
**recorded as `waived` in this first contract**, for reasons fixed here in advance:

- It is measured on the **full task**, over a **different cohort** (n = 98 mice reaching
  full-task proficiency), requiring `biasedChoiceWorld` sessions this benchmark does not
  retrieve.
- The full-task proficiency criterion as printed contains two apparent sign/typographical
  errors ("the bias above 5", median reaction time "over 2 s" where `trained_1b` required
  *under* 2 s). Implementing it requires guessing the authors' intent, which is precisely
  the freedom this protocol exists to remove.

Waiving it here is a scope declaration made before computation, not a response to a result.
It is the obvious next extension and is recorded as such in the README.

## Gate and classification

`benchmark.py` writes `result.json` through `benchmarks/provenance.stamp()` and registers
`contract_passed`, true only if **every non-waived claim** is inside its declared tolerance.
`published_claims.json` records each claim's observed value and `pass`/`fail` status and is
picked up automatically by `tests/test_published_parity.py`.

Classification follows the repository's evidence ladder: **`published-parity`** if
`contract_passed` is true, **`failed-parity`** if it is not.

## Prior exposure

Full disclosure, because this protocol's value rests on it. Before freezing, during a
feasibility check of whether the release contained enough sessions to attempt the work at
all, a **crude count of pre-protocol-transition training dates** was computed and its mean
observed. That quantity is *not* any of the six claims — it uses protocol transition rather
than the trained criterion, and over a different subject set — but it is adjacent to
`training_days_to_proficiency` and I have seen it. It is recorded here rather than omitted.
No psychometric parameter, no lapse or error rate, and no proficiency-based quantity had
been computed at freeze time.

## What this protocol does not license

Reproducing these values tests that the public release plus the published Methods are
sufficient to recover the published numbers. It does not validate the IBL's conclusions,
does not establish that standardisation caused the cross-laboratory agreement, and — since
the reproduced cohort is drawn from a curated public release rather than the full 206 mice
that entered training — does not independently reproduce the paper's attrition.

## Retention

If any claim fails, the failure is committed as `failed-parity` and kept. Tolerances
declared here are not revised after seeing results, and the cohort is not narrowed to move
a number. A revision to this protocol requires a new dated section appended below, stating
what changed and why, with the original text left intact.

---

## Amendment 1 — 2026-07-28, before any claim was computed

This amendment was made after retrieving the IBL's own published analysis source
(`cortex-lab/psychofit`, `int-brain-lab/paper-behavior`, `int-brain-lab/ibllib`) and
**before running any analysis**. It corrects specification errors in the text above, which
is left intact. No tolerance, no claim, and no cohort criterion is loosened here.

### 1. The paper uses two different psychometric fits, not one

The original text declared a single set of fit bounds. That is wrong. The IBL's code
contains two distinct fits, and they are used for different purposes:

| Purpose | Source | `parstart` | `parmin` | `parmax` |
| --- | --- | --- | --- | --- |
| **Trained-criterion evaluation** | `ibllib.brainbox.behavior.training.compute_psychometric` | `[mean(c), 20, .05, .05]` | `[min(c), 0, 0, 0]` | `[max(c), 100, 1, 1]` |
| **Reported figure values** | `paper_behavior_functions.fit_psychfunc` | `[0, 20, .05, .05]` | `[min(c), 5, 0, 0]` | `[max(c), 40, 1, 1]` |

The reported fit clamps threshold σ to **[5, 40]% contrast**; the criterion fit allows
[0, 100]. `ibllib` documents that the criterion parameters are deliberately "sub-optimal"
and frozen for consistency across subjects.

Accordingly:

- **Criterion evaluation** (cohort membership) uses the `compute_psychometric` parameters.
- **`threshold_during_training_pct` and `threshold_pct_at_proficiency`** use the
  `fit_psychfunc` parameters, because those are the fits that produced the published
  numbers.

Note the consequence, recorded in advance: because the reported fit clamps σ ≥ 5, the
reproduced threshold means are bounded below by 5 by construction. This does not help the
reproduction — both published values (17.8, 14.3) sit well inside [5, 40].

### 2. Optimiser and determinism

`psychofit.mle_fit_psycho` minimises the binomial negative log-likelihood with
`scipy.optimize.fmin` (unconstrained Nelder–Mead), enforcing `parmin`/`parmax` as a hard
`1e7` penalty inside the objective, with `nfits = 5`: the first start is `parstart` and the
subsequent four are **uniform-random** draws inside the box. The best-likelihood fit wins.

The random restarts make the published pipeline seed-dependent. This benchmark reimplements
the same objective, penalty, optimiser and restart schedule, and **pins a fixed seed
(`20210101`) and a per-fit deterministic restart stream**, so the committed result is
reproducible. Seed dependence is measured and reported in `result.json` as
`threshold_seed_sensitivity_pct`, the spread of the proficiency-threshold mean across five
alternative seeds. This is diagnostic, not gating.

A fit requires **at least 4 distinct signed-contrast levels**, matching `fit_psychfunc`;
otherwise the fit is `NaN` and the session or subject is excluded from that claim only.

### 3. Cohort filters transcribed from the released code

Two filters in `paper_behavior_functions.py` were missing above and are added:

- **`CUTOFF_DATE = 2020-03-23`.** `query_subjects` retains only subjects whose
  `date_trained` is on or before this date. Applied.
- **`EXCLUDED_SESSIONS = ['a9fb578a-9d7d-42b4-8dbc-3b419ce9f424']`.** Dropped if present.

`STABLE_HW_DATE` is *not* applied, because the figures replicated here call
`query_sessions` with `stable=False`.

### 4. Training-duration definition made exact

`figure2g_time_to_trained.py` counts, per subject, the rows of `BehavioralSummaryByDate`
with `session_date <= date_trained`, and sums `n_trials_date` over them, where
`date_trained` is the **first** session date carrying `training_status` `trained_1a` or
`trained_1b` — that is, the **last** session of the first qualifying consecutive triplet.
This confirms the original text's definition. Duration is counted in **distinct dates**,
and trials are **all trials on those dates**, including any session on a training day that
is not itself a `trainingChoiceWorld` session.

### 5. Choice coding

IBL's ALF `trials.choice` is `-1 / 0 / +1` and `unspool.adapters.ibl_one` preserves it. The
mapping from that coding to "rightward choice" is **not assumed**: the benchmark derives it
from the data by checking, on unambiguous trials (non-zero contrast, `feedbackType == +1`),
which sign of `choice` co-occurs with a right-side stimulus, and asserts the mapping is
consistent across the whole cohort. Correctness is taken from `feedbackType`, never
recomputed from a guessed choice sign. The derived mapping is written to `result.json`.

### 6. The waived claim, more precisely

The deferred 28.5% bias-shift claim is computed in `figure4de_psychfuncs_biased.py` as
`100 × [f(pars_20:80, 0) − f(pars_80:20, 0)]` — the difference of the **fitted curves
evaluated at zero contrast**, not a difference of fitted bias parameters and not an
empirical choice fraction. The original text above described it as an empirical difference,
following the figure legend; the code is authoritative and the legend is loose. The cohort
is `query_sessions_around_criterion(criterion='ephys', days_from_criterion=[2, 0],
force_cutoff=True)` restricted to biased sessions — the `ready4ephysrig` criterion, not the
basic-task one. It remains **waived** in this contract for the reasons already given.

---

## Amendment 2 — 2026-07-29, after the results were computed

Recorded after computation and dated as such. It is a **documentation correction with no
effect on any number**; the code has behaved this way since before the first run.

The criteria table above prints the completed-trial requirement as "≥ 200" and "≥ 400". The
paper's Methods say "200/400 completed trials" without an inequality, and the reference
implementation `brainbox.behavior.training` uses `np.all(n_trials > 200)` — **strictly
greater**. `benchmark.py` implements the strict form, matching the reference. The table's
"≥" is a transcription slip in this document, not in the analysis.

No result changes: the two forms differ only for a session of exactly 200 or 400 trials.

Nothing else in this protocol is revised. The retained failure on `n_proficient_subjects`
stands, its tolerance is unchanged, and the cohort was not narrowed.
