# Protocol: replication of Ashwood et al. (2022)

**Frozen 2026-07-28, before any number in `result.json` existed.**

This document fixes the analysis, the hyperparameters, the cohort and the acceptance
tolerances in advance. Nothing below may be changed to make a number land. If a published
value does not reproduce under this protocol, the failure is recorded and kept.

## 1. Target

Ashwood Z.C., Roy N.A., Stone I.R., International Brain Laboratory, Urai A.E.,
Churchland A.K., Pouget A., Pillow J.W. (2022). *Mice alternate between discrete strategies
during perceptual decision-making.* **Nature Neuroscience** 25:201-212.
doi [`10.1038/s41593-021-01007-z`](https://doi.org/10.1038/s41593-021-01007-z).

Reference implementation: [`github.com/zashwood/glm-hmm`](https://github.com/zashwood/glm-hmm),
which fits its models with a fork of Scott Linderman's
[`ssm`](https://github.com/lindermanlab/ssm) modified to handle violation trials.

## 2. Data

Figshare `10.6084/m9.figshare.11636748.v7`, file id `21623715`,
`ibl-behavior-data-Dec2019.zip`, 228,602,597 bytes,
MD5 `fd219c14ff0f3caa88d5f8bed9a96443`,
SHA-256 `18bfacccf615a767dd6e3935473b628fe4266e9b12c09200ee7f4eac2c54c4e6`.
Licence **CC BY 4.0**. This is the exact file the reference implementation downloads by
numeric id in `1_preprocess_data/ibl/1_download_data_begin_processing.py`. It is 218 MiB and
is fetched whole; `fetch_data.py` verifies size, MD5 and SHA-256 before use.

The archive is read in place. No member is extracted to disk.

## 3. Cohort

Ashwood's three filters, applied in his order:

1. retain a session only if its `_ibl_trials.probabilityLeft` takes exactly the three values
   `{0.2, 0.5, 0.8}`, that is, the animal had entered the bias-block regime;
2. retain an animal only if it has at least 30 such sessions;
3. within each retained session, keep only the trials with `probabilityLeft == 0.5` — the
   unbiased sub-block, which is the first 90 trials — and drop the whole session if that
   sub-block contains 10 or more no-response trials.

The paper's Methods state criterion 2 as "more than 30 sessions of data during the 'bias
block' regime"; the Results paragraph states it loosely as "at least 3,000 trials". The code
implements criterion 2 and this protocol follows the code.

Expected: **37 animals, 2,017 sessions, 181,530 source trials.**

## 4. Design matrix

Ashwood's M = 4 inputs, reproduced exactly:

| input | construction |
| --- | --- |
| bias | Unspool's `intercept` |
| stimulus | `contrastRight - contrastLeft` with `NaN` read as 0, then z-scored using the mean and standard deviation of the pooled 37-animal cohort |
| previous choice | Unspool's `choice_lag_1`, effect-coded to `{-1, +1}` |
| win-stay/lose-switch | `reward[t-1] * (2 * choice[t-1] - 1)` with `reward` in `{-1, +1}` |

Choice is IBL's `_ibl_trials.choice` remapped as Ashwood remaps it: clockwise `+1` becomes
left `0`, counter-clockwise `-1` becomes right `1`, no-response `0` becomes violation `-1`.
The modelled Bernoulli outcome is therefore "chose right".

**Violation trials are dropped**, not masked. See §6.

## 5. Model and hyperparameters

`unspool.BernoulliGLMHMM` with:

| setting | value | source |
| --- | --- | --- |
| `covariates` | `("stimulus", "wsls")` | paper, M = 4 with intercept and `choice_lags=1` |
| `choice_lags` | `1` | paper |
| `l2` | `0.25` | paper's Gaussian prior with sigma = 2; Unspool's penalty is `0.5 * l2 * ||w||^2`, so `l2 = 1 / sigma^2` |
| `label_by` | `"stimulus"` | the paper's engaged state is the one with the largest stimulus weight |
| `n_restarts` | `2` | the reference implementation uses `N_initializations = 2` for per-animal fits |
| `n_states` | `3` for the population analysis; `{1, 2, 3, 4, 5}` for the selection sweep | paper |
| `max_iterations`, `tolerance` | package defaults (`1000`, `1e-9`), unchanged | not tuned for this benchmark |

Comparison models: `BernoulliHistoryGLM` with the same covariates for K = 1, and
`LapsePsychometric` for the paper's classic lapse model.

## 6. Declared substitutions

Each item is a place where Unspool cannot express what the paper did. Every one of these
moves the numbers, and none of them is silent.

1. **No observation mask.** The paper keeps violation trials in the sequence and replaces
   their emission likelihood with 1. Unspool's GLM-HMM has no mask, so violation rows are
   removed. The IBL violation rate is under 0.1% of these trials. Because both history
   regressors are built from the *retained* choices, removal reproduces Ashwood's own rule of
   carrying the last non-violation choice forward. The residual difference is the first trial
   of each session, where Ashwood seeds the history from that trial's own choice and Unspool
   uses zero.
2. **No Dirichlet transition prior.** The paper places a Dirichlet(alpha = 2) prior on each
   row of the transition matrix, with no stickiness (`kappa = 0` in the reference code).
   Unspool's `stickiness` adds pseudo-counts to the *diagonal* only, which is a different
   prior, so it is left at `0.0` and the transition matrix is fitted without a prior.
3. **Intercept is unpenalized.** The paper's Gaussian prior covers all four weights including
   the bias. Unspool's `l2` deliberately excludes the intercept.
4. **No pooled global fit.** The paper fits a single GLM-HMM to all 37 animals pooled and
   seeds each per-animal fit from it, purely so that state labels align across animals.
   Unspool offers no way to initialize a fit from external parameters; it aligns labels by
   canonicalizing on `label_by`. The pooled fit is also out of compute reach here — see §8.
5. **Filtered, not smoothed, state probabilities.** The paper conditions per-state accuracy
   on the smoothed marginal posterior. Unspool publishes filtered and one-step-ahead
   predictive state probabilities but no smoothed posterior, so the filtered distribution is
   substituted. `GLMHMMFitResult.state_occupancy` *is* a smoothed quantity (the mean posterior
   probability per state) and is used for fractional occupancy; the paper's occupancy is a
   hard count of `argmax` assignments, so both are reported.
6. **Symmetric lapse model.** The paper's classic lapse model has two asymmetric lapse rates.
   `LapsePsychometric` has one symmetric rate capped at 0.2.
7. **Direct optimization, not EM.** The paper maximizes the MAP objective by EM with 300
   iterations. Unspool maximizes the same penalized marginal likelihood directly by L-BFGS-B
   over a multi-start set. The objective is the same up to the prior differences above; the
   optimizer is not.

## 7. Analyses

**A. Cohort.** Counts only. No fitting.

**B. Selection sweep, example animal `CSHL_008`.** Five-fold cross-validation with whole
sessions assigned to folds by a seeded permutation, reproducing the *scheme* of the paper's
randomized session allocation but not its exact partition: the reference code draws its
permutation from NumPy's legacy global generator seeded at 65, and this benchmark draws from
a `default_rng(65)` stream. Which sessions land in which fold therefore differs, and a fold
allocation is arbitrary by construction; what is reproduced is five folds of whole sessions
assigned without regard to time. Candidates: K = 1 (plain GLM), the lapse model, and K = 2, 3, 4, 5.
Score: held-out log-likelihood above a Bernoulli null whose rate is the training-set fraction
of rightward choices, scored on the observed held-out choices, divided by the number of test
trials, in bits. The whole curve over K is reported. The two quantities *checked* against the
paper are the three-state model's advantage over the one-state GLM and over the classic lapse
model; see the amendment in section 9a for why the arg-max over K is reported but not
asserted. The animal is the one the reference implementation plots as its example; the article
names no identifier.

This design is an *interpolation* design, not a prospective one: a held-out session may
precede a training session in time. Unspool's `evaluate_splits` requires prospective folds by
default for good reason. It is used here only because the published number being checked was
produced under it, and it is not offered as an example of the package's recommended practice.

**C. Per-state accuracy, example animal.** Fit K = 3 to all of `CSHL_008`'s trials. Order the
states as the paper does: the engaged state is the one with the largest stimulus weight; of
the remaining two, biased-left is the one with the smaller bias weight. For each state, take
the non-zero-contrast trials whose filtered probability for that state is at least 0.9, and
report the fraction on which the animal chose the rewarded side.

**D. Population summary.** Fit K = 3 to each of the 37 animals separately. Dwell time for
state k is `1 / (1 - A_kk)` from that animal's fitted transition matrix, matching the
reference implementation. Report the median across animals of each state's dwell time, and the
median across animals of the engaged state's fractional occupancy.

## 8. Out of scope, and why

| paper result | reason |
| --- | --- |
| the three-state selection itself | the paper selects three states on plateau and parsimony grounds rather than by an arg-max, so there is no published number to check; see section 9a |
| K selection across all 37 animals (Fig. 4a) | 5 folds x 5 candidates x 37 animals is roughly 40 hours of single-core time with this implementation |
| pooled global fit and cross-animal state alignment (Methods, Algorithm 1) | Unspool cannot initialize a fit from external parameters, and a 181,530-trial fit is out of compute reach |
| population predictive-accuracy gains of 4.2% and 2.8% (Results) | requires the full 37-animal cross-validation above |
| response-time signatures of state (Fig. 6) | Unspool's GLM-HMM has Bernoulli emissions only |
| Odoemene et al. mice and human participants (Figs. 5, 7) | different datasets, not fetched |

## 9. Acceptance tolerances

Fixed here, before the numbers exist. The **where printed** column is load-bearing: a value
the article prints is a published claim, and a value recoverable only from the authors'
released code is not. This benchmark checks only the former.

| claim | published | where printed | tolerance | rationale |
| --- | --- | --- | --- | --- |
| animals | 37 | Results; Fig. 4 title | exact | a cohort size is an integer; a difference means the selection was reimplemented wrongly |
| sessions | 2,017 | Fig. 4c caption; Results ("83% of 2,017 sessions") | exact | as above. Note this is the session count entering those two analyses, not a Methods-level dataset descriptor |
| source trials | 181,530 | Extended Data Fig. 8 caption | exact | as above. It appears once, in an Extended Data caption, and matches the `assert len(master_inpt) == 181530` in the reference code |
| example-mouse trials | 5,040 | Results | exact | identifies the same animal and the same unbiased-block rule |
| example-mouse sessions | 56 | Results | exact | as above |
| per-state accuracy | 90 / 60 / 58 % | Results (Fig. 2f) | +/- 5 percentage points | the paper prints whole percents for one animal; the band covers substitutions 1, 2, 3, 5 and 7, and nothing wider would still be the same result |
| median dwell times | 24 / 13 / 12 trials | Results (Fig. 4e) | +/- 25% relative | dwell time is `1/(1 - A_kk)`, which is violently nonlinear near `A_kk = 1`; substitution 2 removes the prior that acts directly on `A` |
| median engaged occupancy | 69% | Results (Fig. 4d) | +/- 5 percentage points | the paper prints whole percents; the band covers the soft/hard occupancy difference in substitution 5 |
| bits per trial over the lapse model | 0.09 | Results (Fig. 2b) | +/- 0.02 bits | the paper prints two decimals; the band covers substitution 6, a three-parameter symmetric comparator standing in for a four-parameter asymmetric one |
| bits per trial over the one-state GLM | 0.13 | Results (Fig. 2b) | +/- 0.02 bits | as above, without the comparator substitution |

A claim that falls outside its band is recorded with `status: "fail"` in
`published_claims.json` and reported in `result.json` under `contract_failures`. It is not
retried with different hyperparameters.

## 9a. Amendment, 2026-07-28, before any result was committed

Sections 7 and 9 originally declared a checkable claim that a three-state model *wins*
cross-validation — that `argmax_K` of held-out log-likelihood equals 3. Verification against
the published article shows the paper does not make that claim, and the amendment removes it.

What the article actually says about the example mouse is that "the multi-state GLM-HMM
outperformed both the standard (one-state) GLM and the classic lapse model, both in test
log-likelihood and percent correct, with the improvement approximately levelling off at three
latent states"; the population sentence is bounded downward only — three states "substantially
outperformed models with *fewer* states". Selection is then stated as a choice, not a maximum:
Figure 2b's caption highlights the three-state log-likelihood "which we used for all subsequent
analyses", and for the Odoemene dataset the authors pick K explicitly on non-arg-max grounds
("the four-state model balanced simplicity and interpretability"). The paper never claims the
three-state model beats the four- or five-state model.

Encoding an arg-max would therefore have invented a published value. Three things change:

1. `selected_state_count` is **removed from the numerical gate** and from
   `published_claims.json` as a checkable claim. It is still computed and reported in
   `result.json`, alongside the full bits-per-trial curve over K and the gain from three to
   five states, so a reader can see the plateau the paper describes.
2. The paper's example-mouse claims that *are* numeric — 0.09 bits/trial over the classic
   lapse model and 0.13 bits/trial over the one-state GLM — take its place in the gate.
3. A waived claim records the three-state selection itself as a parsimony judgement that this
   benchmark does not attempt to reproduce.

This amendment **removes** a claim and loosens no tolerance. It was made before any number in
`result.json` existed, and the original text above is retained rather than rewritten.

Two further provenance corrections from the same verification pass:

- **181,530 trials and 2,017 sessions are printed in the article** and may be cited to it,
  with the locations given in the table above. An earlier draft of this protocol treated them
  as code-only assertions.
- **The example animal's identifier is not in the article.** The paper says only "an example
  mouse"; `CSHL_008` appears nowhere in its text, captions, Methods or Extended Data. It comes
  solely from the reference implementation, which branches on `animal == "CSHL_008"` when
  drawing Figures 2, 3 and 4. The example-mouse claims above are therefore claims about the
  animal the *code* identifies, and are only as good as that identification.

## 10. Reproduce

```bash
uv run python -m benchmarks.ashwood2022_glmhmm.fetch_data
uv run python -m benchmarks.ashwood2022_glmhmm.benchmark --output benchmarks/ashwood2022_glmhmm/result.json
```
