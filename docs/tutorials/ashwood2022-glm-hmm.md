# Replicating the canonical GLM-HMM paper

[Ashwood et al. (2022)](https://doi.org/10.1038/s41593-021-01007-z) is the reference GLM-HMM
analysis of perceptual decision-making: mice alternate between a small number of discrete
strategies rather than drifting smoothly. It has public data, public reference code, and a
model family Behavio ships. That combination makes it the right place to ask a question a
package cannot answer about itself — does this implementation recover numbers somebody else
published?

This chapter reports the answer, including the parts that did not reproduce and the parts
Behavio cannot express at all.

!!! info "Read the protocol first"

    [`benchmarks/ashwood2022_glmhmm/PROTOCOL.md`](https://github.com/aeronjl/behavio/tree/main/benchmarks/ashwood2022_glmhmm/PROTOCOL.md)
    was frozen before any of these numbers existed. It fixes the cohort, the covariates, the
    prior scale, every declared substitution, and every acceptance tolerance in advance.
    Nothing in it was changed after a number came out.

## Honest positioning: `ssm` and `psytrack`

Ashwood's [reference implementation](https://github.com/zashwood/glm-hmm) fits its models
with a fork of Scott Linderman's [`ssm`](https://github.com/lindermanlab/ssm), modified so
violation trials can be treated as missing observations. `ssm` is the mature general
state-space library for this work: many emission and transition families, hierarchical
structure, variational and Laplace-EM inference, observation masks. The smooth-drift
alternative that Ashwood argues against is implemented in Roy and Pillow's
[`psytrack`](https://github.com/nicholas-roy/psytrack).

Behavio's `BernoulliGLMHMM` is **narrower than both, and is not a replacement for either.**
It has one emission family, no hierarchy, no observation mask, no Dirichlet transition prior,
and no procedure anywhere in the package for choosing the number of states. If you want to
fit a GLM-HMM to your data, use `ssm`. If you want the smooth-drift account, use `psytrack`.

What this benchmark measures is a different axis: given that narrower implementation, wrapped
in a frozen protocol with checksum-pinned inputs and retained failures, how much of a
published result comes back?

## The data and the cohort

Figshare [`10.6084/m9.figshare.11636748`](https://doi.org/10.6084/m9.figshare.11636748), the
International Brain Laboratory behavioural release, CC BY 4.0, 218 MiB, SHA-256 pinned. It is
the exact archive the reference code downloads by numeric file id.

Ashwood's cohort rule has three parts, reimplemented here in his order: keep sessions in the
bias-block regime, keep animals with at least thirty such sessions, then model only each
session's unbiased 50/50 sub-block and drop any session with ten or more no-response trials
in it.

That cohort reproduces **bit for bit**: 37 animals, 2,017 sessions, 181,530 trials, of which
165 (0.09%) are no-response violations. The animal the reference code plots as its example has
exactly the 5,040 trials over 56 sessions the paper's Results text reports.

Five exact integer matches is a stronger check than it looks. Selection rules are where
reimplementations usually diverge quietly, and an off-by-one in any of the three filters
would move all five numbers at once.

## What reproduced, and what did not

Eight of fourteen checkable claims pass, six fail, and six more are waived. The benchmark is
classified `failed-parity` and the failures are kept.

| published value | paper | here | |
| --- | --- | --- | --- |
| mice / sessions / trials | 37 / 2,017 / 181,530 | **37 / 2,017 / 181,530** | pass |
| example mouse trials / sessions | 5,040 / 56 | **5,040 / 56** | pass |
| engaged-state accuracy | 90% | **93.9%** | pass |
| biased-left, biased-right accuracy | 60%, 58% | **40.9%, 41.0%** | fail |
| median dwell times | 24 / 13 / 12 | **30.8 / 16.5 / 10.3** | fail, fail, pass |
| median engaged occupancy | 69% | **58.2%** | fail |
| bits/trial over the 1-state GLM | 0.13 | **0.1309** | pass |
| bits/trial over the lapse model | 0.09 | **0.1274** | fail |

**Every quantity that needed no substitution reproduced, and every failure moved in the
direction its declared substitution predicts.** That is the most useful thing this exercise
produced, because it converts a list of missing features into measured consequences:

- Attributing trials to states by *filtered* rather than smoothed probability drives the
  biased-state accuracies **below chance**. The filter is updated by the current trial's own
  outcome, so thresholding it selects exactly the trials that most strongly evidence a biased
  state — the ones where the animal chose its favoured side against the stimulus. Only 257 and
  273 trials survive, against 1,991 for the engaged state, which is unaffected and passes.
- Fitting the transition matrix with **no prior** — because `stickiness` can only add
  pseudo-counts to the diagonal, not the Dirichlet(alpha = 2) the paper puts on every entry —
  leaves self-transitions too high, and dwell time is `1/(1 - A_kk)`. The two long-dwell states
  run 28% and 27% long; the short one passes.
- Measuring occupancy as a **mean smoothed posterior probability** rather than a hard `argmax`
  count gives 58.2% against a published 69%. The hard count is also computed — 64.4%, inside
  the band — but the protocol declared the soft quantity as the gated one in advance, and the
  gate was not switched afterwards.
- The **symmetric one-rate lapse model** standing in for the paper's asymmetric two-rate one is
  a weaker opponent, scoring 0.3049 bits/trial against the plain GLM's 0.3014. So the GLM-HMM
  beats it by *more* than published: this claim fails by being too good.

The cleanest result is the one comparison with nothing substituted. The three-state model's
advantage over the one-state GLM comes out at **0.1309 bits per trial against a published
0.13** — within a thousandth of a bit.

### The plateau, and why the protocol amendment mattered

| states | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| bits/trial | 0.3014 | 0.3922 | **0.4323** | 0.4323 | 0.4323 |

Going from three states to five buys **0.000018 bits per trial**. This is precisely the
plateau the paper describes, and it is why encoding "three states win the arg-max" would have
been a mistake: the arg-max here is K = 5, by a margin of eighteen millionths of a bit. A gate
asserting an arg-max would have recorded a failure with no scientific content, against a claim
the paper never made.

## Where Behavio ran out of expressiveness

The interesting output of this exercise is not the table above; it is the list of things the
package could not say. Every item here changes a number, and every one is written into the
protocol rather than discovered afterwards.

| the paper does | Behavio can | consequence |
| --- | --- | --- |
| treats violation trials as missing choice data, replacing their emission term with 1 | nothing — there is no observation mask | violation rows are dropped instead (0.09% of these trials) |
| places a Dirichlet(alpha = 2) prior on every transition row, with no stickiness | only add pseudo-counts to the *diagonal*, via `stickiness` | the transition matrix is fitted with no prior at all |
| applies a Gaussian prior to all four weights including the bias | penalize every coefficient except the intercept | the bias term is unregularized |
| fits one pooled model over all 37 animals and seeds each per-animal fit from it, so state labels align | canonicalize labels by a named coefficient | no pooled fit; and 181,530 trials in one fit is out of compute reach here |
| conditions per-state accuracy on the smoothed marginal posterior | publish filtered and one-step-ahead predictive state probabilities only | filtered probabilities are substituted |
| compares against a lapse model with two asymmetric lapse rates | offer `mix(Psychometric(...), UniformChoiceGuess())` with one symmetric rate | a three-parameter comparator stands in for a four-parameter one |
| cross-validates a sweep over the number of states | *nothing in the package does this* | the sweep is assembled inside the benchmark from `Study.take` and `pointwise_log_prob` |

That last row is the one worth dwelling on. Ashwood's headline structure **is** a
model-selection result, and Behavio ships no procedure for selecting a state count anywhere.
`nested_select_model` selects among a supplied list of models under a *prospective* nested
scheme; it is not a K-sweep, and the paper's design is not prospective. The sweep in this
benchmark is hand-assembled, which is a fair description of a gap, not of a feature.

### What the paper does *not* claim, and what this benchmark therefore does not check

It is tempting to encode "a three-state model wins cross-validation" as the headline parity
claim. The paper does not say that. For the example mouse it says the multi-state GLM-HMM
"outperformed both the standard (one-state) GLM and the classic lapse model … with the
improvement approximately levelling off at three latent states", and for the cohort that three
states "substantially outperformed models with *fewer* states". Three states are then adopted
because the authors "used [them] for all subsequent analyses" — a parsimony judgement, of the
same kind they make explicitly for a different dataset ("the four-state model balanced
simplicity and interpretability"). The four- and five-state models are never claimed to be
worse.

So the state count is computed here, and the whole bits-per-trial curve over K is retained in
`result.json`, but the arg-max is **reported, not asserted**. What is checked instead are the
two numbers the paper does print: the three-state model's advantage over the one-state GLM and
over the lapse model. Section 9a of the protocol records this correction, which removes a
claim rather than loosening a tolerance.

One more attribution worth stating plainly: the paper names no example animal. `CSHL_008`
comes from the reference implementation, which branches on that identifier when drawing
Figures 2, 3 and 4. Every example-mouse number here is therefore a claim about the animal the
*code* points at, and the exact match on its trial and session counts is part of what makes
that identification credible.

## The validation boundary this analysis does not respect

Ashwood cross-validates by assigning whole sessions to five folds *at random*. A held-out
session can precede a training session in time. Behavio's `evaluate_splits` refuses
non-prospective folds unless you say `require_prospective=False`, and the rest of this
documentation argues at length for why.

The design is used here anyway, deliberately, because the published number being checked was
produced under it. Reproducing somebody's number means reproducing their design, including
the parts you would not choose. It is not an endorsement, and the recommended practice
remains the prospective boundary used everywhere else in these chapters.

## Reproduce it

```bash
uv run python -m benchmarks.ashwood2022_glmhmm.fetch_data
uv run python -m benchmarks.ashwood2022_glmhmm.benchmark \
    --output benchmarks/ashwood2022_glmhmm/result.json
```

The [committed benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ashwood2022_glmhmm)
retains every fold score, every per-animal transition matrix and emission vector, the fit
diagnostics behind them, and
[`published_claims.json`](https://github.com/aeronjl/behavio/tree/main/benchmarks/ashwood2022_glmhmm/published_claims.json),
which `tests/test_published_parity.py` discovers automatically and checks offline.

## A separate, narrower study: does a latent state predict a future session?

The study below predates the replication above and asks a different, smaller question. It is
retained because it is the prospective counterpart to the interpolation design used for
parity: after choosing the number of latent states entirely in earlier sessions, does a
GLM-HMM predict an untouched IBL session better than a stationary history GLM?

!!! warning "Structural analogue, not reproduction"

    This worked study borrows its scientific question from Ashwood et al. (2022), but uses a
    smaller covariate set, one animal, a different session boundary, and Behavio's own
    prospective state-count procedure. It does not reproduce their paper.

The public source and outcome-blind subject rule are identical to the
[choice/response-time study](ibl2021-choice-response-time.md). The experimental unit is a
left/right-choice trial from `CSHL045`. Up to 150 source rows per session are fixed before
no-go trials are removed, leaving 893 eligible choices across six ordinal endpoint windows.

### Nested prospective boundary

The comparison has two validation layers:

1. fit 2-, 3-, and 4-state GLM-HMMs on positions 0–3 and select by mean log loss on
   position 4;
2. refit only the selected state count on positions 0–4, then compare it with a stationary
   stimulus-plus-choice-history GLM on untouched position 5.

Both candidates score choice only. Within the test session, probabilities are filtered
one step ahead: an observed choice may update the state distribution used for the next
trial, but no future choice is visible before it is scored.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped">
  <img src="../../assets/ibl-glmhmm-states.svg" alt="Four panels show filtered latent-state probabilities in the untouched session, fitted state-specific emission coefficients, the fitted transition matrix, and inner state-count selection beside outer static-GLM and GLM-HMM log losses.">
  <figcaption><strong>Literature-shaped · latent structure stays attached to its validation boundary.</strong> State probabilities and coefficients describe the selected fit; the loss panels show selection and untouched-session performance. Near-tied state counts and warning-level diagnostics preclude a claim of four biological strategies.<span class="doc-figure__meta"><strong>Unit:</strong> eligible choice from one animal · <strong>n:</strong> 893 choices, including 150 untouched choices · <strong>Estimand:</strong> training-only state-count selection and future-session log loss · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

### Result

Position-4 selection log losses are `0.67037`, `0.49956`, and `0.49954` for 2, 3, and 4
states. The declared rule therefore selects four states, although the three- and
four-state results are practically tied to five decimal places.

After refitting, the selected GLM-HMM achieves test log loss `0.4447`, compared with
`0.6692` for the static GLM—an improvement of `0.2245` per trial across 150 held-out
choices. Its filtered distribution spends 93.8% of the test session in the highest
sensory-weight state and changes maximum-probability label three times.

The fit audit is a central part of the result, not a footnote. It warns about an
ill-conditioned Hessian, boundary estimates, and disagreement among converged restarts.
The evidence supports better prediction by this selected procedure for this one session;
it does not support a confident claim that four biological strategies exist.

### Interpretation boundary

State labels are canonicalized by fitted stimulus weight. The plotted filtered
probabilities condition on choices already observed in the held-out session and are
model-dependent summaries, not directly measured neural states. Near-tied state-count
selection and numerical warnings should motivate design-matched state and model recovery
before substantive interpretation.

```bash
uv run --extra ibl python -m benchmarks.ibl2021_decision_models.benchmark
uv run --group docs python -m scripts.plot_documentation_figures --skip-cell
```

The [committed benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021_decision_models)
retains all candidate scores, restart evidence, fit audits, transition and emission
parameters, and trialwise predictive and filtered state probabilities. Read it alongside
the [GLM-HMM method contract](../glm-hmm.md) and the
[model-recovery study](model-recovery-design.md).
