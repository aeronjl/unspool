# Ashwood 2022 GLM-HMM replication

A bounded replication of Ashwood et al., "Mice alternate between discrete strategies during
perceptual decision-making," *Nature Neuroscience* 25:201-212 (2022),
doi [`10.1038/s41593-021-01007-z`](https://doi.org/10.1038/s41593-021-01007-z).

This is the canonical GLM-HMM paper for perceptual decision-making, it has public data and
public reference code, and Unspool ships a GLM-HMM. That makes it the natural place to ask
whether Unspool's implementation recovers numbers somebody else published, rather than only
numbers Unspool generated.

`PROTOCOL.md` was frozen before any of these numbers existed. It states what would be
computed, with which hyperparameters, on which cohort, and inside which tolerances. Read it
first — everything below is downstream of it.

## Relationship to `ssm` and `psytrack`

Ashwood's reference implementation,
[`github.com/zashwood/glm-hmm`](https://github.com/zashwood/glm-hmm), fits its models with a
fork of Scott Linderman's [`ssm`](https://github.com/lindermanlab/ssm), modified so that
violation trials can be handled as missing observations. `ssm` is the mature, general
state-space modelling library in this space: it supports many emission and transition
families, hierarchical structure, variational and Laplace-EM inference, and observation
masks. Nick Roy and Jonathan Pillow's
[`psytrack`](https://github.com/nicholas-roy/psytrack) is the companion tool for the
*smoothly drifting* account of the same behaviour, which is the alternative hypothesis
Ashwood's discrete-state model is argued against.

Unspool's `BernoulliGLMHMM` is **narrower than both** and is not a replacement for either.
It fits one emission family, has no hierarchy, no observation mask, no Dirichlet transition
prior and no procedure for selecting the number of states. What it adds is a different thing:
the surrounding contract — frozen protocols, checksum-pinned inputs, provenance-stamped
results, retained failures. If you want to fit a GLM-HMM to your data, use `ssm`. This
benchmark exists to measure how far Unspool's narrower implementation gets on somebody
else's published numbers, and to say plainly where it stops.

## Data

Figshare [`10.6084/m9.figshare.11636748.v7`](https://doi.org/10.6084/m9.figshare.11636748),
file id `21623715`, `ibl-behavior-data-Dec2019.zip`, **218 MiB** (228,602,597 bytes),
SHA-256 `18bfacccf615a767dd6e3935473b628fe4266e9b12c09200ee7f4eac2c54c4e6`. Released by the
International Brain Laboratory under **CC BY 4.0**. It is the exact file the reference
implementation downloads by numeric id, and the same underlying study as the `ibl2021_*`
benchmarks here, which instead read the newer OpenAlyx Parquet distribution.

`fetch_data.py` verifies size, MD5 and SHA-256. The archive is read in place; nothing is
extracted.

```bash
uv run python -m benchmarks.ashwood2022_glmhmm.fetch_data
uv run python -m benchmarks.ashwood2022_glmhmm.benchmark \
    --output benchmarks/ashwood2022_glmhmm/result.json
```

## What is reproduced

**8 of 14 checkable claims pass; 6 fail; 6 more are waived.** The benchmark is classified
`failed-parity`, and the failures are kept.

| published value | paper | this benchmark | outcome |
| --- | --- | --- | --- |
| mice in the cohort | 37 | **37** | published-parity |
| sessions | 2,017 | **2,017** | published-parity |
| trials | 181,530 | **181,530** | published-parity |
| example mouse trials | 5,040 | **5,040** | published-parity |
| example mouse sessions | 56 | **56** | published-parity |
| engaged-state accuracy | 90% | **93.9%** | published-parity |
| biased-left accuracy | 60% | **40.9%** | failed-parity |
| biased-right accuracy | 58% | **41.0%** | failed-parity |
| median engaged dwell | 24 trials | **30.8** | failed-parity |
| median biased-left dwell | 13 trials | **16.5** | failed-parity |
| median biased-right dwell | 12 trials | **10.3** | published-parity |
| median engaged occupancy | 69% | **58.2%** | failed-parity |
| bits/trial over the 1-state GLM | 0.13 | **0.1309** | published-parity |
| bits/trial over the lapse model | 0.09 | **0.1274** | failed-parity |

The pattern is the point. **Every quantity that required no substitution reproduces, and
every failure moves in the direction its declared substitution predicts.**

### The cohort reproduces bit for bit

All five counting claims land exactly. Ashwood's three selection filters, reimplemented from
the paper and cross-checked against the reference code, select the same 37 animals, the same
2,017 sessions and the same 181,530 trials — of which 165 (0.09%) are no-response violations
that this benchmark drops and the paper masks. The example animal identified from the
reference code has exactly the 5,040 trials over 56 sessions the Results text reports, which
is what makes that code-derived identification credible.

### The state-count curve reproduces the paper's actual claim

| states | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| bits/trial | 0.3014 | 0.3922 | **0.4323** | 0.4323 | 0.4323 |

The gain from three states to five is **0.000018 bits per trial**. This is exactly the
plateau the paper describes — "the improvement approximately levelling off at three latent
states" — and it is why the protocol's amendment matters: the arg-max here is K = 5, so a
gate asserting "three states win" would have recorded a failure with no scientific content
whatsoever. The paper never made that claim, and this benchmark does not check it.

The advantage over the one-state GLM, **0.1309 bits/trial against a published 0.13**, is the
cleanest single result here: the one model comparison in which no component was substituted
lands within a thousandth of a bit.

### Why each failure fails

- **Biased-state accuracies, 41% against a published 60% and 58%.** Both fall *below chance*,
  which is the signature of the substitution rather than of the model. Unspool exposes no
  smoothed posterior, so trials are attributed to a state by *filtered* probability — and the
  filter is updated by the current trial's own outcome. Conditioning on filtered probability
  >= 0.9 therefore preferentially selects the trials that most strongly evidence a biased
  state, namely those where the animal chose its favoured side *against* the stimulus. Only
  257 and 273 trials survive that threshold, against 1,991 for the engaged state. The engaged
  state, where choice and stimulus agree, is unaffected and passes at 93.9%.
- **Dwell times, +28% and +27% on the two long-dwell states.** Dwell is `1/(1 - A_kk)`. The
  paper's Dirichlet(alpha = 2) prior adds pseudo-counts to *every* transition entry, pulling
  self-transitions away from one; Unspool's `stickiness` can only add them to the diagonal, so
  the transition matrix here is fitted with no prior at all and self-transitions sit higher.
  The short-dwell biased-right state, least affected, passes.
- **Engaged occupancy, 58.2% against a published 69%.** The paper counts hard `argmax` state
  assignments; `state_occupancy` is the mean *smoothed posterior probability* per state. The
  protocol declared in advance that the soft quantity would be gated, and it fails. The hard
  count is also computed: **64.4%**, which would fall inside the +/- 5 point band. Neither is
  precisely the paper's quantity — the hard count here is an arg-max over filtered rather than
  smoothed probabilities — and the gate has not been switched after the fact. It is reported
  because it locates the failure in the definition rather than in the fit.
- **Advantage over the lapse model, 0.1274 against a published 0.09.** This one fails by being
  *too good*. `LapsePsychometric` has a single symmetric lapse rate; the paper's comparator has
  two asymmetric ones. The weaker stand-in scored 0.3049 bits/trial, barely above the plain
  GLM's 0.3014, so the GLM-HMM's margin over it is inflated by roughly the amount the missing
  fourth parameter is worth.

All 37 per-animal fits converged. Total runtime: **59 minutes** on 8 worker processes.

## What is not reproduced

Six results from the paper are recorded in `published_claims.json` as `waived`, each with a
written rationale, rather than omitted. A waiver means the claim is machine-readably
*unchecked* — not quietly implied by the neighbouring claims that were checked.

| waived claim | why |
| --- | --- |
| **the three-state selection itself** (Figs. 2b, 4a) | The paper never reports an arg-max. It says the improvement "approximately levels off at three latent states" and that three states are what "we used for all subsequent analyses"; K = 4 and K = 5 are never claimed to be worse. Encoding an arg-max would invent a published value. The state count and the whole bits-per-trial curve over K are computed and reported in `result.json` without being asserted. |
| **K selection across all 37 animals** (Fig. 4a) | Only the example mouse is cross-validated. Five folds x five candidates x 37 animals is roughly forty hours of single-core time with this implementation. |
| **the pooled global fit and cross-animal state alignment** (Methods, Algorithm 1) | Unspool cannot initialize a fit from externally supplied parameters, so the paper's seeding procedure is inexpressible; a 181,530-trial fit is also out of compute reach. |
| **population predictive-accuracy gains of 4.2% and 2.8%** (Results) | Both are averages over the 37-animal cross-validation waived above. |
| **response-time signatures of state** (Fig. 6) | `BernoulliGLMHMM` has Bernoulli choice emissions only and cannot attach a latency distribution to a latent state. Structurally inexpressible, not merely expensive. |
| **the Odoemene mice and human participants** (Figs. 5, 7) | Different datasets under different accessions. The declared data boundary is the single IBL archive. |

The first row is the important one. Ashwood's best-known result is usually paraphrased as
"three states win cross-validation". Reading the paper closely, that is not what it claims,
and this benchmark declines to check a number the article does not print. `PROTOCOL.md`
section 9a records that correction, which **removes** a claim and loosens no tolerance.

## What Unspool could not express

This is the most useful output of the exercise. Each row is a place where the paper's
procedure has no equivalent in this package. All are declared in `PROTOCOL.md` section 6,
before the computation, because each one moves the numbers.

| the paper does | Unspool can | what was done instead |
| --- | --- | --- |
| keeps violation trials in the sequence and replaces their emission likelihood with 1 | nothing — `BernoulliGLMHMM` has no observation mask | the violation rows are dropped |
| places a Dirichlet(alpha = 2) prior on every transition row, with `kappa = 0` | only add pseudo-counts to the *diagonal*, through `stickiness` | `stickiness = 0`; the transition matrix is fitted with no prior |
| applies its Gaussian prior (sigma = 2) to all four weights, bias included | penalize every coefficient *except* the intercept | `l2 = 1/sigma^2 = 0.25`, intercept unpenalized |
| fits one pooled GLM-HMM over all 37 animals and seeds every per-animal fit from it, aligning state labels | not initialize a fit from externally supplied parameters at all | labels are canonicalized by the fitted stimulus weight; no pooled fit |
| conditions per-state accuracy on the smoothed marginal posterior | publish filtered and one-step-ahead predictive state probabilities only | filtered probabilities are substituted |
| measures fractional occupancy as a hard count of `argmax` state assignments | expose `state_occupancy`, the *mean smoothed posterior probability* per state | the soft occupancy is compared; the hard one is also reported |
| compares against a lapse model with two asymmetric lapse rates | offer `LapsePsychometric` with one symmetric rate, capped at 0.2 | a three-parameter comparator stands in for a four-parameter one |
| maximizes the MAP objective by EM | maximize the same penalized marginal likelihood directly by L-BFGS-B multi-start | same objective up to the prior differences above, different optimizer |
| selects the number of states by cross-validating a sweep over K | **nothing in the package does this** | the sweep is hand-assembled in `benchmark.py` from `Study.take` and `pointwise_log_prob` |

The last row is the structural one. `nested_select_model` selects among a supplied list of
models under a *prospective* nested scheme; it is not a sweep over state count, and the
paper's cross-validation is not prospective. There is no K-selection procedure in Unspool.

Two further constraints are computational rather than structural. A pooled fit over 181,530
trials, and a five-fold sweep over five candidates repeated for all 37 animals, are both out
of reach: the forward-backward recursion in `BernoulliGLMHMM` costs roughly 50 microseconds
per trial per objective evaluation, and `fit` additionally builds a numerical Hessian costing
`2 * n_parameters` further evaluations.

## Published parity

[`published_claims.json`](published_claims.json) records each published value, the tolerance
declared for it in `PROTOCOL.md` before the computation, the value this benchmark produced,
and a `pass`/`fail`/`waived` status. `tests/test_published_parity.py` discovers it
automatically and runs offline in milliseconds, so a drift away from a published number
cannot survive a default test run.

Failures are kept. No tolerance and no hyperparameter in this directory was changed after
seeing a number.
