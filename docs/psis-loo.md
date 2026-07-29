# PSIS-LOO predictive evaluation

`psis_loo` estimates leave-one-out expected log predictive density from the pointwise log
likelihood retained in a [`PosteriorResult`](posterior-results.md). It delegates the
importance-sampling calculation to ArviZ, then returns a small immutable result with:

- total log-scale expected log pointwise predictive density (`elpd_loo`);
- its standard error and the effective number of parameters (`p_loo`);
- labelled pointwise ELPD and Pareto-$k$ arrays;
- the sample-size-dependent reliability threshold;
- the cross-validation estimand actually computed (`block`, `estimand`);
- the retained posterior convergence audit (`convergence`); and
- stable warning codes localized to the influential observations or blocks.

This is an interoperability layer around an established method, not a new information
criterion.

## Basic use

```python
from behavio import psis_loo

posterior = backend.sample(model, study, task=task)
loo = psis_loo(posterior)

print(loo.estimand, loo.elpd_loo, loo.se, loo.p_loo)
print(loo.status, loo.issue_codes)
for issue in loo.issues:
    print(issue.code, issue.severity, issue.targets)
```

When the result contains one log-likelihood variable, its name is inferred. Select it
explicitly for a joint model:

```python
choice_loo = psis_loo(posterior, log_likelihood_name="choice")
```

An aggregated log likelihood is rejected. PSIS-LOO needs one retained contribution per
scored observation, leading with the same `chain` and `draw` axes as the posterior. The
returned pointwise arrays preserve the remaining axes and coordinates, such as `trial`,
`subject`, or `session`.

## Convergence gating

ELPD computed from a posterior that did not converge is not evidence. `psis_loo` therefore runs
[`audit_posterior`](posterior-diagnostics.md) on every call and retains the whole audit on
`loo.convergence`. The policy is injectable:

```python
from behavio import PosteriorAuditPolicy

loo = psis_loo(posterior, policy=PosteriorAuditPolicy(max_rhat=1.005))
```

Following the same idiom as `FitAuditStatus.FAIL` in the runner, a failed audit does not raise.
The number is retained and marked, so a report can show *what* was computed alongside *why it
must not be believed*. `PSISLOOResult.status` folds two independent sources of doubt into one
verdict by worst severity:

| Source | Issue codes | Severity |
| --- | --- | --- |
| Importance sampling | `psis.high-pareto-k`, `psis.backend-warning`, `psis.few-blocks` | warning |
| Importance sampling | `psis.nonfinite` | error |
| Convergence audit | `psis.posterior-warning` | warning |
| Convergence audit | `psis.posterior-not-converged` | error |

Any error gives `FAIL`, any remaining issue gives `WARNING`, and an empty issue list gives
`PASS`. A `FAIL` result is never ranked by
[`compare_posterior_models`](#comparing-models-on-elpd).

## Blocked LOO: choosing the held-out unit

The default estimand is leave-one-*observation*-out. For the hierarchical backends in this
package the declared likelihood carries `dims="trial"`, so the default is leave-one-**trial**-out,
and on multi-subject, multi-session data that is usually the wrong question. The held-out
trial's own subject stays in the fit, its subject-level parameters were estimated partly from
that trial, and neighbouring trials carry history features derived from the held-out response.
Trial-level ELPD therefore systematically **overstates** predictive performance relative to
"how well does this model predict a subject, or a session, it has not seen?".

`block` changes the estimand. Naming a grouping variable sums the pointwise log likelihood
**within each group before** importance sampling, so PSIS reweights draws for the removal of a
whole subject or session:

```python
trials = psis_loo(posterior)
sessions = psis_loo(posterior, block="trial_session")
subjects = psis_loo(posterior, block="trial_subject")

print(trials.estimand, trials.elpd_loo)  # leave-one-observation-out
print(sessions.estimand, sessions.elpd_loo)  # leave-one-trial_session-out
print(subjects.estimand, subjects.elpd_loo)  # leave-one-trial_subject-out
```

Summing on the log scale *before* PSIS is what changes the estimand. Summing the pointwise ELPD
*afterwards* would only re-aggregate leave-one-trial-out numbers and would keep every bit of the
optimism.

### Where the grouping comes from

Nothing is inferred from coordinate names or heuristics. `block` must name a one-dimensional
variable in `constant_data` or `observed_data` laid out over the scored dimension. The PyMC
backend retains `trial_subject`, `trial_session`, `trial_in_session`, and `trial_session_order`
for exactly this purpose. An unknown name raises and lists the retained candidates.

For posteriors that do not carry the grouping, supply the labels explicitly. The length is
validated against the scored dimension, and `block` then names the estimand:

```python
loo = psis_loo(posterior, block="animal", block_values=study["subject"])
```

Blocks are ordered by first appearance, so subject and session boundaries stay in the order the
study presents them. The returned `dims` become `(block,)` and the pointwise arrays hold one
value per block.

### Reading blocked diagnostics

- **`p_loo` and `se` are computed on the blocked scale.** They come from the blocked pointwise
  ELPD, not from re-aggregated trial-level numbers. `p_loo` normally rises under blocking,
  because removing a subject removes the information that identified that subject's parameters.
- **`good_k` is unchanged by blocking.** The threshold $k_{good}$ is a function of the number of
  posterior draws $S$, not of the number of observations. What blocking changes is the *meaning*
  of crossing it: leaving out a whole subject is a far larger perturbation than leaving out one
  trial, so high $k$ is both more likely and more consequential, and there are far fewer values
  in which to hide it. The `psis.high-pareto-k` message reports the affected share of blocks.
- **`se` is a normal approximation over the pointwise unit.** Blocking can collapse twenty
  thousand trials into eight subjects. Below ten blocks, `psis.few-blocks` fires rather than
  reporting a confidently narrow interval.

`block=None` reproduces the unblocked calculation exactly, value for value.

## Comparing models on ELPD

`compare_posterior_models` reports **paired** ELPD differences. The standard error of a total
ELPD is dominated by how hard the observations are; that difficulty cancels observation by
observation, so the standard error of the difference is computed from the pointwise differences:

\[
\widehat{\text{elpd}}_A - \widehat{\text{elpd}}_B = \sum_i \left(\widehat{\text{elpd}}_{A,i} -
\widehat{\text{elpd}}_{B,i}\right), \qquad
\text{se}_\text{diff} = \sqrt{n \operatorname{Var}_i\!\left(\widehat{\text{elpd}}_{A,i} -
\widehat{\text{elpd}}_{B,i}\right)}.
\]

It is never `A.se - B.se` and never $\sqrt{\text{se}_A^2 + \text{se}_B^2}$; both ignore the
positive correlation between models and inflate the interval.

```python
from behavio.posterior_comparison import compare_posterior_models

comparison = compare_posterior_models(
    {"hierarchical": hierarchical_posterior, "pooled": pooled_posterior},
    block="trial_subject",
)

print(comparison.estimand, comparison.status, comparison.best_model, comparison.reason)
for model in comparison.models:
    print(model.name, model.elpd_loo, model.status, model.issue_codes)
for difference in comparison.differences:
    print(
        difference.left_model,
        difference.right_model,
        difference.elpd_difference,
        difference.se,
        difference.excludes_zero,
    )
```

The mapping accepts either `PSISLOOResult` values or `PosteriorResult` values; posteriors are
scored here under the shared `log_likelihood_name`, `block`, `block_values`, and `policy`, so
every model is scored the same way by construction.

### What it refuses to do

The pairing is only meaningful when index $i$ means the same observation in both models, so
alignment is checked rather than assumed. A mismatch raises; it is never repaired by
broadcasting, reindexing, or truncation, because every such repair silently changes the
estimand.

| Refusal | Raised when |
| --- | --- |
| different estimands | one result was blocked and another was not, or they used different blocks |
| different dimensions | the pointwise units do not share the same labelled dimensions |
| different observation counts | the results score a different number of units |
| different coordinates | the same dimension carries different labels, or the same labels in a different order |

### What it reports rather than hides

| Issue code | Meaning |
| --- | --- |
| `comparison.posterior-fail` | one or more posteriors failed convergence or produced non-finite ELPD; excluded from ranking |
| `comparison.posterior-warning` | one or more models carry unresolved diagnostic warnings |
| `comparison.high-pareto-k` | a model's pointwise contributions rest on unstable importance weights |
| `comparison.likelihood-name-mismatch` | the models score differently named likelihood variables |
| `comparison.few-observations` | too few units to support the normal approximation behind the interval |

Ranking follows the same stance as the [protocol runner](protocols/auditing.md): a model whose
posterior failed is never ranked, and `status` stays `unresolved` with `best_model is None`
whenever a paired interval fails to exclude zero. There is no automatic winner, and there is no
threshold at which an overlapping interval is quietly resolved in favour of the larger point
estimate.

## Reading Pareto $k$

PSIS approximates each leave-one-out posterior by reweighting draws from the full
posterior. Pareto $k$ diagnoses the stability of those importance weights. Behavio retains
ArViZ's sample-size-dependent threshold

\[
k_{good} = \min\!\left(1 - \frac{1}{\log_{10}(S)},\; 0.7\right),
\]

where $S$ is the number of retained posterior samples. A value above that threshold emits
`psis.high-pareto-k` with a target such as `choice[trial=173]`. Non-finite pointwise output
emits `psis.nonfinite`; an unlocalized upstream warning is retained as
`psis.backend-warning`.

The current method and diagnostic follow [Vehtari et al.'s PSIS
paper](https://jmlr.org/papers/v25/19-556.html) and ArviZ's
[PSIS-LOO implementation](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.loo.html).
A high $k$ is evidence that the approximation is unreliable for that observation. It may
motivate a more robust model, exact or moment-matched LOO, or $K$-fold validation; it is not
a licence to delete the influential trial after seeing the result.

## The longitudinal boundary

Blocking fixes the *grouping* boundary. It does not fix the *time* boundary, and the two are
different problems.

`block="trial_subject"` removes a whole subject, so nothing about that subject's parameters is
learned from the held-out data. `block="trial_session"` does the same for a session. Both are
genuine grouped cross-validation and both stay inside the Bayesian pipeline. What neither does
is recreate the information set available at an earlier point in time: later sessions from
*other* subjects remain visible to the fit, and the model is not being asked to forecast forward.

So the boundary is now specific:

- use **blocked PSIS-LOO** to ask whether the model generalizes to an unseen subject or session,
  and `compare_posterior_models` to compare candidates on that estimand;
- use **unblocked PSIS-LOO** to inspect pointwise predictive fit and locate influential
  observations under a declared likelihood;
- use [prospective validation](validation.md) when the question is genuinely temporal — later
  behaviour, drift, or forecasting forward — because no leave-one-group-out estimand answers it;
- keep model selection inside the training portion of a nested prospective protocol; and
- use [recovery](model-recovery.md) to ask whether the design can distinguish the candidate
  mechanisms at all.

These quantities can coexist in one report because they answer different questions. A good
blocked ELPD with reliable Pareto $k$ establishes generalization across the blocked grouping; it
does not establish temporal generalization, parameter recovery, or model adequacy.

## Compatibility and serialization

Install the optional dependency with:

```bash
pip install "behavio[probabilistic]"
```

Behavio tests the same `psis_loo` result against legacy ArviZ `InferenceData` on Python
3.11 and current `DataTree`/ArviZ Stats on Python 3.12 and later. `loo.to_dict()` produces a
JSON-compatible record containing the summary, flattened pointwise values, dimension
metadata, coordinates, warnings, and backend provenance.
