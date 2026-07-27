# PSIS-LOO predictive evaluation

`psis_loo` estimates leave-one-out expected log predictive density from the pointwise log
likelihood retained in a [`PosteriorResult`](posterior-results.md). It delegates the
importance-sampling calculation to ArviZ, then returns a small immutable result with:

- total log-scale expected log pointwise predictive density (`elpd_loo`);
- its standard error and the effective number of parameters (`p_loo`);
- labelled pointwise ELPD and Pareto-$k$ arrays;
- the sample-size-dependent reliability threshold; and
- stable warning codes localized to the influential observations.

This is an interoperability layer around an established method, not a new information
criterion.

## Basic use

```python
from unspool import psis_loo

posterior = backend.sample(model, study, task=task)
loo = psis_loo(posterior)

print(loo.elpd_loo, loo.se, loo.p_loo)
print(loo.status, loo.issue_codes)
for issue in loo.issues:
    print(issue.code, issue.targets)
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

## Reading Pareto $k$

PSIS approximates each leave-one-out posterior by reweighting draws from the full
posterior. Pareto $k$ diagnoses the stability of those importance weights. Unspool retains
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

Observation-level LOO asks what happens when one scored observation is omitted while the
rest of the fitted dataset remains available. It does not recreate the information set at
an earlier session. In a behavioural sequence, surrounding trials may contain history
derived from the held-out response, and later sessions remain visible to the fit.

Therefore:

- use PSIS-LOO to inspect pointwise predictive fit and influential observations under a
  declared likelihood;
- use [prospective validation](validation.md) to assess genuinely later behaviour;
- keep model selection inside the training portion of a nested prospective protocol; and
- use [recovery](model-recovery.md) to ask whether the design can distinguish the candidate
  mechanisms at all.

These quantities can coexist in one report because they answer different questions. A
good ELPD with reliable Pareto $k$ does not establish temporal generalization, parameter
recovery, or model adequacy.

## Compatibility and serialization

Install the optional dependency with:

```bash
pip install "unspool[probabilistic]"
```

Unspool tests the same `psis_loo` result against legacy ArviZ `InferenceData` on Python
3.11 and current `DataTree`/ArviZ Stats on Python 3.12 and later. `loo.to_dict()` produces a
JSON-compatible record containing the summary, flattened pointwise values, dimension
metadata, coordinates, warnings, and backend provenance.
