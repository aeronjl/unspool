# Posterior convergence diagnostics

`audit_posterior` applies one backend-neutral convergence policy to any
[`PosteriorResult`](posterior-results.md). It computes rank-normalized split
$\widehat R$, bulk effective sample size (ESS), and tail ESS for every labelled natural
parameter. When an HMC backend retained the relevant sample statistics, it also counts
divergent transitions and transitions that reached maximum tree depth.

The result is an immutable `PosteriorAudit`: a compact pass/warning decision, the exact
policy, all numeric diagnostics, and stable issue codes with labelled parameter targets.
It does not silently turn a sampler-specific summary table into a scientific verdict.

## The routine workflow

```python
from behavio import audit_posterior
from behavio.posterior import PosteriorAuditPolicy

posterior = backend.sample(model, study, task=task)

policy = PosteriorAuditPolicy(
    max_rhat=1.01,
    min_ess_bulk=400,
    min_ess_tail=400,
    max_divergences=0,
    max_treedepth_hits=0,
)
audit = audit_posterior(posterior, policy=policy)

print(audit.status)  # "pass" or "warning"
print(audit.issue_codes)  # stable, machine-readable codes
for issue in audit.issues:
    print(issue.code, issue.targets)
```

For a labelled coefficient array, a warning target is precise:
`population_coefficient[coefficient='choice_lag_1']`. Scalar parameters retain their
plain name. The diagnostic arrays and their coordinates are also available through
`audit.diagnostics` for tables or figures.

Install the optional diagnostic dependency with:

```bash
pip install "behavio[probabilistic]"
```

The same public API is tested against ArviZ's `InferenceData` representation on Python
3.11 and its current `DataTree` representation on Python 3.12 and later.

## What the defaults mean

| Check | Default | Warning means |
| --- | ---: | --- |
| rank-normalized split $\widehat R$ | $> 1.01$ | chains have not mixed closely enough for the declared policy |
| bulk ESS | $< 400$ | central posterior summaries have too little effective information |
| tail ESS | $< 400$ | interval endpoints or tail summaries have too little effective information |
| divergences | $> 0$ | the HMC trajectory reported a numerical pathology |
| maximum-tree-depth hits | $> 0$ | NUTS may be exploring the posterior inefficiently |

The $\widehat R$ and ESS definitions follow [Vehtari et al.
(2021)](https://doi.org/10.1214/20-BA1221) and ArviZ's
[rank-normalized $\widehat R$ implementation](https://python.arviz.org/en/stable/api/generated/arviz.rhat.html).
The defaults are an explicit screening policy, not universal laws. A protocol may require
larger ESS for small Monte Carlo error, and a difficult scientific model may need a more
specific criterion. Record any changed policy alongside the result.

Maximum tree depth is kept distinct from divergences. The [Stan diagnostics
guide](https://mc-stan.org/learn-stan/diagnostics-warnings.html) describes high $\widehat
R$, low ESS, and divergences as validity concerns, while maximum-tree-depth saturation is
primarily an efficiency warning. Behavio retains that distinction but reports both rather
than discarding backend evidence.

## Stable issue codes

| Code | Trigger |
| --- | --- |
| `posterior.rhat` | at least one labelled parameter exceeds `max_rhat` |
| `posterior.ess-bulk` | at least one labelled parameter is below `min_ess_bulk` |
| `posterior.ess-tail` | at least one labelled parameter is below `min_ess_tail` |
| `posterior.divergences` | retained divergent-transition count exceeds policy |
| `posterior.max-treedepth` | retained maximum-tree-depth count exceeds policy |
| `posterior.nonfinite-diagnostic` | ArviZ returns a non-finite $\widehat R$ or ESS |

Sampler statistics are backend capabilities. If `sample_stats.diverging` or
`sample_stats.reached_max_treedepth` is absent, its audit count is `None`; absence is not
misreported as zero.

## Interpretation boundary

A passing convergence audit says that these retained draws did not violate this policy.
It does **not** establish that the model is identified, correctly specified, predictively
useful, or scientifically calibrated. Use it alongside:

- posterior predictive checks for observable mismatch;
- simulation-based calibration and exact-design recovery for inferential calibration;
- prospective prediction for longitudinal generalization;
- [analysis sensitivity](sensitivity-analysis.md) for prior and modelling choices; and
- pointwise predictive comparison where its assumptions are appropriate.

Those procedures remain separate because they answer different questions. Convergence is
necessary evidence for an MCMC fit, not a substitute for model criticism.
