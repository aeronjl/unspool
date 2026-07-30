# Posterior-predictive checks

Posterior-predictive checks ask whether data replicated from a fitted model reproduce
scientifically relevant features of the observations. Behavio supplies a small discrepancy
contract rather than a fixed dashboard: each check declares what feature it measures, the
relevant reference tail, and a stable signature retained with the result.

This follows the realized-discrepancy framework of [Gelman, Meng, and Stern
(1996)](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n41/j6n41.htm) and the iterative
model-criticism workflow described by [Gabry et al.
(2019)](https://doi.org/10.1111/rssa.12378).

## A minimal behavioural check

```python
from behavio import posterior_predictive_check
from behavio.posterior import CategoryRateDiscrepancy, MeanDiscrepancy, SwitchRateDiscrepancy

audit = posterior_predictive_check(
    posterior,
    (
        CategoryRateDiscrepancy(1),
        MeanDiscrepancy(),
        SwitchRateDiscrepancy(),
    ),
    variable_name="choice",
)

for check in audit.checks:
    print(check.discrepancy_name, check.observed, check.interval)
```

Each check retains the observed statistic, its complete chain-by-draw replicated reference
distribution, a central predictive interval, lower and upper reference probabilities, and
the declared tail probability. Values are immutable and `audit.to_dict()` retains the
evidence for a report or bundle.

The built-in discrepancies are intentionally elementary:

| Discrepancy | Default tail | Typical question |
| --- | --- | --- |
| `MeanDiscrepancy` | two-sided | Does the model reproduce the response mean? |
| `VarianceDiscrepancy` | upper | Is the observed variability unusually large? |
| `CategoryRateDiscrepancy(value)` | two-sided | Does it reproduce a choice or omission rate? |
| `SwitchRateDiscrepancy` | two-sided | Does it reproduce adjacent response switching? |

Simple summaries are useful because their meaning is visible. They are not a universal
battery. A good analysis should declare discrepancies that expose failures relevant to its
scientific claim, such as psychometric shape, reward-conditional switching, block
adaptation, response-time tails, or learning-trajectory landmarks.

## Grouped checks

Replicated behaviour often needs to be checked per animal, session, lab, or condition:

```python
by_subject = posterior_predictive_check(
    posterior,
    (CategoryRateDiscrepancy(1), SwitchRateDiscrepancy()),
    variable_name="choice",
    groupby=("trial_subject",),
)
```

Grouping variables come from `constant_data` and must use exactly the observed outcome's
dimensions and coordinates. This prevents an apparently plausible summary from silently
joining reordered trials. The PyMC hierarchical GLM adapter retains `trial_subject`,
`trial_session`, `trial_in_session`, and `trial_session_order` for this purpose.

`SwitchRateDiscrepancy` uses the supplied row order. Group by subject and session before
interpreting it as within-session switching; the discrepancy deliberately does not infer
chronology or bridge session boundaries on the user's behalf.

## Policies and warnings

```python
from behavio.posterior import PosteriorPredictivePolicy
from behavio.posterior.predictive import PredictiveMultiplicity

policy = PosteriorPredictivePolicy(
    interval_probability=0.9,
    tail_probability_warning=0.05,
    multiplicity=PredictiveMultiplicity.BENJAMINI_HOCHBERG,
    family_discovery_rate=0.05,
)
```

The policy is recorded, and an extreme check emits `ppc.extreme-discrepancy` with its
discrepancy signature and group labels. The tail probability is a posterior-predictive
reference summary, not a classical uniformly distributed p-value. The threshold is a
screening convention—not an automatic accept/reject rule.

## Many checks at once

One call evaluates `groups × discrepancies` checks against the same threshold. Thirty
subjects and four discrepancies is one hundred and twenty simultaneous checks, so about six
of them fall below `0.05` *by construction* even when the model is perfect. One warning per
extreme check therefore teaches readers to ignore the warning.

`audit.family` makes the family explicit and is always present, flagged or not:

```python
family = audit.family
print(family.n_checks, family.n_extreme, family.expected_extreme)
print(family.excess_probability, family.adjusted_threshold, family.n_flagged)
```

`n_extreme` still counts checks below `tail_probability_warning` exactly as before, and
every per-check `tail_probability` remains unadjusted and fully retained. What changed is
which extreme checks become issues: only those surviving the declared `multiplicity`
adjustment at `family_discovery_rate`. A family of one check is never adjusted, so an
ungrouped single-discrepancy audit behaves exactly as it always did. When the observed rate
of extreme checks exceeds chance but no single check survives adjustment, the audit emits
one `ppc.extreme-discrepancy-rate` summary issue rather than a scatter of per-group
warnings. Set `multiplicity=PredictiveMultiplicity.NONE` to restore per-comparison
behaviour explicitly.

## Convergence gating

`posterior_predictive_check` audits the posterior's convergence and retains the audit on
`audit.convergence`. If the audit fails, the predictive audit's `status` is `FAIL` and it
carries `ppc.unconverged-posterior` at `ERROR` severity: replicated draws from chains that
never mixed are not draws from the posterior, so no tail probability below them is
interpretable. Nothing is discarded—every reference distribution is still there—so the
layer above decides. Pass `audit_policy=` a `PosteriorAuditPolicy` naming the severities
you are downgrading if you want the numbers anyway; the downgrade is recorded in
`to_dict()`.

## What a pass does not establish

A model can reproduce a few selected summaries and still be scientifically wrong. A PPC
also reuses the observations that informed the posterior. It therefore does not establish:

- more than the gate above about the sampler—the audit is retained, but a passing
  convergence audit is a precondition, not evidence of fit; see
  [posterior convergence diagnostics](posterior-diagnostics.md);
- calibration of the inference implementation—use simulation-based calibration;
- discriminability of candidate mechanisms—use [recovery](model-recovery.md); or
- prediction of genuinely later sessions—use [prospective validation](validation.md).

Use ArViZ's [generic PPC plots](https://python.arviz.org/en/stable/api/generated/arviz.plot_ppc.html)
through `posterior.to_arviz()` when distributional visualization is useful. Behavio's layer
adds behavioural discrepancy identity, labelled grouping, immutable evidence, and explicit
interpretation boundaries; it does not replace ArViZ's plotting ecosystem.
