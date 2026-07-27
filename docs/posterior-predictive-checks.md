# Posterior-predictive checks

Posterior-predictive checks ask whether data replicated from a fitted model reproduce
scientifically relevant features of the observations. Unspool supplies a small discrepancy
contract rather than a fixed dashboard: each check declares what feature it measures, the
relevant reference tail, and a stable signature retained with the result.

This follows the realized-discrepancy framework of [Gelman, Meng, and Stern
(1996)](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n41/j6n41.htm) and the iterative
model-criticism workflow described by [Gabry et al.
(2019)](https://doi.org/10.1111/rssa.12378).

## A minimal behavioural check

```python
from unspool import (
    CategoryRateDiscrepancy,
    MeanDiscrepancy,
    SwitchRateDiscrepancy,
    posterior_predictive_check,
)

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
from unspool import PosteriorPredictivePolicy

policy = PosteriorPredictivePolicy(
    interval_probability=0.9,
    tail_probability_warning=0.05,
)
```

The policy is recorded, and an extreme check emits `ppc.extreme-discrepancy` with its
discrepancy signature and group labels. The tail probability is a posterior-predictive
reference summary, not a classical uniformly distributed p-value. The threshold is a
screening convention—not an automatic accept/reject rule—and checking many discrepancies
after seeing the data creates multiplicity and researcher-degree-of-freedom concerns.

## What a pass does not establish

A model can reproduce a few selected summaries and still be scientifically wrong. A PPC
also reuses the observations that informed the posterior. It therefore does not establish:

- convergence of the sampler—use [posterior convergence diagnostics](posterior-diagnostics.md);
- calibration of the inference implementation—use simulation-based calibration;
- discriminability of candidate mechanisms—use [recovery](model-recovery.md); or
- prediction of genuinely later sessions—use [prospective validation](validation.md).

Use ArViZ's [generic PPC plots](https://python.arviz.org/en/stable/api/generated/arviz.plot_ppc.html)
through `posterior.to_arviz()` when distributional visualization is useful. Unspool's layer
adds behavioural discrepancy identity, labelled grouping, immutable evidence, and explicit
interpretation boundaries; it does not replace ArViZ's plotting ecosystem.
