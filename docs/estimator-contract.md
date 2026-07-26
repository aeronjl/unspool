# Estimators, generators, and scored observations

Unspool separates three questions that are often collapsed into one model API:

1. Can this object be fitted and scored prospectively?
2. Can it simulate a study from named parameters?
3. Which observed variables does each pointwise likelihood actually score?

That separation lets a third-party estimator enter comparison before it has a simulator,
without letting it enter simulation-based recovery prematurely. It also establishes the
comparison boundary needed for future response-time models.

## The two structural protocols

`BehaviourEstimator` is the forecasting contract. It exposes:

- stable `model_name` and configuration-specific `signature` values;
- the complete tuple of `scored_columns` used by its likelihood;
- its supported filtered or smoothed prediction modes;
- `fit`, `predict`, and `pointwise_log_prob` methods.

`GenerativeBehaviourModel` extends that contract with stable `parameter_names` and
`simulate`. `BehaviourModel` remains the backwards-compatible public name for the full
generative contract.

The protocols are structural: a plugin does not need to inherit an Unspool base class.
`model_capabilities()` performs semantic validation in addition to checking method
presence:

```python
from unspool import model_capabilities

capabilities = model_capabilities(plugin)
print(capabilities.scored_columns)
print(capabilities.can_simulate)
print(capabilities.can_recover_parameters)
```

Evaluation and comparison accept any valid `BehaviourEstimator`. Parameter and model
recovery require a `GenerativeBehaviourModel` for every scenario generator.

## Why scored columns are part of the contract

Pointwise log probabilities are comparable only when they are densities or probabilities
for the same observed event. All current models declare:

```python
scored_columns = ("choice",)
```

A future joint drift-diffusion model should instead declare, for example:

```python
scored_columns = ("choice", "response_time")
```

Unspool will reject a direct likelihood ranking between those two models. A choice-only
probability and a joint choice/response-time density answer different predictive questions;
their numerical log scores are not interchangeable. Joint models must compete with other
models scoring the same joint observation, or expose a separately configured choice-only
marginal estimator.

`Prediction.probability` remains the probability of the declared binary `outcome_column`
and supports Brier scoring. `pointwise_log_prob` may score a larger declared observation.
Prospective comparison records both `outcome_column` and `scored_columns`, preserving the
distinction in serialized reports.

## Fit-result invariants

At every fold, Unspool verifies that:

- `fit()` returned a `FitResult`;
- its model name and signature match the estimator that produced it;
- its observation count equals the training-study length;
- `predict()` returned one prediction per requested row;
- `pointwise_log_prob()` returned one finite score per requested row.

These checks are particularly important for plugins: structural typing alone can establish
that methods exist, but not that their outputs refer to the model and data supplied.

## Recovery eligibility

`run_parameter_recovery` validates every truth mapping before simulation, retains the
complete `FitAudit` for every run, and uses audit status rather than optimizer convergence
alone when constructing summaries. Warning fits remain eligible. Failed fits remain in the
report but do not contribute to bias, RMSE, correlation, or coverage.

Coverage has its own denominator, `n_with_uncertainty`, because an otherwise usable
estimate may lack finite local uncertainty. `ParameterRecoveryReport.to_dict()` preserves
truth, estimates, standard errors, seeds, optimizer messages, audits, and both denominators
without emitting non-standard JSON `NaN` values.

## Reaction-time extension boundary

This contract makes reaction-time models safe to add; it does not yet provide them. The
first such family should add:

- an explicit positive response-time column and unit metadata in the study adapter;
- a joint choice/response-time likelihood with numerically stable tail handling;
- simulation and parameter recovery over identifiable and boundary-near regimes;
- prospective comparisons only against candidates scoring the same joint observation;
- diagnostics for contaminant responses, non-decision-time boundaries, and optimizer
  failures.

That benchmark should precede interpretive claims about latent decision parameters.
