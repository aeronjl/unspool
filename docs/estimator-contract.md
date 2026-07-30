# Estimators, generators, and scored observations

Behavio separates three questions that are often collapsed into one model API:

1. Can this object be fitted and scored prospectively?
2. Can it simulate a study from named parameters?
3. Which observed variables does each pointwise likelihood actually score?

That separation lets a third-party estimator enter comparison before it has a simulator,
without letting it enter simulation-based recovery prematurely. It also establishes the
comparison boundary used by joint response-time models.

## The two structural protocols

`BehaviourEstimator` is the forecasting contract. It exposes:

- stable `model_name` and configuration-specific `signature` values;
- the complete tuple of `scored_columns` used by its likelihood;
- `required_task_columns`, the predictive context it reads but does not score;
- its supported filtered or smoothed prediction modes;
- `fit`, `predict`, and `pointwise_log_prob` methods.

`required_task_columns` is mandatory and may be `()`. It was briefly carried by a separate
`TaskColumnEstimator` protocol so that models which had not yet declared it were not
evicted from the base contract; that protocol is gone. A model whose likelihood reads
nothing but the column it scores declares the empty tuple, which is an answer.

`GenerativeBehaviourModel` extends that contract with stable `parameter_names` and
`simulate`. `BehaviourModel` remains the backwards-compatible public name for the full
generative contract.

Models can additionally expose the structural `ParameterSpaceProvider` protocol. Its
[`ParameterSpace`](parameter-spaces.md) makes natural and optimizer coordinates, bounds,
fixed values, and priors available to fitting adapters without adding an inheritance
requirement or making parameter metadata a condition for basic prospective scoring.

The protocols are structural: a plugin does not need to inherit an Behavio base class.
`model_capabilities()` performs semantic validation in addition to checking method
presence:

```python
from behavio.models import model_capabilities

capabilities = model_capabilities(plugin)
print(capabilities.scored_columns)
print(capabilities.can_simulate)
print(capabilities.can_recover_parameters)
```

Evaluation and comparison accept any valid `BehaviourEstimator`. Parameter and model
recovery require a `GenerativeBehaviourModel` for every scenario generator.

## Why scored columns are part of the contract

Pointwise log probabilities are comparable only when they are densities or probabilities
for the same observed event. Choice-only models declare:

```python
scored_columns = ("choice",)
```

`WienerDriftDiffusion` instead declares, by default:

```python
scored_columns = ("choice", "response_time")
```

Behavio will reject a direct likelihood ranking between those two models. A choice-only
probability and a joint choice/response-time density answer different predictive questions;
their numerical log scores are not interchangeable. Joint models must compete with other
models scoring the same joint observation, or expose a separately configured choice-only
marginal estimator.

A model scoring a joint observation may name its prediction columns with **composite
categories**: each entry of `CategoricalPrediction.categories` is a tuple, and
`category_factors` names the tuple positions. `MetaSDT` scores the joint
`(response, confidence)` cell and labels its columns `(0, 3)` rather than `"no-3"`, so
`prediction.marginal("response")` sums out confidence exactly instead of a caller
splitting a string. The factorisation is required whenever the categories are tuples: an
unlabelled tuple is no more readable than the string it replaces.

`Prediction.probability` is a vector for a declared binary `outcome_column`.
`CategoricalPrediction.probability` is a trial-by-category matrix on an explicit stable
coordinate. Categorical estimators additionally expose `categories` and
`outcome_codes(study)`, so evaluation never guesses how source labels map onto probability
columns. Both forms support Brier scoring; `pointwise_log_prob` may score a larger declared
observation. Prospective comparison records both `outcome_column` and `scored_columns`,
preserving the distinction in serialized reports.

## Fit-result invariants

At every fold, Behavio verifies that:

- `fit()` returned a `FitResult`;
- its model name and signature match the estimator that produced it;
- its observation count equals the training-study length;
- `predict()` returned one prediction per requested row;
- `pointwise_log_prob()` returned one finite score per requested row.

These checks are particularly important for plugins: structural typing alone can establish
that methods exist, but not that their outputs refer to the model and data supplied.

A `FitResult` carries the estimates, their uncertainty, the numerical diagnostics, and the
scalar `derived` quantities a scientist publishes. A model-specific subclass is justified
only by evidence that *fitting itself produced* and that nothing short of refitting
recomputes — the retained restarts of a [`MultistartFit`](diagnostics.md), for instance.
It is not justified by wanting typed names for entries of `derived`, nor by a structured
record that is a deterministic function of the study and the model's own configuration:
that belongs on the model, as a method. `EqualVarianceSDT.summarize(study)` and
`MetaSDT.type1_summary(study)` are where the detection-theory summaries live for exactly
that reason.

Third-party packages can bind conforming factories to explicit names with
`EstimatorRegistry`, which is the allowlist a frozen protocol's `implementation` string is
resolved through. Registries are local to the workflow, reject duplicate or drifting model
identities, and expose a non-executable provider/version manifest. The corresponding
[`FitArtifact`](fit-artifacts.md) is the portable common result; it does not pickle model
objects or erase richer model-specific live results.

## Recovery eligibility

`run_parameter_recovery` validates every truth mapping before simulation, retains the
complete `FitAudit` for every run, and uses audit status rather than optimizer convergence
alone when constructing summaries. Warning fits remain eligible. Failed fits remain in the
report but do not contribute to bias, RMSE, correlation, or coverage.

Coverage has its own denominator, `n_with_uncertainty`, because an otherwise usable
estimate may lack finite local uncertainty. `ParameterRecoveryReport.to_dict()` preserves
truth, estimates, standard errors, seeds, optimizer messages, audits, and both denominators
without emitting non-standard JSON `NaN` values.

## Reaction-time implementation boundary

The first response-time family now supplies an explicit positive column and unit metadata,
a joint Wiener choice/response-time likelihood with finite tail handling, generative
simulation, deterministic restart diagnostics, and dedicated parameter-recovery
benchmarks. Mixing it with a uniform response process through `mix()` leaves the scored
observation unchanged, so the two can be compared directly. See the
[drift-diffusion guide](drift-diffusion.md).

It intentionally stops short of across-trial Wiener-parameter variation, hierarchical
pooling, or longitudinal parameter drift. Neither Wiener configuration can be meaningfully
ranked by joint log score against Behavio's choice-only families.
