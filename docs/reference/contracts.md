# Extension contract API

`behavio.contracts` is the single import address for every protocol a downstream package
implements, and for the dataclasses those protocols declare structurally. The surfaces
described in [Extend Behavio](../extensions.md) all resolve here.

Every name below is also re-exported from the module it used to live in, so existing
imports such as `from behavio.models.base import BehaviourEstimator` continue to work
unchanged.

`behavio.contracts` is a runtime leaf: it imports only `behavio.study`, `behavio.clocks`
and `behavio.posterior`, none of which import it back. Implementation modules
(`behavio.models.base`, `behavio.inference`, `behavio.parameters`, `behavio.transforms`,
`behavio.validation`, `behavio.posterior_predictive`, `behavio.diagnostics`) depend on it
in one direction only.

## Estimators, fits, and predictions

`BehaviourEstimator` requires `required_task_columns`. It used to live on a separate
`TaskColumnEstimator` protocol, which has been removed: every estimator can say which
columns it reads but does not score, and one that reads nothing beyond its scored column
answers `()`. `model_capabilities(...).required_task_columns` surfaces the validated
answer, and `behavio.task.TaskSpec.validate_model` checks that each named column carries a
declared task role.

`CategoricalPrediction.categories` accepts **tuples** as well as scalars. A tuple category
names one cell of a joint observation — meta-d' scores `(response, confidence)` together —
and must be accompanied by `category_factors`, which names the tuple positions. Given
that, `marginal(factor)` sums out the other factors exactly, so a caller wanting one
margin of a jointly scored model never parses a label. Before tuples were admitted, such a
model had to flatten a cell into a string like `"no-3"` even though its `scored_columns`
already declared the observation joint.

::: behavio.contracts.estimator
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Fit evidence and audit vocabulary

`FitDiagnostics.converged` and `.status` are optional, alongside the five search-shaped
diagnostics that were already optional. A procedure that solves rather than searches — the
equal-variance z-transform, say — leaves them absent, because *it did not converge* and
*there was nothing to converge* are different claims and a boolean cannot hold both.

The distinction is reported rather than merely tolerated. `ConvergenceStatus` is
three-valued and both `FitDiagnostics.convergence` and `FitAudit.convergence` return it,
`FitAudit.to_dict()` carries it under `"convergence"`, and `audit_fit` raises
`optimizer_nonconvergence` only for `NOT_CONVERGED`. Consumers deciding whether a run is
usable should ask `FitDiagnostics.failed_to_converge` rather than `not converged`; the
latter answers the question wrongly for a closed-form fit, whose flag is absent rather
than false.

`optimizer` stays mandatory. Every fit can answer *what computed this* — for a searching
estimator the optimizer, for a closed-form one the solution method — so only the questions
a non-searching procedure genuinely cannot answer were widened.

::: behavio.contracts.audit
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## The natural parameterisation

The coordinate a model is estimated in is not in general the coordinate it is reported in.
This optional contract names the second one and gives the delta method what it needs to
carry uncertainty onto it. A model that declares nothing here behaves exactly as it did
before the contract existed.

::: behavio.contracts.natural
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Sampled estimators and the point-summary projection

::: behavio.contracts.posterior
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Optimization backends

::: behavio.contracts.backend
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Parameter semantics

::: behavio.contracts.parameters
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Predictive discrepancies

::: behavio.contracts.discrepancy
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Study transforms

::: behavio.contracts.transform
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Validation folds

::: behavio.contracts.fold
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Data-source adapters

The conformance harness that executes this contract lives beside the adapters, in
[`behavio.adapters.conformance`](data-adapters.md#adapter-conformance), so that
`behavio.contracts` stays a runtime leaf.

::: behavio.contracts.adapter
    options:
      members_order: source
      show_root_heading: false
      show_source: false
