# `behavio.contracts` API

`behavio.contracts` is the single import address for every protocol a downstream package
implements, and for the dataclasses those protocols declare structurally. The surfaces
described in [Extend Behavio](../extensions.md) all resolve here.

Every name below is also re-exported at the friendly implementation-side home an author is
already reading — `from behavio.models.base import BehaviourEstimator` resolves to the same
object. A contract is declared once; it is surfaced where it is implemented.

`behavio.contracts` is a runtime leaf: it imports only `behavio.trials` and
`behavio.posterior.result`, neither of which imports anything from inside the package at all.
Implementation modules (`behavio.models.base`, `behavio.inference.optimize`,
`behavio.inference.parameters`, `behavio.time.transforms`, `behavio.evaluate.splits`,
`behavio.posterior.predictive`, `behavio.diagnostics`) depend on it in one direction only, and
`tests/test_contracts.py` fails if that direction ever reverses.

## Estimators, fits, and predictions

`BehaviourEstimator` requires `required_task_columns`. It used to live on a separate
`TaskColumnEstimator` protocol, which has been removed: every estimator can say which
columns it reads but does not score, and one that reads nothing beyond its scored column
answers `()`. `model_capabilities(...).required_task_columns` surfaces the validated
answer, and `behavio.task.TaskSpec.validate_model` checks that each named column carries a
declared task role.

`ModelPrediction` is `Prediction | CategoricalPrediction | DensityPrediction`. The third is
the shape a response-time, confidence or race model produces: a density on an explicit
outcome grid, optionally *defective* across named categories. It is a full member of the
union, not a side channel — `evaluate_splits` slices it to a fold's scored rows and retains
each row's observed category beside it, and `compare_models` scores it. The log score is
joint over the whole observation; the Brier score, being a scoring rule for a probability,
reads the density's discrete margin only, and refuses outright for a density that has none.

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
four-valued and both `FitDiagnostics.convergence` and `FitAudit.convergence` return it,
`FitAudit.to_dict()` carries it under `"convergence"`, and `audit_fit` raises
`optimizer_nonconvergence` only for `NOT_CONVERGED`. Consumers deciding whether a run is
usable should ask `FitDiagnostics.failed_to_converge` rather than `not converged`; the
latter answers the question wrongly for a closed-form fit, whose flag is absent rather
than false.

The fourth value is `UNREPORTED`: **the procedure searched and said nothing**. That is the
normal state of a wrapper around a third-party fitter whose stopping rule is private —
PyDDM 0.9 discards SciPy's `success` and never populates its `message` — and all three of
the other values misdescribe it. `CONVERGED` claims a success nobody measured,
`NOT_CONVERGED` makes the audit `FAIL` and evicts the candidate from every comparison, and
`INAPPLICABLE` asserts that no search happened. `audit_fit` reports `UNREPORTED` as the
warning `optimizer_convergence_unreported`, so the candidate stays eligible while the gap
stays on the record; `failed_to_converge` is `False`, because absence of evidence is not
evidence of failure. Write it as `converged=ConvergenceStatus.UNREPORTED` with
`status=None`; the other three states keep their `True` / `False` / `None` spellings and
passing any other enum member is refused.

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
