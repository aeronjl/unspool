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

::: behavio.contracts.estimator
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Fit evidence and audit vocabulary

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
