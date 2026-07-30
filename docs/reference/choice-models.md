# `behavio.models` and `behavio.compose`: observable choice API

These models express structure directly in observed choices and predictors. They are the
first matched alternatives for more elaborate latent or mechanistic accounts.

## Shared estimator contract

::: behavio.models.base
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Canonical baselines

::: behavio.models.baselines
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Psychometric functions

::: behavio.models.psychometric
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Signal detection theory

::: behavio.models.sdt
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Economic and value-based choice

::: behavio.models.economic
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Bernoulli history GLM

::: behavio.models.glm
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Combinators: smoothness, hierarchy and mixture

The smooth, hierarchical, and hierarchical-smooth GLMs are not classes, and neither are
their multinomial counterparts, and neither is a lapse model. They are
[`smooth()`, `hierarchical()` and `mix()`](../composing-models.md) applied to the model
above or to `MultinomialLogit`. The contract a family implements to be composable is
`behavio.contracts.compose.PenalisedLinearEstimator`; the contract a *simpler process*
implements to be mixable is `behavio.contracts.mixture.MixtureComponent`.

::: behavio.contracts.compose
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.smoothness
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.hierarchy
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.formula
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.trajectory
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.contracts.mixture
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.mixture
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.compose.components
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Multinomial choice

::: behavio.models.multinomial
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Scalar timing and patch leaving

Neither of these families scores a choice, so neither belongs on a page called *observable
choice*; they are here because the reference nav has no timing-and-foraging page yet. Both
compose through `behavio.contracts.bounded.BoundedCoordinateEstimator` alongside the economic
families above, and both estimate nothing but logarithms of positive quantities. The
identifiability arguments behind them are in
[SDR-0060](../decisions/0060-bisect-time-by-the-ratio-rule.md) and
[SDR-0061](../decisions/0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md).

::: behavio.models.scalar_timing
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.models.patch_leaving
    options:
      members_order: source
      show_root_heading: false
      show_source: false
