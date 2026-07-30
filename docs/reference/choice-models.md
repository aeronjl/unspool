# Observable choice-model API

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

## Bernoulli history GLM

::: behavio.models.glm
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Combinators: smoothness and hierarchy

The smooth, hierarchical, and hierarchical-smooth GLMs are not classes, and neither are
their multinomial counterparts. They are
[`smooth()` and `hierarchical()`](../composing-models.md) applied to the model above or to
`MultinomialLogit`, and the contract a family implements to be composable is
`behavio.contracts.compose.PenalisedLinearEstimator`.

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

## Multinomial choice

::: behavio.models.multinomial
    options:
      members_order: source
      show_root_heading: false
      show_source: false
