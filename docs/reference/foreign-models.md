# `behavio.foreign` API

Behavio estimators backed by third-party model implementations. Every wrapper here is a
real [`BehaviourEstimator`](contracts.md) and keeps its dependency behind its own extra;
`import behavio` never requires one. Read
[wrapped models](../foreign-models.md) first for the compatibility and licence matrix, the
jax conflict, and the places PyDDM strains the contract.

## The package

::: behavio.foreign
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## PyDDM drift diffusion

::: behavio.foreign.pyddm
    options:
      members_order: source
      show_root_heading: false
      show_source: false
