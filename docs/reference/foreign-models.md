# `behavio.foreign` API

Behavio estimators backed by third-party model implementations. Every wrapper here is a
real [`BehaviourEstimator`](contracts.md) — or, for a package that fits by sampling, a real
[`PosteriorBehaviourEstimator`](contracts.md) — and keeps its dependency behind its own
extra; `import behavio` never requires one. Read
[wrapped models](../foreign-models.md) first for the compatibility and licence matrix, the
Python floor, the jax conflict, and the places PyDDM and Bambi strain the contract.

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

## Bambi regression

::: behavio.foreign.bambi
    options:
      members_order: source
      show_root_heading: false
      show_source: false
