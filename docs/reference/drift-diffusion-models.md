# Drift-diffusion model API

These models jointly score choice and response time. Unit declarations, the declared
support of any mixture component, and the prediction mode remain part of the fitted
contract.

## Static DDM

::: behavio.models.ddm
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Session-varying and partially pooled drift diffusion

There is no separate smooth or hierarchical drift-diffusion class. A drift-diffusion model
that varies across a clock, or that pools across animals, is the static model above passed
through [`smooth`][behavio.compose.smooth] and
[`hierarchical`][behavio.compose.hierarchical]:

```python
from behavio import WienerDriftDiffusion
from behavio.compose import hierarchical, smooth

paths = smooth(
    WienerDriftDiffusion(covariates=("stimulus",)),
    over="session_order",
    knots=(0.0, 2.0),
    parameters=("drift.stimulus", "boundary"),
)
pooled = hierarchical(paths, over="subject", parameters=("drift.stimulus", "boundary"))
```

Hierarchy is the outer combinator: it fits a joint coordinate whose width depends on how
many groups the study contains, so nothing outside it can expand that coordinate. See
[Composing models](../composing-models.md) for the contract a model satisfies to be
composable, and [Session-varying trajectories](../smooth-ddm.md) and
[Partially pooled trajectories](../hierarchical-smooth-ddm.md) for what the two mean
scientifically.

::: behavio.compose
    options:
      members_order: source
      show_root_heading: false
      show_source: false
