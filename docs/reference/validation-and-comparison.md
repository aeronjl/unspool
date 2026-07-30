# `behavio.evaluate` and `behavio.compare` API

Splitters define the deployment boundary. Evaluation fits within each boundary;
comparison preserves matched scores and declared aggregation units.

## Splitters

::: behavio.evaluate.splits
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Fold evaluation

::: behavio.evaluate.folds
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Model comparison and nested selection

::: behavio.compare.models
    options:
      members_order: source
      show_root_heading: false
      show_source: false

### The shared simultaneous-inference records

`ComparisonMultiplicity` and `ComparisonFamily` are imported into `behavio.compare.models` and
into the top-level `behavio` namespace under exactly these names. They are defined beside
the step-up arithmetic itself so that `behavio.protocol.schema` — which freezes the adjustment in
`ComparisonSpec` and imports nothing else from the package — and
`behavio.posterior.comparison` — which sizes an ELPD family without the estimator stack —
reach the same two types rather than growing copies of them.

::: behavio.compare.models.ComparisonMultiplicity
    options:
      show_root_heading: true
      show_source: false

::: behavio.compare.models.ComparisonFamily
    options:
      members_order: source
      show_root_heading: true
      show_source: false

## Parameter-trajectory shapes

::: behavio.compare.parameter_trajectories
    options:
      members_order: source
      show_root_heading: false
      show_source: false
