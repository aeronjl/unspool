# Validation and comparison API

Splitters define the deployment boundary. Evaluation fits within each boundary;
comparison preserves matched scores and declared aggregation units.

## Splitters

::: behavio.validation
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Fold evaluation

::: behavio.evaluation
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Model comparison and nested selection

::: behavio.comparison
    options:
      members_order: source
      show_root_heading: false
      show_source: false

### The shared simultaneous-inference records

`ComparisonMultiplicity` and `ComparisonFamily` are imported into `behavio.comparison` and
into the top-level `behavio` namespace under exactly these names. They are defined beside
the step-up arithmetic itself so that `behavio.protocol` — which freezes the adjustment in
`ComparisonSpec` and imports nothing else from the package — and
`behavio.posterior_comparison` — which sizes an ELPD family without the estimator stack —
reach the same two types rather than growing copies of them.

::: behavio.comparison.ComparisonMultiplicity
    options:
      show_root_heading: true
      show_source: false

::: behavio.comparison.ComparisonFamily
    options:
      members_order: source
      show_root_heading: true
      show_source: false
