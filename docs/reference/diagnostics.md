# `behavio.diagnostics` and `behavio.posterior` checks API

These APIs answer different credibility questions. Numerical audits, posterior checks,
simulation calibration, sensitivity, and repeatability should not be collapsed into one
generic pass/fail label.

## Fit audits

::: behavio.diagnostics
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Posterior diagnostics and predictive checks

::: behavio.posterior.diagnostics
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.posterior.predictive
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Simulation-based calibration

::: behavio.posterior.simulation_based_calibration
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Sensitivity and reliability

::: behavio.posterior.sensitivity
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.posterior.reliability
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## PSIS-LOO and posterior model comparison

::: behavio.posterior.loo
    options:
      members_order: source
      show_root_heading: false
      show_source: false

::: behavio.posterior.comparison
    options:
      members_order: source
      show_root_heading: false
      show_source: false
