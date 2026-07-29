# Observed behaviour API

Typed boundaries for pose, ethograms, and continuous behavioural covariates, the
readers that fill them from community tools, the matched-pulse clock transform
that moves them between devices, and the ordered interval policies that decide
which bouts enter an analysis.

Start from [Observed behaviour and behaviour-tool
interoperability](../observed-behaviour.md) for the contracts these signatures
implement, and from [Auditable interval and bout policies](../interval-policy.md)
for the policy algebra.

The file readers need the optional `readers` extra:

```bash
pip install "behavio[readers]"
```

## Pose and pose readers

See [Pose trajectories](../pose.md).

::: behavio.pose
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Ethograms and ethogram readers

See [Ethograms](../ethograms.md).

::: behavio.ethograms
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Behavioural covariates

See [Behavioural covariates](../covariates.md).

::: behavio.covariates
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Clock synchronisation

See [Clock synchronisation](../clock-synchronization.md). This is hardware time
in seconds, not the [longitudinal clocks](../clocks-and-transforms.md) in
`behavio.clocks`.

::: behavio.sync
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Interval policies

::: behavio.interval_policy
    options:
      members_order: source
      show_root_heading: false
      show_source: false
