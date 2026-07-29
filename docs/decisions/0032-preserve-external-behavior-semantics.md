# SDR-0032: Preserve external behavior semantics at typed boundaries

- Status: **Accepted**
- Date: 2026-07-27

!!! note "Moved from fipha"
    This record was made in the photometry package `fipha` and moved here with
    the code it governs when the general behaviour surface moved to Behavio.
    It keeps its original SDR number so that reports which
    cite it stay resolvable. Its module references have been updated; its
    decision has not been changed.

## Context

Neuroscience experiments commonly combine a recorded signal with markerless pose,
automatically discovered behavioral states, or human ethograms. DeepLabCut and
SLEAP produce coordinates with confidence and identity structure. Keypoint-MoSeq
produces frame-level latent-state labels. BORIS distinguishes point events from
positive-duration state annotations.

A single generic event table would make integration easy only by discarding scorer
identity, track identity, confidence, interval duration, clock evidence or units.
Implementing pose or state discovery inside Behavio would instead duplicate
the source tools and blur responsibility for their validation.

## Decision

Behavio will consume external behaviour through separate typed pose,
continuous-covariate, point-event and interval boundaries. Native adapters will be
thin and dependency-light. They must require ambiguous identities and array axes to
be declared, retain confidence-derived missingness, name clocks and units, and
refuse implicit cross-clock alignment.

Projecting an interval to its onset or offset is an explicit analysis operation; it
does not replace the source interval. Time-normalized progress supplements rather
than replaces physical start, stop and duration. Behavioural discovery remains in
the source tool. Longitudinal behavioural modelling belongs to Behavio's `Study`
contract; `fipha`'s side of that handoff is
[SDR-0030](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0030-delegate-behavioral-trajectories-to-behavio.md).

## Alternatives considered

- **Accept arbitrary dataframes directly in the event-kernel API:** rejected
  because row grain, units, clocks and missingness would remain implicit.
- **Convert every source to point events:** rejected because it discards state
  duration and makes onset, offset and progress analyses indistinguishable.
- **Add pose estimation and behavioral clustering here:** rejected because those
  are independently validated scientific methods with mature packages.
- **Depend directly on all source packages:** rejected because a modelling-only
  installation should not inherit several large machine-learning stacks.

## Consequences

- Scientists can compose familiar behaviour tools with their recordings without
  changing the source tools or installing them as Behavio dependencies.
- The initial API is experimental. Official SLEAP and BORIS files plus
  writer-contract DeepLabCut 3.0.0 and Keypoint-MoSeq 0.6.8 files now provide a
  complete one-version parity matrix without conflating fixture provenance.
- Multi-animal identity audits and real acquisition-specific clock validation remain
  visible product gaps; later records close the initial validity-mask, duration, and
  `ndx-pose` implementation gaps.
- A boundary may be extracted into a shared ecosystem package only after it has an
  independent object model and at least two real consumers.

## Revisit trigger

Revisit after real-file fixtures cover two versions of each source format, after
`fipha` or another peer package needs more of this behaviour boundary, or if a community
standard supersedes these types.

## Evidence added later

[SDR-0038](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0038-model-variable-duration-behavior-with-physical-intervals-and-progress.md)
closed the first duration/progress-kernel gap with an aligned interval bundle and a
full-denominator normalized-progress design. Merge/split/filter and overlap policies
were subsequently closed by
[SDR-0049](0049-make-interval-policy-order-explicit-and-auditable.md).
[SDR-0050](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0050-preserve-ndx-pose-values-and-declare-link-omissions.md) adds native
loss-aware `ndx-pose` 0.3 inspection and NWB round trips.
