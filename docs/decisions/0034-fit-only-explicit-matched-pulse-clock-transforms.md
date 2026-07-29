# SDR-0034: Fit only explicit matched-pulse clock transforms

- Status: Accepted
- Date: 2026-07-27
- Decision owners: project maintainers
- Related protocol/report: [clock synchronisation contract](../clock-synchronization.md)

!!! note "Moved from fipha"
    This record was made in the photometry package `fipha` and moved here with
    the code it governs when the general behaviour surface moved to Behavio.
    It keeps its original SDR number so that reports which
    cite it stay resolvable. Its module references have been updated; its
    decision has not been changed.

## Context

Pose, behavioural annotations and recorded signals frequently originate from different
devices. Equal units or similar sampling rates do not establish a shared clock.
Direct interpolation can conceal an offset, stable rate drift, nonlinear timing
error or mistaken pulse correspondence.

SDR-0004 already requires explicit timestamps for derived-data alignment. The
behavioral interoperability boundary made `clock_id` explicit and refused
cross-clock interpolation, but offered no auditable path to establish a shared
clock. The immediate evidence available is a sequence of acquisition pulses that
an experimenter has identified on both clocks.

## Decision

Behavio accepts only explicit, ordered, one-to-one pulse pairs for its
first clock-synchronization boundary. It does not automatically match pulses.

The v0.1 model is affine: target time equals intercept plus scale times source
time. A specification prospectively declares maximum absolute pulse residual,
maximum drift magnitude, minimum matched-pulse count and minimum source-clock
span. At least three pairs are required. The fit fails rather than returning an
inadmissible transform when any threshold is exceeded.

The artifact retains every pulse pair, fitted timestamp and residual; offset,
scale and drift; RMS, median and maximum residual error; all thresholds; and a
stable content-derived synchronization ID. Transforming pose, covariates or
annotations appends that ID to their provenance lineage.

Transformation refuses timestamps outside the matched-pulse domain unless the
caller explicitly declares a bounded extrapolation allowance. Interpolation onto
the target sampling grid remains a later, separate operation.

## Consequences

The package now offers a valid path from distinct behaviour and acquisition clocks
to a shared time coordinate without treating a name change as synchronization.
Failures are visible and machine-readable evidence can be archived beside model
results.

The affine model will reject some real acquisitions with nonlinear drift or pulse
dropout. Threshold selection remains study-specific. The package cannot determine
whether two rows are the same physical pulse; that acquisition-level evidence
remains the scientist's responsibility.

## Alternatives considered

- **Rename both clock IDs:** rejected because it records no timing evidence.
- **Interpolate directly between clocks:** rejected because interpolation assumes
  the correspondence that synchronization must establish.
- **Automatically pair nearest pulses:** rejected because offsets, missing pulses
  and drift can make nearest-neighbor matching confidently wrong.
- **Fit a higher-order spline by default:** rejected because it can absorb bad
  matches, extrapolates poorly and lacks current validation evidence.
- **Allow unlimited extrapolation after a passing fit:** rejected because an
  in-domain residual does not validate the mapping outside the pulse span.

## Revisit trigger

Add robust pulse matching or piecewise/nonlinear transforms only after frozen
simulations and real acquisition fixtures cover missing, duplicated and jittered
pulses; each method must expose held-out timing error and an explicit complexity
selection policy. Revisit default minimum evidence after multiple acquisition
systems have been validated.

## Evidence added later

None.
