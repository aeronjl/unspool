# Observed behaviour and behaviour-tool interoperability

Behavio does not rediscover behaviour. It needs a loss-aware boundary to the
tools that already estimate pose, discover behavioural states, or record an
ethogram, so that what those tools observed can be modelled, related to a
neural signal, or carried across sessions without silently losing confidence,
missingness, or clock identity.

That boundary is four small modules, one per thing being carried:

| Module | Holds | Page |
|---|---|---|
| `behavio.pose` | `PoseTrajectory`, and the DeepLabCut, SLEAP and `movement` readers | [Pose trajectories](pose.md) |
| `behavio.ethograms` | `BehaviorInterval`, `BehaviorAnnotations`, and the Keypoint-MoSeq and BORIS readers | [Ethograms](ethograms.md) |
| `behavio.covariates` | `BehaviorCovariate` | [Behavioural covariates](covariates.md) |
| `behavio.sync` | `ClockSynchronization` and `fit_clock_synchronization()` | [Clock synchronisation](clock-synchronization.md) |

Readers live with the type they produce rather than in a module of their own,
because a reader is only ever a thin translation into one of these types.

!!! info "Experimental v0.1 boundary"
    Typed in-memory and file adapters are implemented. Checksum-pinned official
    SLEAP and BORIS fixtures pass; DeepLabCut and Keypoint-MoSeq currently have
    documented-schema fixtures only. See the
    [validation matrix](evidence/interoperability-validation-v0.1.md).

!!! note "Moved from fipha"
    These types, readers, and the [interval policy](interval-policy.md) were
    first written inside the photometry package `fipha`, because that is where
    they were first needed. They are not photometry concepts, so they now live
    here and `fipha` depends on Behavio for them. The fixtures, guarantees, and
    decision records moved unchanged.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/behavior-ecosystem-v0.1.svg" alt="DeepLabCut and SLEAP pose trajectories, Keypoint-MoSeq state bouts, and BORIS annotations enter typed pose, covariate, point-event, and interval boundaries. A neural package relates them to recorded signals and exports trial summaries to Behavio for longitudinal modelling.">
  <figcaption><strong>Each package keeps the job it is good at.</strong> Pose estimators retain keypoints and confidence, behaviour tools retain point-versus-state semantics, Behavio owns the shared behavioural types and longitudinal models, and a recording package owns signal identity and neural alignment.</figcaption>
</figure>

## What each package owns

| Package or tool | Owns | Behavio consumes |
|---|---|---|
| DeepLabCut | markerless pose inference, scorer and individual identity | keypoint coordinates and likelihood by video frame |
| SLEAP | single- and multi-animal pose inference and tracking | node coordinates, track identity and point scores |
| movement | pose input/output across tools, kinematics, regions of interest | its `poses` dataset, via `pose_from_movement()` |
| Keypoint-MoSeq | unsupervised behavioural state discovery | frame-level syllable labels, run-length encoded as bouts |
| BORIS | human ethogram annotation | distinct point events and state intervals |
| Behavio | the shared behavioural types below, interval policies, clocks, models and prospective validation | typed pose, covariates, point events and intervals |
| fipha | optical signal identity, QC, preprocessing, neural alignment and inference | Behavio's types, through `fipha[behavior]` |

This division follows the scientific capabilities of the source tools.
DeepLabCut exports MultiIndex pose columns containing scorer, body part,
coordinates and likelihood. SLEAP Analysis HDF5 retains tracks, nodes and
confidence-like scores. Keypoint-MoSeq returns a syllable label per time point.
BORIS explicitly distinguishes point from state events. Flattening those outputs
to an anonymous `time,value` CSV would discard information that affects
analysis.

## Four exchange shapes

The public Python boundary deliberately uses a few small types rather than one
universal table.

| Shape | Required meaning | Current type | Typical source |
|---|---|---|---|
| Pose trajectory | one keypoint, one tracked individual, coordinates, confidence and time | `PoseTrajectory` | DeepLabCut, SLEAP |
| Continuous covariate | one named value and validity mask per source timestamp | `BehaviorCovariate` | confidence-gated speed, pupil area, state probability |
| Point events | named instantaneous occurrences | `BehaviorAnnotations.point_events` | BORIS POINT, cue timestamps |
| Intervals | named half-closed behavioural bouts with physical start and stop | `BehaviorInterval` | BORIS STATE, MoSeq syllable run |

## Clock, identity, confidence and units

Interoperability is valid only when these fields remain explicit:

- `subject` and `session` identify the biological and recording units;
- `individual` distinguishes tracks in multi-animal pose output;
- `clock_id` names the time coordinate used by every timestamp;
- `coordinate_unit` and covariate `unit` prevent pixels from masquerading as
  centimetres;
- confidence-like scores or likelihood determine a visible validity mask rather
  than an invisible fill operation;
- `source_version` and `source_artifact` can retain the producing software
  version and path, URI or digest-bearing artifact identity.

Every type in this boundary carries them, so a value never arrives without the
clock it was measured on.

!!! note "Not the longitudinal clocks"
    `clock_id` here names a hardware time coordinate in seconds. It is
    unrelated to the [longitudinal clocks](clocks-and-transforms.md) that place
    a `Study` in learning time. A session has both.

## Installing

The BORIS readers need only the standard library. Pose and Keypoint-MoSeq file
readers need HDF5 and pandas, which is what the `readers` extra installs:

```bash
pip install "behavio[readers]"
```

## From observed behaviour to models

The composition is:

1. transform [pose](pose.md) into a declared
   [covariate](covariates.md), such as confidence-gated speed;
2. [synchronise](clock-synchronization.md) it to the target clock from explicit
   matched pulses and retain the synchronisation artifact;
3. apply an [auditable interval policy](interval-policy.md) to the discovered
   [bouts](ethograms.md), so merging, filtering and contextualisation are
   ordered and recorded;
4. align it without crossing invalid spans and retain the resulting mask; then
5. either model it here, or hand the point events, interval edges, covariate
   values and masks to a recording package.

See the [worked interoperability tutorial](tutorials/behavior-tool-interoperability.md).
For photometry specifically, `fipha` consumes `interval_encoding_inputs()` and
`normalized_progress()` directly in its event-kernel models; the return of a
trial-level neural summary to a Behavio `Study` is documented in its
[Behavio interoperability contract](https://aeronjl.github.io/fipha/behavio-interoperability/).

## Gaps exposed by the examples

| Priority | Missing ecosystem capability | Likely home |
|---|---|---|
| P0 | extend the now-complete one-version fixture matrix across a second released version and one real camera-to-recording synchronisation record | adapter validation here |
| P1 | broaden interval-policy fixtures to real multi-label annotations and a second source-tool version | ethogram validation here |
| P1 | multi-animal identity-switch diagnostics at the alignment boundary | source-tool QC plus consumer preflight |
| P1 | versioned behavioural interchange artifact with hashes and confidence semantics | here, once a second external consumer adopts it |
| P1 | bounded remote `ndx-pose` series access and a real camera-to-photometry synchronisation fixture | `fipha` I/O and validation |
| P2 | adapters for SimBA, B-SOiD and user-supplied state probabilities | thin adapters here |

A gap should become a separate library only when it has an independent object
model, at least two consuming packages, and useful validation outside a single
recording modality. That test is what moved this boundary out of `fipha`:
pose, ethograms and interval policies had an independent object model and a
second consumer. Recording-specific kernels and optical QC remain in `fipha`.

## Sources

- [DeepLabCut output contract](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md) and [Mathis et al. (2018)](https://doi.org/10.1038/s41593-018-0209-y)
- [SLEAP export documentation](https://docs.sleap.ai/latest/tutorial/exporting-the-results/) and [Pereira et al. (2022)](https://doi.org/10.1038/s41592-022-01426-1)
- [Keypoint-MoSeq I/O contract](https://keypoint-moseq.readthedocs.io/en/latest/io.html) and [Weinreb et al. (2024)](https://doi.org/10.1038/s41592-024-02318-2)
- [BORIS aggregated-event export](https://www.boris.unito.it/user_guide/export_events/) and [Friard & Gamba (2016)](https://doi.org/10.1111/2041-210X.12584)
