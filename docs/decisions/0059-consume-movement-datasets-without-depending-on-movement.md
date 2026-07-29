# SDR-0059: Consume movement datasets without depending on movement

- **Status:** Accepted
- **Date:** 2026-07-28
- **Related decisions:** [SDR-0032](0032-preserve-external-behavior-semantics.md), [SDR-0033](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0033-retain-validity-masks-without-compressing-time.md), [SDR-0050](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0050-preserve-ndx-pose-values-and-declare-link-omissions.md)

!!! note "Moved from fipha"
    This record was made in the photometry package `fipha` and moved here with
    the code it governs when the general behaviour surface moved to Behavio.
    It keeps its original SDR number so that reports which
    cite it stay resolvable. Its module references have been updated; its
    decision has not been changed.

## Context

[`movement`](https://github.com/neuroinformatics-unit/movement) is the maintained
community package for animal pose input and output, built by the Neuroinformatics
Unit on the same substrate this package uses. It is BSD-3-Clause, matching this
package's licence, and healthy: 288 stars, releases roughly every six weeks, last
push on the day of this record. It provides `movement.io.load_poses.from_dlc_file`,
`from_sleap_file` and `from_nwb_file`, `movement.filtering.filter_by_confidence`,
and `movement.kinematics.compute_speed`.

This package's own documentation says it "does not need to rediscover behavior",
so a reviewer proposed deleting this package's pose readers and delegating to
`movement`. That proposal was evaluated by installing `movement` 0.17.0 on Python
3.12.13 and running it against this repository's checksum-pinned fixtures. Four
measurements decided it.

**The current SLEAP writer cannot be read.** `_ds_from_sleap_analysis_file` applies
a hardcoded `f["tracks"][:].transpose(3, 1, 2, 0)` and never reads the `dims`
attribute of the `tracks` dataset. That permutation is correct only for the legacy
layout. Against `sleap-io-0.9.2-standard.analysis.h5` — `tracks` of shape
`(1100, 27, 24, 2)` with `dims = ["frame", "track", "node", "xy"]` — it raises
`ValueError: Expected 'position_array' to have 2 or 3 spatial dimensions, but got
27`. The hardcoded transpose is still present on the upstream default branch. On
the legacy fixture `movement` agrees with this package but casts to `float32`,
returning `382.74652099609375` where this package returns `382.74652904163474`.

**The Python floor conflicts.** `movement` requires `>=3.12.0`. This package
declares `requires-python = ">=3.11"` and classifies 3.11. Adding `movement` to the
`behavior` extra would make that extra uninstallable on a supported interpreter.

**The dependency footprint is the opposite of the stated win.** Resolving
`movement` 0.17.0 alone produces **119 packages and 720 MB**, including Qt
(`qt-niu`, `qtpy`, `superqt`), `opencv-python-headless` via `napari-video`,
`skia-python`, `imageio-ffmpeg`, `aiohttp`, `numba`/`llvmlite`, `cartopy`/`pyproj`,
and `matplotlib`/`seaborn` via `xarray[viz]`. It pins `netCDF4<1.7.3`, imposing an
upper bound on any environment installing the extra. Critically, `tables>=3.10.1`
is a **core** dependency of `movement`, so the hoped-for removal of PyTables from
the `behavior` extra is not available: PyTables returns transitively, and
`movement`'s own DeepLabCut reader calls `pd.read_hdf` with a hardcoded
`key="df_with_missing"`.

**Speed semantics differ, and not in this package's favour.** `compute_speed` is
the norm of `xarray.DataArray.differentiate`, a second-order central difference.
This package uses a backward pairwise difference whose step is invalid unless both
endpoints pass the confidence threshold. On `x = [0, 1, 2, 30, 31]` with
`confidence = [0.99, 0.99, 0.2, 0.99, 0.99]` and a 0.9 threshold:

| index | `compute_speed(filter_by_confidence(...))` | `PoseTrajectory.speed()` value | valid |
|---|---|---|---|
| 0 | 1.0 | `NaN` | no |
| 1 | `NaN` | 1.0 | yes |
| 2 | **14.5** | 1.0 | no |
| 3 | `NaN` | 28.0 | no |
| 4 | 1.0 | 1.0 | yes |

The central-difference stencil straddles the gated sample's `NaN`, so `movement`
reports a speed **at** the confidence-gated frame while blanking its two
well-estimated neighbours. It also has no validity concept: gating destroys the
value rather than marking it, which SDR-0033 forbids.

Finally, `movement` has no counterpart for most of what this package's behavior
layer does. It has no clock, synchronisation or foreign-clock alignment API — only
`filtering.interpolate_over_time`, which fills gaps along a pose's own time axis —
and no ethogram or annotation types at all. `PoseTrajectory` is also constructed
and consumed by `io/ndx_pose.py`, so the type survives regardless.

## Decision

Do not depend on `movement`. Keep this package's pose readers, covariate type,
clock-alignment layer and annotation types first-party.

Instead, consume `movement`'s output. Add `pose_from_movement()` to
`behavio/pose.py`, which accepts a `movement` poses `Dataset` and returns a
`PoseTrajectory`. It duck-types on the xarray interface — already a core
dependency — and never imports `movement`, so it costs no new dependency, no
Python floor change and no `behavior` extra change. Its rules:

- Accept both the singular and plural keypoint/individual dimension spellings used
  across `movement` releases.
- Require explicit `keypoint` and `individual` whenever the dataset declares more
  than one, rather than returning the first. `movement.io.load_poses` returns every
  scorer and individual silently; a caller arriving through this bridge keeps the
  ambiguity guard that `pose_from_deeplabcut` enforces.
- Read time from the dataset coordinate only when it declares
  `time_unit == "seconds"`. Frame-indexed datasets require an explicit `time_s` or
  `fps` rather than having frame numbers silently read as seconds.
- Copy confidence through unmodified and leave gating to
  `PoseTrajectory.speed(minimum_confidence=...)`, preserving the value/mask
  separation of SDR-0033. Absent confidence becomes `NaN`, not one, per SDR-0050.
- Refuse non-`poses` datasets, and state in the docstring that `movement` stores
  `float32`, so the returned `float64` arrays carry `float32` precision.

`tables` remains in the `behavior` extra because `pd.read_hdf` still backs the
DeepLabCut HDF5 path and no delegation removes it.

## Consequences

Scientists already using `movement` can load pose with it — including its
`from_nwb_file`, Anipose and Lightning Pose readers, which this package does not
implement — and hand the result to this package's clock synchronisation, covariate
alignment and encoding layers. Neither package has to absorb the other's
dependencies, and `movement` is now named in the documentation as the package that
owns pose IO.

This package continues to read the current SLEAP analysis format that `movement`
cannot, and continues to own the confidence-gated validity contract. Two pose
readers therefore exist in the ecosystem, and a user who loads the same SLEAP file
through both will get `float32` values through `movement` and `float64` through
this package.

## Alternatives considered

- **Delete this package's readers and depend on `movement`:** rejected because it
  would turn a working read of the current SLEAP writer into a hard failure, raise
  the Python floor above the declared support matrix, add 119 packages and 720 MB,
  retain PyTables anyway, and replace a validity-mask contract with a central
  difference that reports speeds at gated samples.
- **Delegate only the DeepLabCut reader:** rejected because it is the one path
  where values already agree exactly, so it buys about 115 lines at the cost of the
  entire dependency footprint and the Python floor, and would lose the tested
  scorer/individual ambiguity guard.
- **Vendor `movement`'s loaders:** rejected because copying BSD code to avoid a
  dependency inherits the SLEAP `dims` defect without inheriting the maintenance.
- **Make `movement` an optional `behavior` extra rather than a hard dependency:**
  rejected because the Python floor and the resolver footprint apply to anyone who
  installs the extra, and the bridge achieves the interoperability benefit without
  either.
- **Convert `PoseTrajectory` to a `movement` dataset internally:** rejected because
  `io/ndx_pose.py` constructs and consumes the dataclass, and because `movement`'s
  attrs carry no clock identity, subject, session or synchronisation lineage.
- **Upstream the SLEAP `dims` fix and then delegate:** deferred rather than
  rejected; see the revisit trigger. It would not address the Python floor, the
  footprint, or the speed semantics.

## Revisit trigger

Reopen delegation if `movement` reads the `dims` attribute of SLEAP analysis files,
its required Python version falls to this package's floor or this package raises
its floor to 3.12, and its core dependency set sheds the Qt and video stack. Revisit
the speed contract only if `movement` gains a validity-mask representation that
survives gating. Revisit the bridge itself whenever `movement` changes its poses
dataset dimension names or `time_unit` contract, both of which
`pose_from_movement()` reads by name.
