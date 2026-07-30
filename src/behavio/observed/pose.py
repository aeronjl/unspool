"""Tracked keypoints over time, and the readers that produce them.

:class:`PoseTrajectory` holds one keypoint of one tracked individual with its
coordinates, confidence-like score and timestamps. The readers for DeepLabCut,
SLEAP and ``movement`` live here because a pose is what they produce. They are
deliberately thin: they translate a tool's file or in-memory shape into this
type and refuse ambiguous selections rather than guessing, so that missingness,
confidence and clock identity survive the boundary.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from behavio.observed._arrays import _identity, _readonly_float, _validate_time
from behavio.observed.covariates import BehaviorCovariate


def _time_from_fps(
    length: int,
    *,
    time_s: ArrayLike | None,
    fps: float | None,
) -> NDArray[np.float64]:
    if (time_s is None) == (fps is None):
        raise ValueError("provide exactly one of time_s or fps")
    if fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        return _readonly_float(np.arange(length, dtype=float) / fps, name="time_s")
    assert time_s is not None
    result = _readonly_float(time_s, name="time_s")
    if len(result) != length:
        raise ValueError("time_s length must match the source frames")
    return result


@dataclass(frozen=True)
class PoseTrajectory:
    """One tracked 2D or 3D keypoint from an external pose tool.

    ``subject`` and ``session`` accept any hashable identifier, matching
    :class:`behavio.trials.Study`.
    """

    subject: Hashable
    session: Hashable
    keypoint: str
    time_s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    confidence: NDArray[np.float64]
    coordinate_unit: str
    source: str
    clock_id: str
    z: NDArray[np.float64] | None = None
    reference_frame: str | None = None
    confidence_definition: str | None = None
    individual: str | None = None
    source_version: str | None = None
    source_artifact: str | None = None
    clock_synchronization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        time_s = _readonly_float(self.time_s, name="time_s")
        x = _readonly_float(self.x, name="x")
        y = _readonly_float(self.y, name="y")
        z = _readonly_float(self.z, name="z") if self.z is not None else None
        confidence = _readonly_float(self.confidence, name="confidence")
        lengths = {len(time_s), len(x), len(y), len(confidence)}
        if z is not None:
            lengths.add(len(z))
        if len(lengths) != 1:
            raise ValueError("pose arrays must have equal length")
        _validate_time(time_s)
        if not self.keypoint.strip() or not self.coordinate_unit.strip():
            raise ValueError("keypoint and coordinate_unit must be non-empty")
        if not self.source.strip() or not self.clock_id.strip():
            raise ValueError("pose source and clock_id must be non-empty")
        for value, name in (
            (self.reference_frame, "reference_frame"),
            (self.confidence_definition, "confidence_definition"),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when provided")
        synchronization_ids = tuple(str(value) for value in self.clock_synchronization_ids)
        if any(not value.strip() for value in synchronization_ids):
            raise ValueError("clock synchronization IDs must be non-empty")
        finite_confidence = confidence[np.isfinite(confidence)]
        if np.any(finite_confidence < 0):
            raise ValueError("confidence-like scores must be non-negative")
        object.__setattr__(self, "subject", _identity(self.subject, name="subject"))
        object.__setattr__(self, "session", _identity(self.session, name="session"))
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "clock_synchronization_ids", synchronization_ids)

    def speed(
        self,
        *,
        minimum_confidence: float,
        coordinate_scale: float = 1.0,
        output_unit: str | None = None,
        name: str | None = None,
    ) -> BehaviorCovariate:
        """Calculate pairwise speed while retaining confidence-derived missingness."""

        if not np.isfinite(minimum_confidence) or minimum_confidence < 0:
            raise ValueError("minimum_confidence must be finite and non-negative")
        if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        valid_pose = (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & np.isfinite(self.confidence)
            & (self.confidence >= minimum_confidence)
        )
        if self.z is not None:
            valid_pose &= np.isfinite(self.z)
        values = np.full(len(self.time_s), np.nan, dtype=float)
        valid = np.zeros(len(self.time_s), dtype=bool)
        if len(self.time_s) > 1:
            squared_delta = np.diff(self.x) ** 2 + np.diff(self.y) ** 2
            if self.z is not None:
                squared_delta += np.diff(self.z) ** 2
            delta = np.sqrt(squared_delta) * coordinate_scale
            values[1:] = delta / np.diff(self.time_s)
            valid[1:] = valid_pose[1:] & valid_pose[:-1] & np.isfinite(values[1:])
        unit = output_unit or f"{self.coordinate_unit}/s"
        return BehaviorCovariate(
            subject=self.subject,
            session=self.session,
            name=name or f"{self.keypoint}_speed",
            time_s=self.time_s,
            values=values,
            valid=valid,
            unit=unit,
            source=self.source,
            clock_id=self.clock_id,
            source_version=self.source_version,
            source_artifact=self.source_artifact,
            clock_synchronization_ids=self.clock_synchronization_ids,
        )


def pose_from_deeplabcut(
    frame: Any,
    *,
    subject: str,
    session: str,
    keypoint: str,
    time_s: ArrayLike | None = None,
    fps: float | None = None,
    scorer: str | None = None,
    individual: str | None = None,
    coordinate_unit: str = "px",
    clock_id: str = "video",
    source_version: str | None = None,
    source_artifact: str | None = None,
) -> PoseTrajectory:
    """Read one keypoint from a DeepLabCut MultiIndex-like result table."""

    if not hasattr(frame, "columns") or not hasattr(frame, "__getitem__"):
        raise TypeError("frame must provide dataframe-like columns and column access")
    columns = [
        tuple(column) if isinstance(column, tuple) else (column,) for column in frame.columns
    ]
    matches: dict[str, list[tuple[Any, ...]]] = {"x": [], "y": [], "likelihood": []}
    for column in columns:
        if len(column) < 3 or str(column[-2]) != keypoint:
            continue
        coordinate = str(column[-1]).lower()
        if coordinate not in matches:
            continue
        if scorer is not None and str(column[0]) != scorer:
            continue
        if individual is not None and (len(column) < 4 or str(column[-3]) != individual):
            continue
        matches[coordinate].append(column)
    ambiguous = {name: values for name, values in matches.items() if len(values) != 1}
    if ambiguous:
        counts = {name: len(values) for name, values in ambiguous.items()}
        raise ValueError(
            "DeepLabCut keypoint columns are missing or ambiguous; "
            f"observed matches {counts}. Declare scorer/individual explicitly."
        )
    x = np.asarray(frame[matches["x"][0]], dtype=float)
    y = np.asarray(frame[matches["y"][0]], dtype=float)
    confidence = np.asarray(frame[matches["likelihood"][0]], dtype=float)
    finite_likelihood = confidence[np.isfinite(confidence)]
    if np.any((finite_likelihood < 0) | (finite_likelihood > 1)):
        raise ValueError("DeepLabCut likelihood must lie between zero and one")
    time = _time_from_fps(len(x), time_s=time_s, fps=fps)
    return PoseTrajectory(
        subject=subject,
        session=session,
        keypoint=keypoint,
        time_s=time,
        x=x,
        y=y,
        confidence=confidence,
        coordinate_unit=coordinate_unit,
        source="deeplabcut",
        clock_id=clock_id,
        individual=individual,
        source_version=source_version,
        source_artifact=source_artifact,
    )


def pose_from_deeplabcut_file(
    path: str | Path,
    *,
    subject: str,
    session: str,
    keypoint: str,
    time_s: ArrayLike | None = None,
    fps: float | None = None,
    scorer: str | None = None,
    individual: str | None = None,
    csv_header_rows: Literal[3, 4] = 3,
    coordinate_unit: str = "px",
    clock_id: str = "video",
    source_version: str | None = None,
) -> PoseTrajectory:
    """Read one keypoint from a DeepLabCut prediction HDF5 or CSV file."""

    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError(
            "DeepLabCut file input requires the 'behavior' optional dependency"
        ) from error
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        frame = pd.read_hdf(source)
    elif suffix == ".csv":
        frame = pd.read_csv(
            source,
            header=list(range(csv_header_rows)),
            index_col=0,
        )
    else:
        raise ValueError("DeepLabCut file must have .h5, .hdf5, or .csv suffix")
    return pose_from_deeplabcut(
        frame,
        subject=subject,
        session=session,
        keypoint=keypoint,
        time_s=time_s,
        fps=fps,
        scorer=scorer,
        individual=individual,
        coordinate_unit=coordinate_unit,
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=str(source),
    )


def pose_from_sleap(
    tracks: ArrayLike,
    *,
    subject: str,
    session: str,
    node_names: Sequence[str],
    node: str,
    dims: tuple[str, str, str, str],
    time_s: ArrayLike | None = None,
    fps: float | None = None,
    track: int = 0,
    point_scores: ArrayLike | None = None,
    score_dims: tuple[str, str, str] | None = None,
    coordinate_unit: str = "px",
    clock_id: str = "video",
    source_version: str | None = None,
    source_artifact: str | None = None,
) -> PoseTrajectory:
    """Read one node from a SLEAP Analysis HDF5-shaped array."""

    expected = {"frame", "track", "node", "xy"}
    if set(dims) != expected or len(set(dims)) != 4:
        raise ValueError(f"dims must contain exactly {sorted(expected)}")
    values = np.asarray(tracks, dtype=float)
    if values.ndim != 4:
        raise ValueError("tracks must be four-dimensional")
    canonical = np.moveaxis(
        values,
        [dims.index(name) for name in ("frame", "track", "node", "xy")],
        range(4),
    )
    if canonical.shape[3] != 2:
        raise ValueError("the SLEAP xy dimension must have length two")
    if len(node_names) != canonical.shape[2] or node not in node_names:
        raise ValueError("node_names must match the node dimension and include node")
    if not 0 <= track < canonical.shape[1]:
        raise ValueError("track index is out of range")
    node_index = list(node_names).index(node)
    selected = canonical[:, track, node_index, :]

    if point_scores is None:
        confidence = np.ones(canonical.shape[0], dtype=float)
    else:
        score_values = np.asarray(point_scores, dtype=float)
        score_order = score_dims or tuple(name for name in dims if name != "xy")
        if set(score_order) != {"frame", "track", "node"}:
            raise ValueError("score_dims must contain frame, track, and node")
        if score_values.ndim != 3:
            raise ValueError("point_scores must be three-dimensional")
        canonical_scores = np.moveaxis(
            score_values,
            [score_order.index(name) for name in ("frame", "track", "node")],
            range(3),
        )
        if canonical_scores.shape != canonical.shape[:3]:
            raise ValueError("point_scores dimensions must match tracks")
        confidence = canonical_scores[:, track, node_index]
    time = _time_from_fps(canonical.shape[0], time_s=time_s, fps=fps)
    return PoseTrajectory(
        subject=subject,
        session=session,
        keypoint=node,
        time_s=time,
        x=selected[:, 0],
        y=selected[:, 1],
        confidence=confidence,
        coordinate_unit=coordinate_unit,
        source="sleap",
        clock_id=clock_id,
        individual=str(track),
        source_version=source_version,
        source_artifact=source_artifact,
    )


def pose_from_sleap_analysis_h5(
    path: str | Path,
    *,
    subject: str,
    session: str,
    node: str,
    time_s: ArrayLike | None = None,
    fps: float | None = None,
    track: int = 0,
    dims: tuple[str, str, str, str] | None = None,
    score_dims: tuple[str, str, str] | None = None,
    coordinate_unit: str = "px",
    clock_id: str = "video",
    source_version: str | None = None,
) -> PoseTrajectory:
    """Read one node from a SLEAP Analysis HDF5 file."""

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError("SLEAP file input requires the 'behavior' optional dependency") from error
    source = Path(path)
    with h5py.File(source, "r") as file:
        tracks = file["tracks"][:]
        node_names = [_decode_text(value) for value in file["node_names"][:]]
        point_scores = file["point_scores"][:] if "point_scores" in file else None
        stored_dims = _hdf5_dims(file["tracks"].attrs.get("dims"))
        stored_score_dims = (
            _hdf5_dims(file["point_scores"].attrs.get("dims")) if "point_scores" in file else None
        )
    selected_dims = dims or stored_dims
    if selected_dims is None or len(selected_dims) != 4:
        raise ValueError("SLEAP file has no usable tracks dimension metadata; declare dims")
    selected_score_dims = score_dims or stored_score_dims
    return pose_from_sleap(
        tracks,
        subject=subject,
        session=session,
        node_names=node_names,
        node=node,
        dims=(
            selected_dims[0],
            selected_dims[1],
            selected_dims[2],
            selected_dims[3],
        ),
        time_s=time_s,
        fps=fps,
        track=track,
        point_scores=point_scores,
        score_dims=(
            (
                selected_score_dims[0],
                selected_score_dims[1],
                selected_score_dims[2],
            )
            if selected_score_dims is not None
            else None
        ),
        coordinate_unit=coordinate_unit,
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=str(source),
    )


_MOVEMENT_DIMENSION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "keypoint": ("keypoint", "keypoints"),
    "individual": ("individual", "individuals"),
}


def _movement_dimension(dataset: Any, name: str) -> str:
    present = [alias for alias in _MOVEMENT_DIMENSION_ALIASES[name] if alias in dataset.sizes]
    if len(present) != 1:
        raise ValueError(
            f"movement dataset must declare exactly one {name} dimension; "
            f"observed {sorted(dataset.sizes)}"
        )
    return present[0]


def _movement_label(
    dataset: Any,
    dimension: str,
    requested: str | None,
    *,
    kind: str,
) -> str:
    labels = [str(value) for value in dataset.coords[dimension].values.ravel()]
    if requested is None:
        if len(labels) != 1:
            raise ValueError(
                f"movement {kind} is missing or ambiguous; observed {labels}. "
                f"Declare {kind} explicitly."
            )
        return labels[0]
    if requested not in labels:
        raise ValueError(f"movement dataset has no {kind} {requested!r}; observed {labels}")
    return requested


def _movement_selection(array: Any, selector: Mapping[str, str]) -> Any:
    return array.sel({key: value for key, value in selector.items() if key in array.dims})


def _movement_time(
    dataset: Any,
    length: int,
    *,
    time_s: ArrayLike | None,
    fps: float | None,
) -> NDArray[np.float64]:
    if time_s is not None or fps is not None:
        return _time_from_fps(length, time_s=time_s, fps=fps)
    if str(dataset.attrs.get("time_unit", "")) != "seconds":
        raise ValueError("movement dataset time is not expressed in seconds; declare time_s or fps")
    return _readonly_float(dataset.coords["time"].values, name="time_s")


def pose_from_movement(
    dataset: Any,
    *,
    subject: str,
    session: str,
    keypoint: str | None = None,
    individual: str | None = None,
    time_s: ArrayLike | None = None,
    fps: float | None = None,
    coordinate_unit: str = "px",
    clock_id: str = "video",
    source: str | None = None,
    reference_frame: str | None = None,
    confidence_definition: str | None = None,
    source_version: str | None = None,
    source_artifact: str | None = None,
) -> PoseTrajectory:
    """Read one keypoint from a ``movement`` poses dataset.

    ``movement`` (neuroinformatics-unit, BSD-3-Clause) is the maintained community
    package for pose input and output. This adapter consumes its ``poses`` dataset
    so that pose estimates loaded through ``movement.io.load_poses`` reach this
    package's clock alignment, covariate and validity contracts. It duck-types on
    the xarray interface and never imports ``movement``, so it adds no dependency
    and no Python version floor. See SDR-0059 for why ``movement`` is consumed
    rather than depended upon.

    The dataset must carry ``position`` over ``time``, ``space`` and one keypoint
    and individual dimension, and ``space`` labels ``x``/``y`` or ``x``/``y``/``z``.
    Both the singular and plural dimension spellings used across ``movement``
    releases are accepted.

    ``keypoint`` and ``individual`` may be omitted only when the dataset declares
    exactly one of each; otherwise selection is ambiguous and this function raises
    rather than silently returning the first. Time is read from the dataset's own
    coordinate only when it declares ``time_unit == "seconds"``; frame-indexed
    datasets require an explicit ``time_s`` or ``fps``.

    Confidence is copied through unmodified so that gating stays this package's
    decision. Supply the dataset *before* ``movement.filtering.filter_by_confidence``
    and threshold with :meth:`PoseTrajectory.speed`, which invalidates a step when
    either endpoint fails the threshold. ``movement.kinematics.compute_speed`` uses
    central differences, so on a confidence-filtered dataset it both emits a speed
    at the gated sample and blanks its two well-estimated neighbours; this adapter
    preserves the value/mask separation instead. Datasets whose ``confidence``
    lacks the keypoint dimension are broadcast across keypoints, which is the only
    reading their shape supports.

    ``movement`` stores coordinates and confidence as ``float32``. Values are
    widened to ``float64`` here, so the returned trajectory carries ``float32``
    precision in a ``float64`` array and will not round-trip bit-identically
    against a ``float64`` first-party reader.
    """

    for attribute in ("sizes", "coords", "attrs", "data_vars"):
        if not hasattr(dataset, attribute):
            raise TypeError("dataset must provide an xarray-like Dataset interface")
    declared_type = str(dataset.attrs.get("ds_type", "poses"))
    if declared_type != "poses":
        raise ValueError(f"movement dataset must be a poses dataset, not {declared_type!r}")
    if "position" not in dataset.data_vars:
        raise ValueError("movement dataset must contain a position variable")
    if "time" not in dataset.sizes or "space" not in dataset.sizes:
        raise ValueError("movement dataset must declare time and space dimensions")

    keypoint_dimension = _movement_dimension(dataset, "keypoint")
    individual_dimension = _movement_dimension(dataset, "individual")
    selected_keypoint = _movement_label(dataset, keypoint_dimension, keypoint, kind="keypoint")
    selected_individual = _movement_label(
        dataset, individual_dimension, individual, kind="individual"
    )
    selector = {
        keypoint_dimension: selected_keypoint,
        individual_dimension: selected_individual,
    }

    space = [str(value) for value in dataset.coords["space"].values.ravel()]
    if space not in (["x", "y"], ["x", "y", "z"]):
        raise ValueError(f"movement space labels must be x/y or x/y/z; observed {space}")
    position = _movement_selection(dataset["position"], selector).transpose("time", "space")
    coordinates = np.asarray(position.values, dtype=float)

    if "confidence" in dataset.data_vars:
        confidence = np.asarray(
            _movement_selection(dataset["confidence"], selector).transpose("time").values,
            dtype=float,
        )
    else:
        confidence = np.full(coordinates.shape[0], np.nan, dtype=float)

    software = source or str(dataset.attrs.get("source_software", "")).lower()
    if not software.strip():
        raise ValueError("movement dataset declares no source_software; declare source explicitly")
    stored_artifact = dataset.attrs.get("source_file")
    return PoseTrajectory(
        subject=subject,
        session=session,
        keypoint=selected_keypoint,
        time_s=_movement_time(dataset, coordinates.shape[0], time_s=time_s, fps=fps),
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        z=coordinates[:, 2] if len(space) == 3 else None,
        confidence=confidence,
        coordinate_unit=coordinate_unit,
        source=software,
        clock_id=clock_id,
        individual=selected_individual,
        reference_frame=reference_frame,
        confidence_definition=confidence_definition,
        source_version=source_version,
        source_artifact=(
            source_artifact
            if source_artifact is not None
            else (str(stored_artifact) if stored_artifact is not None else None)
        ),
    )


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _hdf5_dims(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        items = value.tolist() if isinstance(value, np.ndarray) else value
        return tuple(_decode_text(item) for item in items)
    decoded = _decode_text(value)
    if decoded.startswith("["):
        import json

        loaded = json.loads(decoded)
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("HDF5 dims attribute must be a JSON string list")
        return tuple(loaded)
    raise ValueError("HDF5 dims attribute has an unsupported representation")
