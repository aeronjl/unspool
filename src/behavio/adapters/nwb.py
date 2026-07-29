"""Lossless, session-aware adapters between NWB trials and :class:`behavio.Study`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.study import REQUIRED_COLUMNS, Study

ADAPTER_NAME = "behavio.nwb"
ADAPTER_VERSION = "1"

_EMBEDDED_SUBJECT = "behavio_subject"
_EMBEDDED_SESSION = "behavio_session"
_EMBEDDED_TRIAL = "behavio_trial"
_EMBEDDED_SESSION_ORDER = "behavio_session_order"
_EMBEDDED_COLUMNS = {
    _EMBEDDED_SUBJECT,
    _EMBEDDED_SESSION,
    _EMBEDDED_TRIAL,
    _EMBEDDED_SESSION_ORDER,
}
_STANDARD_INTERVAL_COLUMNS = {"start_time", "stop_time"}


class NWBAdapterError(ValueError):
    """Raised when an NWB trial table cannot satisfy the longitudinal contract."""


@dataclass(frozen=True, slots=True)
class NWBSessionSource:
    """One local NWB session plus the chronology needed to import it safely.

    The source doubles as a :class:`behavio.contracts.adapter.StudyAdapter`: its
    chronology policy is ``RECORDED`` because ``session_order`` is either embedded
    losslessly by :func:`add_study_trials` or supplied by the caller, never inferred.
    """

    path: Path
    session_order: int | None = None
    subject: Any | None = None
    session: Any | None = None
    columns: tuple[str, ...] | None = None
    column_map: Mapping[str, str] | None = None

    adapter_name: ClassVar[str] = ADAPTER_NAME
    adapter_version: ClassVar[str] = ADAPTER_VERSION
    source_type: ClassVar[SourceType] = SourceType.LOCAL_FILE
    session_order_policy: ClassVar[SessionOrderPolicy] = SessionOrderPolicy.RECORDED

    def __post_init__(self) -> None:
        path = Path(self.path)
        _validate_optional_order(self.session_order)
        columns = _validate_columns(self.columns)
        column_map = _validate_column_map(self.column_map)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "column_map", column_map)

    def read(self) -> Study:
        """Read this declared NWB session into a canonical study."""

        return read_nwb(self)


def study_from_nwbfile(
    nwbfile: Any,
    *,
    session_order: int | None = None,
    subject: Any | None = None,
    session: Any | None = None,
    columns: Sequence[str] | None = None,
    column_map: Mapping[str, str] | None = None,
    source_columns: Mapping[str, Any] | None = None,
) -> Study:
    """Copy one NWB session's scalar trial table into a canonical ``Study``.

    ``session_order`` is never inferred from file names, timestamps, or input order. It
    must be embedded by :func:`add_study_trials` or supplied explicitly.
    """

    trials = getattr(nwbfile, "trials", None)
    if trials is None or len(trials.id) == 0:
        raise NWBAdapterError("NWBFile must contain a non-empty trials table")
    _validate_optional_order(session_order)
    selected = _validate_columns(columns)
    mapping = _validate_column_map(column_map)
    available = tuple(str(name) for name in trials.colnames)
    if selected is None:
        selected = available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise NWBAdapterError(f"requested NWB trial columns are missing: {missing}")
    unmapped_sources = sorted(set(mapping) - set(selected))
    if unmapped_sources:
        raise NWBAdapterError(
            f"column_map names columns that were not selected: {unmapped_sources}"
        )

    n_rows = len(trials.id)
    embedded_subject = _constant_embedded(trials, _EMBEDDED_SUBJECT, n_rows)
    embedded_session = _constant_embedded(trials, _EMBEDDED_SESSION, n_rows)
    embedded_order = _constant_embedded(trials, _EMBEDDED_SESSION_ORDER, n_rows)
    resolved_subject = _resolve_identity(
        name="subject",
        explicit=subject,
        embedded=embedded_subject,
        metadata=_nwb_subject_id(nwbfile),
    )
    resolved_session = _resolve_identity(
        name="session",
        explicit=session,
        embedded=embedded_session,
        metadata=getattr(nwbfile, "session_id", None),
    )
    resolved_order = _resolve_session_order(session_order, embedded_order)
    trial_values = (
        _read_scalar_column(trials, _EMBEDDED_TRIAL, n_rows)
        if _EMBEDDED_TRIAL in available
        else [_python_scalar(value) for value in trials.id[:]]
    )
    _validate_trial_values(trial_values)

    imported: dict[str, Any] = {
        "subject": [resolved_subject] * n_rows,
        "session": [resolved_session] * n_rows,
        "trial": trial_values,
        "session_order": [resolved_order] * n_rows,
    }
    for source_name in selected:
        if source_name in _EMBEDDED_COLUMNS:
            continue
        target_name = mapping.get(source_name, source_name)
        if target_name in REQUIRED_COLUMNS:
            raise NWBAdapterError(
                f"NWB column {source_name!r} collides with required Study column "
                f"{target_name!r}; map it to a source-specific name"
            )
        if target_name in _EMBEDDED_COLUMNS:
            raise NWBAdapterError(f"Study column name {target_name!r} is reserved")
        if target_name in imported:
            raise NWBAdapterError(f"multiple NWB columns map to Study column {target_name!r}")
        imported[target_name] = _read_scalar_column(trials, source_name, n_rows)

    provenance = dict(source_columns or {})
    identifier = getattr(nwbfile, "identifier", None)
    if identifier is not None:
        provenance.setdefault("source_nwb_identifier", _python_scalar(identifier))
    for name, value in provenance.items():
        if not isinstance(name, str) or not name:
            raise NWBAdapterError("source column names must be non-empty strings")
        if name in REQUIRED_COLUMNS:
            raise NWBAdapterError(f"source column {name!r} cannot replace a required column")
        value = _python_scalar(value)
        if not _is_scalar(value):
            raise NWBAdapterError(f"source column {name!r} must contain one scalar value")
        if name in imported:
            observed = imported[name]
            if not all(_values_equal(item, value) for item in observed):
                raise NWBAdapterError(f"source column {name!r} conflicts with NWB trial data")
            continue
        imported[name] = [_python_scalar(value)] * n_rows
    return Study.from_columns(imported)


def read_nwb(
    source: NWBSessionSource | str | Path,
    *,
    session_order: int | None = None,
    subject: Any | None = None,
    session: Any | None = None,
    columns: Sequence[str] | None = None,
    column_map: Mapping[str, str] | None = None,
) -> Study:
    """Read one local NWB file without making PyNWB a core dependency."""

    if isinstance(source, NWBSessionSource):
        separate_arguments = (session_order, subject, session, columns, column_map)
        if any(value is not None for value in separate_arguments):
            raise TypeError("NWBSessionSource cannot be combined with separate mapping arguments")
        path = source.path
        session_order = source.session_order
        subject = source.subject
        session = source.session
        columns = source.columns
        column_map = source.column_map
    else:
        path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    NWBHDF5IO, _, _ = _require_pynwb()
    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        return study_from_nwbfile(
            io.read(),
            session_order=session_order,
            subject=subject,
            session=session,
            columns=columns,
            column_map=column_map,
            source_columns={"source_nwb_path": str(path.resolve())},
        )


def read_nwb_sessions(sources: Sequence[NWBSessionSource]) -> Study:
    """Read and concatenate explicitly ordered local NWB sessions."""

    if not sources:
        raise NWBAdapterError("at least one NWB session source is required")
    studies = [read_nwb(source) for source in sources]
    expected = studies[0].columns
    for index, study in enumerate(studies[1:], start=1):
        if study.columns != expected:
            raise NWBAdapterError(
                f"NWB session {index} has different Study columns; "
                f"expected {expected!r}, observed {study.columns!r}"
            )
    return Study.from_columns(
        {name: np.concatenate([study[name] for study in studies]) for name in expected}
    )


def add_study_trials(
    nwbfile: Any,
    study: Study,
    *,
    column_descriptions: Mapping[str, str] | None = None,
) -> None:
    """Add one single-subject, single-session ``Study`` to an ``NWBFile`` in place."""

    subject, session, _ = _single_session_identity(study)
    existing_trials = getattr(nwbfile, "trials", None)
    if existing_trials is not None and len(existing_trials.id) > 0:
        raise NWBAdapterError("NWBFile already contains trials")
    _validate_nwb_identity(nwbfile, subject, session)
    if "start_time" not in study.columns or "stop_time" not in study.columns:
        raise NWBAdapterError("NWB export requires explicit start_time and stop_time columns")
    starts = np.asarray(study["start_time"], dtype=np.float64)
    stops = np.asarray(study["stop_time"], dtype=np.float64)
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(stops)):
        raise NWBAdapterError("NWB trial intervals must be finite")
    if np.any(stops < starts):
        raise NWBAdapterError("NWB trial stop_time cannot precede start_time")

    descriptions = dict(column_descriptions or {})
    export_columns: dict[str, list[Any]] = {
        _EMBEDDED_SUBJECT: [_nwb_scalar(value, _EMBEDDED_SUBJECT) for value in study["subject"]],
        _EMBEDDED_SESSION: [_nwb_scalar(value, _EMBEDDED_SESSION) for value in study["session"]],
        _EMBEDDED_TRIAL: [_nwb_scalar(value, _EMBEDDED_TRIAL) for value in study["trial"]],
        _EMBEDDED_SESSION_ORDER: [
            _nwb_scalar(value, _EMBEDDED_SESSION_ORDER) for value in study["session_order"]
        ],
    }
    for name in study.columns:
        if name in REQUIRED_COLUMNS or name in _STANDARD_INTERVAL_COLUMNS:
            continue
        if name in _EMBEDDED_COLUMNS or name == "id":
            raise NWBAdapterError(f"Study column {name!r} is reserved by the NWB adapter")
        export_columns[name] = [_nwb_scalar(value, name) for value in study[name]]
    for name, values in export_columns.items():
        _validate_nwb_column(values, name)
        nwbfile.add_trial_column(
            name=name,
            description=descriptions.get(name, f"Behavio Study column {name!r}."),
        )
    for row in range(len(study)):
        values = {name: column[row] for name, column in export_columns.items()}
        nwbfile.add_trial(start_time=float(starts[row]), stop_time=float(stops[row]), **values)


def write_nwb(
    study: Study,
    path: str | Path,
    *,
    session_description: str,
    identifier: str,
    session_start_time: datetime,
    column_descriptions: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one Study session as a schema-valid NWB HDF5 file."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if not isinstance(session_description, str) or not session_description:
        raise ValueError("session_description must be a non-empty string")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    if not isinstance(session_start_time, datetime) or session_start_time.utcoffset() is None:
        raise ValueError("session_start_time must be a timezone-aware datetime")
    subject, session, _ = _single_session_identity(study)
    NWBHDF5IO, NWBFile, Subject = _require_pynwb()
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        session_id=str(session),
        subject=Subject(subject_id=str(subject)),
    )
    add_study_trials(nwbfile, study, column_descriptions=column_descriptions)
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)
    return path


def _require_pynwb() -> tuple[Any, Any, Any]:
    try:
        from pynwb import NWBHDF5IO, NWBFile
        from pynwb.file import Subject
    except ImportError as error:
        raise RuntimeError(
            "NWB interoperability requires optional dependencies; install `behavio[nwb]`"
        ) from error
    return NWBHDF5IO, NWBFile, Subject


def _nwb_subject_id(nwbfile: Any) -> Any | None:
    subject = getattr(nwbfile, "subject", None)
    return None if subject is None else getattr(subject, "subject_id", None)


def _resolve_identity(*, name: str, explicit: Any, embedded: Any, metadata: Any) -> Any:
    if explicit is not None and embedded is not None and not _values_equal(explicit, embedded):
        raise NWBAdapterError(
            f"explicit {name} {explicit!r} conflicts with embedded value {embedded!r}"
        )
    value = explicit if explicit is not None else embedded if embedded is not None else metadata
    if value is None:
        raise NWBAdapterError(
            f"{name} is absent from NWB metadata; supply it explicitly when importing"
        )
    return _python_scalar(value)


def _resolve_session_order(explicit: int | None, embedded: Any) -> int:
    if embedded is not None:
        if isinstance(embedded, (bool, np.bool_)) or not isinstance(embedded, (int, np.integer)):
            raise NWBAdapterError("embedded session_order must be a non-negative integer")
        if explicit is not None and int(explicit) != int(embedded):
            raise NWBAdapterError(
                f"explicit session_order {explicit} conflicts with embedded value {embedded}"
            )
        return int(embedded)
    if explicit is None:
        raise NWBAdapterError(
            "session_order cannot be inferred from NWB; supply it explicitly when importing"
        )
    return explicit


def _constant_embedded(trials: Any, name: str, n_rows: int) -> Any | None:
    if name not in trials.colnames:
        return None
    values = _read_scalar_column(trials, name, n_rows)
    first = values[0]
    if not all(_values_equal(value, first) for value in values[1:]):
        raise NWBAdapterError(f"embedded {name!r} must be constant within one NWB session")
    return first


def _read_scalar_column(trials: Any, name: str, n_rows: int) -> list[Any]:
    values = []
    column = trials[name]
    for row in range(n_rows):
        value = _python_scalar(column[row])
        if not _is_scalar(value):
            raise NWBAdapterError(
                f"NWB trial column {name!r} contains non-scalar cells; "
                "select scalar columns explicitly"
            )
        values.append(value)
    return values


def _single_session_identity(study: Study) -> tuple[Any, Any, int]:
    subjects = _unique(study["subject"])
    sessions = _unique(study["session"])
    orders = _unique(study["session_order"])
    if len(subjects) != 1 or len(sessions) != 1 or len(orders) != 1:
        raise NWBAdapterError("NWB export requires exactly one subject and one session")
    return subjects[0], sessions[0], int(orders[0])


def _validate_nwb_identity(nwbfile: Any, subject: Any, session: Any) -> None:
    metadata_subject = _nwb_subject_id(nwbfile)
    if metadata_subject is None:
        raise NWBAdapterError("NWBFile must declare Subject.subject_id before adding trials")
    if str(metadata_subject) != str(subject):
        raise NWBAdapterError(
            f"NWB subject_id {metadata_subject!r} conflicts with Study subject {subject!r}"
        )
    metadata_session = getattr(nwbfile, "session_id", None)
    if metadata_session is None:
        raise NWBAdapterError("NWBFile must declare session_id before adding trials")
    if str(metadata_session) != str(session):
        raise NWBAdapterError(
            f"NWB session_id {metadata_session!r} conflicts with Study session {session!r}"
        )


def _validate_optional_order(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("session_order must be a non-negative integer or None")


def _validate_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    values = tuple(columns)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("columns must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("columns must be unique")
    return values


def _validate_column_map(mapping: Mapping[str, str] | None) -> Mapping[str, str]:
    if mapping is None:
        return MappingProxyType({})
    copied = dict(mapping)
    if any(
        not isinstance(source, str) or not source or not isinstance(target, str) or not target
        for source, target in copied.items()
    ):
        raise ValueError("column_map must map non-empty strings to non-empty strings")
    if len(set(copied.values())) != len(copied):
        raise ValueError("column_map targets must be unique")
    return MappingProxyType(copied)


def _validate_nwb_column(values: list[Any], name: str) -> None:
    kinds = {_scalar_kind(value) for value in values}
    if "unsupported" in kinds or "missing" in kinds:
        raise NWBAdapterError(f"Study column {name!r} contains values unsupported by NWB")
    if len(kinds - {"integer", "float"}) > 1 or (kinds & {"string"} and len(kinds) > 1):
        raise NWBAdapterError(f"Study column {name!r} mixes incompatible scalar types")


def _validate_trial_values(values: list[Any]) -> None:
    for row, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NWBAdapterError(
                f"NWB trial IDs must be non-negative integers; row {row} is {value!r}"
            )
    if len(set(values)) != len(values):
        raise NWBAdapterError("NWB trial IDs must be unique")


def _nwb_scalar(value: Any, name: str) -> Any:
    value = _python_scalar(value)
    if not _is_scalar(value):
        raise NWBAdapterError(f"Study column {name!r} contains non-scalar values")
    return value


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, bool, int, float))


def _scalar_kind(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "integer"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "unsupported"


def _unique(values: Sequence[Any]) -> list[Any]:
    unique: list[Any] = []
    for raw in values:
        value = _python_scalar(raw)
        if not any(_values_equal(value, observed) for observed in unique):
            unique.append(value)
    return unique


def _values_equal(first: Any, second: Any) -> bool:
    try:
        equal = first == second
    except (TypeError, ValueError):
        return False
    return bool(equal) if isinstance(equal, (bool, np.bool_)) else False
