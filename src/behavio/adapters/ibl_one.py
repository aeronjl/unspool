"""Exact IBL trial-table import through the optional ONE client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar
from urllib.parse import urlparse
from uuid import UUID

import numpy as np

from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.study import REQUIRED_COLUMNS, Study

DEFAULT_IBL_ALYX_URL = "https://openalyx.internationalbrainlab.org"
DEFAULT_PUBLIC_PASSWORD = "international"
ADAPTER_NAME = "behavio.ibl_one"
ADAPTER_VERSION = "1"


class IBLONEAdapterError(ValueError):
    """Raised when ONE cannot resolve an exact declared IBL trial table."""


@dataclass(frozen=True, slots=True)
class IBLONETrialSource:
    """One checksum-pinned IBL trial table and its longitudinal identity."""

    session_id: str
    dataset_id: str
    dataset_path: str
    file_size: int
    md5: str
    release_tag: str
    session_order: int
    subject: Any
    session: Any | None = None
    lab: Any | None = None
    columns: tuple[str, ...] | None = None
    column_map: Mapping[str, str] | None = None
    source_columns: Mapping[str, Any] | None = None
    alyx_url: str = DEFAULT_IBL_ALYX_URL

    adapter_name: ClassVar[str] = ADAPTER_NAME
    adapter_version: ClassVar[str] = ADAPTER_VERSION
    source_type: ClassVar[SourceType] = SourceType.REMOTE_ARCHIVE
    session_order_policy: ClassVar[SessionOrderPolicy] = SessionOrderPolicy.RECORDED

    def read(self) -> Study:
        """Load this checksum-pinned ONE trial table into a canonical study."""

        return study_from_ibl_one(self)

    def __post_init__(self) -> None:
        session_id = _validated_uuid(self.session_id, "session_id")
        dataset_id = _validated_uuid(self.dataset_id, "dataset_id")
        if (
            not isinstance(self.dataset_path, str)
            or not self.dataset_path
            or self.dataset_path.startswith("/")
            or ".." in self.dataset_path.split("/")
            or not self.dataset_path.endswith(".pqt")
        ):
            raise ValueError("dataset_path must be a normalized relative Parquet path")
        if (
            isinstance(self.file_size, bool)
            or not isinstance(self.file_size, int)
            or self.file_size < 1
        ):
            raise ValueError("file_size must be a positive integer")
        if (
            not isinstance(self.md5, str)
            or len(self.md5) != 32
            or any(character not in "0123456789abcdef" for character in self.md5)
        ):
            raise ValueError("md5 must contain 32 lowercase hexadecimal digits")
        if not isinstance(self.release_tag, str) or not self.release_tag:
            raise ValueError("release_tag must be a non-empty string")
        if (
            isinstance(self.session_order, bool)
            or not isinstance(self.session_order, int)
            or self.session_order < 0
        ):
            raise ValueError("session_order must be a non-negative integer")
        subject = _validated_identifier(self.subject, "subject")
        session = (
            session_id if self.session is None else _validated_identifier(self.session, "session")
        )
        lab = None if self.lab is None else _validated_identifier(self.lab, "lab")
        parsed_url = urlparse(self.alyx_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.query:
            raise ValueError("alyx_url must be an HTTPS origin without a query")
        columns = _validated_columns(self.columns)
        column_map = _validated_column_map(self.column_map)
        source_columns = _validated_source_columns(self.source_columns)
        selected = columns or tuple(column_map)
        if selected and not set(column_map).issubset(selected):
            raise ValueError("column_map sources must be included in columns")
        targets = tuple(column_map.get(name, name) for name in selected)
        if set(targets) & set(REQUIRED_COLUMNS):
            raise ValueError("trial-table columns may not replace canonical identity columns")
        if len(set(targets)) != len(targets):
            raise ValueError("selected trial-table columns must map to unique targets")
        reserved = set(REQUIRED_COLUMNS) | set(targets) | _PROVENANCE_COLUMNS
        if lab is not None:
            reserved.add("lab")
        overlap = reserved & set(source_columns)
        if overlap:
            raise ValueError(f"source_columns collide with reserved columns: {sorted(overlap)}")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "lab", lab)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "column_map", MappingProxyType(column_map))
        object.__setattr__(self, "source_columns", MappingProxyType(source_columns))
        object.__setattr__(self, "alyx_url", self.alyx_url.rstrip("/"))


_PROVENANCE_COLUMNS = {
    "source_alyx_url",
    "source_ibl_release_tag",
    "source_ibl_session_id",
    "source_ibl_dataset_id",
    "source_ibl_dataset_path",
    "source_ibl_dataset_size",
    "source_ibl_dataset_md5",
}


def study_from_ibl_one(source: IBLONETrialSource, *, client: Any | None = None) -> Study:
    """Load one exact dataset UUID through ONE and preserve trial-level provenance.

    The IBL ``choice`` field is retained in its source coding. This adapter does not infer
    left/right conventions or silently recode it to the binary convention required by
    Behavio's Bernoulli models.
    """

    one = client if client is not None else _public_one(source.alyx_url)
    frame, details = _load_exact_dataset(one, source)
    if not hasattr(frame, "columns") or not hasattr(frame, "__getitem__"):
        raise IBLONEAdapterError("ONE did not return a dataframe-like trial table")
    available = tuple(frame.columns)
    if any(not isinstance(name, str) or not name for name in available):
        raise IBLONEAdapterError("ONE trial-table columns must be non-empty strings")
    selected = source.columns or available
    missing = tuple(name for name in selected if name not in available)
    if missing:
        raise IBLONEAdapterError(f"ONE trial table is missing selected columns: {missing}")
    mapping = dict(source.column_map or {})
    n_trials = _frame_length(frame, available)
    columns: dict[str, Any] = {
        "subject": [source.subject] * n_trials,
        "session": [source.session] * n_trials,
        "trial": np.arange(n_trials, dtype=np.int64),
        "session_order": np.full(n_trials, source.session_order, dtype=np.int64),
    }
    for name in selected:
        target = mapping.get(name, name)
        if target in columns:
            raise IBLONEAdapterError(f"trial-table target column {target!r} is reserved")
        values = np.asarray(frame[name])
        if values.ndim != 1 or len(values) != n_trials:
            raise IBLONEAdapterError(f"trial-table column {name!r} is not one-dimensional")
        columns[target] = values
    if source.lab is not None:
        if "lab" in columns:
            raise IBLONEAdapterError("trial table already contains a lab column")
        columns["lab"] = [source.lab] * n_trials
    provenance = {
        "source_alyx_url": source.alyx_url,
        "source_ibl_release_tag": source.release_tag,
        "source_ibl_session_id": source.session_id,
        "source_ibl_dataset_id": source.dataset_id,
        "source_ibl_dataset_path": source.dataset_path,
        "source_ibl_dataset_size": source.file_size,
        "source_ibl_dataset_md5": source.md5,
        **dict(source.source_columns or {}),
    }
    for name, value in provenance.items():
        if name in columns:
            raise IBLONEAdapterError(f"source provenance column {name!r} is reserved")
        columns[name] = [value] * n_trials
    _verify_details(details, source)
    return Study(columns)


def read_ibl_one_sessions(
    sources: tuple[IBLONETrialSource, ...] | list[IBLONETrialSource],
    *,
    client: Any | None = None,
) -> Study:
    """Load several exact ONE trial tables and preserve input session order."""

    declared = tuple(sources)
    if not declared:
        raise ValueError("sources must contain at least one IBL ONE trial table")
    alyx_urls = {source.alyx_url for source in declared}
    if len(alyx_urls) != 1:
        raise ValueError("all IBL ONE sources must use the same Alyx origin")
    one = client if client is not None else _public_one(declared[0].alyx_url)
    studies = tuple(study_from_ibl_one(source, client=one) for source in declared)
    expected = studies[0].columns
    for index, study in enumerate(studies[1:], start=1):
        if study.columns != expected:
            raise IBLONEAdapterError(
                f"adapted session {index} has different columns; "
                f"expected {expected}, observed {study.columns}"
            )
    return Study({name: np.concatenate([study[name] for study in studies]) for name in expected})


def _load_exact_dataset(client: Any, source: IBLONETrialSource) -> tuple[Any, Any]:
    try:
        loaded = client.load_dataset_from_id(
            source.dataset_id,
            details=True,
            check_hash=True,
        )
    except Exception:
        try:
            client.list_datasets(source.session_id, details=True, query_type="remote")
            loaded = client.load_dataset_from_id(
                source.dataset_id,
                details=True,
                check_hash=True,
            )
        except Exception as error:
            raise IBLONEAdapterError(
                f"ONE could not load dataset {source.dataset_id!r} from "
                f"session {source.session_id!r}: {error}"
            ) from error
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise IBLONEAdapterError("ONE details=True must return data and metadata")
    return loaded


def _verify_details(details: Any, source: IBLONETrialSource) -> None:
    if hasattr(details, "to_dict"):
        metadata = details.to_dict()
    elif isinstance(details, Mapping):
        metadata = dict(details)
    else:
        raise IBLONEAdapterError("ONE dataset metadata is not mapping-like")
    observed_path = str(metadata.get("rel_path") or "")
    observed_hash = str(metadata.get("hash") or "")
    try:
        observed_size = int(metadata["file_size"])
    except (KeyError, TypeError, ValueError):
        raise IBLONEAdapterError("ONE dataset metadata omitted a valid byte size") from None
    mismatches = {
        "dataset_path": (observed_path, source.dataset_path),
        "file_size": (observed_size, source.file_size),
        "md5": (observed_hash, source.md5),
    }
    failed = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in mismatches.items()
        if observed != expected
    }
    if failed:
        raise IBLONEAdapterError(f"ONE dataset provenance mismatch: {failed}")


def _public_one(alyx_url: str) -> Any:
    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError(
            "IBL ONE import requires optional dependencies; install `behavio[ibl]`"
        ) from error
    return ONE(
        base_url=alyx_url,
        password=DEFAULT_PUBLIC_PASSWORD,
        silent=True,
    )


def _frame_length(frame: Any, columns: tuple[str, ...]) -> int:
    if not columns:
        raise IBLONEAdapterError("ONE trial table must contain at least one column")
    try:
        length = len(frame[columns[0]])
    except Exception as error:
        raise IBLONEAdapterError("ONE trial table does not expose column lengths") from error
    if length < 1:
        raise IBLONEAdapterError("ONE trial table must contain at least one trial")
    return length


def _validated_uuid(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical UUID string")
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{name} must be a canonical UUID string") from None
    if value != canonical:
        raise ValueError(f"{name} must be a canonical lowercase UUID string")
    return canonical


def _validated_identifier(value: Any, name: str) -> Any:
    identifier = value.item() if isinstance(value, np.generic) else value
    if identifier is None:
        raise ValueError(f"{name} must not be missing")
    if isinstance(identifier, (float, complex)) and np.isnan(identifier):
        raise ValueError(f"{name} must not be missing")
    try:
        hash(identifier)
    except TypeError:
        raise TypeError(f"{name} must be hashable") from None
    return identifier


def _validated_columns(values: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    columns = tuple(values)
    if not columns or any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("columns must contain non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must be unique")
    return columns


def _validated_column_map(values: Mapping[str, str] | None) -> dict[str, str]:
    mapping = dict(values or {})
    if any(
        not isinstance(source, str) or not source or not isinstance(target, str) or not target
        for source, target in mapping.items()
    ):
        raise ValueError("column_map must map non-empty strings to non-empty strings")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("column_map targets must be unique")
    return mapping


def _validated_source_columns(values: Mapping[str, Any] | None) -> dict[str, Any]:
    mapping = dict(values or {})
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError("source_columns names must be non-empty strings")
    for name, value in mapping.items():
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError(f"source_columns value {name!r} must be scalar")
    return mapping
