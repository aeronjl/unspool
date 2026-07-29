"""Dependency-free CSV/TSV ingest, with Parquet behind one optional extra.

A psychologist with ``trials.csv`` is the most common entry point into Behavio, so the
delimited path deliberately uses nothing but the standard library and NumPy: no pandas, no
pyarrow, no optional install. Parquet is a binary container that genuinely needs a reader,
so it -- and only it -- requires ``behavio[parquet]``.

Three source facts are never guessed:

- ``session_order`` is chronology. If the table does not carry it, the caller must name a
  derivation (:func:`session_order_from_column`, :func:`session_order_from_explicit`,
  :func:`session_order_from_appearance`) and that choice is recorded on every trial. This
  is the same refusal the NWB adapter makes, expressed as an opt-in rule rather than a
  silent guess.
- ``trial`` numbering from row position is likewise opt-in
  (``number_trials_by_row_order=True``) and recorded.
- Column types are inferred by an explicit, documented ladder and can be declared per
  column with ``dtypes``; failures name the column, the row, the offending cell, and the
  two ways to fix it.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.study import REQUIRED_COLUMNS, Study

ADAPTER_NAME = "behavio.table"
ADAPTER_VERSION = "1"

#: Cell contents treated as missing before any type inference. Comparison is exact, after
#: surrounding whitespace is stripped.
DEFAULT_MISSING_VALUES: tuple[str, ...] = (
    "",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NaN",
    "NAN",
    "nan",
    "NULL",
    "null",
    "None",
)

PROVENANCE_COLUMNS = (
    "source_table_path",
    "source_session_order_rule",
    "source_trial_rule",
)

_INTEGER_TEXT = re.compile(r"[+-]?(0|[1-9][0-9]*)")
_ANY_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")
_FLOAT_TEXT = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")
#: A zero-padded run of digits is an identifier, not a number: participant `007` and
#: condition code `010` must survive a read unchanged. Inference therefore leaves any
#: column containing one as text, and only an explicit `dtypes` declaration converts it.
_PADDED_INTEGER_TEXT = re.compile(r"[+-]?0[0-9]+")
_TRUE_TEXT = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_TEXT = frozenset({"false", "f", "no", "n", "0"})


class TableReadError(ValueError):
    """Raised when a delimited or Parquet table cannot become a valid ``Study``."""


class TableFormat(StrEnum):
    """Supported tabular containers."""

    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"


class ColumnType(StrEnum):
    """Column types that may be declared explicitly instead of inferred."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"


class SessionOrderRule(StrEnum):
    """The named ways a caller may derive chronology that the table does not carry."""

    COLUMN = "column"
    EXPLICIT = "explicit"
    APPEARANCE = "appearance"


_SUFFIX_FORMATS = {
    ".csv": TableFormat.CSV,
    ".tsv": TableFormat.TSV,
    ".tab": TableFormat.TSV,
    ".parquet": TableFormat.PARQUET,
    ".pqt": TableFormat.PARQUET,
}
_FORMAT_DELIMITERS = {TableFormat.CSV: ",", TableFormat.TSV: "\t"}


@dataclass(frozen=True, slots=True)
class SessionOrderDerivation:
    """One explicitly named way to obtain ``session_order`` from a table that lacks it.

    A derivation is a recorded scientific choice. The rule that produced the chronology is
    written to the ``source_session_order_rule`` column of every trial, so a study read
    this way can never be mistaken for one whose source declared its own chronology.
    """

    rule: SessionOrderRule
    column: str | None = None
    ordering: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        rule = SessionOrderRule(self.rule)
        if rule is SessionOrderRule.COLUMN:
            if not isinstance(self.column, str) or not self.column:
                raise ValueError("the column rule requires a non-empty ordering column name")
            if self.ordering is not None:
                raise ValueError("the column rule does not take an explicit ordering")
        elif rule is SessionOrderRule.EXPLICIT:
            if self.column is not None:
                raise ValueError("the explicit rule does not take an ordering column")
            ordering = tuple(self.ordering or ())
            if not ordering:
                raise ValueError("the explicit rule requires at least one session identifier")
            try:
                unique = len(set(ordering)) == len(ordering)
            except TypeError:
                raise ValueError("explicit session identifiers must be hashable") from None
            if not unique:
                raise ValueError("explicit session identifiers must be unique")
            object.__setattr__(self, "ordering", ordering)
        elif self.column is not None or self.ordering is not None:
            raise ValueError("the appearance rule takes no further configuration")
        object.__setattr__(self, "rule", rule)

    @property
    def basis(self) -> str:
        """A short, stable record of the derivation, stored on every trial."""

        if self.rule is SessionOrderRule.COLUMN:
            return f"column:{self.column}"
        return self.rule.value


def session_order_from_column(column: str) -> SessionOrderDerivation:
    """Rank each subject's sessions by a column that carries time, such as a date."""

    return SessionOrderDerivation(rule=SessionOrderRule.COLUMN, column=column)


def session_order_from_explicit(ordering: Sequence[Any]) -> SessionOrderDerivation:
    """Rank sessions by a caller-supplied ordering of session identifiers."""

    return SessionOrderDerivation(rule=SessionOrderRule.EXPLICIT, ordering=tuple(ordering))


def session_order_from_appearance() -> SessionOrderDerivation:
    """Rank each subject's sessions by first appearance in source row (and file) order.

    This is the rule the NWB adapter refuses to apply on its own. Naming it here makes the
    assumption -- that the table was written in chronological order -- an explicit, recorded
    claim by the analyst rather than an inference by the library.
    """

    return SessionOrderDerivation(rule=SessionOrderRule.APPEARANCE)


@dataclass(frozen=True, slots=True)
class TableSource:
    """One delimited or Parquet trial table and the declarations needed to read it."""

    path: Path
    format: TableFormat | None = None
    delimiter: str | None = None
    encoding: str = "utf-8-sig"
    subject_column: str = "subject"
    session_column: str = "session"
    trial_column: str = "trial"
    session_order_column: str = "session_order"
    session_order: SessionOrderDerivation | None = None
    number_trials_by_row_order: bool = False
    columns: tuple[str, ...] | None = None
    column_map: Mapping[str, str] | None = None
    dtypes: Mapping[str, ColumnType] | None = None
    missing_values: tuple[str, ...] = DEFAULT_MISSING_VALUES
    #: Record each trial's absolute source path in a ``source_table_path`` column, as the
    #: NWB adapter does. Set it to False when a machine-independent protocol fingerprint
    #: matters more than knowing which file on this machine a trial came from.
    record_source_path: bool = True

    adapter_name: ClassVar[str] = ADAPTER_NAME
    adapter_version: ClassVar[str] = ADAPTER_VERSION
    source_type: ClassVar[SourceType] = SourceType.LOCAL_FILE

    def __post_init__(self) -> None:
        path = Path(self.path)
        table_format = None if self.format is None else TableFormat(self.format)
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or len(self.delimiter) != 1
        ):
            raise ValueError("delimiter must be exactly one character")
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError("encoding must be a non-empty string")
        identity = {
            "subject_column": self.subject_column,
            "session_column": self.session_column,
            "trial_column": self.trial_column,
            "session_order_column": self.session_order_column,
        }
        for name, value in identity.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if len(set(identity.values())) != len(identity):
            raise ValueError("identity columns must name four different source columns")
        if self.session_order is not None and not isinstance(
            self.session_order, SessionOrderDerivation
        ):
            raise TypeError("session_order must be a SessionOrderDerivation or None")
        if not isinstance(self.number_trials_by_row_order, bool):
            raise TypeError("number_trials_by_row_order must be a boolean")
        if not isinstance(self.record_source_path, bool):
            raise TypeError("record_source_path must be a boolean")
        columns = _validated_columns(self.columns)
        column_map = _validated_column_map(self.column_map)
        dtypes = _validated_dtypes(self.dtypes)
        missing = tuple(self.missing_values)
        if any(not isinstance(value, str) for value in missing):
            raise ValueError("missing_values must contain strings")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "format", table_format)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "column_map", column_map)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "missing_values", missing)

    @property
    def session_order_policy(self) -> SessionOrderPolicy:
        """``DERIVED`` when a chronology rule was named, ``RECORDED`` otherwise."""

        if self.session_order is None:
            return SessionOrderPolicy.RECORDED
        return SessionOrderPolicy.DERIVED

    @property
    def resolved_format(self) -> TableFormat:
        """The declared container, or the one implied by the file suffix."""

        if self.format is not None:
            return self.format
        suffix = self.path.suffix.lower()
        resolved = _SUFFIX_FORMATS.get(suffix)
        if resolved is None:
            supported = ", ".join(sorted(_SUFFIX_FORMATS))
            raise TableReadError(
                f"{self.path}: cannot tell the table format from suffix {suffix!r}; "
                f"pass format='csv', 'tsv', or 'parquet' explicitly. "
                f"Recognized suffixes: {supported}."
            )
        return resolved

    def read(self) -> Study:
        """Read the declared table into a canonical study."""

        return read_table(self)


def read_table(source: TableSource | str | Path, **options: Any) -> Study:
    """Read one CSV, TSV, or Parquet trial table into a canonical ``Study``.

    ``read_table("trials.csv")`` is the shortest supported path from a file to a study.
    Every keyword accepted by :class:`TableSource` may be passed alongside a path; a
    prepared :class:`TableSource` may not be combined with separate arguments.
    """

    return read_tables([_as_source(source, options)])


def read_tables(sources: Iterable[TableSource]) -> Study:
    """Read several identically declared tables as one study, preserving file order.

    Rows keep their source order within each file and the files keep the given order, so
    :func:`session_order_from_appearance` derives chronology from file order when each
    file holds one session.
    """

    declared = tuple(sources)
    if not declared:
        raise TableReadError("at least one table source is required")
    if any(not isinstance(source, TableSource) for source in declared):
        raise TypeError("sources must contain TableSource values")
    first = declared[0]
    for index, source in enumerate(declared[1:], start=1):
        differing = _differing_options(first, source)
        if differing:
            raise TableReadError(
                "read_tables requires every source to declare the same reading options; "
                f"source {index} differs in {differing}"
            )
    raw = [_read_raw(source) for source in declared]
    names = raw[0].names
    for table in raw[1:]:
        if table.names != names:
            raise TableReadError(
                f"{table.path}: columns {list(table.names)} differ from "
                f"{raw[0].path} columns {list(names)}"
            )
    return _build_study(raw, first)


@dataclass(frozen=True, slots=True)
class _RawTable:
    """One source table reduced to Python scalars, with ``None`` marking missing cells."""

    path: Path
    names: tuple[str, ...]
    cells: dict[str, list[Any]]
    n_rows: int
    delimited: bool

    def location(self, row: int) -> str:
        """Describe one row the way a person would find it in an editor."""

        if self.delimited:
            return f"data row {row + 1} (line {row + 2}) of {self.path}"
        return f"row {row + 1} of {self.path}"


def _as_source(source: TableSource | str | Path, options: Mapping[str, Any]) -> TableSource:
    if isinstance(source, TableSource):
        if options:
            raise TypeError("TableSource cannot be combined with separate reading options")
        return source
    return TableSource(Path(source), **dict(options))


def _differing_options(first: TableSource, other: TableSource) -> list[str]:
    ignored = {"path", "format", "delimiter"}
    return [
        field.name
        for field in fields(first)
        if field.name not in ignored and getattr(first, field.name) != getattr(other, field.name)
    ]


def _read_raw(source: TableSource) -> _RawTable:
    table_format = source.resolved_format
    if not source.path.is_file():
        raise FileNotFoundError(source.path)
    if table_format is TableFormat.PARQUET:
        return _read_parquet(source)
    return _read_delimited(source, table_format)


def _read_delimited(source: TableSource, table_format: TableFormat) -> _RawTable:
    delimiter = source.delimiter or _FORMAT_DELIMITERS[table_format]
    with source.path.open("r", encoding=source.encoding, newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    rows = [row for row in rows if row not in ([], [""])]
    if not rows:
        raise TableReadError(f"{source.path} is empty; a table needs a header and one data row")
    header = [name.strip() for name in rows[0]]
    if any(not name for name in header):
        raise TableReadError(f"{source.path}: every header field must be a non-empty name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise TableReadError(f"{source.path}: duplicate header names {duplicates}")
    data = rows[1:]
    if not data:
        raise TableReadError(f"{source.path}: the header is present but there are no data rows")
    cells: dict[str, list[Any]] = {name: [] for name in header}
    missing = frozenset(source.missing_values)
    for row_index, row in enumerate(data):
        if len(row) != len(header):
            raise TableReadError(
                f"{source.path}: line {row_index + 2} has {len(row)} fields but the header "
                f"declares {len(header)}"
            )
        for name, value in zip(header, row, strict=True):
            text = value.strip()
            cells[name].append(None if text in missing else text)
    return _RawTable(
        path=source.path,
        names=tuple(header),
        cells=cells,
        n_rows=len(data),
        delimited=True,
    )


def _read_parquet(source: TableSource) -> _RawTable:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet ingest requires optional dependencies; install `behavio[parquet]`. "
            "CSV and TSV need no optional dependencies."
        ) from error
    table = parquet.read_table(source.path)
    header = [str(name) for name in table.column_names]
    if any(not name for name in header):
        raise TableReadError(f"{source.path}: every Parquet column must have a non-empty name")
    if len(set(header)) != len(header):
        raise TableReadError(f"{source.path}: Parquet column names must be unique")
    if table.num_rows == 0:
        raise TableReadError(f"{source.path}: the Parquet table contains no rows")
    missing = frozenset(source.missing_values)
    cells: dict[str, list[Any]] = {}
    for name, column in zip(header, table.columns, strict=True):
        values = [_parquet_scalar(value, missing) for value in column.to_pylist()]
        cells[name] = values
    return _RawTable(
        path=source.path,
        names=tuple(header),
        cells=cells,
        n_rows=table.num_rows,
        delimited=False,
    )


def _parquet_scalar(value: Any, missing: frozenset[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return None if text in missing else text
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _build_study(raw: Sequence[_RawTable], source: TableSource) -> Study:
    names = raw[0].names
    identity = {
        source.subject_column: "subject",
        source.session_column: "session",
        source.trial_column: "trial",
        source.session_order_column: "session_order",
    }
    _require_identity_columns(raw[0], source)
    retained = _retained_columns(raw[0], source, identity)
    typed: dict[str, list[Any]] = {}
    for name in retained:
        typed[name] = _typed_column(name, raw, source)
    canonical = _canonical_names(retained, source, identity)

    n_rows = sum(table.n_rows for table in raw)
    columns: dict[str, list[Any]] = {}
    for name in retained:
        columns[canonical[name]] = typed[name]

    subjects = columns["subject"]
    _require_present(subjects, "subject", raw)
    sessions = columns["session"]
    _require_present(sessions, "session", raw)

    if "trial" not in columns:
        columns["trial"] = _derive_trials(subjects, sessions)
    else:
        _require_non_negative_integers(columns["trial"], "trial", raw)
    if "session_order" not in columns:
        columns["session_order"] = _derive_session_order(
            subjects, sessions, typed, source, raw, names
        )
    else:
        _require_non_negative_integers(columns["session_order"], "session_order", raw)

    provenance: dict[str, list[Any]] = {}
    if source.record_source_path:
        provenance["source_table_path"] = [
            str(table.path.resolve()) for table in raw for _ in range(table.n_rows)
        ]
    if source.session_order is not None:
        provenance["source_session_order_rule"] = [source.session_order.basis] * n_rows
    if source.number_trials_by_row_order:
        provenance["source_trial_rule"] = ["row-order"] * n_rows
    for name, values in provenance.items():
        if name in columns:
            raise TableReadError(
                f"{raw[0].path}: column {name!r} is reserved for adapter provenance; "
                "rename it with column_map"
            )
        columns[name] = values

    ordered_names = list(REQUIRED_COLUMNS)
    ordered_names += [name for name in columns if name not in REQUIRED_COLUMNS]
    ordered = {name: _array(columns[name]) for name in ordered_names}
    try:
        return Study.from_columns(ordered)
    except ValueError as error:
        raise TableReadError(f"{raw[0].path}: {error}") from error


def _require_identity_columns(raw: _RawTable, source: TableSource) -> None:
    available = list(raw.names)
    problems: list[str] = []
    for column, canonical in (
        (source.subject_column, "subject"),
        (source.session_column, "session"),
    ):
        if column not in available:
            problems.append(
                f"{raw.path} has no column {column!r}, which Behavio needs for {canonical!r} "
                f"identity. Name the source column with {canonical}_column=... ."
            )
    if source.session_order_column not in available and source.session_order is None:
        problems.append(
            f"{raw.path} has no column {source.session_order_column!r}. Behavio never infers "
            "session chronology from row, file, or filename order. Either add the column, "
            "name it with session_order_column=..., or record an explicit derivation with "
            "session_order=session_order_from_column('date'), "
            "session_order_from_explicit([...]), or session_order_from_appearance()."
        )
    if source.session_order_column in available and source.session_order is not None:
        problems.append(
            f"{raw.path} already has a {source.session_order_column!r} column; "
            "read it directly instead of deriving a chronology, or map it elsewhere."
        )
    if source.trial_column not in available and not source.number_trials_by_row_order:
        problems.append(
            f"{raw.path} has no column {source.trial_column!r}. Behavio does not number trials "
            "from row position unless asked to: either add the column, name it with "
            "trial_column=..., or record the choice with number_trials_by_row_order=True."
        )
    if source.trial_column in available and source.number_trials_by_row_order:
        problems.append(
            f"{raw.path} already has a {source.trial_column!r} column; "
            "drop number_trials_by_row_order=True or map that column elsewhere."
        )
    if problems:
        problems.append(f"Available columns: {available}.")
        raise TableReadError(" ".join(problems))


def _retained_columns(
    raw: _RawTable, source: TableSource, identity: Mapping[str, str]
) -> tuple[str, ...]:
    available = tuple(raw.names)
    required = {name for name in identity if name in available}
    if source.session_order is not None and source.session_order.rule is SessionOrderRule.COLUMN:
        ordering_column = source.session_order.column
        if ordering_column not in available:
            raise TableReadError(
                f"{raw.path} has no ordering column {ordering_column!r} for the declared "
                f"session_order derivation. Available columns: {list(available)}."
            )
        required.add(str(ordering_column))
    if source.columns is None:
        return available
    unknown = sorted(set(source.columns) - set(available))
    if unknown:
        raise TableReadError(
            f"{raw.path}: requested columns are missing: {unknown}. "
            f"Available columns: {list(available)}."
        )
    selected = set(source.columns) | required
    return tuple(name for name in available if name in selected)


def _canonical_names(
    retained: Sequence[str], source: TableSource, identity: Mapping[str, str]
) -> dict[str, str]:
    canonical: dict[str, str] = {}
    mapping = dict(source.column_map or {})
    unknown = sorted(set(mapping) - set(retained))
    if unknown:
        raise TableReadError(f"column_map names columns that were not read: {unknown}")
    for name in retained:
        if name in identity:
            if name in mapping:
                raise TableReadError(
                    f"column {name!r} is already mapped to {identity[name]!r}; "
                    "remove it from column_map"
                )
            canonical[name] = identity[name]
            continue
        target = mapping.get(name, name)
        if target in REQUIRED_COLUMNS:
            raise TableReadError(
                f"column {name!r} would replace the canonical {target!r} column; "
                "map it to a source-specific name"
            )
        if target in PROVENANCE_COLUMNS:
            raise TableReadError(f"column name {target!r} is reserved for adapter provenance")
        canonical[name] = target
    targets = list(canonical.values())
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    if duplicates:
        raise TableReadError(f"columns collide after mapping: {duplicates}")
    return canonical


def _typed_column(name: str, raw: Sequence[_RawTable], source: TableSource) -> list[Any]:
    declared = (source.dtypes or {}).get(name)
    values: list[Any] = []
    for table in raw:
        cells = table.cells[name]
        if declared is not None:
            values.extend(
                _coerce(value, declared, name=name, row=row, table=table)
                for row, value in enumerate(cells)
            )
        elif table.delimited:
            values.extend(_infer_text(cells))
        else:
            values.extend(cells)
    return values


def _coerce(value: Any, declared: ColumnType, *, name: str, row: int, table: _RawTable) -> Any:
    if value is None:
        return None
    if declared is ColumnType.STR:
        return value if isinstance(value, str) else str(value)
    text = value if isinstance(value, str) else None
    if declared is ColumnType.INT:
        if text is not None and _ANY_INTEGER_TEXT.fullmatch(text):
            return int(text)
        if text is None and isinstance(value, int) and not isinstance(value, bool):
            return value
        raise _cell_error(table, name, row, value, "int", ColumnType.INT)
    if declared is ColumnType.FLOAT:
        if text is not None and _FLOAT_TEXT.fullmatch(text):
            return float(text)
        if text is None and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise _cell_error(table, name, row, value, "float", ColumnType.FLOAT)
    if isinstance(value, bool):
        return value
    lowered = text.lower() if text is not None else None
    if lowered in _TRUE_TEXT:
        return True
    if lowered in _FALSE_TEXT:
        return False
    raise _cell_error(table, name, row, value, "bool", ColumnType.BOOL)


def _cell_error(
    table: _RawTable, name: str, row: int, value: Any, expected: str, declared: ColumnType
) -> TableReadError:
    if declared is ColumnType.BOOL:
        hint = (
            "Booleans are read from true/false, yes/no, t/f, or 1/0. "
            "Add other spellings to missing_values, or read the column as text with "
            f"dtypes={{{name!r}: 'str'}}."
        )
    else:
        hint = (
            f"Add it to missing_values if it means 'missing', or read the column as text "
            f"with dtypes={{{name!r}: 'str'}}."
        )
    return TableReadError(
        f"could not convert column {name!r} to {expected}: "
        f"{table.location(row)} contains {value!r}. {hint}"
    )


def _infer_text(cells: Sequence[Any]) -> list[Any]:
    """Apply the inference ladder: padded digits stay text, then integer, float, text."""

    present = [value for value in cells if value is not None]
    if not present or any(_PADDED_INTEGER_TEXT.fullmatch(value) for value in present):
        return list(cells)
    if all(_INTEGER_TEXT.fullmatch(value) for value in present):
        return [None if value is None else int(value) for value in cells]
    if all(_FLOAT_TEXT.fullmatch(value) for value in present):
        return [None if value is None else float(value) for value in cells]
    return list(cells)


def _require_present(values: Sequence[Any], name: str, raw: Sequence[_RawTable]) -> None:
    for row, value in enumerate(values):
        if value is None:
            raise TableReadError(
                f"{name!r} has no value at {_locate(raw, row)}; identity may not be missing. "
                "Fill the cell, or remove the row before reading."
            )


def _require_non_negative_integers(
    values: Sequence[Any], name: str, raw: Sequence[_RawTable]
) -> None:
    offenders = [row for row, value in enumerate(values) if not _is_index(value)]
    if not offenders:
        return
    # Report the cell that actually reads as wrong. When a stray `2.5` or an empty cell
    # turns the whole column into floats, pointing at row 1 because `0.0` is not an `int`
    # would send the reader to a row that looks perfectly fine in their editor.
    row = min(offenders, key=lambda index: _is_whole_number(values[index]))
    raise TableReadError(
        f"column {name!r} must contain non-negative integers: "
        f"{_locate(raw, row)} contains {values[row]!r}. "
        f"Declare the column with dtypes={{{name!r}: 'int'}} if it is text, or fix "
        "the offending rows."
    )


def _is_index(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_whole_number(value: Any) -> bool:
    """True when a value merely has the wrong type but reads like a valid trial number."""

    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return value >= 0 and float(value).is_integer()
    return False


def _locate(raw: Sequence[_RawTable], row: int) -> str:
    offset = row
    for table in raw:
        if offset < table.n_rows:
            return table.location(offset)
        offset -= table.n_rows
    return f"row {row + 1}"


def _derive_trials(subjects: Sequence[Any], sessions: Sequence[Any]) -> list[int]:
    counters: dict[tuple[Any, Any], int] = {}
    trials: list[int] = []
    for subject, session in zip(subjects, sessions, strict=True):
        key = (subject, session)
        position = counters.get(key, 0)
        counters[key] = position + 1
        trials.append(position)
    return trials


def _derive_session_order(
    subjects: Sequence[Any],
    sessions: Sequence[Any],
    typed: Mapping[str, list[Any]],
    source: TableSource,
    raw: Sequence[_RawTable],
    names: Sequence[str],
) -> list[int]:
    derivation = source.session_order
    if derivation is None:  # pragma: no cover - guaranteed by _require_identity_columns
        raise TableReadError("session_order is absent and no derivation was declared")
    if derivation.rule is SessionOrderRule.EXPLICIT:
        ordering = {session: rank for rank, session in enumerate(derivation.ordering or ())}
        unknown = sorted({str(session) for session in sessions if session not in ordering})
        if unknown:
            raise TableReadError(
                f"{raw[0].path}: the explicit session ordering does not mention {unknown}. "
                "Every session identifier in the table must appear in the ordering."
            )
        return [ordering[session] for session in sessions]

    if derivation.rule is SessionOrderRule.APPEARANCE:
        seen: dict[Any, dict[Any, int]] = {}
        orders: list[int] = []
        for subject, session in zip(subjects, sessions, strict=True):
            ranks = seen.setdefault(subject, {})
            if session not in ranks:
                ranks[session] = len(ranks)
            orders.append(ranks[session])
        return orders

    column = str(derivation.column)
    if column not in typed:
        raise TableReadError(
            f"{raw[0].path} has no ordering column {column!r} for the declared session_order "
            f"derivation. Available columns: {list(names)}."
        )
    keys = typed[column]
    session_keys: dict[tuple[Any, Any], Any] = {}
    for row, (subject, session, key) in enumerate(zip(subjects, sessions, keys, strict=True)):
        if key is None:
            raise TableReadError(
                f"ordering column {column!r} has no value at {_locate(raw, row)}; "
                "session chronology cannot be derived from a missing value."
            )
        known = session_keys.setdefault((subject, session), key)
        if known != key:
            raise TableReadError(
                f"{raw[0].path}: ordering column {column!r} is not constant within "
                f"subject {subject!r} session {session!r} ({known!r} and {key!r}); "
                "session chronology must be a property of the session, not of the trial."
            )
    ranks: dict[tuple[Any, Any], int] = {}
    by_subject: dict[Any, list[tuple[Any, Any]]] = {}
    for (subject, session), key in session_keys.items():
        by_subject.setdefault(subject, []).append((key, session))
    for subject, entries in by_subject.items():
        try:
            ordered = sorted(entries, key=lambda entry: entry[0])
        except TypeError:
            raise TableReadError(
                f"{raw[0].path}: ordering column {column!r} mixes value types for subject "
                f"{subject!r}; declare one type with dtypes={{{column!r}: 'str'}}."
            ) from None
        for rank, (_key, session) in enumerate(ordered):
            ranks[(subject, session)] = rank
    return [ranks[(subject, session)] for subject, session in zip(subjects, sessions, strict=True)]


def _array(values: Sequence[Any]) -> NDArray[Any]:
    kinds = {_kind(value) for value in values if value is not None}
    missing = any(value is None for value in values)
    if not kinds:
        return np.full(len(values), np.nan, dtype=np.float64)
    if kinds == {"bool"} and not missing:
        return np.asarray(values, dtype=bool)
    if kinds == {"int"} and not missing:
        return np.asarray(values, dtype=np.int64)
    if kinds <= {"int", "float"}:
        return np.asarray(
            [np.nan if value is None else float(value) for value in values], dtype=np.float64
        )
    if kinds == {"str"} and not missing:
        return np.asarray(values, dtype=np.str_)
    return np.asarray(values, dtype=object)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "other"


def _validated_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    values = tuple(columns)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("columns must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("columns must be unique")
    return values


def _validated_column_map(mapping: Mapping[str, str] | None) -> Mapping[str, str]:
    if mapping is None:
        return MappingProxyType({})
    copied = dict(mapping)
    if any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in copied.items()
    ):
        raise ValueError("column_map must map non-empty strings to non-empty strings")
    if len(set(copied.values())) != len(copied):
        raise ValueError("column_map targets must be unique")
    return MappingProxyType(copied)


def _validated_dtypes(dtypes: Mapping[str, ColumnType] | None) -> Mapping[str, ColumnType]:
    if dtypes is None:
        return MappingProxyType({})
    copied: dict[str, ColumnType] = {}
    for name, value in dict(dtypes).items():
        if not isinstance(name, str) or not name:
            raise ValueError("dtypes keys must be non-empty column names")
        try:
            copied[name] = ColumnType(value)
        except ValueError:
            supported = ", ".join(sorted(member.value for member in ColumnType))
            raise ValueError(
                f"unknown declared type {value!r} for column {name!r}; use one of {supported}"
            ) from None
    return MappingProxyType(copied)
