"""Deterministic serialization, content addressing, and name validation for declarations.

Behavio now content-addresses two different declarations: the study protocol in
:mod:`behavio.protocol.schema` and the task family and task protocol in
:mod:`behavio.task.ontology`. A fingerprint is an identity only if two declarations that
say the same thing hash the same way, so the canonical JSON writer, the enum-and-tuple
flattening it depends on, and the scalar validation that keeps a payload writable at all
are defined once here instead of once per declaration.

Nothing in this module imports the rest of the package, and every helper takes the error
type its caller raises. A declaration keeps its own exception -- a protocol still fails
with :class:`~behavio.protocol.schema.ProtocolValidationError` -- while the rule being
enforced exists in exactly one place.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | tuple["JSONValue", ...]


class DeclarationError(ValueError):
    """Raised when a declaration payload is not deterministically serializable."""


def json_ready(value: Any) -> Any:
    """Flatten enums and tuples into the plain JSON types a canonical writer accepts."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically, refusing NaN and infinity."""

    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_fingerprint(text: str) -> str:
    """Return the lowercase SHA-256 hex digest of one canonical serialization."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_scalar(value: JSONScalar) -> str:
    """Return a hashable canonical spelling of one scalar, distinguishing ``0`` from ``False``."""

    return canonical_json(value)


def json_value(
    value: Any,
    label: str,
    *,
    error: type[ValueError] = DeclarationError,
) -> JSONValue:
    """Validate one finite JSON scalar or nested tuple, converting lists to tuples."""

    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        return tuple(json_value(item, label, error=error) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise error(f"{label} must be a finite JSON scalar or tuple")


def require_name(value: Any, label: str, *, error: type[ValueError] = DeclarationError) -> None:
    """Require one non-empty, non-blank string."""

    if not isinstance(value, str) or not value.strip():
        raise error(f"{label} must be a non-empty string")


def require_names(
    values: Any,
    label: str,
    *,
    allow_empty: bool = False,
    error: type[ValueError] = DeclarationError,
) -> None:
    """Require a tuple of unique, non-empty strings."""

    if isinstance(values, str):
        raise error(f"{label} must be a tuple")
    values = tuple(values)
    if not allow_empty and not values:
        raise error(f"{label} must not be empty")
    for value in values:
        require_name(value, label, error=error)
    if len(set(values)) != len(values):
        raise error(f"{label} must be unique")


def require_fingerprint(
    value: Any, label: str, *, error: type[ValueError] = DeclarationError
) -> None:
    """Require a lowercase SHA-256 hex digest."""

    if not isinstance(value, str) or len(value) != 64:
        raise error(f"{label} must be a lowercase SHA-256 hex digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError:
        parsed = b""
    if len(parsed) != 32 or value != value.lower():
        raise error(f"{label} must be a lowercase SHA-256 hex digest")
