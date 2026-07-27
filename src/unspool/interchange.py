"""Portable, non-executable records for task-validated model fits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from unspool.study import Study
from unspool.task import FittedModel, TaskSpec

FIT_ARTIFACT_SCHEMA: Final = "unspool.fit-artifact/1"


class FitArtifactError(ValueError):
    """Raised when a fit artifact is invalid or cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class FitArtifact:
    """Immutable common record for one task-validated fit.

    This record is an interchange and provenance view, not a serialized Python model.
    Model-specific live results remain available on :class:`FittedModel`.
    """

    schema_version: str
    unspool_version: str
    model_name: str
    model_signature: str
    result_type: str
    task: Mapping[str, Any]
    data: Mapping[str, Any]
    parameters: tuple[Mapping[str, Any], ...]
    covariance: tuple[tuple[float | None, ...], ...]
    diagnostics: Mapping[str, Any]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != FIT_ARTIFACT_SCHEMA:
            raise FitArtifactError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {FIT_ARTIFACT_SCHEMA!r}"
            )
        for value, label in (
            (self.unspool_version, "unspool_version"),
            (self.model_name, "model_name"),
            (self.model_signature, "model_signature"),
            (self.result_type, "result_type"),
        ):
            if not isinstance(value, str) or not value:
                raise FitArtifactError(f"{label} must be a non-empty string")
        parameters = tuple(_freeze_json(dict(parameter)) for parameter in self.parameters)
        names = tuple(parameter.get("name") for parameter in parameters)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise FitArtifactError("parameters must have non-empty names")
        if len(set(names)) != len(names):
            raise FitArtifactError("parameter names must be unique")
        covariance = tuple(tuple(row) for row in self.covariance)
        if len(covariance) != len(parameters) or any(
            len(row) != len(parameters) for row in covariance
        ):
            raise FitArtifactError("covariance must be square with one row per parameter")
        for row in covariance:
            if any(value is not None and not np.isfinite(value) for value in row):
                raise FitArtifactError("finite covariance entries or null are required")
        object.__setattr__(self, "task", _freeze_json(dict(self.task)))
        object.__setattr__(self, "data", _freeze_json(dict(self.data)))
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "diagnostics", _freeze_json(dict(self.diagnostics)))
        object.__setattr__(self, "audit", _freeze_json(dict(self.audit)))

    @property
    def fingerprint(self) -> str:
        """SHA-256 content address of the complete common record."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh standards-compliant JSON representation."""

        return {
            "schema_version": self.schema_version,
            "unspool_version": self.unspool_version,
            "model": {
                "name": self.model_name,
                "signature": self.model_signature,
                "result_type": self.result_type,
            },
            "task": _thaw_json(self.task),
            "data": _thaw_json(self.data),
            "parameters": [_thaw_json(parameter) for parameter in self.parameters],
            "covariance": [list(row) for row in self.covariance],
            "diagnostics": _thaw_json(self.diagnostics),
            "audit": _thaw_json(self.audit),
        }

    def canonical_json(self) -> str:
        """Serialize deterministically without NaN or executable objects."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def export_fit(fitted: FittedModel, study: Study) -> FitArtifact:
    """Bind a common fit record to its task, complete study content, and audit."""

    if not isinstance(fitted, FittedModel):
        raise TypeError("fitted must be a FittedModel")
    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    validation = fitted.task.validate(study)
    if validation != fitted.validation:
        raise FitArtifactError("study task denominators do not match the fitted result")
    result = fitted.result
    if len(study) != result.n_observations:
        raise FitArtifactError("study row count does not match the fitted result")

    parameters = tuple(
        {
            "name": name,
            "estimate": _finite_or_none(estimate),
            "standard_error": _finite_or_none(standard_error),
        }
        for name, estimate, standard_error in zip(
            result.parameter_names,
            result.estimates,
            result.standard_errors,
            strict=True,
        )
    )
    diagnostics = _sanitize_json(
        {
            "converged": result.diagnostics.converged,
            "optimizer": result.diagnostics.optimizer,
            "status": result.diagnostics.status,
            "message": result.diagnostics.message,
            "n_iterations": result.diagnostics.n_iterations,
            "objective": result.diagnostics.objective,
            "gradient_norm": result.diagnostics.gradient_norm,
            "hessian_condition": result.diagnostics.hessian_condition,
            "boundary_estimate": result.diagnostics.boundary_estimate,
        }
    )
    return FitArtifact(
        schema_version=FIT_ARTIFACT_SCHEMA,
        unspool_version=_package_version(),
        model_name=result.model_name,
        model_signature=result.model_signature,
        result_type=type(result).__name__,
        task=_task_record(fitted.task),
        data=_study_record(study),
        parameters=parameters,
        covariance=tuple(
            tuple(_finite_or_none(value) for value in row) for row in result.covariance
        ),
        diagnostics=diagnostics,
        audit=_sanitize_json(fitted.audit().to_dict()),
    )


def fit_artifact_from_dict(payload: Mapping[str, Any]) -> FitArtifact:
    """Validate a decoded fit artifact without recreating executable model objects."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    expected = {
        "schema_version",
        "unspool_version",
        "model",
        "task",
        "data",
        "parameters",
        "covariance",
        "diagnostics",
        "audit",
    }
    if set(payload) != expected:
        raise FitArtifactError(
            f"fit artifact fields differ; missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    model = payload["model"]
    if not isinstance(model, Mapping) or set(model) != {"name", "signature", "result_type"}:
        raise FitArtifactError("model must contain name, signature, and result_type")
    for field in ("task", "data", "diagnostics", "audit"):
        if not isinstance(payload[field], Mapping):
            raise FitArtifactError(f"{field} must be an object")
    if isinstance(payload["parameters"], (str, bytes)) or not isinstance(
        payload["parameters"], Sequence
    ):
        raise FitArtifactError("parameters must be an array")
    if isinstance(payload["covariance"], (str, bytes)) or not isinstance(
        payload["covariance"], Sequence
    ):
        raise FitArtifactError("covariance must be an array")
    try:
        return FitArtifact(
            schema_version=payload["schema_version"],
            unspool_version=payload["unspool_version"],
            model_name=model["name"],
            model_signature=model["signature"],
            result_type=model["result_type"],
            task=payload["task"],
            data=payload["data"],
            parameters=tuple(payload["parameters"]),
            covariance=tuple(tuple(row) for row in payload["covariance"]),
            diagnostics=payload["diagnostics"],
            audit=payload["audit"],
        )
    except (TypeError, KeyError) as error:
        raise FitArtifactError("fit artifact contains invalid value types") from error


def fit_artifact_from_json(payload: str) -> FitArtifact:
    """Decode and validate one fit artifact from JSON text."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise FitArtifactError("payload is not valid JSON") from error
    return fit_artifact_from_dict(decoded)


def _task_record(task: TaskSpec) -> dict[str, Any]:
    choice = task.choice
    reward = task.reward
    response_time = task.response_time
    return _sanitize_json(
        {
            "choice": {
                "column": choice.column,
                "options": list(choice.options),
                "omission_values": list(choice.omission_values),
                "missing_is_omission": choice.missing_is_omission,
                "available_options_column": choice.available_options_column,
            },
            "predictors": list(task.predictors),
            "reward": (
                {
                    "column": reward.column,
                    "minimum": reward.minimum,
                    "maximum": reward.maximum,
                    "allow_missing": reward.allow_missing,
                }
                if reward is not None
                else None
            ),
            "response_time": (
                {"column": response_time.column, "unit": response_time.unit.value}
                if response_time is not None
                else None
            ),
            "block_column": task.block_column,
            "episode_column": task.episode_column,
        }
    )


def _study_record(study: Study) -> dict[str, Any]:
    content = {
        "columns": list(study.columns),
        "values": {
            column: [_json_value(value) for value in study[column]] for column in study.columns
        },
    }
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "n_trials": len(study),
        "n_subjects": len(study.subjects),
        "columns": list(study.columns),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_or_none(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, (datetime, date, time)):
        return {"type": type(value).__name__, "iso": value.isoformat()}
    if isinstance(value, timedelta):
        return {"type": "timedelta", "seconds": value.total_seconds()}
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise FitArtifactError(
        f"study or task value of type {type(value).__name__!r} is not portable JSON data"
    )


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return _finite_or_none(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _sanitize_json(value.value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize_json(item) for item in value]
    raise FitArtifactError(f"value of type {type(value).__name__!r} is not JSON-compatible")


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _package_version() -> str:
    try:
        return importlib.metadata.version("unspool")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"
