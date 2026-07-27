"""Backend-neutral parameter coordinates, bounds, and priors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

PARAMETER_SPACE_SCHEMA: Final = "unspool.parameter-space/1"


class ParameterSpaceError(ValueError):
    """Raised when parameter metadata or coordinates are invalid."""


class ParameterRole(StrEnum):
    """Whether inference estimates a parameter or holds it fixed."""

    FREE = "free"
    FIXED = "fixed"


class ParameterTransform(StrEnum):
    """One-to-one map from a natural parameter to an optimizer coordinate."""

    IDENTITY = "identity"
    LOG = "log"
    BOUNDED_LOGIT = "bounded-logit"


class PriorFamily(StrEnum):
    """Portable prior families defined on natural parameter coordinates."""

    NORMAL = "normal"
    HALF_NORMAL = "half-normal"
    BETA = "beta"
    UNIFORM = "uniform"


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """A JSON-safe scalar prior defined on the natural coordinate."""

    family: PriorFamily
    arguments: Mapping[str, float]

    def __post_init__(self) -> None:
        try:
            family = PriorFamily(self.family)
        except ValueError as error:
            raise ParameterSpaceError(f"unsupported prior family {self.family!r}") from error
        if not isinstance(self.arguments, Mapping):
            raise ParameterSpaceError("prior arguments must be a mapping")
        try:
            arguments = {str(name): float(value) for name, value in self.arguments.items()}
        except (TypeError, ValueError):
            raise ParameterSpaceError("prior arguments must be finite numeric values") from None
        if any(not name or not np.isfinite(value) for name, value in arguments.items()):
            raise ParameterSpaceError("prior arguments require non-empty names and finite values")
        expected = {
            PriorFamily.NORMAL: {"location", "scale"},
            PriorFamily.HALF_NORMAL: {"scale"},
            PriorFamily.BETA: {"alpha", "beta"},
            PriorFamily.UNIFORM: {"lower", "upper"},
        }[family]
        if set(arguments) != expected:
            raise ParameterSpaceError(
                f"{family.value} prior arguments must be exactly {sorted(expected)}"
            )
        if family in (PriorFamily.NORMAL, PriorFamily.HALF_NORMAL):
            if arguments["scale"] <= 0:
                raise ParameterSpaceError("prior scale must be positive")
        elif family == PriorFamily.BETA:
            if arguments["alpha"] <= 0 or arguments["beta"] <= 0:
                raise ParameterSpaceError("beta prior shapes must be positive")
        elif arguments["lower"] >= arguments["upper"]:
            raise ParameterSpaceError("uniform prior lower must be below upper")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "arguments", MappingProxyType(arguments))

    @classmethod
    def normal(cls, location: float, scale: float) -> PriorSpec:
        """Construct a normal prior."""

        return cls(PriorFamily.NORMAL, {"location": location, "scale": scale})

    @classmethod
    def half_normal(cls, scale: float) -> PriorSpec:
        """Construct a zero-centred half-normal prior."""

        return cls(PriorFamily.HALF_NORMAL, {"scale": scale})

    @classmethod
    def beta(cls, alpha: float, beta: float) -> PriorSpec:
        """Construct a beta prior on the open unit interval."""

        return cls(PriorFamily.BETA, {"alpha": alpha, "beta": beta})

    @classmethod
    def uniform(cls, lower: float, upper: float) -> PriorSpec:
        """Construct a bounded uniform prior."""

        return cls(PriorFamily.UNIFORM, {"lower": lower, "upper": upper})

    def log_prob(self, value: float) -> float:
        """Return the normalized scalar log density on the natural scale."""

        value = _finite_float(value, "prior value")
        arguments = self.arguments
        if self.family == PriorFamily.NORMAL:
            standardized = (value - arguments["location"]) / arguments["scale"]
            return float(
                -0.5 * standardized**2
                - math.log(arguments["scale"])
                - 0.5 * math.log(2.0 * math.pi)
            )
        if self.family == PriorFamily.HALF_NORMAL:
            if value < 0:
                return -math.inf
            standardized = value / arguments["scale"]
            return float(
                math.log(2.0)
                - 0.5 * standardized**2
                - math.log(arguments["scale"])
                - 0.5 * math.log(2.0 * math.pi)
            )
        if self.family == PriorFamily.BETA:
            if not 0 < value < 1:
                return -math.inf
            alpha = arguments["alpha"]
            beta = arguments["beta"]
            return float(
                (alpha - 1.0) * math.log(value)
                + (beta - 1.0) * math.log1p(-value)
                - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
            )
        if not arguments["lower"] <= value <= arguments["upper"]:
            return -math.inf
        return float(-math.log(arguments["upper"] - arguments["lower"]))

    def grad_log_prob(self, value: float) -> float:
        """Derivative of the scalar log density on its supported natural scale."""

        value = _finite_float(value, "prior value")
        arguments = self.arguments
        if self.family == PriorFamily.NORMAL:
            return float(-(value - arguments["location"]) / arguments["scale"] ** 2)
        if self.family == PriorFamily.HALF_NORMAL:
            if value < 0:
                raise ParameterSpaceError("half-normal gradient is undefined outside support")
            return float(-value / arguments["scale"] ** 2)
        if self.family == PriorFamily.BETA:
            if not 0 < value < 1:
                raise ParameterSpaceError("beta gradient is undefined outside support")
            return float(
                (arguments["alpha"] - 1.0) / value - (arguments["beta"] - 1.0) / (1.0 - value)
            )
        if not arguments["lower"] <= value <= arguments["upper"]:
            raise ParameterSpaceError("uniform gradient is undefined outside support")
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a portable prior record."""

        return {"family": self.family.value, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One scientific parameter and its inference representation.

    ``bounds`` and ``plausible_bounds`` are always expressed on the natural scale.
    ``optimizer_bounds`` are optional numerical safeguards on the transformed scale;
    they do not redefine the scientific domain.
    """

    name: str
    optimizer_name: str | None = None
    role: ParameterRole = ParameterRole.FREE
    transform: ParameterTransform = ParameterTransform.IDENTITY
    bounds: tuple[float | None, float | None] = (None, None)
    plausible_bounds: tuple[float | None, float | None] | None = None
    optimizer_bounds: tuple[float | None, float | None] | None = None
    fixed_value: float | None = None
    prior: PriorSpec | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ParameterSpaceError("parameter name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ParameterSpaceError("parameter description must be a string")
        try:
            role = ParameterRole(self.role)
            transform = ParameterTransform(self.transform)
        except ValueError as error:
            raise ParameterSpaceError("invalid parameter role or transform") from error
        bounds = _interval(self.bounds, "bounds")
        plausible = (
            None
            if self.plausible_bounds is None
            else _interval(self.plausible_bounds, "plausible_bounds")
        )
        optimizer_bounds = (
            None
            if self.optimizer_bounds is None
            else _interval(self.optimizer_bounds, "optimizer_bounds")
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "plausible_bounds", plausible)
        object.__setattr__(self, "optimizer_bounds", optimizer_bounds)
        if role == ParameterRole.FIXED:
            if self.optimizer_name is not None:
                raise ParameterSpaceError("fixed parameters cannot have an optimizer_name")
            if self.fixed_value is None:
                raise ParameterSpaceError("fixed parameters require fixed_value")
        else:
            if self.fixed_value is not None:
                raise ParameterSpaceError("free parameters cannot have fixed_value")
            if self.optimizer_name is not None and (
                not isinstance(self.optimizer_name, str) or not self.optimizer_name
            ):
                raise ParameterSpaceError("optimizer_name must be a non-empty string or null")
        if transform == ParameterTransform.LOG:
            if bounds[0] is None or bounds[1] is not None:
                raise ParameterSpaceError(
                    "log transforms require a finite lower and no upper bound"
                )
        elif transform == ParameterTransform.BOUNDED_LOGIT and (
            bounds[0] is None or bounds[1] is None
        ):
            raise ParameterSpaceError("bounded-logit transforms require two finite bounds")
        if plausible is not None:
            _require_interval_inside(plausible, bounds, "plausible_bounds")
            for value in plausible:
                if value is not None:
                    self._forward(value)
        if optimizer_bounds is not None and plausible is not None:
            transformed = tuple(
                None if value is None else self._forward(value) for value in plausible
            )
            _require_interval_inside(transformed, optimizer_bounds, "transformed plausible_bounds")
        fixed_value = (
            None if self.fixed_value is None else _finite_float(self.fixed_value, "fixed_value")
        )
        if fixed_value is not None:
            self._forward(fixed_value)
        if self.prior is not None and not isinstance(self.prior, PriorSpec):
            raise ParameterSpaceError("prior must be a PriorSpec or null")
        object.__setattr__(self, "fixed_value", fixed_value)

    @property
    def resolved_optimizer_name(self) -> str | None:
        """Optimizer coordinate name, or null for a fixed parameter."""

        if self.role == ParameterRole.FIXED:
            return None
        return self.name if self.optimizer_name is None else self.optimizer_name

    def to_optimizer(self, natural_value: float) -> float:
        """Encode one valid natural value without applying numerical fit bounds."""

        return self._forward(_finite_float(natural_value, self.name))

    def to_natural(self, optimizer_value: float) -> float:
        """Decode one finite optimizer value into the scientific coordinate."""

        optimizer_value = _finite_float(optimizer_value, self.resolved_optimizer_name or self.name)
        lower, upper = self.bounds
        try:
            if self.transform == ParameterTransform.IDENTITY:
                natural = optimizer_value
            elif self.transform == ParameterTransform.LOG:
                natural = float(lower + math.exp(optimizer_value))
            else:
                natural = float(lower + (upper - lower) * expit(optimizer_value))
        except OverflowError:
            raise ParameterSpaceError(f"decoded {self.name!r} is not finite") from None
        if not np.isfinite(natural):
            raise ParameterSpaceError(f"decoded {self.name!r} is not finite")
        self._validate_natural(natural)
        return natural

    def log_abs_det_inverse_jacobian(self, optimizer_value: float) -> float:
        """Log absolute derivative of natural with respect to optimizer coordinate."""

        value = _finite_float(optimizer_value, self.resolved_optimizer_name or self.name)
        if self.transform == ParameterTransform.IDENTITY:
            return 0.0
        if self.transform == ParameterTransform.LOG:
            return value
        lower, upper = self.bounds
        return float(math.log(upper - lower) - np.logaddexp(0.0, -value) - np.logaddexp(0.0, value))

    def inverse_derivative(self, optimizer_value: float) -> float:
        """Derivative of the natural coordinate with respect to optimizer coordinate."""

        value = _finite_float(optimizer_value, self.resolved_optimizer_name or self.name)
        if self.transform == ParameterTransform.IDENTITY:
            return 1.0
        if self.transform == ParameterTransform.LOG:
            try:
                derivative = math.exp(value)
            except OverflowError:
                raise ParameterSpaceError(f"derivative for {self.name!r} is not finite") from None
            if not np.isfinite(derivative):
                raise ParameterSpaceError(f"derivative for {self.name!r} is not finite")
            return float(derivative)
        lower, upper = self.bounds
        probability = float(expit(value))
        return float((upper - lower) * probability * (1.0 - probability))

    def grad_log_abs_det_inverse_jacobian(self, optimizer_value: float) -> float:
        """Derivative of the log inverse-transform Jacobian."""

        value = _finite_float(optimizer_value, self.resolved_optimizer_name or self.name)
        if self.transform == ParameterTransform.IDENTITY:
            return 0.0
        if self.transform == ParameterTransform.LOG:
            return 1.0
        return float(1.0 - 2.0 * expit(value))

    def transformed_bounds(self) -> tuple[float | None, float | None]:
        """Return numerical optimizer bounds, falling back to transformed hard bounds."""

        if self.optimizer_bounds is not None:
            return self.optimizer_bounds
        if self.transform in (ParameterTransform.LOG, ParameterTransform.BOUNDED_LOGIT):
            return (None, None)
        return self.bounds

    def transformed_plausible_bounds(self) -> tuple[float | None, float | None] | None:
        """Return plausible bounds in the optimizer coordinate when declared."""

        if self.plausible_bounds is None:
            return None
        return tuple(
            None if value is None else self._forward(value) for value in self.plausible_bounds
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a portable parameter record."""

        return {
            "name": self.name,
            "optimizer_name": self.resolved_optimizer_name,
            "role": self.role.value,
            "transform": self.transform.value,
            "bounds": list(self.bounds),
            "plausible_bounds": (
                None if self.plausible_bounds is None else list(self.plausible_bounds)
            ),
            "optimizer_bounds": (
                None if self.optimizer_bounds is None else list(self.optimizer_bounds)
            ),
            "fixed_value": self.fixed_value,
            "prior": None if self.prior is None else self.prior.to_dict(),
            "description": self.description,
        }

    def _forward(self, value: float) -> float:
        value = _finite_float(value, self.name)
        self._validate_natural(value)
        lower, upper = self.bounds
        if self.transform == ParameterTransform.IDENTITY:
            return value
        if self.transform == ParameterTransform.LOG:
            return float(math.log(value - lower))
        return float(math.log(value - lower) - math.log(upper - value))

    def _validate_natural(self, value: float) -> None:
        lower, upper = self.bounds
        lower_invalid = lower is not None and (
            value <= lower
            if self.transform in (ParameterTransform.LOG, ParameterTransform.BOUNDED_LOGIT)
            else value < lower
        )
        upper_invalid = upper is not None and (
            value >= upper if self.transform == ParameterTransform.BOUNDED_LOGIT else value > upper
        )
        if lower_invalid or upper_invalid:
            qualifier = (
                "strictly inside" if self.transform != ParameterTransform.IDENTITY else "inside"
            )
            raise ParameterSpaceError(
                f"parameter {self.name!r} must lie {qualifier} natural bounds {self.bounds!r}"
            )


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Ordered, content-addressed parameter semantics shared by inference backends."""

    parameters: tuple[ParameterSpec, ...]
    schema_version: str = PARAMETER_SPACE_SCHEMA

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if self.schema_version != PARAMETER_SPACE_SCHEMA:
            raise ParameterSpaceError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {PARAMETER_SPACE_SCHEMA!r}"
            )
        if not parameters or any(not isinstance(item, ParameterSpec) for item in parameters):
            raise ParameterSpaceError("parameters must contain at least one ParameterSpec")
        natural_names = [item.name for item in parameters]
        optimizer_names = [
            item.resolved_optimizer_name for item in parameters if item.role == ParameterRole.FREE
        ]
        if len(set(natural_names)) != len(natural_names):
            raise ParameterSpaceError("natural parameter names must be unique")
        if len(set(optimizer_names)) != len(optimizer_names):
            raise ParameterSpaceError("optimizer parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """All scientific coordinates, including fixed values."""

        return tuple(item.name for item in self.parameters)

    @property
    def free_names(self) -> tuple[str, ...]:
        """Scientific coordinates estimated by an inference backend."""

        return tuple(item.name for item in self.parameters if item.role == ParameterRole.FREE)

    @property
    def optimizer_names(self) -> tuple[str, ...]:
        """Ordered transformed coordinates presented to an optimizer."""

        return tuple(
            item.resolved_optimizer_name
            for item in self.parameters
            if item.role == ParameterRole.FREE
        )

    @property
    def optimizer_bounds(self) -> tuple[tuple[float | None, float | None], ...]:
        """Ordered hard or numerical bounds for optimizer backends."""

        return tuple(
            item.transformed_bounds() for item in self.parameters if item.role == ParameterRole.FREE
        )

    @property
    def optimizer_plausible_bounds(
        self,
    ) -> tuple[tuple[float | None, float | None] | None, ...]:
        """Ordered transformed plausible bounds, notably for PyBADS."""

        return tuple(
            item.transformed_plausible_bounds()
            for item in self.parameters
            if item.role == ParameterRole.FREE
        )

    def encode(self, natural: Mapping[str, float]) -> NDArray[np.float64]:
        """Encode an exact natural mapping into the ordered free optimizer vector."""

        values = self._natural_mapping(natural)
        encoded: list[float] = []
        for item in self.parameters:
            value = values[item.name]
            if item.role == ParameterRole.FIXED:
                if not np.isclose(value, item.fixed_value, rtol=0.0, atol=1e-12):
                    raise ParameterSpaceError(
                        f"fixed parameter {item.name!r} must equal {item.fixed_value!r}"
                    )
            else:
                encoded.append(item.to_optimizer(value))
        return _protected_vector(encoded)

    def encode_mapping(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode an exact natural mapping keyed by optimizer coordinate name."""

        vector = self.encode(natural)
        return MappingProxyType(dict(zip(self.optimizer_names, vector.tolist(), strict=True)))

    def decode(self, optimizer: Sequence[float] | NDArray[np.floating[Any]]) -> Mapping[str, float]:
        """Decode an ordered free optimizer vector and restore fixed parameters."""

        try:
            vector = np.asarray(optimizer, dtype=np.float64)
        except (TypeError, ValueError):
            raise ParameterSpaceError("optimizer values must be finite numeric values") from None
        if vector.shape != (len(self.optimizer_names),) or not np.all(np.isfinite(vector)):
            raise ParameterSpaceError(
                f"optimizer vector must contain {len(self.optimizer_names)} finite values"
            )
        decoded: dict[str, float] = {}
        free_index = 0
        for item in self.parameters:
            if item.role == ParameterRole.FIXED:
                decoded[item.name] = float(item.fixed_value)
            else:
                decoded[item.name] = item.to_natural(float(vector[free_index]))
                free_index += 1
        return MappingProxyType(decoded)

    def decode_mapping(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        """Decode an exact optimizer-coordinate mapping into natural parameters."""

        if not isinstance(optimizer, Mapping):
            raise TypeError("optimizer must be a mapping")
        expected = set(self.optimizer_names)
        observed = set(optimizer)
        if observed != expected:
            raise ParameterSpaceError(
                "optimizer parameters must match exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        return self.decode([optimizer[name] for name in self.optimizer_names])

    def log_prior(self, natural: Mapping[str, float], *, require_all: bool = False) -> float:
        """Sum declared natural-scale priors without a transform Jacobian."""

        values = self._natural_mapping(natural)
        total = 0.0
        for item in self.parameters:
            item.to_optimizer(values[item.name])
            if item.prior is None:
                if require_all and item.role == ParameterRole.FREE:
                    raise ParameterSpaceError(f"free parameter {item.name!r} has no prior")
                continue
            total += item.prior.log_prob(values[item.name])
        return float(total)

    def log_abs_det_inverse_jacobian(
        self, optimizer: Sequence[float] | NDArray[np.floating[Any]]
    ) -> float:
        """Sum transform Jacobians for density conversion into optimizer space."""

        vector = np.asarray(optimizer, dtype=np.float64)
        if vector.shape != (len(self.optimizer_names),) or not np.all(np.isfinite(vector)):
            raise ParameterSpaceError(
                f"optimizer vector must contain {len(self.optimizer_names)} finite values"
            )
        return float(
            sum(
                item.log_abs_det_inverse_jacobian(float(value))
                for item, value in zip(
                    (item for item in self.parameters if item.role == ParameterRole.FREE),
                    vector,
                    strict=True,
                )
            )
        )

    def grad_log_prior_optimizer(
        self,
        optimizer: Sequence[float] | NDArray[np.floating[Any]],
        *,
        require_all: bool = False,
    ) -> NDArray[np.float64]:
        """Gradient of natural-scale priors with respect to optimizer coordinates."""

        vector = self._optimizer_vector(optimizer)
        natural = self.decode(vector)
        gradient: list[float] = []
        free_index = 0
        for item in self.parameters:
            if item.role == ParameterRole.FIXED:
                continue
            if item.prior is None:
                if require_all:
                    raise ParameterSpaceError(f"free parameter {item.name!r} has no prior")
                gradient.append(0.0)
            else:
                gradient.append(
                    item.prior.grad_log_prob(natural[item.name])
                    * item.inverse_derivative(float(vector[free_index]))
                )
            free_index += 1
        return _protected_vector(gradient)

    def grad_log_abs_det_inverse_jacobian(
        self, optimizer: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Gradient of the summed inverse-transform log Jacobian."""

        vector = self._optimizer_vector(optimizer)
        return _protected_vector(
            [
                item.grad_log_abs_det_inverse_jacobian(float(value))
                for item, value in zip(
                    (item for item in self.parameters if item.role == ParameterRole.FREE),
                    vector,
                    strict=True,
                )
            ]
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 content address of the complete parameter semantics."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a standards-compliant JSON representation."""

        return {
            "schema_version": self.schema_version,
            "parameters": [item.to_dict() for item in self.parameters],
        }

    def canonical_json(self) -> str:
        """Serialize deterministically without non-finite JSON constants."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _natural_mapping(self, natural: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(natural, Mapping):
            raise TypeError("natural parameters must be a mapping")
        expected = set(self.natural_names)
        observed = set(natural)
        if observed != expected:
            raise ParameterSpaceError(
                "natural parameters must match exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            return {name: _finite_float(natural[name], name) for name in self.natural_names}
        except (TypeError, ValueError):
            raise ParameterSpaceError("natural parameters must contain finite values") from None

    def _optimizer_vector(
        self, optimizer: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(optimizer, dtype=np.float64)
        except (TypeError, ValueError):
            raise ParameterSpaceError("optimizer values must be finite numeric values") from None
        if vector.shape != (len(self.optimizer_names),) or not np.all(np.isfinite(vector)):
            raise ParameterSpaceError(
                f"optimizer vector must contain {len(self.optimizer_names)} finite values"
            )
        return vector


@runtime_checkable
class ParameterSpaceProvider(Protocol):
    """Object exposing stable parameter semantics to inference and interchange code."""

    @property
    def parameter_space(self) -> ParameterSpace: ...


def parameter_space_from_dict(payload: Mapping[str, Any]) -> ParameterSpace:
    """Validate and restore a decoded parameter-space record."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if set(payload) != {"schema_version", "parameters"}:
        raise ParameterSpaceError("parameter-space fields must be schema_version and parameters")
    records = payload["parameters"]
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ParameterSpaceError("parameters must be an array")
    parameters: list[ParameterSpec] = []
    expected = {
        "name",
        "optimizer_name",
        "role",
        "transform",
        "bounds",
        "plausible_bounds",
        "optimizer_bounds",
        "fixed_value",
        "prior",
        "description",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ParameterSpaceError("every parameter record must contain the canonical fields")
        prior_record = record["prior"]
        prior = None
        if prior_record is not None:
            if not isinstance(prior_record, Mapping) or set(prior_record) != {
                "family",
                "arguments",
            }:
                raise ParameterSpaceError("prior must contain family and arguments")
            prior = PriorSpec(prior_record["family"], prior_record["arguments"])
        optimizer_name = record["optimizer_name"]
        if record["role"] == ParameterRole.FREE.value and optimizer_name == record["name"]:
            optimizer_name = None
        parameters.append(
            ParameterSpec(
                name=record["name"],
                optimizer_name=optimizer_name,
                role=record["role"],
                transform=record["transform"],
                bounds=_interval(record["bounds"], "bounds"),
                plausible_bounds=(
                    None
                    if record["plausible_bounds"] is None
                    else _interval(record["plausible_bounds"], "plausible_bounds")
                ),
                optimizer_bounds=(
                    None
                    if record["optimizer_bounds"] is None
                    else _interval(record["optimizer_bounds"], "optimizer_bounds")
                ),
                fixed_value=record["fixed_value"],
                prior=prior,
                description=record["description"],
            )
        )
    return ParameterSpace(tuple(parameters), schema_version=payload["schema_version"])


def parameter_space_from_json(payload: str) -> ParameterSpace:
    """Decode and validate a parameter space from JSON text."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ParameterSpaceError("payload is not valid JSON") from error
    return parameter_space_from_dict(decoded)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ParameterSpaceError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ParameterSpaceError(f"{label} must be a finite number") from None
    if not np.isfinite(result):
        raise ParameterSpaceError(f"{label} must be a finite number")
    return result


def _interval(value: Sequence[float | None], label: str) -> tuple[float | None, float | None]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ParameterSpaceError(f"{label} must contain lower and upper values")
    lower = None if value[0] is None else _finite_float(value[0], f"{label} lower")
    upper = None if value[1] is None else _finite_float(value[1], f"{label} upper")
    if lower is not None and upper is not None and lower >= upper:
        raise ParameterSpaceError(f"{label} lower must be below upper")
    return (lower, upper)


def _require_interval_inside(
    inner: tuple[float | None, float | None],
    outer: tuple[float | None, float | None],
    label: str,
) -> None:
    if inner[0] is not None and outer[0] is not None and inner[0] < outer[0]:
        raise ParameterSpaceError(f"{label} must lie inside bounds")
    if inner[1] is not None and outer[1] is not None and inner[1] > outer[1]:
        raise ParameterSpaceError(f"{label} must lie inside bounds")


def _protected_vector(values: Sequence[float]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result
