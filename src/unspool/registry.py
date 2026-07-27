"""Instance-scoped extension registry for behavioural estimators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from unspool.models.base import BehaviourEstimator, model_capabilities

EstimatorFactory = Callable[[Mapping[str, Any]], BehaviourEstimator]


class RegistryError(ValueError):
    """Raised when an estimator registration or lookup is invalid."""


@dataclass(frozen=True, slots=True)
class EstimatorRegistration:
    """One named estimator factory and its package provenance."""

    name: str
    factory: EstimatorFactory
    provider: str
    version: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.name, "name"),
            (self.provider, "provider"),
            (self.version, "version"),
        ):
            if not isinstance(value, str) or not value:
                raise RegistryError(f"registration {label} must be a non-empty string")
        if not callable(self.factory):
            raise RegistryError("registration factory must be callable")

    def to_dict(self) -> dict[str, str]:
        """Return non-executable registration provenance."""

        return {"name": self.name, "provider": self.provider, "version": self.version}


class EstimatorRegistry:
    """Explicit local registry for core or third-party estimator factories.

    Registries are not global singletons: callers decide which implementations are in
    scope for an analysis and can serialize the resulting non-executable manifest.
    """

    __slots__ = ("_registrations",)

    def __init__(self, registrations: tuple[EstimatorRegistration, ...] = ()) -> None:
        self._registrations: dict[str, EstimatorRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, EstimatorRegistration):
                raise TypeError("registrations must contain EstimatorRegistration values")
            self.register(registration)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered names in deterministic sorted order."""

        return tuple(sorted(self._registrations))

    @property
    def registrations(self) -> Mapping[str, EstimatorRegistration]:
        """Read-only view of the exact active registrations."""

        return MappingProxyType(dict(self._registrations))

    def register(self, registration: EstimatorRegistration) -> None:
        """Add one registration, rejecting accidental replacement."""

        if not isinstance(registration, EstimatorRegistration):
            raise TypeError("registration must be an EstimatorRegistration")
        if registration.name in self._registrations:
            raise RegistryError(f"estimator {registration.name!r} is already registered")
        self._registrations[registration.name] = registration

    def add(
        self,
        name: str,
        factory: EstimatorFactory,
        *,
        provider: str,
        version: str,
    ) -> None:
        """Construct and register one named estimator factory."""

        self.register(EstimatorRegistration(name, factory, provider, version))

    def create(self, name: str, config: Mapping[str, Any] | None = None) -> BehaviourEstimator:
        """Create and validate an estimator from explicit JSON-like configuration."""

        try:
            registration = self._registrations[name]
        except KeyError:
            raise RegistryError(
                f"unknown estimator {name!r}; registered names are {self.names!r}"
            ) from None
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        model = registration.factory(MappingProxyType(dict(config)))
        model_capabilities(model)
        if model.model_name != registration.name:
            raise RegistryError(
                f"factory registered as {registration.name!r} returned "
                f"model_name {model.model_name!r}"
            )
        return model

    def manifest(self) -> tuple[dict[str, str], ...]:
        """Return deterministic non-executable package provenance for all entries."""

        return tuple(self._registrations[name].to_dict() for name in self.names)
