"""The registry that actually resolves a declared implementation into an estimator.

Why this exists
---------------
A frozen protocol declares each candidate as an ``implementation`` *string*. Turning that
string into an object is the one place where data becomes code, and it must never be done
by importing the string: a protocol is untrusted data, and ``importlib.import_module`` on
an attacker-supplied path is arbitrary code execution. The resolution therefore goes
through an explicit allowlist.

This module *is* that allowlist. It used to be documented as the extension point while a
hard-coded ``_MODELS`` dict inside :mod:`behavio.cli` did the actual resolving, so the
public registry and the running one were two different objects and only one of them
could be extended. :func:`builtin_estimator_registry` now returns the registry the command
line resolves through, and a caller who wants their own model in a protocol run builds a
registry, registers it, and passes it in.

What a registration buys
------------------------
A registration is not only a factory. It also declares what the factory *produces*, which
is what lets :func:`behavio.protocol.runner.verify_candidate_declarations` decide -- rather than
shrug at -- whether the object handed to the runner is the object the protocol froze.
Without a registry that check can only compare class names and consult already-imported
modules, so it reports ``unverifiable`` for anything it has not seen; with one, a declared
implementation the registry knows is always either verified or contradicted.

Composition
-----------
A protocol candidate is one implementation name and a flat list of scalar settings, which
cannot spell a nested constructor call. It can spell a *reference*: the ``base`` setting
names another registered implementation and every setting prefixed ``base.`` configures
it, recursively. That is what lets a frozen protocol declare
``hierarchical(smooth(BernoulliHistoryGLM(...)))`` without the registry regrowing one entry
per composition.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from behavio.compose import HierarchicalModel, SmoothModel, hierarchical, smooth
from behavio.models import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BinaryQLearning,
    WienerDriftDiffusion,
)
from behavio.models.base import BehaviourEstimator

EstimatorFactory = Callable[..., BehaviourEstimator]

BASE_SETTING = "base"
"""The hyperparameter naming the model a combinator candidate wraps."""

_BASE_PREFIX = f"{BASE_SETTING}."


class RegistryError(ValueError):
    """Raised when an estimator registration or lookup is invalid."""


@dataclass(frozen=True, slots=True)
class EstimatorRegistration:
    """One named estimator factory, what it produces, and its package provenance.

    ``name`` is the string a protocol declares. For a first-party model that is its public
    import path (``behavio.models.BernoulliHistoryGLM``); for an extension it is whatever
    the extension chooses, as long as protocols and registry agree.

    ``produces`` is the class the factory returns. Declaring it is what makes a candidate
    declaration decidable: the runner can check ``isinstance(model, produces)`` without
    importing anything. ``model_name`` optionally pins the stable model name the factory's
    output must report, which catches a factory whose identity has drifted from the
    registration it was made under.

    ``base_attribute`` marks a combinator: a factory whose first positional argument is
    another model, resolved from the ``base`` setting. Its value names the attribute the
    constructed object exposes that wrapped model under, so declaration verification can
    reach it. Both built-in combinators call it ``model``; an extension that calls it
    ``inner`` says so here rather than being silently treated as a non-combinator.
    """

    name: str
    factory: EstimatorFactory
    provider: str
    version: str
    produces: type[Any] | None = None
    model_name: str | None = None
    base_attribute: str | None = None

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
        if self.produces is not None and not isinstance(self.produces, type):
            raise RegistryError("registration produces must be a class")
        if self.model_name is not None and not self.model_name:
            raise RegistryError("registration model_name must be null or non-empty")
        if self.base_attribute is not None and not self.base_attribute:
            raise RegistryError("registration base_attribute must be null or non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return non-executable registration provenance."""

        return {
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "produces": None if self.produces is None else self.produces.__qualname__,
            "model_name": self.model_name,
            "base_attribute": self.base_attribute,
        }


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

    def __contains__(self, name: object) -> bool:
        return name in self._registrations

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
        produces: type[Any] | None = None,
        model_name: str | None = None,
        base_attribute: str | None = None,
    ) -> None:
        """Construct and register one named estimator factory."""

        self.register(
            EstimatorRegistration(
                name,
                factory,
                provider,
                version,
                produces=produces,
                model_name=model_name,
                base_attribute=base_attribute,
            )
        )

    def registration_for(self, name: str) -> EstimatorRegistration:
        """Return one registration, naming the alternatives when there is no match."""

        try:
            return self._registrations[name]
        except KeyError:
            raise RegistryError(
                f"unknown estimator {name!r}; registered names are {self.names!r}"
            ) from None

    def create(self, name: str, settings: Mapping[str, Any] | None = None) -> BehaviourEstimator:
        """Create and validate an estimator from explicit JSON-like configuration.

        ``settings`` is the flat scalar mapping a protocol candidate declares. A ``base``
        entry naming another registered implementation is resolved first, recursively, and
        the resulting model is passed to this factory as its first positional argument.
        """

        registration = self.registration_for(name)
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        model = self._build(registration, dict(settings))
        if registration.produces is not None and not isinstance(model, registration.produces):
            raise RegistryError(
                f"factory registered as {registration.name!r} returned "
                f"{type(model).__qualname__!r}, not {registration.produces.__qualname__!r}"
            )
        if registration.model_name is not None and model.model_name != registration.model_name:
            raise RegistryError(
                f"factory registered as {registration.name!r} returned "
                f"model_name {model.model_name!r}"
            )
        return model

    def _build(
        self,
        registration: EstimatorRegistration,
        settings: Mapping[str, Any],
    ) -> BehaviourEstimator:
        nested = {
            key[len(_BASE_PREFIX) :]: value
            for key, value in settings.items()
            if key.startswith(_BASE_PREFIX)
        }
        direct = {
            key: value
            for key, value in settings.items()
            if key != BASE_SETTING and not key.startswith(_BASE_PREFIX)
        }
        if BASE_SETTING not in settings:
            if nested:
                raise RegistryError(
                    f"{BASE_SETTING}.* settings need a {BASE_SETTING} implementation"
                )
            return registration.factory(**direct)
        reference = settings[BASE_SETTING]
        if not isinstance(reference, str):
            raise RegistryError(f"candidate {BASE_SETTING} must name an implementation")
        base = self.create(reference, nested)
        return registration.factory(base, **direct)

    def verify(self, name: str, model: Any) -> bool | None:
        """Whether ``model`` is what ``name`` resolves to, or ``None`` if undecidable.

        ``None`` means this registry cannot speak to the question -- the name is not
        registered, or the registration declined to declare what it produces. It never
        means "probably fine".
        """

        registration = self._registrations.get(name)
        if registration is None or registration.produces is None:
            return None
        return isinstance(model, registration.produces)

    def base_of(self, name: str, model: Any) -> Any | None:
        """Return the model a combinator candidate wraps, when there is one.

        Used to decide ``base.``-prefixed hyperparameter declarations, which name fields of
        the wrapped model rather than of the candidate itself. The attribute is read from
        the registration rather than guessed, so a combinator that does not happen to call
        its wrapped model ``model`` is still decidable.
        """

        registration = self._registrations.get(name)
        if registration is None or registration.base_attribute is None:
            return None
        return getattr(model, registration.base_attribute, None)

    def manifest(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic non-executable package provenance for all entries."""

        return tuple(self._registrations[name].to_dict() for name in self.names)


def builtin_estimator_registry() -> EstimatorRegistry:
    """Return a fresh registry holding every implementation the package ships.

    This is the registry :mod:`behavio.cli` resolves protocol candidates through, and the
    default one :func:`behavio.protocol.runner.verify_candidate_declarations` checks declarations
    against. It is constructed per call rather than shared, so an extension that registers
    into it cannot leak into an unrelated analysis in the same process.
    """

    registry = EstimatorRegistry()
    provider = "behavio"
    version = _package_version()
    for model_type in (
        BernoulliHistoryGLM,
        BernoulliGLMHMM,
        BinaryQLearning,
        WienerDriftDiffusion,
    ):
        registry.add(
            f"behavio.models.{model_type.__name__}",
            model_type,
            provider=provider,
            version=version,
            produces=model_type,
        )
    registry.add(
        "behavio.compose.smooth",
        smooth,
        provider=provider,
        version=version,
        produces=SmoothModel,
        base_attribute="model",
    )
    registry.add(
        "behavio.compose.hierarchical",
        hierarchical,
        provider=provider,
        version=version,
        produces=HierarchicalModel,
        base_attribute="model",
    )
    return registry


def _package_version() -> str:
    """Read the installed distribution version without importing the package root."""

    try:
        return importlib.metadata.version("behavio")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        return "source-tree"
