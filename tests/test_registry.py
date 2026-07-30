"""Tests for the registry that resolves declared implementations into estimators."""

import pytest

from behavio import BernoulliHistoryGLM, EstimatorRegistry, builtin_estimator_registry
from behavio.compose import HierarchicalModel, SmoothModel
from behavio.protocol.runner import DeclarationCheck, verify_candidate_declarations
from behavio.protocol.schema import CandidateSpec, Setting
from behavio.registry import EstimatorRegistration, RegistryError


def glm_factory(**config):
    return BernoulliHistoryGLM(**config)


class _Protocol:
    """The one member :func:`verify_candidate_declarations` reads off a protocol."""

    def __init__(self, *candidates: CandidateSpec) -> None:
        self.candidates = candidates


def declared(implementation: str, **settings) -> CandidateSpec:
    return CandidateSpec(
        name="candidate",
        implementation=implementation,
        hyperparameters=tuple(Setting(key, value) for key, value in settings.items()),
        scored_columns=("choice",),
    )


def findings(candidate: CandidateSpec, model, registry=None) -> dict[str, DeclarationCheck]:
    """Subject-to-status for everything the registry decides.

    ``inference`` is asserted here and then dropped: it is decided by the contract the
    supplied object satisfies rather than by any registry, so repeating it in every
    expectation below would say nothing about resolution. Its own behaviour is checked in
    ``tests/test_runner.py``.
    """

    verification = verify_candidate_declarations(
        _Protocol(candidate), {"candidate": model}, registry=registry
    )[0]
    statuses = {finding.subject: finding.status for finding in verification.findings}
    assert statuses.pop("inference") is DeclarationCheck.VERIFIED
    return statuses


def test_registry_constructs_validated_external_estimators_and_manifests_provenance() -> None:
    registry = EstimatorRegistry()
    registry.add(
        "mylab.models.HistoryGLM",
        glm_factory,
        provider="example-extension",
        version="2.1.0",
        produces=BernoulliHistoryGLM,
        model_name="bernoulli-history-glm",
    )

    model = registry.create(
        "mylab.models.HistoryGLM",
        {"predictors": ("stimulus",), "choice_lags": 0},
    )

    assert model.model_name == "bernoulli-history-glm"
    assert registry.names == ("mylab.models.HistoryGLM",)
    assert "mylab.models.HistoryGLM" in registry
    assert registry.manifest() == (
        {
            "name": "mylab.models.HistoryGLM",
            "provider": "example-extension",
            "version": "2.1.0",
            "produces": "BernoulliHistoryGLM",
            "model_name": "bernoulli-history-glm",
            "base_attribute": None,
        },
    )


def test_registry_rejects_duplicate_and_unknown_names() -> None:
    registration = EstimatorRegistration(
        "mylab.models.HistoryGLM",
        glm_factory,
        provider="example-extension",
        version="2.1.0",
    )
    registry = EstimatorRegistry((registration,))

    with pytest.raises(RegistryError, match="already registered"):
        registry.register(registration)
    with pytest.raises(RegistryError, match="unknown estimator"):
        registry.create("missing")


def test_registry_rejects_a_factory_whose_model_identity_drifts() -> None:
    registry = EstimatorRegistry()
    registry.add(
        "mylab.models.Wrong",
        glm_factory,
        provider="extension",
        version="1",
        model_name="not-what-it-returns",
    )

    with pytest.raises(RegistryError, match="returned model_name"):
        registry.create("mylab.models.Wrong")


def test_registry_rejects_a_factory_that_returns_the_wrong_class() -> None:
    registry = EstimatorRegistry()
    registry.add(
        "mylab.models.Wrong",
        glm_factory,
        provider="extension",
        version="1",
        produces=SmoothModel,
    )

    with pytest.raises(RegistryError, match="not 'SmoothModel'"):
        registry.create("mylab.models.Wrong")


def test_the_builtin_registry_resolves_every_implementation_the_cli_accepts() -> None:
    registry = builtin_estimator_registry()

    assert registry.names == (
        "behavio.compose.hierarchical",
        "behavio.compose.smooth",
        "behavio.models.BernoulliGLMHMM",
        "behavio.models.BernoulliHistoryGLM",
        "behavio.models.BinaryQLearning",
        "behavio.models.WienerDriftDiffusion",
    )
    model = registry.create(
        "behavio.models.BernoulliHistoryGLM",
        {"predictors": ("stimulus",), "choice_lags": 0},
    )
    assert isinstance(model, BernoulliHistoryGLM)


def test_the_registry_resolves_a_composed_candidate_from_flat_settings() -> None:
    registry = builtin_estimator_registry()

    model = registry.create(
        "behavio.compose.hierarchical",
        {
            "base": "behavio.compose.smooth",
            "base.base": "behavio.models.BernoulliHistoryGLM",
            "base.base.predictors": ("stimulus",),
            "base.base.choice_lags": 0,
            "base.over": "session_order",
            "base.knots": (0.0, 2.0),
            "over": "subject",
        },
    )

    assert isinstance(model, HierarchicalModel)
    assert isinstance(model.model, SmoothModel)
    assert isinstance(model.model.model, BernoulliHistoryGLM)


def test_base_settings_without_a_base_implementation_are_refused() -> None:
    registry = builtin_estimator_registry()

    with pytest.raises(RegistryError, match="need a base implementation"):
        registry.create("behavio.models.BernoulliHistoryGLM", {"base.predictors": ("stimulus",)})


def test_a_registered_declaration_is_decidable_rather_than_unverifiable() -> None:
    """The point of wiring the registry in: no shrugging about a name it knows."""

    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)

    verified = findings(declared("behavio.models.BernoulliHistoryGLM", choice_lags=0), model)
    contradicted = findings(declared("behavio.models.BinaryQLearning"), model)

    assert verified == {
        "implementation": DeclarationCheck.VERIFIED,
        "hyperparameter:choice_lags": DeclarationCheck.VERIFIED,
    }
    assert contradicted == {"implementation": DeclarationCheck.CONTRADICTED}


def test_a_composed_candidate_verifies_through_the_registry() -> None:
    """Before the registry this was reported as a contradiction and refused the run."""

    from behavio.compose import hierarchical

    model = hierarchical(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0), over="subject"
    )

    statuses = findings(
        declared(
            "behavio.compose.hierarchical",
            base="behavio.models.BernoulliHistoryGLM",
            **{"base.choice_lags": 0},
        ),
        model,
    )

    assert statuses == {
        "implementation": DeclarationCheck.VERIFIED,
        "hyperparameter:base": DeclarationCheck.VERIFIED,
        "hyperparameter:base.choice_lags": DeclarationCheck.VERIFIED,
    }


def test_a_composed_candidate_wrapping_the_wrong_base_is_contradicted() -> None:
    from behavio.compose import hierarchical

    model = hierarchical(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0), over="subject"
    )

    statuses = findings(
        declared(
            "behavio.compose.hierarchical",
            base="behavio.models.BinaryQLearning",
            **{"base.choice_lags": 3},
        ),
        model,
    )

    assert statuses == {
        "implementation": DeclarationCheck.VERIFIED,
        "hyperparameter:base": DeclarationCheck.CONTRADICTED,
        "hyperparameter:base.choice_lags": DeclarationCheck.CONTRADICTED,
    }


def test_an_unregistered_declaration_still_falls_back_to_the_import_free_check() -> None:
    """A registry that has never heard of a name must not manufacture a verdict."""

    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)

    statuses = findings(
        declared("some.unimported.package.BernoulliHistoryGLM"),
        model,
        registry=EstimatorRegistry(),
    )

    assert statuses == {"implementation": DeclarationCheck.UNVERIFIABLE}


def test_an_extension_registry_makes_its_own_model_decidable() -> None:
    registry = EstimatorRegistry()
    registry.add(
        "mylab.models.HistoryGLM",
        glm_factory,
        provider="example-extension",
        version="2.1.0",
        produces=BernoulliHistoryGLM,
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)

    assert findings(declared("mylab.models.HistoryGLM"), model, registry=registry) == {
        "implementation": DeclarationCheck.VERIFIED
    }
