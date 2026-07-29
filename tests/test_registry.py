import pytest

from behavio import (
    BernoulliHistoryGLM,
    EstimatorRegistration,
    EstimatorRegistry,
    RegistryError,
)


def glm_factory(config):
    return BernoulliHistoryGLM(**config)


def test_registry_constructs_validated_external_estimators_and_manifests_provenance() -> None:
    registry = EstimatorRegistry()
    registry.add(
        "bernoulli-history-glm",
        glm_factory,
        provider="example-extension",
        version="2.1.0",
    )

    model = registry.create(
        "bernoulli-history-glm",
        {"covariates": ("stimulus",), "choice_lags": 0},
    )

    assert model.model_name == "bernoulli-history-glm"
    assert registry.names == ("bernoulli-history-glm",)
    assert registry.manifest() == (
        {
            "name": "bernoulli-history-glm",
            "provider": "example-extension",
            "version": "2.1.0",
        },
    )


def test_registry_rejects_duplicate_and_unknown_names() -> None:
    registration = EstimatorRegistration(
        "bernoulli-history-glm",
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
    registry.add("wrong-name", glm_factory, provider="extension", version="1")

    with pytest.raises(RegistryError, match="returned model_name"):
        registry.create("wrong-name")
