import math
from types import MappingProxyType

import numpy as np
import pytest

from behavio import (
    BinaryQLearning,
    ChoiceSpec,
    ParameterSpace,
    RewardSpec,
    Study,
    TaskSpec,
    export_fit,
)
from behavio.inference import (
    PARAMETER_SPACE_SCHEMA,
    ParameterRole,
    ParameterSpaceError,
    ParameterSpaceProvider,
    ParameterSpec,
    ParameterTransform,
    PriorSpec,
    parameter_space_from_json,
)
from behavio.models import FitDiagnostics, FitResult
from behavio.task import FittedModel


def example_space() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterSpec(
                name="learning_rate",
                optimizer_name="learning_rate_logit",
                transform=ParameterTransform.BOUNDED_LOGIT,
                bounds=(0.0, 1.0),
                plausible_bounds=(0.05, 0.95),
                prior=PriorSpec.beta(2.0, 3.0),
            ),
            ParameterSpec(
                name="scale",
                optimizer_name="scale_log",
                transform=ParameterTransform.LOG,
                bounds=(0.0, None),
                plausible_bounds=(0.1, 10.0),
                prior=PriorSpec.half_normal(2.0),
            ),
            ParameterSpec(
                name="bias",
                bounds=(-5.0, 5.0),
                plausible_bounds=(-2.0, 2.0),
                prior=PriorSpec.normal(0.0, 1.0),
            ),
            ParameterSpec(
                name="lapse",
                role=ParameterRole.FIXED,
                bounds=(0.0, 0.2),
                fixed_value=0.01,
                prior=PriorSpec.uniform(0.0, 0.2),
            ),
        )
    )


def natural_values() -> dict[str, float]:
    return {"learning_rate": 0.25, "scale": 2.0, "bias": -0.5, "lapse": 0.01}


def test_parameter_space_round_trips_natural_optimizer_and_fixed_coordinates() -> None:
    space = example_space()

    encoded = space.encode(natural_values())
    decoded = space.decode(encoded)

    assert space.natural_names == ("learning_rate", "scale", "bias", "lapse")
    assert space.free_names == ("learning_rate", "scale", "bias")
    assert space.optimizer_names == ("learning_rate_logit", "scale_log", "bias")
    assert isinstance(decoded, MappingProxyType)
    assert decoded == pytest.approx(natural_values())
    assert not encoded.flags.writeable
    assert space.encode_mapping(natural_values())["scale_log"] == pytest.approx(math.log(2.0))
    assert space.decode_mapping(space.encode_mapping(natural_values())) == pytest.approx(
        natural_values()
    )


def test_bounds_priors_and_density_jacobians_retain_coordinate_meaning() -> None:
    space = example_space()
    encoded = space.encode(natural_values())

    expected_prior = sum(
        spec.prior.log_prob(natural_values()[spec.name])
        for spec in space.parameters
        if spec.prior is not None
    )
    expected_jacobian = math.log(0.25 * 0.75) + math.log(2.0) + 0.0

    assert space.log_prior(natural_values(), require_all=True) == pytest.approx(expected_prior)
    assert space.log_abs_det_inverse_jacobian(encoded) == pytest.approx(expected_jacobian)
    assert space.optimizer_plausible_bounds[0] == pytest.approx(
        (math.log(0.05 / 0.95), math.log(0.95 / 0.05))
    )
    assert PriorSpec.half_normal(1.0).log_prob(-0.1) == -math.inf
    assert PriorSpec.beta(1.0, 1.0).log_prob(0.0) == -math.inf


def test_prior_and_jacobian_gradients_match_optimizer_finite_differences() -> None:
    space = example_space()
    vector = space.encode(natural_values())
    analytic_prior = space.grad_log_prior_optimizer(vector, require_all=True)
    analytic_jacobian = space.grad_log_abs_det_inverse_jacobian(vector)
    numeric_prior = np.empty_like(vector)
    numeric_jacobian = np.empty_like(vector)
    for index in range(len(vector)):
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += 1e-6
        negative[index] -= 1e-6
        numeric_prior[index] = (
            space.log_prior(space.decode(positive), require_all=True)
            - space.log_prior(space.decode(negative), require_all=True)
        ) / 2e-6
        numeric_jacobian[index] = (
            space.log_abs_det_inverse_jacobian(positive)
            - space.log_abs_det_inverse_jacobian(negative)
        ) / 2e-6

    np.testing.assert_allclose(analytic_prior, numeric_prior, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(analytic_jacobian, numeric_jacobian, atol=1e-6, rtol=1e-6)
    assert not analytic_prior.flags.writeable


def test_parameter_space_is_strictly_portable_and_content_addressed() -> None:
    space = example_space()

    restored = parameter_space_from_json(space.canonical_json())

    assert restored.schema_version == PARAMETER_SPACE_SCHEMA
    assert restored.to_dict() == space.to_dict()
    assert restored.fingerprint == space.fingerprint
    assert len(space.fingerprint) == 64


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ParameterSpec("rate", transform=ParameterTransform.LOG, bounds=(None, None)),
            "finite lower",
        ),
        (
            lambda: ParameterSpec(
                "fixed", role=ParameterRole.FIXED, optimizer_name="fixed", fixed_value=1.0
            ),
            "cannot have",
        ),
        (
            lambda: ParameterSpace((ParameterSpec("duplicate"), ParameterSpec("duplicate"))),
            "must be unique",
        ),
        (lambda: PriorSpec.beta(0.0, 1.0), "must be positive"),
    ],
)
def test_invalid_parameter_declarations_fail_early(factory, message: str) -> None:
    with pytest.raises(ParameterSpaceError, match=message):
        factory()


def test_parameter_values_require_exact_names_domains_and_fixed_values() -> None:
    space = example_space()

    with pytest.raises(ParameterSpaceError, match="match exactly"):
        space.encode({"learning_rate": 0.2})
    with pytest.raises(ParameterSpaceError, match="strictly inside"):
        space.encode({**natural_values(), "learning_rate": 1.0})
    with pytest.raises(ParameterSpaceError, match="must equal"):
        space.encode({**natural_values(), "lapse": 0.02})
    with pytest.raises(ParameterSpaceError, match="3 finite values"):
        space.decode([0.0, 1.0])
    with pytest.raises(ParameterSpaceError, match="not finite"):
        space.parameters[1].to_natural(1_000.0)
    with pytest.raises(ParameterSpaceError, match="has no prior"):
        ParameterSpace((ParameterSpec("x"),)).log_prior({"x": 0.0}, require_all=True)


def test_q_learning_uses_the_shared_space_without_breaking_legacy_fit_coordinates() -> None:
    model = BinaryQLearning(n_restarts=1)
    encoded = model.parameters_from_components(
        learning_rate=0.25,
        inverse_temperature=4.0,
        choice_bias=0.1,
        perseveration=0.2,
    )

    assert isinstance(model, ParameterSpaceProvider)
    assert model.parameter_names == model.parameter_space.optimizer_names
    assert model.parameter_space.natural_names == (
        "learning_rate",
        "inverse_temperature",
        "choice_bias",
        "perseveration",
    )
    assert model.parameter_components(encoded).learning_rate == pytest.approx(0.25)
    assert model.parameter_space.optimizer_bounds == (
        (-12.0, 12.0),
        (-5.0, 5.0),
        (-30.0, 30.0),
        (-30.0, 30.0),
    )


def test_fit_artifact_carries_parameter_space_and_both_coordinate_estimates() -> None:
    model = BinaryQLearning(n_restarts=1)
    design = Study(
        {
            "subject": ["a"] * 8,
            "session": ["s"] * 8,
            "trial": list(range(8)),
            "session_order": [0] * 8,
            "reward_probability_0": [0.2] * 8,
            "reward_probability_1": [0.8] * 8,
        }
    )
    encoded = model.parameters_from_components(
        learning_rate=0.25,
        inverse_temperature=4.0,
        choice_bias=0.1,
        perseveration=0.2,
    )
    study = model.simulate(design, encoded, seed=2)
    estimates = np.asarray([encoded[name] for name in model.parameter_names])
    result = FitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=estimates,
        standard_errors=np.full(4, 0.1),
        covariance=np.eye(4),
        n_observations=len(study),
        diagnostics=FitDiagnostics(True, "known", 0, "known", 0, 0.0, 0.0, 1.0, False),
    )
    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        reward=RewardSpec(column="reward", minimum=0.0, maximum=1.0),
    )
    fitted = FittedModel(model, task, result, task.validate(study))

    artifact = export_fit(fitted, study)

    assert artifact.parameters[0]["name"] == "learning_rate_logit"
    assert artifact.parameters[0]["natural_name"] == "learning_rate"
    assert artifact.parameters[0]["natural_estimate"] == pytest.approx(0.25)
    assert artifact.parameters[0]["coordinate"] == "optimizer"
    assert artifact.diagnostics["parameter_space_fingerprint"] == model.parameter_space.fingerprint
    assert artifact.diagnostics["parameter_space"]["schema_version"] == PARAMETER_SPACE_SCHEMA
