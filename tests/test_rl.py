import numpy as np
import pytest

from behavio import BinaryRLAgent, Study, run_parameter_recovery
from behavio.models import (
    AsymmetricLearning,
    BehaviourModel,
    ChoiceKernel,
    FitDiagnostics,
    FitResult,
    ResetRule,
    SoftmaxPolicy,
    SymmetricLearning,
    UnchosenForgetting,
)


def bandit_design(n_trials: int = 360) -> Study:
    rich = np.tile(np.repeat([0.8, 0.2], 30), n_trials // 60)
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": [f"s{index // 120}" for index in range(n_trials)],
            "trial": [index % 120 for index in range(n_trials)],
            "session_order": [index // 120 for index in range(n_trials)],
            "reward_probability_0": 1.0 - rich,
            "reward_probability_1": rich,
        }
    )


def fixed_fit(model: BinaryRLAgent, parameters, n_trials: int) -> FitResult:
    estimates = np.asarray([parameters[name] for name in model.parameter_names])
    return FitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=estimates,
        standard_errors=np.zeros_like(estimates),
        covariance=np.zeros((len(estimates), len(estimates))),
        n_observations=n_trials,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="known truth",
            status=0,
            message="known truth",
            n_iterations=0,
            objective=0.0,
            gradient_norm=0.0,
            hessian_condition=1.0,
            boundary_estimate=False,
        ),
    )


def test_components_define_stable_optimizer_and_natural_coordinates() -> None:
    model = BinaryRLAgent(
        learning=AsymmetricLearning(),
        forgetting=UnchosenForgetting(),
        choice_kernel=ChoiceKernel(),
        policy=SoftmaxPolicy(maximum_lapse=0.2),
        n_restarts=2,
    )
    natural = {
        "positive_learning_rate": 0.4,
        "negative_learning_rate": 0.15,
        "forgetting_rate": 0.08,
        "choice_kernel_rate": 0.3,
        "choice_kernel_weight": 0.7,
        "inverse_temperature": 4.0,
        "choice_bias": -0.1,
        "lapse_rate": 0.04,
    }

    encoded = model.parameters_from_components(**natural)
    decoded = model.parameter_components(encoded)

    assert tuple(encoded) == model.parameter_names
    assert tuple(decoded) == model.natural_parameter_names
    assert decoded == pytest.approx(natural)
    assert "asymmetric_delta" in model.signature
    assert "exponential_choice_kernel" in model.signature
    assert isinstance(model, BehaviourModel)


def test_trajectory_applies_asymmetric_updates_forgetting_kernel_and_resets() -> None:
    model = BinaryRLAgent(
        learning=AsymmetricLearning(),
        forgetting=UnchosenForgetting(),
        choice_kernel=ChoiceKernel(),
        policy=SoftmaxPolicy(include_bias=False),
    )
    parameters = model.parameters_from_components(
        positive_learning_rate=0.5,
        negative_learning_rate=0.25,
        forgetting_rate=0.2,
        choice_kernel_rate=0.5,
        choice_kernel_weight=1.0,
        inverse_temperature=2.0,
    )
    study = Study(
        {
            "subject": ["a", "a", "a"],
            "session": ["s1", "s1", "s2"],
            "trial": [0, 1, 0],
            "session_order": [0, 0, 1],
            "choice": [1, 0, 1],
            "reward": [1.0, 0.0, 0.0],
        }
    )

    trajectory = model.trajectory(study, fixed_fit(model, parameters, len(study)))

    np.testing.assert_allclose(
        trajectory.pre_choice_values,
        [[0.5, 0.5], [0.5, 0.75], [0.5, 0.5]],
    )
    np.testing.assert_allclose(
        trajectory.post_update_values,
        [[0.5, 0.75], [0.375, 0.7], [0.5, 0.375]],
    )
    np.testing.assert_allclose(
        trajectory.post_update_kernel,
        [[0.0, 0.5], [0.5, 0.25], [0.0, 0.5]],
    )


def test_reset_rule_can_explicitly_carry_state_across_sessions_within_subject() -> None:
    model = BinaryRLAgent(
        reset=ResetRule(("subject",)),
        policy=SoftmaxPolicy(include_bias=False),
    )
    parameters = model.parameters_from_components(
        learning_rate=0.5,
        inverse_temperature=2.0,
    )
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s1", "s2"],
            "trial": [0, 0],
            "session_order": [0, 1],
            "choice": [1, 0],
            "reward": [1.0, 0.0],
        }
    )

    trajectory = model.trajectory(study, fixed_fit(model, parameters, len(study)))

    np.testing.assert_allclose(trajectory.pre_choice_values[1], [0.5, 0.75])


def test_composable_agent_fits_scores_audits_and_recovers() -> None:
    model = BinaryRLAgent(
        learning=SymmetricLearning(),
        policy=SoftmaxPolicy(maximum_lapse=0.15),
        n_restarts=2,
        random_seed=3,
        max_iterations=400,
    )
    parameters = model.parameters_from_components(
        learning_rate=0.25,
        inverse_temperature=4.0,
        choice_bias=0.1,
        lapse_rate=0.03,
    )
    design = bandit_design()
    study = model.simulate(design, parameters, seed=18)

    fit = model.fit(study)

    assert fit.model_signature == model.signature
    assert fit.natural_parameters.keys() == model.parameter_components(fit).keys()
    assert fit.restart_objectives.shape == (2,)
    assert model.predict(study, fit).probability.shape == (len(study),)
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))
    assert fit.audit().model_name == model.model_name

    recovery = run_parameter_recovery(model, design, [parameters], seed=8)
    assert recovery.model_name == model.model_name
    assert recovery.n_runs == 1


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SoftmaxPolicy(maximum_lapse=1.0), "maximum_lapse"),
        (lambda: ResetRule(("session",)), "include subject"),
        (lambda: BinaryRLAgent(n_restarts=0), "n_restarts"),
    ],
)
def test_component_configuration_fails_loudly(factory, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
