"""A sampled model reaches prospective comparison, selection, and recovery.

The reference sampled estimator here is a Laplace-approximated Bernoulli GLM whose draws
are produced analytically rather than by MCMC. That keeps the test fast while exercising
exactly the contract a PyMC- or Stan-backed model would: labelled posterior groups, a
convergence audit, a declared point-summary projection, draw-averaged predictions, and an
explicit map from simulator parameter names onto projected posterior coordinates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from behavio import (
    BernoulliHistoryGLM,
    BiasOnly,
    ModelRecoveryScenario,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    Study,
    compare_models,
    forward_session_splits,
    nested_select_model,
    run_model_recovery,
    run_parameter_recovery,
)
from behavio.contracts.estimator import BehaviourEstimator, FitResult, Prediction, PredictionMode
from behavio.contracts.posterior import (
    GenerativePosteriorBehaviourModel,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    is_posterior_estimator,
    posterior_draw_matrix,
    posterior_log_predictive_density,
    posterior_parameter_columns,
    posterior_point_summary,
)
from behavio.diagnostics import FitAuditStatus
from behavio.evaluation import PosteriorFoldPolicy, evaluate_splits
from behavio.posterior_diagnostics import PosteriorAuditPolicy, PosteriorAuditStatus
from behavio.recovery import (
    POSTERIOR_QUANTILE_INTERVAL,
    RUN_FAILURE_CODE,
    WALD_INTERVAL,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "mle_reports_before_posterior_wiring.json"


# --------------------------------------------------------------------------------------
# A reference sampled estimator.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LaplaceSampledGLM:
    """A sampler-backed Bernoulli GLM with an analytic Gaussian posterior.

    ``diverging`` marks every transition as divergent, which is the cheapest way to force
    :func:`behavio.posterior_diagnostics.audit_posterior` to return ``FAIL`` without
    fabricating an unrealistic R-hat.
    """

    covariate: str = "stimulus"
    prior_precision: float = 1.0
    chains: int = 4
    draws: int = 250
    seed: int = 0
    diverging: bool = False

    model_name = "laplace-sampled-glm"
    scored_columns = ("choice",)
    supported_prediction_modes = (PredictionMode.FILTERED,)
    parameter_names = ("intercept", "stimulus")

    @property
    def signature(self) -> str:
        return (
            f"laplace-sampled-glm[covariate={self.covariate};"
            f"prior_precision={self.prior_precision};diverging={self.diverging}]"
        )

    @property
    def posterior_parameter_labels(self) -> Mapping[str, str]:
        return {
            "intercept": "beta[coefficient='intercept']",
            "stimulus": "beta[coefficient='stimulus']",
        }

    # -- inference ---------------------------------------------------------------------

    def sample(self, study: Study) -> PosteriorResult:
        centre, covariance = self._laplace(study)
        generator = np.random.default_rng(self.seed)
        standard = generator.standard_normal((self.chains, self.draws, 2))
        values = centre + standard @ np.linalg.cholesky(covariance).T
        log_likelihood = self._draw_log_likelihood(study, values.reshape(-1, 2)).reshape(
            self.chains, self.draws, len(study)
        )
        sample_coords = {
            "chain": np.arange(self.chains),
            "draw": np.arange(self.draws),
        }
        trial = np.arange(len(study))
        groups = [
            PosteriorGroup(
                "posterior",
                (
                    PosteriorVariable(
                        "beta",
                        values,
                        ("chain", "draw", "coefficient"),
                        {**sample_coords, "coefficient": np.asarray(["intercept", "stimulus"])},
                    ),
                ),
            ),
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        log_likelihood,
                        ("chain", "draw", "trial"),
                        {**sample_coords, "trial": trial},
                    ),
                ),
            ),
            PosteriorGroup(
                "sample_stats",
                (
                    PosteriorVariable(
                        "diverging",
                        np.full((self.chains, self.draws), self.diverging, dtype=bool),
                        ("chain", "draw"),
                        sample_coords,
                    ),
                ),
            ),
            PosteriorGroup(
                "observed_data",
                (
                    PosteriorVariable(
                        "choice",
                        self._outcomes(study).astype(np.int8),
                        ("trial",),
                        {"trial": trial},
                    ),
                ),
            ),
        ]
        return PosteriorResult(
            model_name=self.model_name,
            model_signature=self.signature,
            inference_library="behavio-test-laplace",
            inference_library_version="1.0",
            parameter_names=("beta",),
            groups=tuple(groups),
        )

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    # -- prediction and scoring --------------------------------------------------------

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        probability = np.mean(self._draw_probabilities(study, _draws(posterior)), axis=0)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability / (1.0 - probability)),
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        # The honest held-out Bayesian score: log of the draw-averaged density, not the
        # density at the posterior mean.
        return posterior_log_predictive_density(self._draw_log_likelihood(study, _draws(posterior)))

    # -- simulation --------------------------------------------------------------------

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        coefficients = np.asarray(
            [parameters["intercept"], parameters["stimulus"]], dtype=np.float64
        )
        probability = _sigmoid(self._matrix(design) @ coefficients)
        columns = {name: design[name] for name in design.columns}
        columns["choice"] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    # -- internals ---------------------------------------------------------------------

    def _matrix(self, study: Study) -> NDArray[np.float64]:
        covariate = np.asarray(study[self.covariate], dtype=np.float64)
        return np.column_stack((np.ones_like(covariate), covariate))

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        return np.asarray(study["choice"], dtype=np.float64)

    def _laplace(self, study: Study) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        design = self._matrix(study)
        outcomes = self._outcomes(study)
        coefficients = np.zeros(2, dtype=np.float64)
        hessian = np.eye(2)
        for _ in range(100):
            probability = _sigmoid(design @ coefficients)
            weights = np.clip(probability * (1.0 - probability), 1e-9, None)
            gradient = design.T @ (outcomes - probability) - self.prior_precision * coefficients
            hessian = design.T @ (design * weights[:, None]) + self.prior_precision * np.eye(2)
            step = np.linalg.solve(hessian, gradient)
            coefficients = coefficients + step
            if np.max(np.abs(step)) < 1e-12:
                break
        return coefficients, np.linalg.inv(hessian)

    def _draw_probabilities(self, study: Study, draws: NDArray[np.float64]) -> NDArray[np.float64]:
        return _sigmoid(draws @ self._matrix(study).T)

    def _draw_log_likelihood(self, study: Study, draws: NDArray[np.float64]) -> NDArray[np.float64]:
        probability = np.clip(self._draw_probabilities(study, draws), 1e-12, 1.0 - 1e-12)
        outcomes = self._outcomes(study)
        return outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(-probability)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-values))


def _draws(posterior: PosteriorResult) -> NDArray[np.float64]:
    return posterior_draw_matrix(posterior)[1]


def sampled_design(*, n_subjects: int = 2, n_sessions: int = 4, n_trials: int = 30) -> Study:
    generator = np.random.default_rng(77)
    rows = n_subjects * n_sessions * n_trials
    return Study(
        {
            "subject": [
                f"m{subject}"
                for subject in range(n_subjects)
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session": [
                f"m{subject}-s{session}"
                for subject in range(n_subjects)
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session_order": [
                session
                for _subject in range(n_subjects)
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * n_subjects,
            "stimulus": generator.normal(size=rows),
        }
    )


def sampled_study(**design: int) -> Study:
    return LaplaceSampledGLM().simulate(
        sampled_design(**design), {"intercept": -0.2, "stimulus": 1.1}, seed=404
    )


# --------------------------------------------------------------------------------------
# The contract itself.
# --------------------------------------------------------------------------------------


def test_the_reference_model_satisfies_the_sampled_contract_and_not_the_frequentist_one() -> None:
    model = LaplaceSampledGLM()

    assert isinstance(model, PosteriorBehaviourEstimator)
    assert isinstance(model, GenerativePosteriorBehaviourModel)
    assert not isinstance(model, BehaviourEstimator)
    assert is_posterior_estimator(model)
    assert not is_posterior_estimator(BernoulliHistoryGLM(choice_lags=0))

    capabilities = any_model_capabilities(model)
    assert capabilities.scored_columns == ("choice",)
    assert capabilities.can_recover_parameters


def test_pointwise_scoring_averages_the_density_over_draws_instead_of_using_the_mean() -> None:
    pytest.importorskip("arviz")
    model = LaplaceSampledGLM()
    study = sampled_study().take(np.arange(60))
    posterior = model.sample(study)

    averaged = model.pointwise_log_prob(study, posterior)

    draws = _draws(posterior)
    per_draw = model._draw_log_likelihood(study, draws)
    at_mean = model._draw_log_likelihood(study, draws.mean(axis=0)[None, :])[0]
    # Jensen: the log of the average density is at least the average log density, and it
    # is a different number from the density at the posterior mean.
    assert np.all(averaged >= per_draw.mean(axis=0) - 1e-12)
    assert not np.allclose(averaged, at_mean)
    np.testing.assert_allclose(averaged, posterior_log_predictive_density(per_draw))


def test_the_predictive_density_helper_survives_impossible_draws() -> None:
    values = np.array([[-1.0, -np.inf], [-2.0, -np.inf]])

    density = posterior_log_predictive_density(values)

    assert np.isfinite(density[0])
    assert np.isneginf(density[1])
    with pytest.raises(ValueError, match="finite or negative infinity"):
        posterior_log_predictive_density(np.array([[np.nan]]))


# --------------------------------------------------------------------------------------
# End-to-end prospective comparison and nested selection.
# --------------------------------------------------------------------------------------


def test_a_sampled_model_completes_a_prospective_comparison_against_an_optimised_one() -> None:
    pytest.importorskip("arviz")
    study = sampled_study()
    splits = forward_session_splits(study, min_train_sessions=3)
    candidates = {
        "sampled": LaplaceSampledGLM(),
        "bias": BiasOnly(),
    }

    report = compare_models(candidates, study, splits, bootstrap_resamples=64, bootstrap_seed=11)

    sampled = report.result_for("sampled")
    assert sampled.audit_status is FitAuditStatus.PASS
    assert report.winner == "sampled"
    assert set(report.eligible_model_order) == {"sampled", "bias"}
    assert len(sampled.posterior_folds) == len(splits)
    assert all(evaluation.from_posterior for evaluation in sampled.evaluations)
    assert not any(
        evaluation.from_posterior for evaluation in report.result_for("bias").evaluations
    )

    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    folds = payload["models"]["sampled"]["posterior_folds"]
    assert [fold["fold"] for fold in folds] == list(range(len(splits)))
    assert all(fold["centre"] == "mean" for fold in folds)
    assert all(fold["convergence_audit"]["status"] == "pass" for fold in folds)
    # The optimizer-facing record says, in words, that this was not an optimizer.
    assert all(
        audit["numerical"]["message"].startswith("posterior mean projected")
        for audit in payload["models"]["sampled"]["fit_audits"]
    )
    assert all(
        audit["numerical"]["gradient_norm"] is None
        for audit in payload["models"]["sampled"]["fit_audits"]
    )
    # A maximum-likelihood candidate is untouched.
    assert "posterior_folds" not in payload["models"]["bias"]


def test_a_failed_convergence_audit_makes_a_sampled_candidate_visibly_ineligible() -> None:
    pytest.importorskip("arviz")
    study = sampled_study()
    splits = forward_session_splits(study, min_train_sessions=3)
    candidates = {
        "sampled": LaplaceSampledGLM(diverging=True),
        "bias": BiasOnly(),
    }

    report = compare_models(candidates, study, splits, bootstrap_resamples=64, bootstrap_seed=11)

    sampled = report.result_for("sampled")
    assert sampled.audit_status is FitAuditStatus.FAIL
    assert report.eligible_model_order == ("bias",)
    # The failing candidate would otherwise have won on score alone.
    assert sampled.unit_balanced_log_loss < report.result_for("bias").unit_balanced_log_loss
    assert report.winner == "bias"

    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    folds = payload["models"]["sampled"]["posterior_folds"]
    assert all(fold["converged"] is False for fold in folds)
    assert all(
        "posterior.divergences" in [issue["code"] for issue in fold["convergence_audit"]["issues"]]
        for fold in folds
    )
    assert all(
        "optimizer_nonconvergence" in [issue["code"] for issue in audit["issues"]]
        for audit in payload["models"]["sampled"]["fit_audits"]
    )
    # The score is retained rather than silently dropped, so the matched aggregation units
    # stay identical across candidates.
    assert sampled.aggregation_units == report.result_for("bias").aggregation_units


def test_a_stricter_posterior_policy_can_fail_an_otherwise_healthy_sampled_candidate() -> None:
    pytest.importorskip("arviz")
    study = sampled_study()
    splits = forward_session_splits(study, min_train_sessions=3)
    strict = PosteriorFoldPolicy(
        centre=PosteriorCentre.MEDIAN,
        audit_policy=PosteriorAuditPolicy(min_ess_bulk=1e9, ess_severity="error"),
    )

    report = compare_models(
        {"sampled": LaplaceSampledGLM(), "bias": BiasOnly()},
        study,
        splits,
        bootstrap_resamples=32,
        bootstrap_seed=2,
        posterior_policy=strict,
    )

    sampled = report.result_for("sampled")
    assert sampled.audit_status is FitAuditStatus.FAIL
    assert all(
        evaluation.posterior is not None
        and evaluation.posterior.status is PosteriorAuditStatus.FAIL
        and evaluation.posterior.centre is PosteriorCentre.MEDIAN
        for evaluation in sampled.evaluations
    )
    assert all(
        fold["centre"] == "median"
        for fold in report.to_dict()["models"]["sampled"]["posterior_folds"]
    )


def test_nested_selection_can_choose_and_score_a_sampled_candidate() -> None:
    pytest.importorskip("arviz")
    study = sampled_study(n_sessions=5)
    report = nested_select_model(
        {"sampled": LaplaceSampledGLM(), "bias": BiasOnly()},
        study,
        forward_session_splits(study, min_train_sessions=4),
        lambda training: forward_session_splits(training, min_train_sessions=2),
        bootstrap_resamples=32,
        inner_bootstrap_resamples=16,
        bootstrap_seed=4,
    )

    assert report.selection_counts["sampled"] == len(report.folds)
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert all(fold["selected_posterior"]["converged"] for fold in payload["folds"])


def test_posterior_policy_is_rejected_rather_than_ignored_for_an_optimised_model() -> None:
    study = sampled_study(n_sessions=2, n_trials=20)

    with pytest.raises(ValueError, match="applies only to a PosteriorBehaviourEstimator"):
        evaluate_splits(
            BernoulliHistoryGLM(choice_lags=0),
            study,
            forward_session_splits(study, min_train_sessions=1),
            posterior_policy=PosteriorFoldPolicy(),
        )


# --------------------------------------------------------------------------------------
# Parameter recovery, name round-tripping, and interval provenance.
# --------------------------------------------------------------------------------------


def test_projected_posterior_names_do_not_round_trip_without_the_declared_mapping() -> None:
    pytest.importorskip("arviz")
    model = LaplaceSampledGLM()
    study = sampled_study().take(np.arange(80))
    fit = model.point_summary(model.sample(study), converged=True)

    # The projection names coordinates, the simulator names scalars. They differ.
    assert fit.parameter_names == (
        "beta[coefficient='intercept']",
        "beta[coefficient='stimulus']",
    )
    assert fit.parameter_names != model.parameter_names

    # The declared mapping is what makes recovery well defined.
    assert posterior_parameter_columns(model, fit.parameter_names) == (0, 1)


def test_an_undeclarable_parameter_mapping_raises_instead_of_being_papered_over() -> None:
    pytest.importorskip("arviz")

    class Mislabelled(LaplaceSampledGLM):
        @property
        def posterior_parameter_labels(self) -> Mapping[str, str]:
            return {"intercept": "beta[coefficient='intercept']", "stimulus": "slope"}

    model = Mislabelled()
    design = sampled_design(n_sessions=1, n_trials=40)

    with pytest.raises(ValueError, match="does not round-trip"):
        run_parameter_recovery(
            model,
            design,
            [{"intercept": 0.0, "stimulus": 1.0}],
            seed=1,
        )


def test_parameter_recovery_of_a_sampled_model_uses_posterior_quantile_coverage() -> None:
    pytest.importorskip("arviz")
    model = LaplaceSampledGLM()
    design = sampled_design(n_subjects=1, n_sessions=2, n_trials=120)
    truths = [{"intercept": -0.3, "stimulus": 1.0}, {"intercept": 0.4, "stimulus": 0.6}]

    report = run_parameter_recovery(model, design, truths, repeats=2, seed=31)

    assert report.parameter_names == ("intercept", "stimulus")
    assert report.n_runs == 4
    assert report.interval_kind == POSTERIOR_QUANTILE_INTERVAL
    assert report.convergence_rate == 1.0
    assert report.audit_pass_rate == 1.0
    assert all(summary.interval_kind == POSTERIOR_QUANTILE_INTERVAL for summary in report.summary())
    assert all(np.isfinite(summary.rmse) for summary in report.summary())
    assert all(summary.n_with_uncertainty == 4 for summary in report.summary())
    assert report.interval_lower is not None and report.interval_upper is not None
    assert np.all(report.interval_lower < report.estimates)
    assert np.all(report.estimates < report.interval_upper)
    # The quantile interval is not the Wald interval built from the same standard errors.
    wald_lower = report.estimates - 1.959963984540054 * report.standard_errors
    assert not np.allclose(report.interval_lower, wald_lower)

    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["coverage_interval"] == POSTERIOR_QUANTILE_INTERVAL
    assert payload["summary"][0]["interval_kind"] == POSTERIOR_QUANTILE_INTERVAL
    assert payload["runs"][0]["posterior_audit"]["status"] == "pass"
    assert set(payload["runs"][0]["interval"]) == {"intercept", "stimulus"}
    assert payload["runs"][0]["message"].startswith("posterior mean projected")


def test_a_failed_posterior_recovery_run_is_gated_and_retained() -> None:
    pytest.importorskip("arviz")
    model = LaplaceSampledGLM(diverging=True)
    design = sampled_design(n_subjects=1, n_sessions=1, n_trials=90)

    report = run_parameter_recovery(model, design, [{"intercept": 0.1, "stimulus": 0.9}], seed=7)

    assert report.convergence_rate == 0.0
    assert report.audit_failure_rate == 1.0
    assert report.posterior_audits[0] is not None
    assert report.posterior_audits[0].status is PosteriorAuditStatus.FAIL
    assert all(summary.n_successful == 0 for summary in report.summary())
    json.dumps(report.to_dict(), allow_nan=False)


def test_a_raising_refit_is_retained_as_evidence_instead_of_aborting_the_experiment() -> None:
    class BrokenGLM(BernoulliHistoryGLM):
        def fit(self, study: Study) -> FitResult:
            raise RuntimeError("optimizer exploded")

    model = BrokenGLM(covariates=("stimulus",), choice_lags=0)
    design = sampled_design(n_subjects=1, n_sessions=1, n_trials=40)

    report = run_parameter_recovery(
        model,
        design,
        [{"intercept": 0.0, "stimulus": 1.0}],
        repeats=2,
        seed=9,
    )

    assert report.n_runs == 2
    assert report.interval_kind == WALD_INTERVAL
    assert report.audit_failure_rate == 1.0
    assert all(audit.issue_codes == (RUN_FAILURE_CODE,) for audit in report.audits)
    assert all(
        message.startswith("RuntimeError: optimizer exploded") for message in report.messages
    )
    assert np.all(np.isnan(report.estimates))
    assert all(summary.n_successful == 0 for summary in report.summary())
    assert all(summary.interval_kind == WALD_INTERVAL for summary in report.summary())
    json.dumps(report.to_dict(), allow_nan=False)


# --------------------------------------------------------------------------------------
# Model recovery.
# --------------------------------------------------------------------------------------


def test_model_recovery_admits_a_sampled_generator_and_a_sampled_candidate() -> None:
    pytest.importorskip("arviz")
    sampled = LaplaceSampledGLM()
    design = sampled_design(n_subjects=1, n_sessions=3, n_trials=60)

    report = run_model_recovery(
        design,
        (
            ModelRecoveryScenario(
                name="sampled-truth",
                truth_label="sampled",
                generator=sampled,
                parameters={"intercept": -0.1, "stimulus": 1.6},
            ),
            ModelRecoveryScenario(
                name="bias-truth",
                truth_label="bias",
                generator=BiasOnly(),
                parameters={"intercept": 0.8},
            ),
        ),
        {"sampled": sampled, "bias": BiasOnly()},
        seed=13,
        min_train_sessions=2,
    )

    assert report.sampled_candidate_labels == ("sampled",)
    assert np.all(report.converged)
    matrix = report.scenario_confusion_matrix()
    sampled_row = matrix.counts[matrix.scenario_names.index("sampled-truth")]
    assert sampled_row[matrix.selected_labels.index("sampled")] == 1
    assert report.selected_labels[0] == "sampled"


def test_a_diverging_sampled_candidate_cannot_win_a_model_recovery_run() -> None:
    pytest.importorskip("arviz")
    design = sampled_design(n_subjects=1, n_sessions=3, n_trials=60)
    generator = LaplaceSampledGLM()

    report = run_model_recovery(
        design,
        (
            ModelRecoveryScenario(
                name="sampled-truth",
                truth_label="sampled",
                generator=generator,
                parameters={"intercept": -0.1, "stimulus": 1.6},
            ),
        ),
        {"sampled": LaplaceSampledGLM(diverging=True), "bias": BiasOnly()},
        seed=13,
        min_train_sessions=2,
    )

    assert report.audit_statuses[0][0] is FitAuditStatus.FAIL
    assert "optimizer_nonconvergence" in report.audit_issue_codes[0][0]
    assert report.selected_labels == ("bias",)
    assert report.overall_accuracy == 0.0


# --------------------------------------------------------------------------------------
# The maximum-likelihood path is a strict, byte-identical no-op.
# --------------------------------------------------------------------------------------


def mle_design(*, n_subjects: int = 2, n_sessions: int = 4, n_trials: int = 25) -> Study:
    generator = np.random.default_rng(101)
    rows = n_subjects * n_sessions * n_trials
    return Study(
        {
            "subject": [
                f"m{subject}"
                for subject in range(n_subjects)
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session": [
                f"m{subject}-s{session}"
                for subject in range(n_subjects)
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session_order": [
                session
                for _subject in range(n_subjects)
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * n_subjects,
            "stimulus": generator.normal(size=rows),
        }
    )


def mle_models() -> tuple[BernoulliHistoryGLM, BiasOnly]:
    return (
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.1),
        BiasOnly(),
    )


def mle_study() -> Study:
    glm, _ = mle_models()
    return glm.simulate(
        mle_design(),
        {"intercept": -0.1, "stimulus": 1.0, "choice_lag_1": 0.4},
        seed=202,
    )


def mle_report_payload() -> dict[str, Any]:
    glm, bias = mle_models()
    candidates = {"glm": glm, "bias": bias}
    study = mle_study()
    splits = forward_session_splits(study, min_train_sessions=2)
    comparison = compare_models(
        candidates,
        study,
        splits,
        bootstrap_resamples=64,
        bootstrap_seed=7,
    )
    nested = nested_select_model(
        candidates,
        study,
        forward_session_splits(study, min_train_sessions=3),
        lambda training: forward_session_splits(training, min_train_sessions=1),
        bootstrap_resamples=32,
        inner_bootstrap_resamples=16,
        bootstrap_seed=3,
    )
    recovery = run_parameter_recovery(
        glm,
        mle_design(n_sessions=2, n_trials=40),
        [
            {"intercept": -0.2, "stimulus": 0.6, "choice_lag_1": 0.2},
            {"intercept": 0.1, "stimulus": 1.2, "choice_lag_1": 0.5},
        ],
        repeats=2,
        seed=123,
    )
    model_recovery = run_model_recovery(
        mle_design(n_sessions=3, n_trials=30),
        (
            ModelRecoveryScenario(
                name="glm-truth",
                truth_label="glm",
                generator=glm,
                parameters={"intercept": -0.1, "stimulus": 1.2, "choice_lag_1": 0.3},
            ),
            ModelRecoveryScenario(
                name="bias-truth",
                truth_label="bias",
                generator=bias,
                parameters={"intercept": 0.4},
            ),
        ),
        candidates,
        seed=5,
        min_train_sessions=2,
    )
    return {
        "comparison": comparison.to_dict(),
        "nested_selection": nested.to_dict(),
        "parameter_recovery": recovery.to_dict(),
        "model_recovery": {
            "candidate_labels": list(model_recovery.candidate_labels),
            "scenario_names": list(model_recovery.scenario_names),
            "truth_labels": list(model_recovery.truth_labels),
            "selected_labels": list(model_recovery.selected_labels),
            "mean_log_probabilities": model_recovery.mean_log_probabilities.tolist(),
            "converged": model_recovery.converged.tolist(),
            "failure_messages": [list(row) for row in model_recovery.failure_messages],
            "audit_statuses": [
                [status.value for status in row] for row in model_recovery.audit_statuses
            ],
            "audit_issue_codes": [
                [list(codes) for codes in row] for row in model_recovery.audit_issue_codes
            ],
            "seeds": [int(seed) for seed in model_recovery.seeds],
            "n_folds": [int(count) for count in model_recovery.n_folds],
            "overall_accuracy": model_recovery.overall_accuracy,
        },
    }


def test_every_maximum_likelihood_report_is_byte_identical_to_the_pre_wiring_output() -> None:
    """The golden file was generated before a single posterior line was wired in.

    It was regenerated once, when ``FitAudit.to_dict`` gained a ``convergence`` key. That
    key exists because ``FitDiagnostics.converged`` became three-valued: a procedure that
    solves rather than searches now leaves it absent, and the audit reports *inapplicable*
    rather than laundering the distinction through a fabricated ``converged=True``. The
    regenerated file differs from its predecessor by twenty-two inserted lines and zero
    deleted ones, every one of them ``"convergence": "converged"``. No estimate, score,
    interval, standard error or study digest moved, which is exactly what this fixture
    exists to establish -- so the pin was recomputed rather than the assertion loosened.
    """

    expected = GOLDEN.read_text(encoding="utf-8")

    produced = json.dumps(mle_report_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    assert produced == expected


def test_the_maximum_likelihood_recovery_report_gains_no_posterior_keys() -> None:
    glm, _ = mle_models()
    report = run_parameter_recovery(
        glm,
        mle_design(n_sessions=1, n_trials=40),
        [{"intercept": 0.0, "stimulus": 1.0, "choice_lag_1": 0.2}],
        seed=17,
    )

    payload = report.to_dict()

    assert report.interval_kind == WALD_INTERVAL
    assert report.interval_lower is None and report.interval_upper is None
    assert report.posterior_audits == ()
    assert "coverage_interval" not in payload
    assert "interval_kind" not in payload["summary"][0]
    assert "posterior_audit" not in payload["runs"][0]
    assert replace(report, interval_kind=WALD_INTERVAL).interval_kind == WALD_INTERVAL
