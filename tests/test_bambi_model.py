"""The Bambi wrapper: does the *sampled* half of the contract accept a foreign model?

Everything here is gated on the optional extra, exactly as the PyDDM, NWB, DANDI, ONE,
PyBADS and Parquet suites are: ``pytest.importorskip`` at module scope, so a checkout
without ``behavio[bambi]`` skips this file cleanly rather than failing it. The three tests
that assert what a machine *without* the extra sees deliberately block the import instead.

Sampling is real MCMC, so every posterior in this file is a module-scoped fixture and the
configurations are the smallest that still pass the default
:class:`~behavio.posterior.PosteriorAuditPolicy` -- four chains, because two make R-hat
noisy enough that a healthy fit fails its own gate, which is a fact about the diagnostic
rather than about Bambi.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from behavio import BiasOnly, Study, compare_models, forward_session_splits
from behavio.adapters import check_behaviour_estimator
from behavio.adapters.conformance import CheckStatus
from behavio.contracts.estimator import (
    BehaviourEstimator,
    CategoricalPrediction,
    FitResult,
    ModelDataError,
    ModelPrediction,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.contracts.posterior import (
    GenerativePosteriorBehaviourModel,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    is_posterior_estimator,
    posterior_log_predictive_density,
    posterior_parameter_columns,
)
from behavio.diagnostics import FitAuditStatus
from behavio.foreign import BAMBI_EXTRA, ForeignPackageUnavailableError
from behavio.inference import PriorSpec
from behavio.posterior import PosteriorResult, compare_posterior_models, psis_loo
from behavio.posterior.diagnostics import PosteriorAuditStatus, audit_posterior
from behavio.recovery import run_parameter_recovery

pytest.importorskip("bambi")

from behavio.foreign.bambi import (
    BLOCKING_VARIABLES,
    SUPPORTED_FAMILIES,
    TRIAL_DIM,
    BambiRegression,
    DesignBoundBambiRegression,
    prior_correspondence,
)

CHAINS = 4
DRAWS = 400


def design(
    *,
    n_subjects: int = 3,
    n_sessions: int = 3,
    n_trials: int = 30,
    seed: int = 7,
) -> Study:
    generator = np.random.default_rng(seed)
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


def flat_model(**overrides: Any) -> BambiRegression:
    settings: dict[str, Any] = {
        "formula": "choice ~ stimulus",
        "draws": DRAWS,
        "tune": DRAWS,
        "chains": CHAINS,
        "cores": 1,
        "seed": 3,
    }
    settings.update(overrides)
    return BambiRegression(**settings)


def hierarchical_model(**overrides: Any) -> BambiRegression:
    settings: dict[str, Any] = {
        "formula": "choice ~ stimulus + (1|subject)",
        # A hierarchical model over a handful of groups has a funnel geometry, and 1600
        # draws leave R-hat at 1.02 for its variance -- Monte Carlo noise that the default
        # audit correctly refuses. Longer chains, not a looser policy.
        "draws": 1000,
        "tune": 1000,
        "target_accept": 0.95,
    }
    settings.update(overrides)
    return flat_model(**settings)


@pytest.fixture(scope="module")
def flat_design() -> Study:
    return design()


@pytest.fixture(scope="module")
def flat_study(flat_design: Study) -> Study:
    bound = flat_model().bind(flat_design)
    return bound.simulate(flat_design, {"Intercept": -0.2, "stimulus": 1.4}, seed=11)


@pytest.fixture(scope="module")
def flat_posterior(flat_study: Study) -> PosteriorResult:
    return flat_model().sample(flat_study)


@pytest.fixture(scope="module")
def hierarchical_study() -> Study:
    grid = design(n_subjects=6, n_sessions=2, n_trials=30, seed=13)
    bound = hierarchical_model().bind(grid)
    truth = {"Intercept": -0.2, "stimulus": 1.3, "1|subject_sigma": 0.6}
    for index in range(6):
        truth[f"1|subject[subject__factor_dim='m{index}']"] = -0.75 + 0.3 * index
    return bound.simulate(grid, truth, seed=5)


@pytest.fixture(scope="module")
def hierarchical_posterior(hierarchical_study: Study) -> PosteriorResult:
    return hierarchical_model().sample(hierarchical_study)


# --------------------------------------------------------------------------------------
# The contract itself.
# --------------------------------------------------------------------------------------


def test_the_wrapper_satisfies_the_sampled_contract_and_not_the_frequentist_one() -> None:
    model = flat_model()

    assert isinstance(model, PosteriorBehaviourEstimator)
    assert not isinstance(model, BehaviourEstimator)
    assert is_posterior_estimator(model)

    capabilities = any_model_capabilities(model)
    assert capabilities.scored_columns == ("choice",)
    assert capabilities.prediction_modes == (PredictionMode.FILTERED,)
    assert capabilities.required_task_columns == ("stimulus",)
    # An unbound model is not generative, and the capability matrix says so rather than
    # promising a recovery run that would fail on parameter_names.
    assert not capabilities.can_simulate
    assert not capabilities.can_recover_parameters


def test_the_formula_alone_answers_what_columns_the_model_reads() -> None:
    model = hierarchical_model()

    assert model.outcome == "choice"
    assert model.scored_columns == ("choice",)
    assert model.formula_columns == ("choice", "stimulus", "subject")
    # ``subject`` is required of every Study, so it is not a *task* column, but it is a
    # formula column and must therefore be retained for a rebuild.
    assert model.required_task_columns == ("stimulus",)
    assert model.grouping_factors == ("subject",)


def test_the_signature_fingerprints_the_specification_and_not_the_installed_sampler() -> None:
    model = flat_model()
    signature = model.signature

    assert "formula=choice ~ stimulus" in signature
    assert "family=bernoulli" in signature
    assert "auto_scale=True" in signature
    assert "seed=3" in signature
    # Sampler settings change the numbers, so they are in.
    assert flat_model(seed=4).signature != signature
    assert flat_model(draws=DRAWS + 1).signature != signature
    assert flat_model(auto_scale=False).signature != signature
    assert flat_model(priors={"stimulus": PriorSpec.normal(0.0, 2.0)}).signature != signature
    # The installed Bambi and PyMC versions are deliberately not, because a sampler is an
    # approximation to a declared target rather than the target itself.
    import bambi

    assert bambi.__version__ not in signature


def test_a_bad_formula_or_an_unsupported_family_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="not a valid Bambi formula"):
        _ = BambiRegression("choice ~~ stimulus").outcome
    with pytest.raises(ValueError, match="family must be one of"):
        BambiRegression("choice ~ stimulus", family="poisson")
    assert "poisson" not in SUPPORTED_FAMILIES
    with pytest.raises(ValueError, match="link for the bernoulli family"):
        BambiRegression("choice ~ stimulus", link="softmax")
    with pytest.raises(ValueError, match="at least two for cross-chain diagnostics"):
        flat_model(chains=1)
    with pytest.raises(ValueError, match="plain column name"):
        _ = BambiRegression("scale(choice) ~ stimulus").outcome


# --------------------------------------------------------------------------------------
# The point-summary projection, on a foreign posterior.
# --------------------------------------------------------------------------------------


def test_the_reference_projection_reduces_a_foreign_posterior_unchanged(
    flat_posterior: PosteriorResult,
    flat_study: Study,
) -> None:
    model = flat_model()
    audit = audit_posterior(flat_posterior)
    assert audit.status is PosteriorAuditStatus.PASS, audit.issue_codes

    fit = model.point_summary(flat_posterior, converged=True)

    assert isinstance(fit, FitResult)
    assert fit.model_name == model.model_name
    assert fit.model_signature == model.signature
    assert fit.n_observations == len(flat_study)
    assert set(fit.parameter_names) == {"Intercept", "stimulus"}
    # Every projected number is a measured posterior quantity.
    names, draws = _draw_matrix(flat_posterior)
    order = [names.index(name) for name in fit.parameter_names]
    np.testing.assert_allclose(fit.estimates, draws[:, order].mean(axis=0))
    np.testing.assert_allclose(fit.standard_errors, draws[:, order].std(axis=0, ddof=1))
    # Every optimizer-shaped field is absent rather than fabricated, and the audit that
    # would have warned about a NaN gradient is silent.
    assert fit.diagnostics.gradient_norm is None
    assert fit.diagnostics.objective is None
    assert fit.diagnostics.hessian_condition is None
    assert fit.diagnostics.boundary_estimate is None
    assert fit.diagnostics.optimizer.startswith("Bambi/")
    assert fit.audit().status is FitAuditStatus.PASS
    assert fit.diagnostics.message.startswith("posterior mean projected")

    median = model.point_summary(flat_posterior, converged=True, centre=PosteriorCentre.MEDIAN)
    assert not np.allclose(median.estimates, fit.estimates)


def test_the_projection_records_the_verdict_it_is_given_and_not_its_own(
    flat_posterior: PosteriorResult,
) -> None:
    refused = flat_model().point_summary(flat_posterior, converged=False)

    assert refused.diagnostics.converged is False
    assert refused.audit().status is FitAuditStatus.FAIL


# --------------------------------------------------------------------------------------
# Scoring: it must be Bambi's likelihood, integrated over draws.
# --------------------------------------------------------------------------------------


def test_the_pointwise_score_reproduces_bambis_own_log_likelihood(
    flat_posterior: PosteriorResult,
    flat_study: Study,
) -> None:
    """The claim the wrapper makes about its score, checked rather than asserted.

    ``pointwise_log_prob`` reads the observed entry of the draw-averaged response simplex.
    Bambi computed a per-draw log likelihood of its own during sampling and retained it, so
    on the training rows the two must agree to floating point -- and they are computed by
    completely different code paths.
    """

    scored = flat_model().pointwise_log_prob(flat_study, flat_posterior)

    retained = np.asarray(flat_posterior["log_likelihood"]["choice"].values, dtype=np.float64)
    expected = posterior_log_predictive_density(retained.reshape(-1, len(flat_study)))
    np.testing.assert_allclose(scored, expected, atol=1e-9)
    # It is emphatically not the density at the posterior mean.
    assert np.all(scored >= retained.reshape(-1, len(flat_study)).mean(axis=0) - 1e-12)


def test_predict_and_the_score_are_one_computation(
    flat_posterior: PosteriorResult,
    flat_study: Study,
) -> None:
    model = flat_model()

    prediction = model.predict(flat_study, flat_posterior)
    scored = model.pointwise_log_prob(flat_study, flat_posterior)

    assert isinstance(prediction, Prediction)
    assert prediction.mode is PredictionMode.FILTERED
    outcomes = np.asarray(flat_study["choice"], dtype=np.float64)
    implied = np.where(outcomes == 1, prediction.probability, 1.0 - prediction.probability)
    np.testing.assert_allclose(np.exp(scored), implied, atol=1e-12)


def test_the_wrapper_refuses_a_mode_it_does_not_declare(
    flat_posterior: PosteriorResult,
    flat_study: Study,
) -> None:
    with pytest.raises(UnsupportedPredictionMode, match="only filtered"):
        flat_model().predict(flat_study, flat_posterior, mode=PredictionMode.SMOOTHED)


def test_a_posterior_from_another_specification_is_refused(
    flat_posterior: PosteriorResult,
    flat_study: Study,
) -> None:
    with pytest.raises(ValueError, match="different model specification"):
        flat_model(seed=99).predict(flat_study, flat_posterior)


def test_sampling_is_deterministic_under_a_declared_seed(flat_study: Study) -> None:
    small = flat_model(draws=60, tune=60)
    first = small.sample(flat_study)
    second = small.sample(flat_study)

    np.testing.assert_array_equal(
        np.asarray(first["posterior"]["stimulus"].values),
        np.asarray(second["posterior"]["stimulus"].values),
    )
    assert not np.array_equal(
        np.asarray(first["posterior"]["stimulus"].values),
        np.asarray(
            flat_model(draws=60, tune=60, seed=4).sample(flat_study)["posterior"]["stimulus"].values
        ),
    )


# --------------------------------------------------------------------------------------
# Blocked LOO: the grouping Bambi does not retain.
# --------------------------------------------------------------------------------------


def test_the_posterior_carries_the_grouping_bambi_never_produces(
    hierarchical_posterior: PosteriorResult,
    hierarchical_study: Study,
) -> None:
    assert audit_posterior(hierarchical_posterior).status is PosteriorAuditStatus.PASS
    assert "constant_data" in hierarchical_posterior.group_names
    retained = hierarchical_posterior["constant_data"].variable_names
    assert set(BLOCKING_VARIABLES) <= set(retained)
    # And the observation dimension is Behavio's name, not Bambi's ``__obs__``.
    likelihood = hierarchical_posterior["log_likelihood"]["choice"]
    assert likelihood.intrinsic_dims == (TRIAL_DIM,)
    np.testing.assert_array_equal(
        np.asarray(hierarchical_posterior["constant_data"]["trial_subject"].values),
        np.asarray(hierarchical_study["subject"]),
    )


def test_blocked_loo_reaches_a_subject_estimand_without_explicit_block_values(
    hierarchical_posterior: PosteriorResult,
) -> None:
    trialwise = psis_loo(hierarchical_posterior)
    blocked = psis_loo(hierarchical_posterior, block="trial_subject")

    assert trialwise.estimand == "leave-one-observation-out"
    assert trialwise.dims == (TRIAL_DIM,)
    assert blocked.estimand == "leave-one-trial_subject-out"
    assert blocked.dims == ("trial_subject",)
    assert blocked.n_data_points == 6
    assert list(blocked.coords["trial_subject"]) == ["m0", "m1", "m2", "m3", "m4", "m5"]
    # Four blocks is below the normal-approximation threshold, and the result says so
    # rather than reporting a confident standard error.
    assert "psis.few-blocks" in blocked.issue_codes


def test_two_bambi_posteriors_can_be_compared_on_one_blocked_estimand(
    hierarchical_posterior: PosteriorResult,
    hierarchical_study: Study,
) -> None:
    plain = hierarchical_model(formula="choice ~ 1 + (1|subject)").sample(hierarchical_study)

    comparison = compare_posterior_models(
        {
            "with-stimulus": psis_loo(hierarchical_posterior, block="trial_subject"),
            "intercept-only": psis_loo(plain, block="trial_subject"),
        }
    )

    assert comparison.block == "trial_subject"
    assert comparison.estimand == "leave-one-trial_subject-out"
    assert comparison.model_names[0] == "with-stimulus"
    assert set(comparison.eligible_models) == {"with-stimulus", "intercept-only"}
    json.dumps(comparison.to_dict(), allow_nan=False)


def test_a_posterior_that_did_not_come_from_the_wrapper_cannot_be_predicted_from(
    flat_study: Study,
) -> None:
    model = flat_model()
    posterior = model.sample(flat_study)
    stripped = PosteriorResult(
        model_name=posterior.model_name,
        model_signature=posterior.model_signature,
        inference_library=posterior.inference_library,
        inference_library_version=posterior.inference_library_version,
        parameter_names=posterior.parameter_names,
        groups=tuple(group for group in posterior.groups if group.name != "constant_data"),
    )

    with pytest.raises(Exception, match="retains no training rows"):
        model.predict(flat_study, stripped)


# --------------------------------------------------------------------------------------
# Out-of-fold behaviour.
# --------------------------------------------------------------------------------------


def test_a_prospective_comparison_runs_end_to_end_against_an_optimised_candidate(
    flat_study: Study,
) -> None:
    splits = forward_session_splits(flat_study, min_train_sessions=2)

    report = compare_models(
        {"bambi": flat_model(), "bias": BiasOnly()},
        flat_study,
        splits,
        bootstrap_resamples=32,
        bootstrap_seed=1,
    )

    result = report.result_for("bambi")
    assert result.audit_status is FitAuditStatus.PASS
    assert report.winner == "bambi"
    assert len(result.posterior_folds) == len(splits)
    assert all(evaluation.from_posterior for evaluation in result.evaluations)

    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    folds = payload["models"]["bambi"]["posterior_folds"]
    assert all(fold["convergence_audit"]["status"] == "pass" for fold in folds)
    assert all(
        audit["numerical"]["message"].startswith("posterior mean projected")
        for audit in payload["models"]["bambi"]["fit_audits"]
    )
    assert "posterior_folds" not in payload["models"]["bias"]


def test_out_of_fold_prediction_reuses_the_training_frames_transform_state(
    flat_design: Study,
) -> None:
    """A stateful ``scale()`` must be fitted in the fold and *reused*, not re-estimated.

    This is the check ``AGENTS.md`` demands of any learned preprocessing, and it is the one
    thing a formula language can get silently wrong. ``formulae`` stores the training mean
    and standard deviation on the design and applies them to new rows; the assertion is that
    a test frame whose covariate is on a wildly different scale is *not* recentred onto
    itself.
    """

    model = flat_model(formula="choice ~ scale(stimulus)")
    study = model.bind(flat_design).simulate(
        flat_design, {"Intercept": 0.1, "scale(stimulus)": 1.2}, seed=2
    )
    posterior = model.sample(study)
    shifted = Study(
        {
            **{name: study[name] for name in study.columns},
            "stimulus": np.asarray(study["stimulus"], dtype=np.float64) * 10.0 + 5.0,
        }
    )

    original = model.predict(study, posterior).probability
    rescaled = model.predict(shifted, posterior).probability

    assert not np.allclose(original, rescaled)
    # And the transform really is the training one: standardising the shifted column
    # against the *training* moments reproduces the shifted prediction exactly.
    reference = np.asarray(study["stimulus"], dtype=np.float64)
    centre, spread = reference.mean(), reference.std()
    manual = Study(
        {
            **{name: study[name] for name in study.columns},
            "stimulus": (np.asarray(shifted["stimulus"], dtype=np.float64) - centre)
            / spread
            * spread
            + centre,
        }
    )
    np.testing.assert_allclose(model.predict(manual, posterior).probability, rescaled)


def test_a_random_effect_refuses_to_predict_a_group_the_fit_never_saw(
    hierarchical_posterior: PosteriorResult,
    hierarchical_study: Study,
) -> None:
    unseen = Study(
        {
            **{name: hierarchical_study[name] for name in hierarchical_study.columns},
            "subject": [f"new-{value}" for value in hierarchical_study["subject"]],
        }
    )

    with pytest.raises(ModelDataError, match="no estimated effect"):
        hierarchical_model().predict(unseen, hierarchical_posterior)

    # The alternative is a declared, different predictive claim rather than a silent one.
    permissive = hierarchical_model(sample_new_groups=True)
    resampled = permissive.sample(hierarchical_study)
    prediction = permissive.predict(
        Study(
            {
                **{name: hierarchical_study[name] for name in hierarchical_study.columns},
                "subject": [f"new-{value}" for value in hierarchical_study["subject"]],
            }
        ),
        resampled,
    )
    assert prediction.n_observations == len(hierarchical_study)


# --------------------------------------------------------------------------------------
# The estimator conformance harness.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PosteriorBackedFit(FitResult):
    """A projected fit that also carries the posterior it was projected from."""

    posterior: PosteriorResult = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _ConformanceShim:
    """The adapter :func:`check_behaviour_estimator` needs, and cannot supply itself.

    The harness is written against :class:`~behavio.contracts.BehaviourEstimator`: it calls
    ``model_capabilities``, then ``model.fit(study)``, then ``model.predict(study, fit)``.
    A :class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` has ``sample`` and
    takes a ``PosteriorResult`` where the harness passes a ``FitResult``, so it cannot be
    checked without something in between. This is that something, and writing it here rather
    than hiding it inside the wrapper is the point: **the leakage and mode checks in
    ``behavio.adapters.estimator_conformance`` have no sampled-model entry point**, and that
    is a gap in the harness, not in Bambi.

    Nothing is faked. ``fit`` runs the real sampler, the real convergence audit and the real
    ``point_summary`` projection, and hands the harness a ``FitResult`` subclass that also
    carries the posterior so ``predict`` and ``pointwise_log_prob`` can be given the object
    the contract says they take.
    """

    model: DesignBoundBambiRegression

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def signature(self) -> str:
        return self.model.signature

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self.model.scored_columns

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return self.model.required_task_columns

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return self.model.supported_prediction_modes

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.model.parameter_names

    def fit(self, study: Study) -> _PosteriorBackedFit:
        posterior = self.model.sample(study)
        audit = audit_posterior(posterior)
        projected = self.model.point_summary(
            posterior, converged=audit.status is not PosteriorAuditStatus.FAIL
        )
        return _PosteriorBackedFit(
            model_name=projected.model_name,
            model_signature=projected.model_signature,
            parameter_names=projected.parameter_names,
            estimates=projected.estimates,
            standard_errors=projected.standard_errors,
            covariance=projected.covariance,
            n_observations=projected.n_observations,
            diagnostics=projected.diagnostics,
            posterior=posterior,
        )

    def predict(
        self,
        study: Study,
        fit: _PosteriorBackedFit,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        return self.model.predict(study, fit.posterior, mode=mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: _PosteriorBackedFit,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        return self.model.pointwise_log_prob(study, fit.posterior, mode=mode)

    def simulate(
        self,
        design_study: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        return self.model.simulate(design_study, parameters, seed=seed)


def test_the_conformance_harness_passes_and_names_the_one_check_it_cannot_run(
    flat_design: Study,
    flat_study: Study,
) -> None:
    shim = _ConformanceShim(flat_model(draws=200, tune=200).bind(flat_study))

    report = check_behaviour_estimator(shim, flat_study, seed=5)

    assert report.passed, report.summary()
    executed = {check.name: check for check in report.checks}
    for name in (
        "declares-model-identity",
        "fit-reports-the-training-study",
        "predicts-one-row-per-trial",
        "scores-one-row-per-trial",
        "refuses-undeclared-prediction-modes",
        "filtered-prediction-ignores-future-rows",
        "filtered-score-ignores-future-rows",
        "simulates-the-columns-it-scores",
    ):
        assert executed[name].status is CheckStatus.PASSED, executed[name].detail
    # Exactly one check does not apply, and it does not apply for a stated reason rather
    # than because the study was too small: a bernoulli regression predicts no continuous
    # outcome, so there is no density to reconcile against the choice probabilities.
    skipped = {check.name for check in report.skipped}
    assert skipped == {"density-agrees-with-the-choice-prediction"}
    assert "no continuous outcome" in executed["density-agrees-with-the-choice-prediction"].detail


# --------------------------------------------------------------------------------------
# Binding, simulation and parameter recovery.
# --------------------------------------------------------------------------------------


def test_an_unbound_model_is_not_generative_and_binding_is_what_makes_it_one(
    flat_design: Study,
) -> None:
    model = flat_model()

    assert not isinstance(model, GenerativePosteriorBehaviourModel)
    assert not hasattr(model, "parameter_names")

    bound = model.bind(flat_design)

    assert isinstance(bound, GenerativePosteriorBehaviourModel)
    assert set(bound.parameter_names) == {"Intercept", "stimulus"}
    # The two vocabularies coincide here because the simulator's names *are* the projected
    # posterior coordinates, so the declared mapping is the identity and round-trips.
    assert dict(bound.posterior_parameter_labels) == {name: name for name in bound.parameter_names}
    assert posterior_parameter_columns(bound, bound.parameter_names) == (0, 1)


def test_binding_names_one_coordinate_per_group_level(flat_design: Study) -> None:
    bound = hierarchical_model().bind(flat_design)

    assert set(bound.parameter_names) == {
        "Intercept",
        "stimulus",
        "1|subject_sigma",
        "1|subject[subject__factor_dim='m0']",
        "1|subject[subject__factor_dim='m1']",
        "1|subject[subject__factor_dim='m2']",
    }


def test_a_bound_model_simulates_only_the_design_it_was_bound_to(flat_design: Study) -> None:
    bound = flat_model().bind(flat_design)
    truth = {"Intercept": 0.0, "stimulus": 1.0}

    first = bound.simulate(flat_design, truth, seed=3)
    second = bound.simulate(flat_design, truth, seed=3)

    assert set(np.unique(np.asarray(first["choice"]))) <= {0, 1}
    np.testing.assert_array_equal(np.asarray(first["choice"]), np.asarray(second["choice"]))
    assert not np.array_equal(
        np.asarray(first["choice"]),
        np.asarray(bound.simulate(flat_design, truth, seed=4)["choice"]),
    )
    with pytest.raises(ValueError, match="only the design it was bound to"):
        bound.simulate(design(n_subjects=2, seed=99), truth, seed=3)
    with pytest.raises(ValueError, match="must match the bound design"):
        bound.simulate(flat_design, {"Intercept": 0.0}, seed=3)


def test_parameter_recovery_of_a_foreign_posterior_uses_quantile_coverage() -> None:
    grid = design(n_subjects=3, n_sessions=1, n_trials=70, seed=21)
    bound = flat_model(draws=300, tune=300, seed=1).bind(grid)

    report = run_parameter_recovery(
        bound, grid, [{"Intercept": 0.0, "stimulus": 1.0}], repeats=1, seed=4
    )

    assert report.parameter_names == bound.parameter_names
    assert report.interval_kind == "posterior-quantile"
    assert report.convergence_rate == 1.0
    assert report.interval_lower is not None and report.interval_upper is not None
    assert np.all(report.interval_lower < report.estimates)
    assert np.all(report.estimates < report.interval_upper)
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["runs"][0]["posterior_audit"]["status"] in {"pass", "warning"}


# --------------------------------------------------------------------------------------
# Families beyond the logit link.
# --------------------------------------------------------------------------------------


def test_a_categorical_family_predicts_a_simplex_and_scores_bambis_own_likelihood() -> None:
    grid = design(n_subjects=2, n_sessions=2, n_trials=50, seed=31)
    model = BambiRegression(
        "action ~ stimulus",
        family="categorical",
        categories=("left", "right", "up"),
        draws=300,
        tune=300,
        chains=CHAINS,
        cores=1,
        seed=2,
    )
    bound = model.bind(grid)
    truth = dict(zip(bound.parameter_names, (0.2, -0.3, 0.8, -0.5), strict=True))
    study = bound.simulate(grid, truth, seed=3)

    posterior = model.sample(study)
    prediction = model.predict(study, posterior)
    scored = model.pointwise_log_prob(study, posterior)

    assert isinstance(prediction, CategoricalPrediction)
    assert prediction.categories == ("left", "right", "up")
    assert prediction.probability.shape == (len(study), 3)
    codes = model.outcome_codes(study)
    np.testing.assert_allclose(
        np.exp(scored), prediction.probability[np.arange(len(study)), codes], atol=1e-12
    )
    retained = np.asarray(posterior["log_likelihood"]["action"].values, dtype=np.float64)
    np.testing.assert_allclose(
        scored,
        posterior_log_predictive_density(retained.reshape(-1, len(study))),
        atol=1e-9,
    )


def test_categorical_levels_are_part_of_the_model_and_must_be_declared_in_bambis_order() -> None:
    with pytest.raises(ValueError, match="requires at least two declared categories"):
        BambiRegression("action ~ stimulus", family="categorical")
    with pytest.raises(ValueError, match="sorted string form"):
        BambiRegression(
            "action ~ stimulus", family="categorical", categories=("up", "left", "right")
        )
    with pytest.raises(ValueError, match="only to the categorical family"):
        BambiRegression("choice ~ stimulus", categories=(0, 1))


def test_a_non_logit_link_is_a_different_model_and_a_different_signature() -> None:
    logit = flat_model()
    probit = flat_model(link="probit")

    assert probit.signature != logit.signature
    assert "link=probit" in probit.signature


# --------------------------------------------------------------------------------------
# Priors.
# --------------------------------------------------------------------------------------


def test_the_prior_translation_is_exact_and_is_published() -> None:
    correspondence = prior_correspondence()

    assert set(correspondence) == {"normal", "half-normal", "beta", "uniform"}
    assert "location->mu" in correspondence["normal"]
    assert "scale->sigma" in correspondence["normal"]

    import bambi

    model = flat_model(priors={"stimulus": PriorSpec.normal(0.5, 2.0)})
    translated = model._bambi_priors(bambi)
    assert translated["stimulus"].name == "Normal"
    assert translated["stimulus"].args["mu"] == 0.5
    assert translated["stimulus"].args["sigma"] == 2.0


def test_a_declared_prior_reaches_bambi_and_is_recorded_on_the_posterior(
    flat_study: Study,
) -> None:
    model = flat_model(
        draws=100,
        tune=100,
        auto_scale=False,
        priors={"stimulus": PriorSpec.normal(0.0, 0.25)},
    )

    posterior = model.sample(flat_study)

    assert "Normal(mu: 0.0, sigma: 0.25)" in posterior.attrs["prior_specification"]
    # A prior that tight visibly shrinks the slope towards zero.
    tight = float(np.mean(np.asarray(posterior["posterior"]["stimulus"].values)))
    loose = float(
        np.mean(
            np.asarray(
                flat_model(draws=100, tune=100).sample(flat_study)["posterior"]["stimulus"].values
            )
        )
    )
    assert abs(tight) < abs(loose)


def test_the_prior_language_refuses_what_it_cannot_express() -> None:
    import bambi

    with pytest.raises(ValueError, match="group-specific term"):
        hierarchical_model(priors={"1|subject": PriorSpec.normal(0.0, 1.0)})
    with pytest.raises(TypeError, match="no JSON-safe form"):
        flat_model(priors={"stimulus": bambi.Prior("Normal", mu=0, sigma=1)})


# --------------------------------------------------------------------------------------
# describe(), and what it says before a sampler starts.
# --------------------------------------------------------------------------------------


def test_describe_reports_the_data_dependent_default_priors_as_a_warning(
    flat_study: Study,
) -> None:
    description = flat_model().describe(flat_study)

    codes = {finding.code for finding in description.findings}
    assert "data_dependent_default_priors" in codes
    assert not description.errors
    assert any("auto_scale" in line for line in description.priors)
    assert "data_dependent_default_priors" not in {
        finding.code for finding in flat_model(auto_scale=False).describe(flat_study).findings
    }


def test_describe_warns_when_a_random_effect_has_one_group_in_this_fold(
    flat_study: Study,
) -> None:
    """``forward_session_splits`` builds one fold per subject; a ``(1|subject)`` model
    fitted inside one of them has an unidentified variance, and the warning says so before
    any sampler starts. ``cohort_forward_session_splits`` is the matching splitter."""

    one_subject = flat_study.take(np.flatnonzero(np.asarray(flat_study["subject"]) == "m0"))

    description = hierarchical_model().describe(one_subject)

    assert "single_group_level" in {finding.code for finding in description.findings}
    assert not description.errors
    assert "single_group_level" not in {
        finding.code for finding in hierarchical_model().describe(flat_study).findings
    }


def test_validate_refuses_a_study_the_formula_cannot_read(flat_study: Study) -> None:
    without = Study({name: flat_study[name] for name in flat_study.columns if name != "stimulus"})

    with pytest.raises(ModelDataError, match="columns the study does not have"):
        flat_model().validate(without)


def test_a_non_binary_outcome_is_an_error_before_any_sampling(flat_study: Study) -> None:
    counts = Study(
        {
            **{name: flat_study[name] for name in flat_study.columns},
            "choice": np.arange(len(flat_study)) % 4,
        }
    )

    with pytest.raises(ModelDataError, match="only zero and one"):
        flat_model().validate(counts)


# --------------------------------------------------------------------------------------
# The extra, and life without it.
# --------------------------------------------------------------------------------------


def test_the_wrapper_module_imports_and_names_the_extra_without_bambi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "bambi", None)

    model = BambiRegression("choice ~ stimulus")

    # Nothing that only reads the declaration needs the package.
    assert model.model_name == "bambi-regression"
    assert model.signature.startswith("bambi-regression[")
    with pytest.raises(ForeignPackageUnavailableError, match=BAMBI_EXTRA):
        _ = model.outcome


def test_an_unsupported_bambi_series_is_refused_rather_than_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import ModuleType

    module = ModuleType("bambi")
    module.__version__ = "0.13.0"
    monkeypatch.setitem(sys.modules, "bambi", module)

    with pytest.raises(ForeignPackageUnavailableError, match="written against Bambi"):
        BambiRegression("choice ~ stimulus").sample(design())


def _draw_matrix(posterior: PosteriorResult) -> tuple[list[str], NDArray[np.float64]]:
    from behavio.contracts.posterior import posterior_draw_matrix

    names, draws = posterior_draw_matrix(posterior)
    return list(names), draws
