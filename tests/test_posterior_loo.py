import importlib
import json

import numpy as np
import pytest

from behavio import (
    PosteriorAuditStatus,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    PSISLOOResult,
    psis_loo,
)
from behavio.contracts.audit import AuditSeverity
from behavio.posterior import PosteriorError
from behavio.posterior_diagnostics import PosteriorAuditPolicy


def posterior_result(*, pathological: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(281)
    chains = 4
    draws = 500
    trials = 12
    posterior_values = generator.normal(scale=0.2, size=(chains, draws))
    probability = 1.0 / (1.0 + np.exp(-posterior_values[..., None]))
    outcomes = np.asarray([0, 1] * (trials // 2))
    log_likelihood = np.where(
        outcomes[None, None, :] == 1,
        np.log(probability),
        np.log1p(-probability),
    )
    if pathological:
        log_likelihood[:, :, 3] = -np.exp(generator.normal(scale=1.2, size=(chains, draws)))
    sample_coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    posterior = PosteriorGroup(
        "posterior",
        (
            PosteriorVariable(
                "intercept",
                posterior_values,
                ("chain", "draw"),
                sample_coords,
            ),
        ),
    )
    likelihood_coords = {**sample_coords, "trial": np.arange(trials)}
    likelihood = PosteriorGroup(
        "log_likelihood",
        (
            PosteriorVariable(
                "choice",
                log_likelihood,
                ("chain", "draw", "trial"),
                likelihood_coords,
            ),
        ),
    )
    return PosteriorResult(
        model_name="test-model",
        model_signature="test-signature",
        inference_library="test-sampler",
        inference_library_version="1",
        parameter_names=("intercept",),
        groups=(posterior, likelihood),
    )


def test_psis_loo_retains_labelled_pointwise_evidence() -> None:
    pytest.importorskip("arviz")

    result = psis_loo(posterior_result())

    assert isinstance(result, PSISLOOResult)
    assert result.status is PosteriorAuditStatus.PASS
    assert result.issue_codes == ()
    assert result.log_likelihood_name == "choice"
    assert result.dims == ("trial",)
    assert result.n_samples == 2_000
    assert result.n_data_points == 12
    assert result.pointwise_elpd.shape == (12,)
    assert result.pareto_k.shape == (12,)
    assert not result.pointwise_elpd.flags.writeable
    assert np.sum(result.pointwise_elpd) == pytest.approx(result.elpd_loo)
    json.dumps(result.to_dict(), allow_nan=False)


def test_psis_loo_localizes_unreliable_importance_sampling() -> None:
    pytest.importorskip("arviz")

    with pytest.warns(UserWarning, match="Pareto"):
        result = psis_loo(posterior_result(pathological=True))

    assert result.status is PosteriorAuditStatus.WARNING
    assert "psis.high-pareto-k" in result.issue_codes
    issue = next(item for item in result.issues if item.code == "psis.high-pareto-k")
    assert "choice[trial=3]" in issue.targets
    assert result.pareto_k[3] > result.good_k


def test_psis_loo_requires_pointwise_finite_named_likelihood() -> None:
    result = posterior_result()
    posterior = result["posterior"]
    sample_coords = {"chain": np.arange(4), "draw": np.arange(500)}
    aggregated = PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(
            posterior,
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        np.zeros((4, 500)),
                        ("chain", "draw"),
                        sample_coords,
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(PosteriorError, match="pointwise rather than aggregated"):
        psis_loo(aggregated)

    values = np.array(result["log_likelihood"]["choice"].values, copy=True)
    values[0, 0, 0] = np.nan
    nonfinite = PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(
            posterior,
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        values,
                        ("chain", "draw", "trial"),
                        {
                            **sample_coords,
                            "trial": np.arange(12),
                        },
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(PosteriorError, match="only finite"):
        psis_loo(nonfinite)

    with pytest.raises(PosteriorError, match="no variable"):
        psis_loo(result, log_likelihood_name="response_time")


def _stats():
    """Resolve PSIS the way :mod:`behavio.posterior_loo` does, under the same gate."""

    pytest.importorskip("arviz")
    try:
        return importlib.import_module("arviz_stats")
    except ImportError:
        return importlib.import_module("arviz")


N_SUBJECTS = 6
N_SESSIONS = 2
N_TRIALS = 30


def hierarchical_result(
    *,
    pooled: bool = False,
    diverging: bool = False,
    seed: int = 913,
) -> PosteriorResult:
    """A multi-subject, multi-session posterior whose fit saw every held-out subject.

    Each subject's intercept is estimated from that subject's own trials, so trial-level LOO
    leaves the subject's parameter almost untouched. That is exactly the optimism blocking is
    meant to remove.
    """

    generator = np.random.default_rng(seed)
    chains, draws = 4, 300
    rows = N_SUBJECTS * N_TRIALS
    subjects = np.asarray([f"subject-{index}" for index in range(N_SUBJECTS)])
    trial_subject = np.repeat(subjects, N_TRIALS)
    trial_session = np.asarray(
        [
            f"subject-{index}-session-{session}"
            for index in range(N_SUBJECTS)
            for session in range(N_SESSIONS)
            for _ in range(N_TRIALS // N_SESSIONS)
        ]
    )
    truth = generator.normal(scale=1.4, size=N_SUBJECTS)
    outcomes = generator.binomial(1, 1.0 / (1.0 + np.exp(-np.repeat(truth, N_TRIALS))))
    per_subject = outcomes.reshape(N_SUBJECTS, N_TRIALS).mean(axis=1)
    centre = np.log((per_subject + 0.5) / (1.5 - per_subject))
    if pooled:
        centre = np.full(N_SUBJECTS, np.log((outcomes.mean() + 0.01) / (1.01 - outcomes.mean())))
    intercept = centre[None, None, :] + generator.normal(
        scale=0.25, size=(chains, draws, N_SUBJECTS)
    )
    probability = 1.0 / (1.0 + np.exp(-np.repeat(intercept, N_TRIALS, axis=2)))
    log_likelihood = np.where(
        outcomes[None, None, :] == 1,
        np.log(probability),
        np.log1p(-probability),
    )
    sample_coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    trial = np.arange(rows)
    groups = [
        PosteriorGroup(
            "posterior",
            (
                PosteriorVariable(
                    "subject_intercept",
                    intercept,
                    ("chain", "draw", "subject"),
                    {**sample_coords, "subject": subjects},
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
            "constant_data",
            (
                PosteriorVariable("trial_subject", trial_subject, ("trial",), {"trial": trial}),
                PosteriorVariable("trial_session", trial_session, ("trial",), {"trial": trial}),
            ),
        ),
    ]
    if diverging:
        flags = np.zeros((chains, draws), dtype=bool)
        flags[0, :7] = True
        groups.append(
            PosteriorGroup(
                "sample_stats",
                (PosteriorVariable("diverging", flags, ("chain", "draw"), sample_coords),),
            )
        )
    return PosteriorResult(
        model_name="pooled-model" if pooled else "hierarchical-model",
        model_signature="pooled-signature" if pooled else "hierarchical-signature",
        inference_library="test-sampler",
        inference_library_version="1",
        parameter_names=("subject_intercept",),
        groups=tuple(groups),
    )


def _backend_field(output, *names):
    """Read one ELPD field under whichever name the installed ArviZ publishes it as.

    ArviZ 1.x renamed ``elpd_loo``, ``p_loo`` and ``loo_i`` to ``elpd``, ``p`` and
    ``elpd_i``. The package supports both, so the parity assertion compares values rather
    than depending on one release's spelling.
    """

    for name in names:
        if hasattr(output, name):
            return getattr(output, name)
    raise AssertionError(f"backend output publishes none of {names}")


def test_unblocked_psis_loo_is_bit_identical_to_the_backend() -> None:
    stats = _stats()
    result = posterior_result()

    expected = stats.loo(result.to_arviz(), pointwise=True, var_name="choice")
    actual = psis_loo(result)

    assert actual.block is None
    assert actual.estimand == "leave-one-observation-out"
    assert actual.dims == ("trial",)
    assert actual.elpd_loo == float(_backend_field(expected, "elpd", "elpd_loo"))
    assert actual.se == float(_backend_field(expected, "se"))
    assert actual.p_loo == float(_backend_field(expected, "p", "p_loo"))
    assert actual.n_data_points == int(_backend_field(expected, "n_data_points"))
    pointwise = _backend_field(expected, "elpd_i", "loo_i")
    assert np.array_equal(actual.pointwise_elpd, np.asarray(pointwise.values))
    assert np.array_equal(actual.pareto_k, np.asarray(_backend_field(expected, "pareto_k").values))


def test_blocking_removes_the_optimism_of_leave_one_trial_out() -> None:
    _stats()
    result = hierarchical_result()

    trials = psis_loo(result)
    sessions = psis_loo(result, block="trial_session")
    subjects = psis_loo(result, block="trial_subject")

    assert trials.n_data_points == N_SUBJECTS * N_TRIALS
    assert sessions.n_data_points == N_SUBJECTS * N_SESSIONS
    assert subjects.n_data_points == N_SUBJECTS
    assert sessions.estimand == "leave-one-trial_session-out"
    assert subjects.estimand == "leave-one-trial_subject-out"
    # Higher ELPD is better, so a strictly coarser held-out unit must not look better.
    assert subjects.elpd_loo < sessions.elpd_loo < trials.elpd_loo
    # The effective number of parameters grows once whole subjects are held out.
    assert subjects.p_loo > trials.p_loo


def test_blocked_summaries_are_computed_on_the_block_scale() -> None:
    _stats()
    result = hierarchical_result()

    subjects = psis_loo(result, block="trial_subject")

    assert subjects.dims == ("trial_subject",)
    assert subjects.pointwise_elpd.shape == (N_SUBJECTS,)
    assert subjects.pareto_k.shape == (N_SUBJECTS,)
    assert list(subjects.coords["trial_subject"]) == [
        f"subject-{index}" for index in range(N_SUBJECTS)
    ]
    assert np.sum(subjects.pointwise_elpd) == pytest.approx(subjects.elpd_loo)
    pointwise = np.asarray(subjects.pointwise_elpd)
    expected_se = float(np.sqrt(pointwise.size * np.var(pointwise)))
    assert subjects.se == pytest.approx(expected_se, rel=1e-6)
    # good_k depends on the number of posterior draws, not on the number of blocks.
    assert subjects.good_k == psis_loo(result).good_k
    assert "psis.few-blocks" in subjects.issue_codes
    json.dumps(subjects.to_dict(), allow_nan=False)


def test_blocking_accepts_explicit_labels_and_validates_their_length() -> None:
    _stats()
    result = hierarchical_result()
    labels = np.repeat(np.arange(N_SUBJECTS), N_TRIALS)

    explicit = psis_loo(result, block="animal", block_values=labels)
    stored = psis_loo(result, block="trial_subject")

    assert explicit.dims == ("animal",)
    assert np.allclose(explicit.pointwise_elpd, stored.pointwise_elpd)

    with pytest.raises(PosteriorError, match="supplies 3 labels"):
        psis_loo(result, block="animal", block_values=np.arange(3))
    with pytest.raises(ValueError, match="requires an explicit block name"):
        psis_loo(result, block_values=labels)


def test_unknown_blocking_variable_names_the_retained_candidates() -> None:
    _stats()
    result = hierarchical_result()

    with pytest.raises(PosteriorError, match=r"constant_data\.trial_subject"):
        psis_loo(result, block="subject_id")
    with pytest.raises(PosteriorError, match="collides with a posterior dimension"):
        psis_loo(result, block="subject", block_values=np.repeat(np.arange(N_SUBJECTS), N_TRIALS))


def test_psis_loo_marks_a_non_convergent_posterior_without_discarding_it() -> None:
    _stats()
    result = hierarchical_result(diverging=True)

    scored = psis_loo(result)

    assert scored.status is PosteriorAuditStatus.FAIL
    assert scored.convergence_status is PosteriorAuditStatus.FAIL
    assert "psis.posterior-not-converged" in scored.issue_codes
    assert "posterior.divergences" in scored.convergence.issue_codes
    # The evidence is retained and marked, not raised away.
    assert np.isfinite(scored.elpd_loo)
    assert scored.pointwise_elpd.shape == (N_SUBJECTS * N_TRIALS,)
    json.dumps(scored.to_dict(), allow_nan=False)


def test_convergence_policy_is_injectable() -> None:
    _stats()
    result = hierarchical_result(diverging=True)

    tolerant = psis_loo(result, policy=PosteriorAuditPolicy(max_divergences=10))
    downgraded = psis_loo(
        result,
        policy=PosteriorAuditPolicy(divergence_severity=AuditSeverity.WARNING),
    )

    assert tolerant.status is PosteriorAuditStatus.PASS
    assert tolerant.issue_codes == ()
    assert downgraded.status is PosteriorAuditStatus.WARNING
    assert "psis.posterior-warning" in downgraded.issue_codes
