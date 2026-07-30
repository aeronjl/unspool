"""Models accept a DesignSpec, and a covariate tuple is the same thing said shorter."""

from __future__ import annotations

import numpy as np
import pytest

from behavio import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    DesignSpec,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
    NumericTerm,
    Study,
    WienerDriftDiffusion,
)
from behavio.compose import smooth
from behavio.models._kernels.design import covariate_design, outcome_history_term
from behavio.models.base import ModelDataError


def panel(*, n_subjects: int = 2, n_sessions: int = 3, n_trials: int = 24) -> Study:
    rng = np.random.default_rng(4)
    columns: dict[str, list[object]] = {
        name: []
        for name in (
            "subject",
            "session",
            "trial",
            "session_order",
            "stimulus",
            "phase",
            "choice",
            "response_time",
        )
    }
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for trial in range(n_trials):
                stimulus = float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
                columns["subject"].append(f"m{subject}")
                columns["session"].append(f"m{subject}-{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["stimulus"].append(stimulus)
                columns["phase"].append(float(session >= n_sessions - 1))
                columns["choice"].append(int(rng.random() < 0.5 + 0.3 * stimulus))
                columns["response_time"].append(float(0.3 + rng.gamma(2.0, 0.1)))
    return Study(columns)


def test_a_design_built_glm_and_a_covariate_built_glm_agree_exactly() -> None:
    study = panel()
    shorthand = BernoulliHistoryGLM(covariates=("stimulus", "phase"), choice_lags=2, l2=0.1)
    written_out = BernoulliHistoryGLM(
        choice_lags=2,
        l2=0.1,
        design=DesignSpec(
            terms=(NumericTerm(column="stimulus"), NumericTerm(column="phase")),
            intercept=True,
        ),
    )

    assert shorthand.parameter_names == written_out.parameter_names
    assert shorthand.parameter_names == (
        "intercept",
        "stimulus",
        "phase",
        "choice_lag_1",
        "choice_lag_2",
    )

    left = shorthand.design_matrix(study)
    right = written_out.design_matrix(study)
    assert np.array_equal(left, right)

    shorthand_fit = shorthand.fit(study)
    written_out_fit = written_out.fit(study)
    assert np.array_equal(shorthand_fit.estimates, written_out_fit.estimates)
    assert np.array_equal(shorthand_fit.standard_errors, written_out_fit.standard_errors)


def test_the_shorthand_history_term_reproduces_the_hand_built_history_column() -> None:
    study = panel()
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=2)

    built = model.design_matrix(study)
    term = outcome_history_term("choice", 2)
    assert term.feature_names == ("choice_lag_1", "choice_lag_2")
    assert np.array_equal(built[:, 2:], term.build(study).values)


def test_a_design_built_ddm_keeps_the_drift_prefix_on_every_column() -> None:
    shorthand = WienerDriftDiffusion(covariates=("stimulus",))
    written_out = WienerDriftDiffusion(
        design=DesignSpec(terms=(NumericTerm(column="stimulus"),), intercept=True)
    )

    assert shorthand.coefficient_names == ("drift.intercept", "drift.stimulus")
    assert written_out.coefficient_names == shorthand.coefficient_names
    assert written_out.parameter_names == shorthand.parameter_names

    study = panel()
    assert np.array_equal(shorthand._feature_matrix(study), written_out._feature_matrix(study))


def test_an_interaction_term_matches_a_hand_built_product_column() -> None:
    study = panel()
    stimulus = np.asarray(study["stimulus"], dtype=np.float64)
    phase = np.asarray(study["phase"], dtype=np.float64)
    hand_built = stimulus * phase

    term = InteractionTerm(left=NumericTerm(column="stimulus"), right=NumericTerm(column="phase"))
    from_term = term.build(study).values[:, 0]

    assert term.feature_names == ("stimulus:phase",)
    assert np.array_equal(from_term, hand_built)
    # This used to allow the term and the hand-written product to disagree on the sign of
    # a zero, because np.einsum normalizes -0.0 to +0.0. Nothing downstream of X @ beta
    # can observe that, but Study columns are hashed into FitArtifact provenance, so the
    # two ways of building the same column produced two content addresses. The exception
    # is gone: the term multiplies, so it matches bit for bit.
    assert from_term.tobytes() == hand_built.tobytes()
    assert any(np.signbit(hand_built)), "the panel must contain a -0.0 for this to bite"


def test_an_interaction_expressed_as_a_term_fits_like_a_materialized_column() -> None:
    study = panel()
    stimulus = np.asarray(study["stimulus"], dtype=np.float64)
    phase = np.asarray(study["phase"], dtype=np.float64)
    materialized = Study(
        {**{name: study[name] for name in study.columns}, "stimulus_phase": stimulus * phase}
    )

    by_column = BernoulliHistoryGLM(
        covariates=("stimulus", "phase", "stimulus_phase"), choice_lags=0, l2=0.05
    ).fit(materialized)
    by_term = BernoulliHistoryGLM(
        choice_lags=0,
        l2=0.05,
        design=DesignSpec(
            terms=(
                NumericTerm(column="stimulus"),
                NumericTerm(column="phase"),
                InteractionTerm(
                    left=NumericTerm(column="stimulus"), right=NumericTerm(column="phase")
                ),
            ),
            intercept=True,
        ),
    ).fit(study)

    assert np.array_equal(by_column.estimates, by_term.estimates)
    assert np.array_equal(by_column.standard_errors, by_term.standard_errors)
    assert by_column.parameter_names[:3] == by_term.parameter_names[:3]
    assert by_term.parameter_names[3] == "stimulus:phase"


def test_a_design_changes_the_signature_and_a_covariate_tuple_does_not() -> None:
    legacy = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.1)
    assert legacy.signature == (
        "bernoulli-history-glm[outcome=choice;covariates=stimulus;choice_lags=1;l2=0.1]"
    )
    assert "design=" not in legacy.signature

    equivalent = BernoulliHistoryGLM(choice_lags=1, l2=0.1, design=covariate_design(("stimulus",)))
    assert equivalent.parameter_names == legacy.parameter_names
    assert equivalent.signature != legacy.signature
    assert "design=" in equivalent.signature

    other = BernoulliHistoryGLM(
        choice_lags=1,
        l2=0.1,
        design=DesignSpec(
            terms=(NumericTerm(column="stimulus", center=1.0, scale=2.0),), intercept=True
        ),
    )
    assert other.signature != equivalent.signature


def test_a_ddm_design_changes_the_signature_and_a_covariate_tuple_does_not() -> None:
    legacy = WienerDriftDiffusion(covariates=("stimulus",))
    equivalent = WienerDriftDiffusion(design=covariate_design(("stimulus",)))

    assert "design=" not in legacy.signature
    assert "design=" in equivalent.signature
    assert equivalent.parameter_names == legacy.parameter_names


def test_passing_both_a_design_and_covariates_is_rejected() -> None:
    with pytest.raises(ValueError, match="either covariates or design"):
        BernoulliHistoryGLM(covariates=("stimulus",), design=covariate_design(("stimulus",)))
    with pytest.raises(TypeError, match="design must be a DesignSpec"):
        BernoulliHistoryGLM(design=("stimulus",))  # type: ignore[arg-type]


def test_a_design_built_model_reports_the_columns_it_reads_as_task_predictors() -> None:
    model = BernoulliHistoryGLM(
        choice_lags=1,
        design=DesignSpec(
            terms=(
                NumericTerm(column="stimulus"),
                InteractionTerm(
                    left=NumericTerm(column="stimulus"), right=NumericTerm(column="phase")
                ),
            ),
            intercept=True,
        ),
    )

    assert model.required_task_columns == ("stimulus", "phase")


def test_a_history_kernel_can_replace_a_lag_tuple_without_new_columns() -> None:
    study = panel()
    model = BernoulliHistoryGLM(
        choice_lags=0,
        design=DesignSpec(
            terms=(
                NumericTerm(column="stimulus"),
                HistoryTerm(column="choice", lags=(1, 2), coding="effect", name="prior"),
            ),
            intercept=True,
        ),
    )

    fit = model.fit(study)
    assert fit.parameter_names == ("intercept", "stimulus", "prior_lag_1", "prior_lag_2")


def test_a_smooth_glm_built_from_a_design_names_its_knots_the_same_way() -> None:
    shorthand = smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0),
        knots=(0.0, 2.0),
        shared_trajectory=True,
    )
    written_out = smooth(
        BernoulliHistoryGLM(choice_lags=0, design=covariate_design(("stimulus",))),
        knots=(0.0, 2.0),
        shared_trajectory=True,
    )

    assert shorthand.parameter_names == written_out.parameter_names
    assert shorthand.parameter_names == (
        "intercept[session_order=0]",
        "intercept[session_order=2]",
        "stimulus[session_order=0]",
        "stimulus[session_order=2]",
    )

    study = panel()
    assert np.array_equal(shorthand.fit(study).estimates, written_out.fit(study).estimates)


def test_a_mistyped_covariate_is_found_before_fit_instead_of_inside_it() -> None:
    study = panel()
    model = BernoulliHistoryGLM(covariates=("stimulis",))

    description = model.describe(study)
    assert [finding.code for finding in description.errors] == ["missing_column"]
    assert "did you mean 'stimulus'?" in description.errors[0].message
    assert "stimulis" in str(description)

    with pytest.raises(ModelDataError, match="stimulis"):
        model.validate(study)
    with pytest.raises(ModelDataError):
        model.fit(study)


def test_knots_outside_the_observed_clock_are_reported_rather_than_ignored() -> None:
    study = panel()
    model = smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0),
        knots=(0.0, 2.0, 40.0),
        shared_trajectory=True,
    )

    description = model.describe(study)
    warnings = [finding for finding in description.warnings if finding.code == "unsupported_knot"]
    assert warnings, description
    assert "extrapolation" in warnings[0].message
    assert "session_order" in warnings[0].message

    # The fit itself still succeeds without complaint, which is exactly the defect: the
    # third knot is pure extrapolation and nothing in the fitted result says so.
    assert model.validate(study).warnings == description.warnings
    assert np.all(np.isfinite(model.fit(study).estimates))


def test_describe_lists_the_design_parameters_bounds_and_priors_without_a_study() -> None:
    model = WienerDriftDiffusion(covariates=("stimulus",))

    description = model.describe()
    assert description.n_observations is None
    assert description.design_columns == ("intercept", "stimulus")
    assert description.parameter_bounds["drift.stimulus"] == (-12.0, 12.0)
    assert description.parameter_bounds["starting_bias"] == (0.02, 0.98)
    assert description.findings == ()

    penalized = BernoulliHistoryGLM(covariates=("stimulus",), l2=0.25)
    assert any("ridge" in prior for prior in penalized.describe().priors)


def test_describe_prints_the_coding_of_every_history_column() -> None:
    """``choice_lag_1`` is two different columns, so the printout has to say which.

    The name is fixed -- it is baked into fitted artefacts and committed benchmark
    results -- so two fits that both report ``choice_lag_1`` can only be told apart if the
    description carries the coding next to it. It does, for a plain lag, for a kernel, and
    for a history term inside an interaction, where the name is no less ambiguous.
    """

    effect = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1).describe()
    identity = BernoulliHistoryGLM(
        design=DesignSpec.from_formula("choice ~ stimulus + lag(choice, 1, coding='identity')"),
        choice_lags=0,
    ).describe()

    assert "choice_lag_1" in effect.design_columns
    assert "choice_lag_1" in identity.design_columns
    assert effect.design_columns == identity.design_columns
    assert effect.design_column_notes["choice_lag_1"] == "[coding=effect]"
    assert identity.design_column_notes["choice_lag_1"] == "[coding=identity]"
    assert "    choice_lag_1  [coding=effect]" in str(effect)
    assert "    choice_lag_1  [coding=identity]" in str(identity)
    # A column whose meaning its name already carries is not annotated.
    assert "stimulus" not in effect.design_column_notes
    assert "    stimulus\n" in str(effect)

    composite = BernoulliHistoryGLM(
        design=DesignSpec(
            terms=(
                InteractionTerm(left=NumericTerm(column="stimulus"), right=HistoryTerm("choice")),
                HistoryKernelTerm("choice", weights=(0.6, 0.4), coding="effect"),
            )
        ),
        choice_lags=0,
    ).describe()

    assert composite.design_column_notes["stimulus:choice_lag_1"] == "[coding=identity]"
    assert composite.design_column_notes["choice_kernel"] == "[coding=effect]"


def test_describe_names_the_required_task_columns_from_the_capability_contract() -> None:
    from behavio import model_capabilities

    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0)
    capabilities = model_capabilities(model)

    assert model.describe().required_task_columns == capabilities.required_task_columns
    assert capabilities.required_task_columns == ("stimulus",)


ALL_COVARIATE_MODELS = (
    BernoulliHistoryGLM,
    BernoulliGLMHMM,
    WienerDriftDiffusion,
)


@pytest.mark.parametrize("model_class", ALL_COVARIATE_MODELS, ids=lambda c: c.__name__)
def test_every_covariate_model_accepts_an_equivalent_design(model_class: type) -> None:
    legacy = model_class(covariates=("stimulus",))
    designed = model_class(design=covariate_design(("stimulus",)))

    assert designed.parameter_names == legacy.parameter_names
    assert designed.coefficient_names == legacy.coefficient_names
    assert "design=" not in legacy.signature
    assert "design=" in designed.signature
    assert designed.signature != legacy.signature


def test_a_formula_built_design_drives_a_model_like_a_hand_built_one() -> None:
    """The join between the formula front end and the model design-acceptance path.

    ``DesignSpec.from_formula`` and the models' ``design=`` argument were written
    independently against :mod:`behavio.design`. This is the test that they meet: the
    formula's desugaring must produce exactly the design a user would write out, and the
    model must name and fit it identically either way.
    """

    study = panel()
    from_formula = DesignSpec.from_formula("choice ~ stimulus * phase")
    written_out = DesignSpec(
        terms=(
            NumericTerm(column="stimulus"),
            NumericTerm(column="phase"),
            InteractionTerm(left=NumericTerm(column="stimulus"), right=NumericTerm(column="phase")),
        ),
        intercept=True,
    )

    assert from_formula.feature_names == written_out.feature_names
    assert from_formula.signature == written_out.signature

    parsed_fit = BernoulliHistoryGLM(design=from_formula, choice_lags=0, l2=0.05).fit(study)
    manual_fit = BernoulliHistoryGLM(design=written_out, choice_lags=0, l2=0.05).fit(study)

    assert parsed_fit.parameter_names == manual_fit.parameter_names
    assert parsed_fit.model_signature == manual_fit.model_signature
    assert np.array_equal(parsed_fit.estimates, manual_fit.estimates)


def test_a_formula_lag_term_reproduces_the_choice_lags_shorthand_when_effect_coded() -> None:
    study = panel()
    shorthand = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)
    parsed = BernoulliHistoryGLM(
        design=DesignSpec.from_formula('choice ~ stimulus + lag(choice, 1, coding="effect")'),
        choice_lags=0,
    )

    assert parsed.parameter_names == shorthand.parameter_names
    assert np.array_equal(parsed.design_matrix(study), shorthand.design_matrix(study))


def test_a_default_formula_lag_is_the_glm_shorthand_rather_than_a_lookalike() -> None:
    """``lag(choice, 1)`` and ``choice_lags=1`` share a name, so they must share a coding.

    This test used to pin the opposite. ``lag()`` inherited ``HistoryTerm``'s
    ``coding="identity"`` and built a 0/1 column, while ``choice_lags=`` has always built
    the effect-coded -1/+1 column; both called it ``choice_lag_1``. The divergence was
    pinned rather than chosen, and it is not a survivable one: the formula is the surface
    users migrate *to*, so the migration has to be safe by default, and a name that means
    two things silently rescales a coefficient.

    ``lag()`` now defaults to ``coding="effect"``. The literal lagged value is still one
    keyword away, and is pinned just below.
    """

    study = panel()
    parsed = BernoulliHistoryGLM(
        design=DesignSpec.from_formula("choice ~ stimulus + lag(choice, 1)"), choice_lags=0
    )
    shorthand = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)

    assert parsed.parameter_names == shorthand.parameter_names
    assert np.array_equal(parsed.design_matrix(study), shorthand.design_matrix(study))
    assert set(np.unique(parsed.design_matrix(study)[:, 2])) == {-1.0, 0.0, 1.0}


def test_an_identity_coded_formula_lag_is_still_one_keyword_away() -> None:
    study = panel()
    identity = BernoulliHistoryGLM(
        design=DesignSpec.from_formula("choice ~ stimulus + lag(choice, 1, coding='identity')"),
        choice_lags=0,
    )

    assert (
        identity.parameter_names
        == BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1).parameter_names
    )
    assert set(np.unique(identity.design_matrix(study)[:, 2])) == {0.0, 1.0}


def test_a_formula_built_ddm_keeps_the_drift_prefix() -> None:
    parsed = WienerDriftDiffusion(design=DesignSpec.from_formula("choice ~ stimulus"))

    assert parsed.parameter_names == WienerDriftDiffusion(covariates=("stimulus",)).parameter_names
    assert "design=" in parsed.signature
