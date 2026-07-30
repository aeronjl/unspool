import numpy as np
import pytest

from behavio import Study
from behavio.design import (
    CategoricalTerm,
    DesignSpec,
    DesignValidationError,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
    NumericTerm,
    StandardizeTerm,
)


def make_study() -> Study:
    return Study(
        {
            "subject": ["a", "a", "a", "a", "b"],
            "session": ["late", "early", "early", "late", "only"],
            "trial": [0, 1, 0, 1, 0],
            "session_order": [1, 0, 0, 1, 0],
            "choice": [1, 0, 1, 0, 1],
            "stimulus": [2.0, -2.0, -1.0, 1.0, 0.0],
            "condition": ["probe", "train", "train", "probe", "train"],
        }
    )


def test_fixed_design_builds_labelled_numeric_categorical_and_history_terms() -> None:
    design = DesignSpec(
        terms=(
            NumericTerm("stimulus", center=0.0, scale=2.0),
            CategoricalTerm("condition", levels=("train", "probe")),
            HistoryTerm("choice", lags=(1, 2), coding="effect"),
        )
    )

    matrix = design.build(make_study())

    assert design.feature_names == (
        "intercept",
        "stimulus",
        "condition['probe']",
        "choice_lag_1",
        "choice_lag_2",
    )
    assert matrix.names == (
        "intercept",
        "stimulus",
        "condition['probe']",
        "choice_lag_1",
        "choice_lag_2",
    )
    assert matrix.values[:, 1].tolist() == [1.0, -1.0, -0.5, 0.5, 0.0]
    assert matrix.values[:, 2].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert matrix.values[:, 3:].tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
    assert "history(" in matrix.specification
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        matrix.values.setflags(write=True)


def test_history_can_cross_sessions_only_when_reset_is_declared_at_subject_level() -> None:
    study = make_study()
    session_history = HistoryTerm("choice", reset_by=("subject",), coding="effect").build(study)

    assert session_history.values[:, 0].tolist() == [-1.0, 1.0, 0.0, 1.0, 0.0]


def test_interactions_cross_all_columns_of_two_terms() -> None:
    interaction = InteractionTerm(
        NumericTerm("stimulus"),
        CategoricalTerm(
            "condition",
            levels=("train", "probe"),
            drop_reference=False,
        ),
    )

    block = interaction.build(make_study())

    assert block.names == (
        "stimulus:condition['train']",
        "stimulus:condition['probe']",
    )
    assert block.values[0].tolist() == [0.0, 2.0]
    assert block.values[1].tolist() == [-2.0, 0.0]


def test_an_interaction_keeps_the_sign_of_a_zero_the_way_multiplication_does() -> None:
    """``-2.0 * 0.0`` is ``-0.0``, so an interaction column must carry ``-0.0``.

    ``np.einsum("ij,ik->ijk", ...)`` returned ``+0.0`` here, because it reduces through a
    sum. The two values are equal as numbers and identical through ``X @ beta``, so this
    never changed a fit -- but Behavio hashes Study columns into ``FitArtifact``
    provenance, and ``json.dumps`` writes ``-0.0`` and ``0.0`` differently. The same data
    therefore produced two content addresses depending on whether an interaction column was
    multiplied out by hand or built by this term. Elementwise multiplication is also simply
    the correct answer: the hand-written version was right and einsum was the deviation.
    """

    study = make_study()
    interaction = InteractionTerm(NumericTerm("stimulus"), NumericTerm("choice"))

    block = interaction.build(study)
    hand = np.asarray(study["stimulus"], dtype=np.float64) * np.asarray(
        study["choice"], dtype=np.float64
    )

    assert block.names == ("stimulus:choice",)
    assert block.values[:, 0].tobytes() == hand.tobytes()
    # Row 1 is stimulus=-2.0 against choice=0; row 4 is stimulus=0.0 against choice=1.
    assert np.signbit(block.values[1, 0])
    assert not np.signbit(block.values[4, 0])


def test_fixed_levels_and_effect_coding_fail_loudly() -> None:
    study = make_study()
    with pytest.raises(DesignValidationError, match="outside the fixed levels"):
        CategoricalTerm("condition", levels=("train", "test")).build(study)

    invalid = Study(
        {
            **{name: study[name] for name in study.columns if name != "choice"},
            "choice": [0, 1, 2, 0, 1],
        }
    )
    with pytest.raises(DesignValidationError) as raised:
        HistoryTerm("choice", coding="effect").build(invalid)
    # A user who wrote coding="effect" is told what is wrong and nothing else. The longer
    # message that explains the default belongs to the formula path, which chose the
    # coding on the user's behalf; see tests/test_formula.py.
    assert str(raised.value) == "effect-coded history requires zero/one values"


def test_design_rejects_feature_name_collisions() -> None:
    design = DesignSpec(terms=(NumericTerm("stimulus", name="x"), NumericTerm("choice", name="x")))
    with pytest.raises(DesignValidationError, match="collide"):
        design.build(make_study())


def test_design_verifies_extension_term_names_before_model_fitting() -> None:
    class DriftingNamesTerm:
        signature = "drifting-names"
        feature_names = ("declared",)
        required_columns = ("stimulus",)

        def build(self, study):
            from behavio.design import FeatureBlock

            return FeatureBlock(("observed",), np.ones((len(study), 1)))

    with pytest.raises(DesignValidationError, match="declared feature_names"):
        DesignSpec((DriftingNamesTerm(),)).build(make_study())


def test_standardization_is_fitted_only_on_the_supplied_training_study() -> None:
    study = make_study()
    training = study.take([1, 2, 4])
    fitted = StandardizeTerm("stimulus").fit(training)

    assert fitted.center == pytest.approx(-1.0)
    assert fitted.scale == pytest.approx(np.sqrt(2.0 / 3.0))
    assert fitted.build(study).values[0, 0] == pytest.approx((2.0 - fitted.center) / fitted.scale)

    constant = Study(
        {**{name: study[name] for name in study.columns if name != "stimulus"}, "stimulus": [1] * 5}
    )
    with pytest.raises(DesignValidationError, match="constant"):
        StandardizeTerm("stimulus").fit(constant)


def test_fixed_history_kernel_respects_session_boundaries() -> None:
    kernel = HistoryKernelTerm(
        "choice",
        weights=(0.75, 0.25),
        coding="effect",
        name="choice_trace",
    )

    block = kernel.build(make_study())

    assert block.names == ("choice_trace",)
    assert block.values[:, 0].tolist() == [0.0, 0.75, 0.0, 0.75, 0.0]
    assert "weights=(0.75, 0.25)" in kernel.signature
