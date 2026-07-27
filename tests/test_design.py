import numpy as np
import pytest

from unspool import (
    CategoricalTerm,
    DesignSpec,
    DesignValidationError,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
    NumericTerm,
    StandardizeTerm,
    Study,
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
    with pytest.raises(DesignValidationError, match="zero/one"):
        HistoryTerm("choice", coding="effect").build(invalid)


def test_design_rejects_feature_name_collisions() -> None:
    design = DesignSpec(terms=(NumericTerm("stimulus", name="x"), NumericTerm("choice", name="x")))
    with pytest.raises(DesignValidationError, match="collide"):
        design.build(make_study())


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
