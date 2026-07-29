"""The formula surface: grammar, precedence, round-trips, errors, and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from behavio import (
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
from behavio.formula import (
    CategoricalFormulaTerm,
    ColumnFormulaTerm,
    Formula,
    FormulaError,
    FormulaSyntaxError,
    GroupTerm,
    LagFormulaTerm,
    StandardizeFormulaTerm,
    describe_design,
    describe_term,
)


def make_study() -> Study:
    return Study(
        {
            "subject": ["a", "a", "a", "a", "b"],
            "session": ["late", "early", "early", "late", "only"],
            "trial": [0, 1, 0, 1, 0],
            "session_order": [1, 0, 0, 1, 0],
            "choice": [1, 0, 1, 0, 1],
            "response_time": [0.4, 0.5, 0.6, 0.7, 0.8],
            "stimulus": [2.0, -2.0, -1.0, 1.0, 0.0],
            "phase": [1.0, 0.0, 1.0, 0.0, 1.0],
            "condition": ["probe", "train", "train", "probe", "train"],
        }
    )


def continuous_study() -> Study:
    """``make_study()`` plus a history column that is not binary, so cannot be effect coded."""

    study = make_study()
    return Study(
        {
            **{name: study[name] for name in study.columns},
            "reward_magnitude": [0.5, 1.5, 0.0, 2.5, 1.0],
        }
    )


# --------------------------------------------------------------------------------------
# Each syntax form
# --------------------------------------------------------------------------------------


def test_a_bare_column_is_a_numeric_term_under_a_named_response() -> None:
    formula = Formula.parse("choice ~ stimulus")

    assert formula.response is not None
    assert formula.response.choice == "choice"
    assert formula.response.response_time is None
    assert formula.response.outcome_columns == ("choice",)
    assert formula.terms == (ColumnFormulaTerm(column="stimulus"),)
    assert formula.to_design() == DesignSpec(terms=(NumericTerm("stimulus"),))


def test_addition_keeps_declaration_order_and_drops_a_repeated_term() -> None:
    formula = Formula.parse("choice ~ stimulus + phase + stimulus")

    assert formula.render() == "choice ~ 1 + stimulus + phase"


def test_a_categorical_term_can_name_its_levels_and_reference() -> None:
    formula = Formula.parse("choice ~ C(condition, ['train', 'probe'], reference='probe')")

    assert formula.to_design() == DesignSpec(
        terms=(CategoricalTerm("condition", levels=("train", "probe"), reference="probe"),)
    )


def test_a_categorical_term_can_keep_its_reference_column() -> None:
    design = Formula.parse("0 + C(condition, ['train', 'probe'], drop_reference=False)").to_design()

    assert design.feature_names == ("condition['train']", "condition['probe']")


def test_lag_maps_onto_a_history_term_with_its_reset_boundary_and_coding() -> None:
    formula = Formula.parse(
        "choice ~ lag(choice, 1, 2, coding='effect', reset_by=['subject'], fill_value=-1.0)"
    )

    assert formula.to_design() == DesignSpec(
        terms=(
            HistoryTerm(
                "choice",
                lags=(1, 2),
                reset_by=("subject",),
                coding="effect",
                fill_value=-1.0,
            ),
        )
    )


def test_lag_and_kernel_default_to_the_effect_coding_the_shorthand_has_always_built() -> None:
    """The formula's history default is ``coding="effect"``, not ``HistoryTerm``'s own.

    This used to inherit ``HistoryTerm(coding="identity")``, which made
    ``lag(choice, 1)`` a 0/1 column named ``choice_lag_1`` while
    ``BernoulliHistoryGLM(choice_lags=1)`` built a -1/+1 column under exactly that name.
    The formula is the surface users migrate to, so the two must agree; the identity coding
    is still reachable, and is spelled.
    """

    assert Formula.parse("~ lag(choice, 1)").to_design() == DesignSpec(
        terms=(HistoryTerm("choice", coding="effect"),)
    )
    assert Formula.parse("~ kernel(choice, [0.6, 0.4])").to_design() == DesignSpec(
        terms=(HistoryKernelTerm("choice", weights=(0.6, 0.4), coding="effect"),)
    )
    assert Formula.parse("~ lag(choice, 1, coding='identity')").to_design() == DesignSpec(
        terms=(HistoryTerm("choice", coding="identity"),)
    )
    # The default is the one that goes unwritten; identity coding renders explicitly.
    assert Formula.parse("~ lag(choice, 1, coding='effect')").render() == "1 + lag(choice, 1)"
    assert (
        Formula.parse("~ lag(choice, 1, coding='identity')").render()
        == "1 + lag(choice, 1, coding='identity')"
    )


def test_a_defaulted_effect_coding_on_a_continuous_column_explains_itself() -> None:
    """The default the user did not write has to name itself, and name the way out."""

    study = continuous_study()
    design = Formula.parse("~ lag(reward_magnitude, 1)").to_design()

    with pytest.raises(DesignValidationError) as raised:
        design.build(study)

    message = str(raised.value)
    assert "effect-coded history requires zero/one values" in message
    assert "'reward_magnitude'" in message
    assert "coding='effect'" in message
    assert "choice_lags=" in message
    assert "lag(reward_magnitude, 1, coding='identity')" in message


def test_a_declared_effect_coding_gets_the_short_message_the_design_layer_gives() -> None:
    """Writing ``coding='effect'`` is asking for it, so nothing needs explaining."""

    study = continuous_study()
    design = Formula.parse("~ lag(reward_magnitude, 1, coding='effect')").to_design()

    with pytest.raises(DesignValidationError) as raised:
        design.build(study)

    assert str(raised.value) == "effect-coded history requires zero/one values"


def test_a_defaulted_kernel_coding_suggests_the_kernel_spelling_not_the_lag_one() -> None:
    study = continuous_study()
    design = Formula.parse("~ kernel(reward_magnitude, [0.6, 0.4])").to_design()

    with pytest.raises(DesignValidationError) as raised:
        design.build(study)

    assert "kernel(reward_magnitude, [0.6, 0.4], coding='identity')" in str(raised.value)


def test_the_coding_hint_is_provenance_and_not_part_of_the_term() -> None:
    """A hinting term and a hand-written one are the same term, and hash the same."""

    hinted = Formula.parse("~ lag(choice, 1)").to_design().terms[0]
    written = HistoryTerm("choice", coding="effect")

    assert hinted == written
    assert hinted.signature == written.signature
    assert "coding_hint" not in repr(hinted)


def test_lags_may_be_written_one_by_one_or_as_a_list() -> None:
    assert Formula.parse("~ lag(choice, 1, 3)") == Formula.parse("~ lag(choice, [1, 3])")
    assert Formula.parse("~ lag(choice, [1, 3])") == Formula.parse("~ lag(choice, lags=[1, 3])")


def test_kernel_maps_onto_a_fixed_weight_history_kernel_term() -> None:
    formula = Formula.parse("choice ~ kernel(choice, [0.6, 0.3, 0.1], coding='effect')")

    assert formula.to_design() == DesignSpec(
        terms=(HistoryKernelTerm("choice", weights=(0.6, 0.3, 0.1), coding="effect"),)
    )


def test_numeric_carries_a_declared_affine_transformation_and_feature_name() -> None:
    design = Formula.parse("~ numeric(stimulus, center=0.5, scale=2.0, name='z')").to_design()

    assert design == DesignSpec(terms=(NumericTerm("stimulus", name="z", center=0.5, scale=2.0),))
    assert design.feature_names == ("intercept", "z")


def test_a_two_part_response_names_the_same_roles_as_a_task_spec() -> None:
    formula = Formula.parse("response_time | choice ~ stimulus")

    assert formula.response is not None
    assert formula.response.choice == "choice"
    assert formula.response.response_time == "response_time"
    # TaskSpec.outcome_columns orders the roles (choice, response_time); so does this.
    assert formula.response.outcome_columns == ("choice", "response_time")
    assert formula.render() == "response_time | choice ~ 1 + stimulus"


def test_the_intercept_is_declared_by_one_and_suppressed_by_zero_or_minus_one() -> None:
    assert Formula.parse("choice ~ 1 + stimulus").intercept
    assert not Formula.parse("choice ~ 0 + stimulus").intercept
    assert not Formula.parse("choice ~ stimulus - 1").intercept
    assert Formula.parse("choice ~ stimulus").intercept


def test_a_backquoted_name_carries_a_column_the_bare_grammar_cannot_spell() -> None:
    formula = Formula.parse("~ `reaction time (s)`")

    assert formula.required_columns == ("reaction time (s)",)
    assert formula.render() == "1 + `reaction time (s)`"


def test_a_response_is_optional_so_a_design_only_formula_parses() -> None:
    assert Formula.parse("1 + stimulus").response is None
    assert Formula.parse("~ 1 + stimulus").response is None


# --------------------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------------------


def test_star_expands_to_main_effects_plus_the_interaction() -> None:
    assert Formula.parse("choice ~ stimulus * phase").render() == (
        "choice ~ 1 + stimulus + phase + stimulus:phase"
    )


def test_colon_is_the_interaction_alone() -> None:
    assert Formula.parse("choice ~ stimulus:phase").render() == "choice ~ 1 + stimulus:phase"


def test_colon_binds_more_tightly_than_star_which_binds_more_tightly_than_plus() -> None:
    # ':' first: 'b:c' is one factor, so the expansion is over {a, b:c}.
    assert Formula.parse("~ a * b:c").render() == "1 + a + b:c + a:b:c"
    # '*' before '+': only 'b' and 'c' are crossed.
    assert Formula.parse("~ a + b * c").render() == "1 + a + b + c + b:c"


def test_a_three_way_star_expands_to_every_non_empty_subset() -> None:
    assert Formula.parse("~ a * b * c").render() == "1 + a + b + a:b + c + a:c + b:c + a:b:c"


def test_parentheses_group_a_sum_before_it_is_crossed() -> None:
    assert Formula.parse("~ (a + b) * c").render() == "1 + a + b + c + a:c + b:c"


def test_subtraction_removes_a_term_that_an_expansion_introduced() -> None:
    assert Formula.parse("~ a * b - a:b").render() == "1 + a + b"


def test_an_interaction_is_left_associated_onto_nested_interaction_terms() -> None:
    design = Formula.parse("0 + a:b:c").to_design()
    stimulus, phase, condition = (NumericTerm(name) for name in ("a", "b", "c"))

    assert design == DesignSpec(
        terms=(InteractionTerm(InteractionTerm(stimulus, phase), condition),), intercept=False
    )


# --------------------------------------------------------------------------------------
# Round-trip canonicalisation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "choice ~ stimulus",
        "choice ~ stimulus + phase",
        "choice ~ stimulus * phase",
        "choice ~ stimulus:phase",
        "choice ~ a * b * c",
        "choice ~ C(condition)",
        "choice ~ C(condition, ['train', 'probe'], reference='probe')",
        "choice ~ C(condition, ['train', 'probe'], drop_reference=False, name='cond')",
        "choice ~ lag(choice, 1)",
        "choice ~ lag(choice, 1, 2, coding='identity')",
        "choice ~ lag(choice, 1, reset_by=['subject'], fill_value=-1.0, name='previous')",
        "choice ~ kernel(choice, [0.6, 0.3, 0.1])",
        "choice ~ kernel(choice, [0.6, 0.3], lags=[1, 4], coding='identity')",
        "choice ~ scale(stimulus)",
        "choice ~ scale(stimulus, ddof=1, name='z')",
        "choice ~ numeric(stimulus, center=0.5, scale=2.0, name='z')",
        "response_time | choice ~ stimulus + phase",
        "choice ~ 0 + stimulus",
        "choice ~ stimulus + (1 | subject)",
        "choice ~ stimulus + (0 + stimulus | subject)",
        "1 + `an odd column`",
    ],
)
def test_every_form_renders_to_a_canonical_string_that_parses_back_to_itself(text: str) -> None:
    parsed = Formula.parse(text)
    canonical = parsed.render()

    reparsed = Formula.parse(canonical)

    assert reparsed == parsed
    assert reparsed.render() == canonical
    assert str(parsed) == canonical


def test_a_hand_built_design_describes_itself_as_a_formula_that_rebuilds_it() -> None:
    stimulus = NumericTerm("stimulus", center=0.0, scale=2.0)
    condition = CategoricalTerm("condition", levels=("train", "probe"))
    spec = DesignSpec(
        terms=(
            stimulus,
            condition,
            HistoryTerm("choice", lags=(1, 2), coding="effect"),
            HistoryKernelTerm("choice", weights=(0.6, 0.4)),
            InteractionTerm(stimulus, condition),
        )
    )

    described = spec.describe()

    assert described == describe_design(spec)
    assert DesignSpec.from_formula(described) == spec
    assert str(spec) == described


def test_a_suppressed_intercept_survives_the_description_round_trip() -> None:
    spec = DesignSpec(terms=(NumericTerm("stimulus"),), intercept=False)

    assert spec.describe() == "0 + stimulus"
    assert DesignSpec.from_formula(spec.describe()) == spec


def test_a_third_party_term_has_no_formula_spelling_and_says_so() -> None:
    class Constant:
        signature = "constant(value=1.0)"
        feature_names = ("constant",)
        required_columns = ()

        def build(self, study: Study) -> None:  # pragma: no cover - never built here
            raise NotImplementedError

    spec = DesignSpec(terms=(Constant(),))

    assert describe_term(Constant()) is None
    with pytest.raises(FormulaError, match="no formula spelling"):
        spec.describe()
    # __str__ still reads, falling back to the term's own signature.
    assert str(spec) == "1 + constant(value=1.0)"


# --------------------------------------------------------------------------------------
# A formula-built design is the hand-built design
# --------------------------------------------------------------------------------------


def test_a_formula_design_matrix_is_identical_to_the_hand_built_equivalent() -> None:
    study = make_study()
    stimulus = NumericTerm("stimulus", center=0.0, scale=2.0)
    condition = CategoricalTerm("condition", levels=("train", "probe"))
    hand = DesignSpec(
        terms=(
            stimulus,
            condition,
            InteractionTerm(stimulus, condition),
            HistoryTerm("choice", lags=(1, 2), coding="effect"),
        )
    )
    parsed = DesignSpec.from_formula(
        "choice ~ numeric(stimulus, scale=2.0) * C(condition, ['train', 'probe'])"
        " + lag(choice, 1, 2, coding='effect')"
    )

    assert parsed == hand
    assert parsed.feature_names == hand.feature_names
    assert parsed.signature == hand.signature
    np.testing.assert_array_equal(parsed.build(study).values, hand.build(study).values)


def test_the_formula_reports_the_source_columns_the_design_will_read() -> None:
    formula = Formula.parse("response_time | choice ~ stimulus + lag(choice, 1) + C(condition)")

    assert formula.required_columns == (
        "choice",
        "response_time",
        "stimulus",
        "subject",
        "session",
        "condition",
    )


# --------------------------------------------------------------------------------------
# The scale() leakage boundary
# --------------------------------------------------------------------------------------


def test_a_data_dependent_term_refuses_to_build_without_a_named_training_study() -> None:
    formula = Formula.parse("choice ~ stimulus + scale(stimulus)")

    with pytest.raises(FormulaError, match="estimates its coordinate from study rows") as error:
        formula.to_design()

    assert error.value.position == 20
    assert "Formula.fit(training_study)" in str(error.value)


def test_an_inferred_category_set_is_data_dependent_for_the_same_reason() -> None:
    assert Formula.parse("~ C(condition)").data_dependent_terms == (
        CategoricalFormulaTerm(column="condition"),
    )
    assert Formula.parse("~ C(condition, ['train', 'probe'])").data_dependent_terms == ()

    with pytest.raises(FormulaError, match="estimates its coordinate"):
        Formula.parse("~ C(condition)").to_design()


def test_fitting_freezes_the_training_estimate_so_test_rows_cannot_move_it() -> None:
    training = make_study()
    testing = Study(
        {
            "subject": ["c", "c"],
            "session": ["only", "only"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "choice": [1, 0],
            "stimulus": [100.0, -100.0],
            "condition": ["train", "probe"],
        }
    )

    design = Formula.parse("choice ~ scale(stimulus)").fit(training)
    frozen = StandardizeTerm("stimulus").fit(training)

    assert design == DesignSpec(terms=(frozen,))
    # The frozen centre and scale are the training ones on both studies.
    np.testing.assert_allclose(
        design.build(testing).values[:, 1], testing["stimulus"] / np.std(training["stimulus"])
    )
    assert design.build(testing).specification == design.build(training).specification


def test_an_inferred_category_set_is_ordered_so_the_reference_is_not_row_order() -> None:
    training = make_study()

    design = Formula.parse("choice ~ C(condition)").fit(training)

    assert design == DesignSpec(
        terms=(CategoricalTerm("condition", levels=("probe", "train"), reference="probe"),)
    )
    assert design.build(training).names == ("intercept", "condition['train']")


def test_an_inferred_category_set_needs_two_categories_in_the_training_fold() -> None:
    training = Study(
        {
            "subject": ["a", "a"],
            "session": ["one", "one"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "condition": ["train", "train"],
        }
    )

    with pytest.raises(FormulaError, match="at least two categories"):
        Formula.parse("~ C(condition)").fit(training)


def test_fit_requires_a_study_rather_than_quietly_accepting_anything() -> None:
    with pytest.raises(TypeError, match="must be a Study"):
        Formula.parse("~ scale(stimulus)").fit({"stimulus": [1.0]})  # type: ignore[arg-type]


def test_design_spec_from_formula_names_the_training_study_in_its_own_signature() -> None:
    training = make_study()

    design = DesignSpec.from_formula("choice ~ scale(stimulus)", training_study=training)

    assert design == Formula.parse("choice ~ scale(stimulus)").fit(training)


# --------------------------------------------------------------------------------------
# Group terms: parsed, structured, refused at use
# --------------------------------------------------------------------------------------


def test_a_group_term_parses_into_a_structured_declaration_rather_than_being_dropped() -> None:
    formula = Formula.parse("choice ~ stimulus + (1 | subject) + (0 + stimulus | session)")

    assert formula.terms == (ColumnFormulaTerm(column="stimulus"),)
    assert formula.groups == (
        GroupTerm(grouping="subject", terms=(), intercept=True),
        GroupTerm(
            grouping="session", terms=(ColumnFormulaTerm(column="stimulus"),), intercept=False
        ),
    )
    assert formula.groups[0].required_columns == ("subject",)
    assert formula.groups[1].required_columns == ("session", "stimulus")


def test_a_group_term_hands_a_combinator_a_grouping_column_and_a_within_group_design() -> None:
    group = Formula.parse("choice ~ (stimulus | subject)").groups[0]

    assert group.grouping == "subject"
    assert group.to_design() == DesignSpec(terms=(NumericTerm("stimulus"),))
    assert group.to_design().feature_names == ("intercept", "stimulus")


def test_using_a_group_term_is_a_loud_error_because_nothing_can_honour_it_yet() -> None:
    formula = Formula.parse("choice ~ stimulus + (1 | subject)")

    with pytest.raises(FormulaError, match="cannot be honoured yet") as error:
        formula.to_design()

    assert error.value.position == 20
    assert "no varying-effect representation" in str(error.value)
    with pytest.raises(FormulaError, match="cannot be honoured yet"):
        formula.fit(make_study())


def test_a_group_term_cannot_be_nested_crossed_or_subtracted() -> None:
    with pytest.raises(FormulaSyntaxError, match="cannot be nested"):
        Formula.parse("choice ~ ((1 | session) | subject)")
    with pytest.raises(FormulaSyntaxError, match="cannot take part in a '\\*' expansion"):
        Formula.parse("choice ~ stimulus * (1 | subject)")
    with pytest.raises(FormulaSyntaxError, match="cannot be subtracted"):
        Formula.parse("choice ~ stimulus - (1 | subject)")


def test_an_unparenthesised_bar_says_how_to_write_a_group_term() -> None:
    with pytest.raises(FormulaSyntaxError, match="must be parenthesised") as error:
        Formula.parse("choice ~ stimulus | subject")

    assert error.value.position == 18


# --------------------------------------------------------------------------------------
# Errors point at the offending position
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "position", "match"),
    [
        ("choice ~ stimulus + )", 20, "expected a term"),
        ("choice ~ stimulus +", 19, "the end of the formula"),
        ("choice ~ spline(stimulus)", 9, "unknown formula function"),
        ("choice ~ 2", 9, "only '1' and '0'"),
        ("choice ~ stimulus & phase", 18, "unexpected character"),
        ("choice ~ stimulus ~ phase", 18, "exactly one '~'"),
        ("choice + phase ~ stimulus", 7, "two-part response"),
        ("2 ~ stimulus", 0, "must be a column name"),
        ("choice ~ lag(choice, 0)", 21, "positive integer"),
        ("choice ~ lag(choice, 1, order=2)", 24, "unknown argument"),
        ("choice ~ lag(choice, 1, coding='exponential')", 24, "must be one of"),
        ("choice ~ scale(stimulus, 1)", 25, "no positional argument"),
        ("choice ~ C(condition, 'train')", 22, "must be a list"),
        ("choice ~ kernel(choice)", 9, "needs its fixed weights"),
        ("choice ~ 1 * stimulus", 11, "cannot take part"),
        ("choice ~ stimulus + (1 | 2)", 25, "grouped by a column name"),
        ("choice ~ (stimulus", 18, "expected '\\)'"),
        ("choice ~ `unterminated", 9, "unterminated backquoted"),
        ("choice ~ 0", 9, "intercept or at least one term"),
        ("choice ~ stimulus - 0", 20, "'- 0' is not meaningful"),
    ],
)
def test_a_parse_error_names_the_problem_and_points_at_it(
    text: str, position: int, match: str
) -> None:
    with pytest.raises(FormulaSyntaxError, match=match) as error:
        Formula.parse(text)

    assert error.value.position == position
    assert error.value.source == text
    rendered = str(error.value).splitlines()
    assert rendered[1] == f"  {text}"
    assert rendered[2] == "  " + " " * position + "^"


def test_a_formula_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        Formula.parse(["choice ~ stimulus"])  # type: ignore[arg-type]


def test_a_design_error_raised_by_a_term_is_re_reported_at_its_position() -> None:
    with pytest.raises(FormulaError, match="lags must be in increasing order") as error:
        Formula.parse("choice ~ stimulus + lag(choice, [2, 1])").to_design()

    assert error.value.position == 20


def test_colliding_feature_names_fail_at_declaration_not_at_the_first_build() -> None:
    with pytest.raises(FormulaError, match="feature names collide"):
        Formula.parse("~ stimulus + numeric(phase, name='stimulus')").to_design()


# --------------------------------------------------------------------------------------
# Column references are checked against a study at declaration time
# --------------------------------------------------------------------------------------


def test_a_mistyped_column_is_caught_at_declaration_with_a_suggestion() -> None:
    formula = Formula.parse("choice ~ stimulis")

    with pytest.raises(FormulaError, match="did you mean 'stimulus'") as error:
        formula.validate(make_study())

    assert error.value.position == 9


def test_the_response_columns_are_checked_too() -> None:
    with pytest.raises(FormulaError, match="no column 'latency'"):
        Formula.parse("latency | choice ~ stimulus").validate(make_study())


def test_a_history_reset_column_is_checked_like_any_other_reference() -> None:
    with pytest.raises(FormulaError, match="no column 'cohort'"):
        Formula.parse("~ lag(choice, 1, reset_by=['cohort'])").validate(make_study())


def test_a_group_grouping_column_is_checked_before_the_group_is_refused() -> None:
    with pytest.raises(FormulaError, match="no column 'litter'"):
        Formula.parse("choice ~ (1 | litter)").validate(make_study())


def test_validation_accepts_a_study_that_declares_every_referenced_column() -> None:
    formula = Formula.parse(
        "response_time | choice ~ stimulus * C(condition, ['train', 'probe']) + lag(choice, 1)"
    )

    assert formula.validate(make_study()) is None


def test_validate_requires_a_study() -> None:
    with pytest.raises(TypeError, match="must be a Study"):
        Formula.parse("~ stimulus").validate(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Term-level records stay usable on their own
# --------------------------------------------------------------------------------------


def test_each_parsed_term_reports_its_columns_and_whether_it_reads_rows() -> None:
    lag = LagFormulaTerm(column="choice", lags=(1,), reset_by=("subject", "session"))
    standardize = StandardizeFormulaTerm(column="stimulus")

    assert lag.required_columns == ("choice", "subject", "session")
    assert not lag.data_dependent
    assert str(lag) == "lag(choice, 1)"
    assert standardize.data_dependent
    assert str(standardize) == "scale(stimulus)"


def test_describe_design_rejects_something_that_is_not_a_design() -> None:
    with pytest.raises(TypeError, match="must be a DesignSpec"):
        describe_design("1 + stimulus")  # type: ignore[arg-type]
