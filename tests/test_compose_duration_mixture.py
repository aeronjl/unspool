"""``mix()`` on the two families whose observation is a bare duration.

The last unopened cell of the combinator grid, and it was never a missing combinator. A
reproduced duration and a patch residence time are one continuous number per row, none of the
three shipped components writes one, and ``require_mixable`` compares scored columns before it
reaches any arithmetic. :class:`~behavio.compose.UniformDurationGuess` is that component.

What is tested here is the three things that are *not* obvious once it exists.

*The bounds are a declaration.* A uniform over a duration does not exist until an interval is
named, so the interval is a constructor argument, it is in the signature, and it is in the
outcome column's own units -- unlike ``UniformResponseGuess``, whose bounds are in canonical
seconds because a drift-diffusion model declares a unit and these families do not.

*A censored row is not scored by a density.* ``PatchLeaving`` scores a visit that was still in
progress by :math:`\\log S(c)`, so the component must contribute the probability its own
duration exceeds the same :math:`c`. Getting this wrong is not a rounding error and the test
below measures it rather than asserting it: the same study fitted with a censoring-blind
component recovers less than half the weight it was simulated with.

*The prediction is a density, not a probability.* A mixture's predicted density is the
weighted average of two densities on the model's own grid, which is a second average from the
one ``blended_prediction`` performs and needs no new member of the component contract -- a
component's ``pointwise_log_density`` is already a function of an outcome, so a grid point is
a question it can already answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from behavio.adapters.estimator_conformance import check_behaviour_estimator
from behavio.compose import (
    UniformChoiceGuess,
    UniformDurationGuess,
    hierarchical,
    mix,
    smooth,
)
from behavio.compose.mixture import MixtureRowModel, blended_density, blended_prediction
from behavio.contracts.estimator import DensityPrediction, ModelDataError, PredictionMode
from behavio.contracts.mixture import MixtureComponent
from behavio.models.multinomial import MultinomialLogit
from behavio.models.patch_leaving import PatchLeaving
from behavio.models.scalar_timing import DurationReproduction, TemporalBisection
from behavio.task.spec import ChoiceSpec
from behavio.trials import Study

REPRODUCTION_BOUNDS = (0.05, 8.0)
RESIDENCE_BOUNDS = (0.0, 3.0)
OBSERVATION_LIMIT = "observation_limit"


def frame(n_rows: int, *, n_subjects: int = 1, n_sessions: int = 1, **columns: Any) -> Study:
    """Return a study skeleton with ``n_rows`` rows per subject and session."""

    return Study(
        {
            "subject": [
                f"m{subject}" for subject in range(n_subjects) for _ in range(n_sessions * n_rows)
            ],
            "session": [
                session
                for _ in range(n_subjects)
                for session in range(n_sessions)
                for _ in range(n_rows)
            ],
            "session_order": [
                session
                for _ in range(n_subjects)
                for session in range(n_sessions)
                for _ in range(n_rows)
            ],
            "trial": list(range(n_rows)) * (n_subjects * n_sessions),
            **{name: np.asarray(value) for name, value in columns.items()},
        }
    )


def reproduction_design(
    *,
    n_rows: int = 250,
    n_subjects: int = 1,
    n_sessions: int = 1,
    targets: Any = None,
    seed: int = 4,
) -> Study:
    generator = np.random.default_rng(seed)
    rows = n_rows * n_subjects * n_sessions
    levels = (0.5, 1.0, 2.0, 4.0) if targets is None else targets
    return frame(
        n_rows,
        n_subjects=n_subjects,
        n_sessions=n_sessions,
        target_duration=generator.choice(np.asarray(levels, dtype=np.float64), rows),
    )


def patch_design(
    *,
    n_rows: int = 200,
    n_subjects: int = 3,
    n_sessions: int = 2,
    limits: Any = (1.0, 1.5, 2.5),
    seed: int = 3,
) -> Study:
    generator = np.random.default_rng(seed)
    rows = n_rows * n_subjects * n_sessions
    return frame(
        n_rows,
        n_subjects=n_subjects,
        n_sessions=n_sessions,
        patch_yield=generator.choice(np.asarray([4.0, 8.0, 16.0]), rows),
        patch_decay=generator.choice(np.asarray([0.4, 0.9]), rows),
        **{OBSERVATION_LIMIT: generator.choice(np.asarray(limits, dtype=np.float64), rows)},
    )


def reproduction_mixture(**changes: Any) -> MixtureRowModel:
    arguments: dict[str, Any] = {"weight_bounds": (0.0, 0.3)}
    arguments.update(changes)
    component = arguments.pop("component", None) or UniformDurationGuess(
        REPRODUCTION_BOUNDS, "reproduced_duration"
    )
    return mix(DurationReproduction(), component, **arguments)


def patch_mixture(**changes: Any) -> MixtureRowModel:
    arguments: dict[str, Any] = {"weight_bounds": (0.0, 0.4)}
    arguments.update(changes)
    component = arguments.pop("component", None) or UniformDurationGuess(
        RESIDENCE_BOUNDS, "residence_time", censoring_time_column=OBSERVATION_LIMIT
    )
    return mix(PatchLeaving(censoring_time_column=OBSERVATION_LIMIT), component, **arguments)


@dataclass(frozen=True, slots=True)
class CensoringBlindGuess:
    """The same uniform, scoring every row by its density: what the shortcut costs.

    A complete :class:`~behavio.contracts.mixture.MixtureComponent` in every other respect,
    and deliberately so -- the point of the measurement below is that nothing about it looks
    wrong. It writes the right column, adds no parameter, simulates what it scores, and
    differs from the shipped component in one line.
    """

    duration_bounds: tuple[float, float]
    outcome: str

    @property
    def component_name(self) -> str:
        return "contaminant"

    @property
    def signature(self) -> str:
        return f"censoring-blind-guess[outcome={self.outcome};bounds={self.duration_bounds}]"

    @property
    def weight_name(self) -> str:
        return "contaminant_rate"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        return ()

    @property
    def prediction_width(self) -> int:
        return 1

    def mixture_refusal(self, model: Any) -> str | None:
        del model
        return None

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        del study
        observed = np.asarray(outcomes, dtype=np.float64)
        lower, upper = self.duration_bounds
        density = np.full(observed.shape, -np.inf, dtype=np.float64)
        inside = (observed >= lower) & (observed <= upper)
        density[inside] = -float(np.log(upper - lower))
        return density

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        lower, upper = self.duration_bounds
        return np.full(len(study), 1.0 / (upper - lower), dtype=np.float64)

    def simulate_outcomes(
        self, study: Study, rows: NDArray[np.intp], *, generator: np.random.Generator
    ) -> Mapping[str, NDArray[Any]]:
        del study
        return {self.outcome: generator.uniform(*self.duration_bounds, len(rows))}


# --------------------------------------------------------------------------------------
# What the component declares
# --------------------------------------------------------------------------------------


def test_the_component_satisfies_the_contract_and_puts_its_interval_in_the_signature() -> None:
    component = UniformDurationGuess((0.25, 4.0), "reproduced_duration")

    assert isinstance(component, MixtureComponent)
    assert component.scored_columns == ("reproduced_duration",)
    assert component.outcome_channels == ()
    assert component.weight_name == "contaminant_rate"
    assert component.component_name == "contaminant"
    assert "duration_bounds=0.25,4" in component.signature
    assert "censoring=none" in component.signature
    assert (
        OBSERVATION_LIMIT
        in UniformDurationGuess(
            (0.0, 3.0), "residence_time", censoring_time_column=OBSERVATION_LIMIT
        ).signature
    )


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"duration_bounds": (4.0, 1.0), "outcome": "d"}, "increasing"),
        ({"duration_bounds": (-1.0, 1.0), "outcome": "d"}, "non-negative"),
        ({"duration_bounds": (0.0, np.inf), "outcome": "d"}, "finite"),
        ({"duration_bounds": (0.0, 1.0), "outcome": ""}, "non-empty column name"),
        ({"duration_bounds": (0.0, 1.0), "outcome": "trial"}, "required Study column"),
        (
            {"duration_bounds": (0.0, 1.0), "outcome": "d", "censoring_time_column": "d"},
            "different columns",
        ),
    ],
)
def test_an_interval_that_is_not_an_interval_is_refused_at_construction(
    arguments: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        UniformDurationGuess(**arguments)


def test_the_declared_interval_is_in_the_outcome_column_s_own_units() -> None:
    """The one place ``UniformResponseGuess``'s answer does not generalise.

    A drift-diffusion model declares a ``ResponseTimeSpec`` and hands components canonical
    seconds; a scalar-timing model hands back the column verbatim. So the same study recorded
    in milliseconds needs bounds a thousand times larger, and the two components disagree.
    """

    seconds = UniformDurationGuess((0.05, 8.0), "reproduced_duration")
    study = reproduction_design(n_rows=64)
    outcomes = np.full(len(study), 4.0)

    assert np.all(np.isfinite(seconds.pointwise_log_density(study, outcomes)))
    milliseconds = UniformDurationGuess((50.0, 8000.0), "reproduced_duration")
    assert np.all(np.isneginf(milliseconds.pointwise_log_density(study, outcomes)))


# --------------------------------------------------------------------------------------
# The two cells, opened
# --------------------------------------------------------------------------------------


def test_a_contaminant_on_a_reproduction_is_one_extra_parameter_and_recovers() -> None:
    model = reproduction_mixture()
    design = reproduction_design(n_rows=900)
    truth = model.parameters_from_weight({"clock_rate": 1.0, "weber_fraction": 0.15}, 0.12)

    assert model.parameter_names == ("clock_rate_log", "weber_fraction_log", "mixture_logit")
    assert model.natural_names == ("clock_rate", "weber_fraction", "contaminant_rate")

    simulation = model.simulate_with_component(design, truth, seed=7)
    fitted = model.to_natural(model.fit(simulation.study).estimates)

    assert simulation.n_from_component == pytest.approx(0.12 * len(design), abs=0.03 * len(design))
    assert fitted["contaminant_rate"] == pytest.approx(0.12, abs=0.04)
    assert fitted["clock_rate"] == pytest.approx(1.0, abs=0.08)
    assert fitted["weber_fraction"] == pytest.approx(0.15, abs=0.04)


def test_a_contaminant_on_patch_leaving_recovers_the_weight_through_a_censored_likelihood() -> None:
    model = patch_mixture()
    design = patch_design()
    truth = model.parameters_from_weight({"giving_up_rate": 1.2, "decision_noise": 0.25}, 0.2)

    simulation = model.simulate_with_component(design, truth, seed=11)
    times = np.asarray(simulation.study["residence_time"], dtype=np.float64)
    limits = np.asarray(simulation.study[OBSERVATION_LIMIT], dtype=np.float64)
    censored = times >= limits - 1e-9
    fitted = model.to_natural(model.fit(simulation.study).estimates)

    assert float(np.mean(censored)) > 0.4, "the demonstration needs censoring to be doing work"
    assert fitted["contaminant_rate"] == pytest.approx(0.2, abs=0.04)
    assert fitted["giving_up_rate"] == pytest.approx(1.2, abs=0.15)


def test_a_simulated_contaminant_never_writes_a_duration_past_its_own_row_s_limit() -> None:
    """The truncation the simulator has to do, and what omitting it would produce.

    A contaminant that emitted an untruncated draw would write durations longer than the
    session that recorded them, and ``read_censoring`` refuses such a study outright -- so the
    omission is not a subtle bias, it is a model that cannot read back what it simulated.
    """

    model = patch_mixture()
    design = patch_design(n_rows=120, n_subjects=1, n_sessions=1)
    truth = model.parameters_from_weight({"giving_up_rate": 1.2, "decision_noise": 0.25}, 0.35)

    simulation = model.simulate_with_component(design, truth, seed=5)
    times = np.asarray(simulation.study["residence_time"], dtype=np.float64)
    limits = np.asarray(simulation.study[OBSERVATION_LIMIT], dtype=np.float64)

    assert simulation.n_from_component > 0
    assert np.all(times <= limits + 1e-12)
    assert np.any(times[simulation.from_component] >= limits[simulation.from_component] - 1e-9)
    # And the study reads back, which is the property the truncation buys.
    model.fit(simulation.study)


# --------------------------------------------------------------------------------------
# What a censored row contributes, and what getting it wrong costs
# --------------------------------------------------------------------------------------


def test_a_censored_row_is_scored_by_the_probability_the_guess_outlasts_the_limit() -> None:
    component = UniformDurationGuess(
        (0.0, 4.0), "residence_time", censoring_time_column=OBSERVATION_LIMIT
    )
    study = frame(
        3,
        patch_yield=np.full(3, 8.0),
        patch_decay=np.full(3, 0.5),
        **{OBSERVATION_LIMIT: np.asarray([2.0, 2.0, 6.0])},
    )
    scores = component.pointwise_log_density(study, np.asarray([1.0, 2.0, 1.0]))

    # An uncensored row is a density; the row that reached its limit is P(X > 2) = 1/2; the
    # row whose limit lies beyond the declared interval is unreachable rather than certain.
    assert scores[0] == pytest.approx(-np.log(4.0))
    assert scores[1] == pytest.approx(np.log(0.5))
    assert scores[2] == pytest.approx(-np.log(4.0))
    assert component.pointwise_log_density(study, np.asarray([1.0, 2.0, 6.0]))[2] == -np.inf


def test_ignoring_the_distinction_biases_the_recovered_weight_downwards() -> None:
    """Measured, not asserted: the same study, two components differing in one line.

    A survival probability is dimensionless and a density is one over time, so a
    censoring-blind component offers a censored row a number an order of magnitude smaller
    than the one it should. Every censored row therefore looks like a row the contaminant
    could not have produced, and with three in five rows censored the recovered weight lands
    at less than half of what the study was simulated with -- while the model's own
    parameters barely move, which is what makes the error hard to notice from a fit table.
    """

    model = patch_mixture()
    design = patch_design()
    truth = model.parameters_from_weight({"giving_up_rate": 1.2, "decision_noise": 0.25}, 0.2)
    simulation = model.simulate_with_component(design, truth, seed=11)

    blind = mix(
        PatchLeaving(censoring_time_column=OBSERVATION_LIMIT),
        CensoringBlindGuess(RESIDENCE_BOUNDS, "residence_time"),
        weight_bounds=(0.0, 0.4),
    )
    aware_fit = model.fit(simulation.study)
    blind_fit = blind.fit(simulation.study)
    aware = model.to_natural(aware_fit.estimates)
    careless = blind.to_natural(blind_fit.estimates)

    assert aware["contaminant_rate"] == pytest.approx(0.2, abs=0.04)
    assert careless["contaminant_rate"] < 0.5 * aware["contaminant_rate"]
    assert careless["giving_up_rate"] == pytest.approx(aware["giving_up_rate"], rel=0.1)
    # The blind mixture is also a worse account of the same data, on the same rows.
    assert float(np.sum(blind.pointwise_log_prob(simulation.study, blind_fit))) < float(
        np.sum(model.pointwise_log_prob(simulation.study, aware_fit))
    )


def test_the_two_processes_must_read_the_same_observation_limit() -> None:
    censored = PatchLeaving(censoring_time_column=OBSERVATION_LIMIT)
    blind_component = UniformDurationGuess(RESIDENCE_BOUNDS, "residence_time")
    aware_component = UniformDurationGuess(
        RESIDENCE_BOUNDS, "residence_time", censoring_time_column=OBSERVATION_LIMIT
    )

    with pytest.raises(TypeError, match="both must read the same limit"):
        mix(censored, blind_component)
    with pytest.raises(TypeError, match="both must read the same limit"):
        mix(PatchLeaving(), aware_component)
    with pytest.raises(TypeError, match="declares no censoring at all"):
        aware = UniformDurationGuess(
            REPRODUCTION_BOUNDS, "reproduced_duration", censoring_time_column="limit"
        )
        mix(DurationReproduction(), aware)


def test_a_missing_or_unreadable_limit_column_is_a_model_data_error() -> None:
    component = UniformDurationGuess(
        RESIDENCE_BOUNDS, "residence_time", censoring_time_column=OBSERVATION_LIMIT
    )
    study = frame(4, patch_yield=np.full(4, 8.0), patch_decay=np.full(4, 0.5))

    with pytest.raises(ModelDataError, match="missing censoring column"):
        component.pointwise_log_density(study, np.zeros(4))


# --------------------------------------------------------------------------------------
# What the mixture predicts, which is a density and not a probability
# --------------------------------------------------------------------------------------


def test_the_mixed_density_is_the_two_component_average_at_every_grid_point() -> None:
    model = reproduction_mixture()
    design = reproduction_design(n_rows=120)
    truth = model.parameters_from_weight({"clock_rate": 1.0, "weber_fraction": 0.2}, 0.15)
    study = model.simulate(design, truth, seed=2)
    fit = model.fit(study)

    prediction = model.predict(study, fit)
    assert isinstance(prediction, DensityPrediction)
    assert prediction.outcome == model.density_outcome == "reproduced_duration"

    weight = float(model.weight(fit))
    inner = np.tile(np.asarray(fit.estimates[:-1], dtype=np.float64), (len(study), 1))
    own = DurationReproduction().predict_rows(study, inner, mode=PredictionMode.FILTERED)
    lower, upper = REPRODUCTION_BOUNDS
    inside = (prediction.grid >= lower) & (prediction.grid <= upper)
    component = np.where(inside, 1.0 / (upper - lower), 0.0)
    expected = (1.0 - weight) * np.asarray(own.density) + weight * component[None, :]

    assert np.array_equal(prediction.grid, own.grid)
    assert np.allclose(np.asarray(prediction.density), expected)
    assert np.all(prediction.total_mass <= 1.0)
    assert np.array_equal(
        np.asarray(model.predict_density(study, fit).density), np.asarray(prediction.density)
    )


def test_a_censored_duration_mixture_retains_the_survival_score() -> None:
    """Mixing carries both event density and survival probability through one prediction."""

    model = patch_mixture()
    design = patch_design(n_rows=60, n_subjects=1, n_sessions=1)
    truth = model.parameters_from_weight({"giving_up_rate": 1.2, "decision_noise": 0.25}, 0.25)
    study = model.simulate(design, truth, seed=9)
    fit = model.fit(study)

    prediction = model.predict(study, fit)
    times = np.asarray(study["residence_time"], dtype=np.float64)
    limits = np.asarray(study[OBSERVATION_LIMIT], dtype=np.float64)
    censored = times >= limits - 1e-9

    assert isinstance(prediction, DensityPrediction)
    assert np.all(np.asarray(prediction.density) >= 0.0)
    assert np.any(censored)
    scored = model.pointwise_log_prob(study, fit)
    tabulated = prediction.observed_log_density(times, censored=censored)
    assert np.allclose(scored, tabulated, atol=1e-3)


def test_a_tabulated_density_is_refused_by_the_probability_average_that_cannot_take_it() -> None:
    model = reproduction_mixture()
    design = reproduction_design(n_rows=40)
    truth = model.parameters_from_weight({"clock_rate": 1.0, "weber_fraction": 0.2}, 0.1)
    study = model.simulate(design, truth, seed=3)
    rows = np.tile(np.asarray([0.0, float(np.log(0.2))]), (len(study), 1))
    density = DurationReproduction().predict_rows(study, rows, mode=PredictionMode.FILTERED)
    weight = np.full(len(study), 0.1)

    with pytest.raises(TypeError, match="blended_density"):
        blended_prediction(density, weight, np.full((len(study), 1), 0.5))
    with pytest.raises(ValueError, match="the model's own grid"):
        blended_density(density, weight, np.zeros((len(study), 3)))


def test_a_mixed_duration_model_is_an_ordinary_estimator_to_everything_downstream() -> None:
    model = reproduction_mixture()
    design = reproduction_design(n_rows=200, n_sessions=2)
    truth = model.parameters_from_weight({"clock_rate": 1.0, "weber_fraction": 0.15}, 0.12)
    study = model.simulate(design, truth, seed=6)

    report = check_behaviour_estimator(model, study)

    assert report.passed, [
        (check.name, check.detail) for check in report.checks if check.status.value == "failed"
    ]


# --------------------------------------------------------------------------------------
# The stack, and what the weight may vary over
# --------------------------------------------------------------------------------------


def test_the_full_stack_composes_over_a_duration_mixture() -> None:
    model = patch_mixture()
    design = patch_design()
    truth = model.parameters_from_weight({"giving_up_rate": 1.2, "decision_noise": 0.25}, 0.2)
    study = model.simulate(design, truth, seed=11)

    stack = hierarchical(
        smooth(model, over="session_order", knots=(0.0, 1.0), parameters=("mixture_logit",)),
        over="subject",
        parameters=("giving_up_rate_log",),
        scale=0.4,
    )
    fit = stack.fit(study)

    assert stack.parameter_names == (
        "giving_up_rate_log",
        "decision_noise_log",
        "mixture_logit[session_order=0]",
        "mixture_logit[session_order=1]",
    )
    assert fit.diagnostics.converged
    assert isinstance(stack.predict(study, fit), DensityPrediction)


def test_a_hierarchy_over_the_weight_gives_a_per_subject_contaminant_rate() -> None:
    model = reproduction_mixture()
    design = reproduction_design(n_rows=250, n_subjects=3)
    truth = model.parameters_from_weight({"clock_rate": 1.0, "weber_fraction": 0.15}, 0.12)
    study = model.simulate(design, truth, seed=8)

    pooled = hierarchical(model, over="subject", parameters=("mixture_logit",), scale=0.6)
    fit = pooled.fit(study)

    per_subject = [
        float(
            model.to_natural(list(fit.parameters_for(f"m{subject}").values()))["contaminant_rate"]
        )
        for subject in range(3)
    ]

    assert pooled.parameter_names == ("clock_rate_log", "weber_fraction_log", "mixture_logit")
    assert pooled.varying_parameters == ("mixture_logit",)
    assert fit.diagnostics.converged
    assert fit.group_parameters.shape == (3, 1)
    assert all(0.0 < rate < 0.3 for rate in per_subject)
    assert len(set(per_subject)) == 3, "each subject gets its own contaminant rate"


# --------------------------------------------------------------------------------------
# Which identifiability findings mean the same thing here
# --------------------------------------------------------------------------------------


def test_a_single_target_duration_cannot_identify_a_contaminant_rate() -> None:
    """``unidentified_mixture`` carries over with its meaning intact.

    One target duration is one reproduction density for every row, so a wider clock with less
    contamination and a narrower one with more predict the same thing everywhere -- the same
    trade-off between a lapse rate and a shallow psychometric slope, in the coordinates of a
    scalar clock.
    """

    base = DurationReproduction()
    design = reproduction_design(n_rows=200, targets=(2.0,))
    study = base.simulate(
        design, base.parameters_from_components(clock_rate=1.0, weber_fraction=0.2), seed=1
    )
    codes = [finding.code for finding in reproduction_mixture().describe(study).findings]

    assert "unidentified_mixture" in codes
    assert "narrow_target_range" in codes, "the wrapped model's own finding is forwarded"


def test_a_single_patch_type_cannot_identify_a_contaminant_rate_either() -> None:
    base = PatchLeaving(censoring_time_column=OBSERVATION_LIMIT)
    design = frame(
        200,
        patch_yield=np.full(200, 8.0),
        patch_decay=np.full(200, 0.5),
        **{OBSERVATION_LIMIT: np.full(200, 5.0)},
    )
    study = base.simulate(
        design, base.parameters_from_components(giving_up_rate=1.2, decision_noise=0.25), seed=2
    )
    codes = [finding.code for finding in patch_mixture().describe(study).findings]

    assert codes == ["unidentified_mixture", "unidentified_leaving_rule"]


def test_a_contaminant_no_reproduction_could_have_come_from_reports_itself() -> None:
    base = DurationReproduction()
    design = reproduction_design(n_rows=150)
    study = base.simulate(
        design, base.parameters_from_components(clock_rate=1.0, weber_fraction=0.2), seed=1
    )
    unreachable = reproduction_mixture(
        component=UniformDurationGuess((50.0, 90.0), "reproduced_duration")
    )

    assert "unreachable_mixture_component" in [
        finding.code for finding in unreachable.describe(study).findings
    ]


def test_unreachability_means_less_under_censoring_and_the_finding_is_right_not_to_fire() -> None:
    """The one finding whose *meaning* shifts, and it shifts in the component's favour.

    A contaminant whose declared interval lies entirely beyond every observed residence time
    is not a process that could have produced nothing. A censored row says only that the visit
    outlasted the session, and a process that always outlasts the session is exactly
    consistent with that -- so the finding stays silent, correctly, where on an uncensored
    family it would fire.
    """

    base = PatchLeaving(censoring_time_column=OBSERVATION_LIMIT)
    design = patch_design(n_rows=150, n_subjects=1, n_sessions=1, limits=(1.0,))
    study = base.simulate(
        design, base.parameters_from_components(giving_up_rate=1.2, decision_noise=0.25), seed=2
    )
    beyond = patch_mixture(
        component=UniformDurationGuess(
            (20.0, 30.0), "residence_time", censoring_time_column=OBSERVATION_LIMIT
        )
    )
    codes = [finding.code for finding in beyond.describe(study).findings]

    assert "unreachable_mixture_component" not in codes
    assert "heavy_censoring" in codes


# --------------------------------------------------------------------------------------
# The components stay sorted by what is observed
# --------------------------------------------------------------------------------------


def test_the_two_ways_of_getting_the_kind_of_outcome_wrong_are_both_refused() -> None:
    """A binary column and a duration column are both one float per row.

    Neither direction is caught by comparing members or column names, which is exactly the
    case ``mixture_refusal`` exists for: a Bernoulli guess on a reproduced duration would
    score a continuous observation as if it were a coin, and a uniform-over-an-interval on a
    bisection report would treat a zero and a one as two points of a continuum.
    """

    with pytest.raises(TypeError, match="declares no density_outcome"):
        mix(TemporalBisection(), UniformDurationGuess((0.0, 1.0), "choice"))
    with pytest.raises(TypeError, match="tabulates a density"):
        mix(DurationReproduction(), UniformChoiceGuess(outcome="reproduced_duration"))

    categorical = UniformDurationGuess((0.0, 1.0), "action").mixture_refusal(
        MultinomialLogit(choice=ChoiceSpec(column="action", options=("a", "b", "c")))
    )
    assert categorical is not None
    assert "rather than a duration" in categorical


def test_the_shipped_components_still_sort_by_what_is_observed() -> None:
    duration = UniformDurationGuess(REPRODUCTION_BOUNDS, "reproduced_duration")

    assert isinstance(UniformChoiceGuess(), MixtureComponent)
    assert isinstance(duration, MixtureComponent)
    assert UniformChoiceGuess().weight_name == "lapse_rate"
    assert duration.weight_name == "contaminant_rate"
    assert duration.scored_columns == DurationReproduction().scored_columns
