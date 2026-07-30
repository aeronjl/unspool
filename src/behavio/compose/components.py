"""The simpler processes a model can be mixed with.

Each of these is a :class:`~behavio.contracts.mixture.MixtureComponent`: a declared
distribution over exactly the columns some model scores, with no estimated parameters of
its own. Three of them cover every mixture the package used to hard-code, and the split
between them is by *what is observed* rather than by which family anticipated them --
:class:`UniformChoiceGuess` scores one binary column whether that column came from a GLM, a
psychometric baseline or anything else that scores a binary column, and
:class:`UniformCategoryGuess` scores a category code whether or not the model that produced
it was a multinomial logit.

Uniform in all three names means *over the outcome*, not over the trial: a guess that
ignores which options a trial offered would put mass on an action that could not be taken,
so :class:`UniformCategoryGuess` guesses uniformly over the options a row actually had.
That is per-row information, which is why a component is handed the study alongside the
outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.estimator import ModelDataError
from behavio.response_times import ResponseTimeSpec
from behavio.study import REQUIRED_COLUMNS, Study
from behavio.task import ChoiceSpec, TaskValidationError

__all__ = [
    "UniformCategoryGuess",
    "UniformChoiceGuess",
    "UniformResponseGuess",
]

LAPSE_RATE = "lapse_rate"
"""What a guessing process calls its weight, in the reported coordinate."""


@dataclass(frozen=True, slots=True)
class UniformChoiceGuess:
    """A stimulus-independent Bernoulli response on one binary outcome column.

    ``probability`` is the chance of emitting a one and is **declared**, not estimated.
    ``0.5`` is the unbiased guess a symmetric two-alternative task implies and is the
    default; a declared asymmetry is a legitimate and cheap thing to say, while an
    *estimated* one is a second free parameter and belongs in a link -- see
    :mod:`behavio.contracts.mixture` for why that line is drawn where it is.
    """

    outcome: str = "choice"
    probability: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS:
            raise ValueError("outcome cannot replace a required Study column")
        if not np.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError("probability must lie strictly between zero and one")

    @property
    def component_name(self) -> str:
        return "lapse"

    @property
    def signature(self) -> str:
        return f"uniform-choice-guess[outcome={self.outcome};probability={self.probability:g}]"

    @property
    def weight_name(self) -> str:
        return LAPSE_RATE

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
        """Refuse a categorical model, whose outcome is a code and not a coin."""

        categories = getattr(model, "categories", None)
        if categories is not None:
            return (
                f"its outcome is one of {len(tuple(categories))} categories rather than a "
                "binary choice; mix it with UniformCategoryGuess instead"
            )
        return None

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the Bernoulli log probability of each observed zero or one."""

        observed = np.asarray(outcomes, dtype=np.float64)
        if observed.ndim != 1:
            raise ValueError("a binary guess scores one value per row")
        density = np.full(observed.shape, -np.inf, dtype=np.float64)
        upper = observed == 1.0
        lower = observed == 0.0
        density[upper] = float(np.log(self.probability))
        density[lower] = float(np.log1p(-self.probability))
        return density

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        """Return the constant guessing probability, one value per row."""

        return np.full(len(study), float(self.probability), dtype=np.float64)

    def simulate_outcomes(
        self,
        study: Study,
        rows: NDArray[np.intp],
        *,
        generator: np.random.Generator,
    ) -> Mapping[str, NDArray[Any]]:
        """Draw independent guesses for the rows this process was chosen on."""

        draws = generator.binomial(1, self.probability, len(rows)).astype(np.int8)
        return {self.outcome: draws}


@dataclass(frozen=True, slots=True)
class UniformCategoryGuess:
    """A uniform draw over the options a trial actually offered.

    The choice specification is the model's own, and :meth:`mixture_refusal` checks that
    the category coordinate agrees: a guess and a model that disagree about which code
    means which action would score the wrong outcome silently, and no arrangement of
    members can be inspected to catch it.
    """

    choice: ChoiceSpec
    include_omission: bool = False
    omission_label: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.choice, ChoiceSpec):
            raise TypeError("choice must be a ChoiceSpec")
        if not isinstance(self.include_omission, bool):
            raise ValueError("include_omission must be boolean")
        if self.include_omission:
            if not self.choice.omission_values:
                raise ValueError("an omission-aware guess requires ChoiceSpec.omission_values")
            label = (
                self.choice.omission_values[0]
                if self.omission_label is None
                else _scalar(self.omission_label)
            )
            if label not in [_scalar(value) for value in self.choice.omission_values]:
                raise ValueError("omission_label must be one of ChoiceSpec.omission_values")
            object.__setattr__(self, "omission_label", label)
        elif self.omission_label is not None:
            raise ValueError("omission_label requires include_omission=True")

    @property
    def categories(self) -> tuple[Any, ...]:
        """The category coordinate this guess is uniform over, in model order."""

        if self.include_omission:
            return (*self.choice.options, self.omission_label)
        return tuple(self.choice.options)

    @property
    def component_name(self) -> str:
        return "lapse"

    @property
    def signature(self) -> str:
        return (
            f"uniform-category-guess[choice={self.choice.column!r};categories={self.categories!r}]"
        )

    @property
    def weight_name(self) -> str:
        return LAPSE_RATE

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.choice.column,)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        return ()

    @property
    def prediction_width(self) -> int:
        return len(self.categories)

    def mixture_refusal(self, model: Any) -> str | None:
        """Refuse a model whose category coordinate is not the one guessed over."""

        categories = getattr(model, "categories", None)
        if categories is None:
            return (
                "it scores a single number rather than a category code; mix it with "
                "UniformChoiceGuess instead"
            )
        if tuple(categories) != self.categories:
            return (
                f"the model's categories {tuple(categories)!r} are not the categories this "
                f"guess is uniform over, {self.categories!r}"
            )
        return None

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return ``-log`` of the number of options each row offered."""

        available = self.availability(study)
        codes = np.asarray(outcomes, dtype=np.intp)
        if codes.ndim != 1 or len(codes) != len(available):
            raise ValueError("a category guess scores one code per row")
        if codes.size and (codes.min() < 0 or codes.max() >= len(self.categories)):
            raise ValueError("category codes must index a declared category")
        offered = available[np.arange(len(codes)), codes]
        counts = available.sum(axis=1)
        density = np.full(codes.shape, -np.inf, dtype=np.float64)
        usable = offered & (counts > 0)
        density[usable] = -np.log(counts[usable].astype(np.float64))
        return density

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        """Return the uniform-over-available distribution, one row per trial."""

        available = self.availability(study).astype(np.float64)
        counts = available.sum(axis=1)
        if np.any(counts <= 0):
            raise ModelDataError("every trial must offer at least one option to guess between")
        return np.asarray(available / counts[:, None], dtype=np.float64)

    def simulate_outcomes(
        self,
        study: Study,
        rows: NDArray[np.intp],
        *,
        generator: np.random.Generator,
    ) -> Mapping[str, NDArray[Any]]:
        """Draw one uniformly chosen available option per selected row."""

        probability = self.prediction_probability(study)
        labels = [
            self.categories[int(generator.choice(len(self.categories), p=probability[int(row)]))]
            for row in rows
        ]
        return {self.choice.column: np.asarray(labels, dtype=object)}

    def availability(self, study: Study) -> NDArray[np.bool_]:
        """Return which categories each row offered, omission always among them."""

        try:
            actions = self.choice.availability(study)
        except TaskValidationError as error:
            raise ModelDataError(str(error)) from error
        if self.include_omission:
            return np.column_stack((actions, np.ones(len(study), dtype=np.bool_)))
        return np.asarray(actions, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class UniformResponseGuess:
    """A response emitted without a decision: uniform latency, independent choice.

    This is the process a drift-diffusion model used to name a *contaminant*. It scores the
    joint observation the model scores -- which boundary and when -- with response time
    uniform over ``time_bounds`` in canonical seconds and choice an independent Bernoulli
    draw. Both are declared; only the weight is estimated.

    ``time_bounds`` is in **seconds** whatever unit the response-time column is recorded
    in, because the outcome coordinate a model hands a component is the canonical one. The
    unit reappears only in :meth:`simulate_outcomes`, which has to write a study column
    back.
    """

    time_bounds: tuple[float, float]
    outcome: str = "choice"
    response_time: ResponseTimeSpec = field(default_factory=ResponseTimeSpec)
    choice_probability: float = 0.5

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.time_bounds)
        if len(bounds) != 2:
            raise ValueError("time_bounds must contain exactly two values")
        if not all(np.isfinite(value) for value in bounds) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("time_bounds must be finite, positive and increasing")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if not isinstance(self.response_time, ResponseTimeSpec):
            raise TypeError("response_time must be a ResponseTimeSpec")
        if self.outcome == self.response_time.column:
            raise ValueError("outcome and response-time columns must be distinct")
        if not np.isfinite(self.choice_probability) or not 0 < self.choice_probability < 1:
            raise ValueError("choice_probability must lie strictly between zero and one")
        object.__setattr__(self, "time_bounds", bounds)

    @property
    def component_name(self) -> str:
        return "contaminant"

    @property
    def signature(self) -> str:
        return (
            f"uniform-response-guess[outcome={self.outcome};"
            f"response_time={self.response_time.column}:{self.response_time.unit.value};"
            f"time_bounds={self.time_bounds};choice_probability={self.choice_probability:g}]"
        )

    @property
    def weight_name(self) -> str:
        return "contaminant_rate"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome, self.response_time.column)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        return ("choice", "response_time")

    @property
    def prediction_width(self) -> int:
        return 1

    def mixture_refusal(self, model: Any) -> str | None:
        """Refuse a model whose response-time column is read in another unit."""

        declared = getattr(model, "response_time", None)
        if isinstance(declared, ResponseTimeSpec) and declared != self.response_time:
            return (
                f"the model reads {declared.column!r} in {declared.unit.value} while this "
                f"process writes {self.response_time.column!r} in "
                f"{self.response_time.unit.value}"
            )
        return None

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the normalized joint log density of each choice and latency."""

        observed = np.asarray(outcomes, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != 2:
            raise ValueError("a joint response guess scores a choice and a latency per row")
        choices = observed[:, 0]
        seconds = observed[:, 1]
        lower, upper = self.time_bounds
        density = np.full(len(observed), -np.inf, dtype=np.float64)
        valid = (
            np.isfinite(seconds)
            & (seconds >= lower)
            & (seconds <= upper)
            & ((choices == 0) | (choices == 1))
        )
        if np.any(valid):
            log_choice = np.where(
                choices[valid] == 1,
                np.log(self.choice_probability),
                np.log1p(-self.choice_probability),
            )
            density[valid] = log_choice - np.log(upper - lower)
        return density

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        """Return the constant upper-boundary probability of a contaminated trial."""

        return np.full(len(study), float(self.choice_probability), dtype=np.float64)

    def simulate_outcomes(
        self,
        study: Study,
        rows: NDArray[np.intp],
        *,
        generator: np.random.Generator,
    ) -> Mapping[str, NDArray[Any]]:
        """Draw an independent choice and a uniform latency for each selected row."""

        count = len(rows)
        lower, upper = self.time_bounds
        choices = generator.binomial(1, self.choice_probability, count).astype(np.int8)
        seconds = generator.uniform(lower, upper, count)
        return {
            self.outcome: choices,
            self.response_time.column: seconds / self.response_time.unit.seconds_per_unit,
        }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
