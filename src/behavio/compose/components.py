"""The simpler processes a model can be mixed with.

Each of these is a :class:`~behavio.contracts.mixture.MixtureComponent`: a declared
distribution over exactly the columns some model scores, with no estimated parameters of
its own. Four of them cover every mixture the package used to hard-code, and the split
between them is by *what is observed* rather than by which family anticipated them --
:class:`UniformChoiceGuess` scores one binary column whether that column came from a GLM, a
psychometric baseline or anything else that scores a binary column, and
:class:`UniformCategoryGuess` scores a category code whether or not the model that produced
it was a multinomial logit.

Uniform in all four names means *over the outcome*, not over the trial: a guess that
ignores which options a trial offered would put mass on an action that could not be taken,
so :class:`UniformCategoryGuess` guesses uniformly over the options a row actually had.
That is per-row information, which is why a component is handed the study alongside the
outcome -- and it is the same channel :class:`UniformDurationGuess` reads a row's
observation limit down.

What "uniform" costs on an unbounded outcome
--------------------------------------------
A binary guess needs no declaration: uniform over two options is one half, and there is
nothing to argue about. A **duration** has unbounded support, so a uniform over it does not
exist until an interval is named, and naming one is a modelling statement rather than a
default -- see :class:`UniformDurationGuess`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.estimator import ModelDataError
from behavio.models._kernels.hazard import CENSORING_TOLERANCE
from behavio.task.response_times import ResponseTimeSpec
from behavio.task.spec import ChoiceSpec, TaskValidationError
from behavio.trials import REQUIRED_COLUMNS, Study

__all__ = [
    "UniformCategoryGuess",
    "UniformChoiceGuess",
    "UniformDurationGuess",
    "UniformResponseGuess",
]

_UNDECLARED: Final = object()
"""Sentinel telling "the model declares no censoring column" from "it declares ``None``"."""

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
        """Refuse a categorical model, and a continuous one, for the same reason.

        A category code is not a coin, and neither is a duration. The second half is the
        mirror of :meth:`UniformDurationGuess.mixture_refusal` and was added when the
        continuous families arrived: a binary column and a duration column are both one float
        per row, so nothing about the members distinguishes them and only a declaration can.
        """

        categories = getattr(model, "categories", None)
        if categories is not None:
            return (
                f"its outcome is one of {len(tuple(categories))} categories rather than a "
                "binary choice; mix it with UniformCategoryGuess instead"
            )
        density = getattr(model, "density_outcome", None)
        if density is not None:
            return (
                f"it tabulates a density over {density!r} rather than scoring a binary "
                "choice; mix it with UniformDurationGuess instead"
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


@dataclass(frozen=True, slots=True)
class UniformDurationGuess:
    """A duration emitted without timing anything: uniform over a declared interval.

    The component the two continuous families needed. A reproduced duration and a patch
    residence time are **bare durations** -- one number per row, no boundary reached and no
    option chosen -- so none of the three processes above writes what they score, and
    ``require_mixable`` refuses on the columns before it reaches any arithmetic. This scores
    exactly that column, so ``mix(DurationReproduction(), ...)`` and
    ``mix(PatchLeaving(), ...)`` become the same expression a lapse on a GLM already was.

    Where the bounds come from, and why they are here
    -------------------------------------------------
    ``duration_bounds`` is the interval the process is uniform over, **in the outcome
    column's own units**. It has to be declared because a duration has unbounded support:
    "uniform over the outcome" is a complete statement for a coin and an empty one for a
    positive real, and there is no default that is not a fabricated one.

    :class:`UniformResponseGuess` faced the same question and answered it with
    ``time_bounds``. Half of that answer generalises and half of it does not. The half that
    does is the *shape*: a two-element interval, declared and never estimated, outside which
    the component's density is zero and an observation is scored ``-inf``. The half that does
    not is the **unit**. A drift-diffusion model reads its latency column through a
    :class:`~behavio.task.response_times.ResponseTimeSpec` and hands components a latency in
    canonical seconds whatever the column is recorded in, so ``time_bounds`` can be in
    seconds and mean one thing. A scalar-timing or patch-leaving model declares no unit at
    all: :meth:`~behavio.models.scalar_timing.DurationReproduction.outcomes` returns the
    column verbatim. So these bounds are in the column's units, they appear in
    :attr:`signature` so a fit cannot be read without them, and a study recorded in
    milliseconds needs bounds in milliseconds.

    Declared rather than read off the data, which is the tempting alternative. Taking the
    interval from the observed minimum and maximum would make the component's normalising
    constant a function of the sample, so the mixture's likelihood would no longer be a
    likelihood -- and the widest observations, the ones a contaminant exists to explain,
    would be the ones setting the density that explains them.

    Why uniform, and why nothing with a parameter in it
    ---------------------------------------------------
    A uniform over an interval is the standard contaminant and is what
    ``UniformResponseTimeContaminant`` was before :func:`~behavio.compose.mix` absorbed it.
    An exponential or a lognormal outlier process would also be expressible here **provided
    its rate or width were declared**, and the reason to prefer the uniform is that its
    support is exactly the thing that has to be declared anyway: the interval is visible in
    the signature, and an observation outside it is reported as unreachable rather than
    absorbed at a small density.

    What is *not* expressible is an outlier process whose spread is estimated, and that line
    is drawn in :mod:`behavio.contracts.mixture` rather than here: a component with an
    estimated parameter is a second model inside the first, needing its own coordinate, its
    own box, its own group prior and its own place in every combinator. A mixture adds one
    parameter whatever it is mixed with, and that is what makes ``hierarchical(smooth(mix
    (model)))`` a sentence rather than a special case.

    Censoring, which is the part a duration mixture cannot skip
    -----------------------------------------------------------
    :class:`~behavio.models.patch_leaving.PatchLeaving` scores a row whose visit was still in
    progress when observation stopped by :math:`\\log S(c)` -- the probability the departure
    is still to come -- rather than by a density. A mixture of two processes is an average of
    what **each** of them says about the observation that was actually made, so on such a row
    the component must contribute the probability that *its* duration exceeds the same limit,

    .. math::

        S_{\\text{mix}}(c) = (1 - \\omega)\\,S_{\\text{model}}(c) + \\omega\\,S_{\\text{comp}}(c),

    and not its density there. Contributing the density instead is not a rounding error: a
    density is in units of one over time and a survival probability is dimensionless, so the
    component looks unable to account for censored rows and the weight is pulled towards the
    floor of its declared range in proportion to how many of them there are.

    ``censoring_time_column`` is therefore declared here as well, and :meth:`mixture_refusal`
    checks that it is the **same column the model reads**. Two processes disagreeing about
    which rows are still in progress would score different observations and average the
    results, and no arrangement of members reveals that; it is the same check
    :class:`UniformResponseGuess` makes about a response-time unit, for the same reason.

    A row counts as censored when its observed duration sits on its limit, which is
    :func:`~behavio.models._kernels.hazard.read_censoring`'s own rule stated so that it also
    answers the other question a mixture asks: *what is your density at this value?* A
    tabulated mixture prediction evaluates this component at grid points that are not
    observations, and a grid point above a row's limit is a hypothetical leaving time rather
    than a censored one. Since a duration may never exceed its declared limit -- the model
    refuses a study in which one does -- keying on the limit itself rather than on
    "at or above" agrees exactly with the model on every real observation, and still returns
    a density when asked about a value the row could not have shown.

    The simulator truncates for the same reason. A drawn duration longer than the row's limit
    is not an observation the study could contain, so it is written back at the limit, which
    is exactly what :meth:`~behavio.models.patch_leaving.PatchLeaving.simulate_rows` does to
    its own draws -- and it is what makes a simulated contaminant row *censored* rather than
    a contradiction the next ``fit`` refuses to read.
    """

    duration_bounds: tuple[float, float]
    outcome: str
    censoring_time_column: str | None = None

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.duration_bounds)
        if len(bounds) != 2:
            raise ValueError("duration_bounds must contain exactly two values")
        if not all(np.isfinite(value) for value in bounds) or not 0.0 <= bounds[0] < bounds[1]:
            raise ValueError("duration_bounds must be finite, non-negative and increasing")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS:
            raise ValueError("outcome cannot replace a required Study column")
        if self.censoring_time_column is not None:
            if (
                not isinstance(self.censoring_time_column, str)
                or not self.censoring_time_column
                or self.censoring_time_column in REQUIRED_COLUMNS
            ):
                raise ValueError("censoring_time_column must be a non-empty column name")
            if self.censoring_time_column == self.outcome:
                raise ValueError("the observation limit and the duration are different columns")
        object.__setattr__(self, "duration_bounds", bounds)

    @property
    def component_name(self) -> str:
        return "contaminant"

    @property
    def signature(self) -> str:
        lower, upper = self.duration_bounds
        censoring = self.censoring_time_column or "none"
        return (
            f"uniform-duration-guess[outcome={self.outcome};"
            f"duration_bounds={lower:g},{upper:g};censoring={censoring}]"
        )

    @property
    def weight_name(self) -> str:
        """``contaminant_rate``: this is a distribution over the outcome, not a guess.

        The distinction :mod:`behavio.contracts.mixture` draws is between a process that
        *guesses between the options a trial offered* and one that is simply a distribution
        over what was scored. A duration offers no options, so there is nothing to guess
        between and the weight is the share of trials a second, unmodelled process produced.
        """

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
        """Refuse a discrete outcome, and any disagreement about the observation limit.

        A component's log density is averaged with the model's, so the two have to be the
        same kind of number. A model that declares no ``density_outcome`` scores a
        *probability* -- one of finitely many outcomes -- and a uniform over an interval
        would silently treat a zero and a one as two points of a continuum, which no
        arrangement of members reveals because a binary column and a duration column are both
        one float per row.
        """

        categories = getattr(model, "categories", None)
        if categories is not None:
            return (
                f"its outcome is one of {len(tuple(categories))} categories rather than a "
                "duration; mix it with UniformCategoryGuess instead"
            )
        density = getattr(model, "density_outcome", None)
        if density is None:
            return (
                "it scores a discrete outcome -- it declares no density_outcome, so its "
                "likelihood is a probability rather than a density, and a probability and a "
                "density do not average; mix it with UniformChoiceGuess instead"
            )
        if density != self.outcome:
            return (
                f"the model tabulates a density over {density!r} while this process writes "
                f"{self.outcome!r}"
            )
        declared = getattr(model, "censoring_time_column", _UNDECLARED)
        if declared is _UNDECLARED:
            if self.censoring_time_column is None:
                return None
            return (
                f"this process reads observation limits from {self.censoring_time_column!r} "
                "while the model declares no censoring at all, so the two would disagree "
                "about which rows are still in progress"
            )
        if declared != self.censoring_time_column:
            model_limits = "none" if declared is None else repr(declared)
            own_limits = (
                "none" if self.censoring_time_column is None else repr(self.censoring_time_column)
            )
            return (
                f"the model reads observation limits from {model_limits} while this process "
                f"reads them from {own_limits}; a censored row is scored by the probability "
                "each process exceeds that row's limit, so both must read the same limit"
            )
        return None

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return this process's log density of each duration, or its log survival if censored.

        The two answers are different quantities and the selection is the same one
        :func:`~behavio.models._kernels.hazard.censored_log_scores` makes on the model's side,
        which is what keeps the mixture an average of two statements about one observation.
        """

        observed = np.asarray(outcomes, dtype=np.float64)
        if observed.ndim != 1:
            raise ValueError("a duration guess scores one value per row")
        lower, upper = self.duration_bounds
        span = upper - lower
        density = np.full(observed.shape, -np.inf, dtype=np.float64)
        inside = np.isfinite(observed) & (observed >= lower) & (observed <= upper)
        density[inside] = -float(np.log(span))
        limits = self.observation_limits(study, len(observed))
        if limits is None:
            return density
        censored = np.abs(observed - limits) <= CENSORING_TOLERANCE * np.maximum(
            1.0, np.abs(limits)
        )
        remaining = np.clip((upper - observed) / span, 0.0, 1.0)
        with np.errstate(divide="ignore"):
            survival = np.where(remaining > 0.0, np.log(remaining), -np.inf)
        return np.asarray(np.where(censored, survival, density), dtype=np.float64)

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        """Return the declared uniform's height, which is what it predicts about a row.

        A continuous outcome has no probability to average, so a mixture over a tabulated
        density blends the two **densities** on the model's own grid rather than reading this
        -- through :meth:`pointwise_log_density`, which is the member that knows where the
        declared interval ends. This is the single number the prediction contract asks a
        component for, and for a uniform it is the same number everywhere inside that
        interval.
        """

        lower, upper = self.duration_bounds
        return np.full(len(study), 1.0 / (upper - lower), dtype=np.float64)

    def simulate_outcomes(
        self,
        study: Study,
        rows: NDArray[np.intp],
        *,
        generator: np.random.Generator,
    ) -> Mapping[str, NDArray[Any]]:
        """Draw a uniform duration per selected row, truncated at that row's own limit."""

        selected = np.asarray(rows, dtype=np.intp)
        lower, upper = self.duration_bounds
        drawn = generator.uniform(lower, upper, len(selected))
        limits = self.observation_limits(study, len(study))
        if limits is not None:
            drawn = np.minimum(drawn, limits[selected])
        return {self.outcome: np.asarray(drawn, dtype=np.float64)}

    def observation_limits(self, study: Study, n_rows: int) -> NDArray[np.float64] | None:
        """Return each row's declared observation limit, or ``None`` when none is declared."""

        if self.censoring_time_column is None:
            return None
        if self.censoring_time_column not in study.columns:
            raise ModelDataError(
                f"study is missing censoring column {self.censoring_time_column!r}"
            )
        try:
            limits = np.asarray(study[self.censoring_time_column], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(
                f"censoring column {self.censoring_time_column!r} must be numeric"
            ) from None
        if limits.ndim != 1 or limits.shape != (n_rows,) or not np.all(np.isfinite(limits)):
            raise ModelDataError(
                f"censoring column {self.censoring_time_column!r} must contain one finite "
                "observation limit per row"
            )
        return limits


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
