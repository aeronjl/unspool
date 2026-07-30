"""A prediction type for a continuous outcome, and the estimator surface that returns one.

:class:`~behavio.contracts.estimator.Prediction` is a probability plus a linear predictor and
:class:`~behavio.contracts.estimator.CategoricalPrediction` is a simplex over named
categories. Both describe a *discrete* outcome. Nothing in the contract described a response
time, a continuous confidence report, or the finishing-time distribution of a race between
accumulators, so a wrapper around a package that predicts one -- PyDDM, an LBA, a
state-space model with a continuous emission -- had exactly two choices: throw the
continuous prediction away and report choice probabilities alone, or invent a private return
type that no Behavio consumer could read. The first is what the estimator contract silently
encouraged, and it discards the half of the prediction that distinguishes a response-time
model from a logistic regression.

:class:`DensityPrediction` is that missing object. It carries a predictive **density on an
explicit outcome grid**, optionally *defective* across named categories, which is the shape
every first-passage-time and race model actually produces:

- an unlabelled continuous outcome (a confidence rating) is a density that integrates to one;
- a two-boundary diffusion is two defective densities whose masses are the choice
  probabilities and whose total integral is one;
- an *n*-accumulator race is *n* defective densities on the same coordinate.

The three facts a caller needs -- what the choice probabilities are, what the density is at
an observed value, and how much mass the grid failed to cover -- are methods rather than
conventions, so no wrapper has to re-derive them and no two wrappers can derive them
differently.

Where this belongs
------------------
In ``behavio.contracts.estimator``, beside the two prediction types it completes, with
``ModelPrediction`` widened to ``Prediction | CategoricalPrediction | DensityPrediction``
and :func:`behavio.evaluate.folds.evaluate_splits` taught to slice it. It is here because
``behavio.contracts`` is owned elsewhere while this was written; nothing about the type is
adapter-specific.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import (
    BehaviourEstimator,
    CategoricalPrediction,
    FitResult,
    PredictionMode,
)
from behavio.trials import Study

LOG_DENSITY_FLOOR: Final = float(np.log(np.finfo(np.float64).tiny))
"""Smallest log density a pointwise score reports, mirroring ``behavio.models._kernels.wiener``.

A density read off a numerical grid underflows to exactly zero far into the tail even where
the analytic density is merely very small, so ``log(0)`` is a statement about the grid rather
than about the model. Flooring keeps a single tail trial from making an entire fold's score
``-inf``, and the floor is a declared constant rather than a magic number inside one wrapper.
"""

_MASS_TOLERANCE: Final = 1e-3
"""How far above one a row's integrated mass may sit before the density is rejected.

Loose on purpose. The trapezoid rule overshoots a convex density, so a legitimate
tabulation on a coarse grid integrates to slightly more than one, and rejecting that would
reject every real solver output. The check exists to catch a density that is wrong *by
construction* -- per-bin masses passed as densities, a missing Jacobian after a unit change,
a normalisation applied twice -- and those are wrong by orders of magnitude, not by 1e-4.
"""


@dataclass(frozen=True, slots=True)
class DensityPrediction:
    """A predictive density for one continuous outcome, on an explicit grid.

    ``grid`` is the strictly increasing outcome coordinate the density is tabulated on, in
    the units of the scored column. ``density`` is ``(n_trials, n_grid)`` for an unlabelled
    continuous outcome and ``(n_trials, n_categories, n_grid)`` when the density is
    *defective* across ``categories`` -- the two absorbing boundaries of a diffusion, the
    accumulators of a race.

    Integrated mass is checked, not assumed: a row's total mass may fall short of one,
    because a finite grid truncates a distribution with unbounded support and because a
    model may leave probability undecided, but it may never exceed one. :attr:`total_mass`
    reports what each row actually integrates to, so a caller can see truncation rather than
    discover it as a bias.
    """

    grid: NDArray[np.float64]
    density: NDArray[np.float64]
    outcome: str
    mode: PredictionMode
    categories: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        grid = protected_array(self.grid, dtype=np.float64)
        density = protected_array(self.density, dtype=np.float64)
        mode = PredictionMode(self.mode)
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("a density prediction must name the outcome column it predicts")
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("the outcome grid must be one-dimensional with at least two points")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("the outcome grid must be finite and strictly increasing")
        categories = None if self.categories is None else tuple(self.categories)
        if categories is None:
            if density.ndim != 2:
                raise ValueError("an unlabelled density must be (n_trials, n_grid)")
        else:
            if len(categories) < 2:
                raise ValueError("a defective density must name at least two categories")
            keys = [_category_key(value) for value in categories]
            if len(set(keys)) != len(keys):
                raise ValueError("density categories must be unique")
            if density.ndim != 3 or density.shape[1] != len(categories):
                raise ValueError("a defective density must be (n_trials, n_categories, n_grid)")
        if density.shape[-1] != grid.size:
            raise ValueError("the density's last axis must match the outcome grid")
        if not density.shape[0]:
            raise ValueError("a density prediction must cover at least one trial")
        if not np.all(np.isfinite(density)) or np.any(density < 0):
            raise ValueError("densities must be finite and non-negative")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "mode", mode)
        mass = self.total_mass
        if np.any(mass > 1.0 + _MASS_TOLERANCE):
            worst = float(np.max(mass))
            raise ValueError(f"a predictive density may not integrate above one; observed {worst}")
        if np.any(mass <= 0.0):
            raise ValueError("every trial's predictive density must carry positive mass")

    @property
    def n_observations(self) -> int:
        """Number of trial rows represented by the prediction."""

        return int(self.density.shape[0])

    @property
    def n_grid(self) -> int:
        """Number of points the density is tabulated on."""

        return int(self.grid.size)

    @property
    def is_defective(self) -> bool:
        """Whether the density is split across named categories that share the grid."""

        return self.categories is not None

    @property
    def category_mass(self) -> NDArray[np.float64]:
        """Integrated mass of each category, ``(n_trials, n_categories)``.

        For a two-boundary diffusion these are the choice probabilities, up to whatever mass
        the grid truncated away.
        """

        if self.categories is None:
            raise ValueError("this prediction declares no categories, so it has no category mass")
        return protected_array(np.trapezoid(self.density, self.grid, axis=-1), dtype=np.float64)

    @property
    def total_mass(self) -> NDArray[np.float64]:
        """Total integrated mass of each trial's prediction, ``(n_trials,)``.

        One minus this is the probability the grid does not account for: truncated tail,
        undecided mass, or both. It is reported rather than normalised away.
        """

        integral = np.trapezoid(self.density, self.grid, axis=-1)
        if self.categories is not None:
            integral = np.sum(integral, axis=-1)
        return protected_array(integral, dtype=np.float64)

    def density_at(
        self, values: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Linearly interpolate the density at one outcome value per trial.

        Returns ``(n_trials,)`` for an unlabelled density and ``(n_trials, n_categories)``
        for a defective one. Values outside the grid evaluate to zero rather than to the
        nearest endpoint, because a grid that does not reach an observation carries no
        information about it and clamping would invent some.

        Interpolation, not nearest-bin lookup, is the point. A package that returns a
        tabulated PDF is usually scored by rounding each observation to its grid index,
        which makes the reported per-trial likelihood a function of the solver's step size.
        """

        observed = np.asarray(values, dtype=np.float64)
        if observed.ndim != 1 or observed.size != self.n_observations:
            raise ValueError("values must contain one outcome per predicted trial")
        inside = np.isfinite(observed) & (observed >= self.grid[0]) & (observed <= self.grid[-1])
        upper = np.clip(np.searchsorted(self.grid, observed, side="left"), 1, self.n_grid - 1)
        lower = upper - 1
        span = self.grid[upper] - self.grid[lower]
        weight = np.where(span > 0, (observed - self.grid[lower]) / span, 0.0)
        weight = np.clip(weight, 0.0, 1.0)
        if self.categories is None:
            left = self.density[np.arange(self.n_observations), lower]
            right = self.density[np.arange(self.n_observations), upper]
            interpolated = left + weight * (right - left)
            return protected_array(np.where(inside, interpolated, 0.0), dtype=np.float64)
        rows = np.arange(self.n_observations)[:, None]
        columns = np.arange(len(self.categories))[None, :]
        left = self.density[rows, columns, lower[:, None]]
        right = self.density[rows, columns, upper[:, None]]
        interpolated = left + weight[:, None] * (right - left)
        return protected_array(np.where(inside[:, None], interpolated, 0.0), dtype=np.float64)

    def observed_log_density(
        self,
        values: Sequence[float] | NDArray[np.floating[Any]],
        categories: Sequence[Any] | NDArray[Any] | None = None,
        *,
        floor: float = LOG_DENSITY_FLOOR,
    ) -> NDArray[np.float64]:
        """Return one floored log density per trial for the observation it actually made.

        This is the pointwise score of a joint discrete/continuous observation: for a
        two-boundary diffusion, the log of the defective density of the *observed* boundary
        at the *observed* time. ``categories`` is required exactly when the prediction is
        defective, and names the observed category of each row.
        """

        if self.categories is None:
            if categories is not None:
                raise ValueError("this prediction declares no categories to select")
            density = self.density_at(values)
        else:
            if categories is None:
                raise ValueError("a defective density needs the observed category of each row")
            observed = list(categories)
            if len(observed) != self.n_observations:
                raise ValueError("categories must contain one observed category per trial")
            index = {_category_key(value): column for column, value in enumerate(self.categories)}
            try:
                columns = np.asarray(
                    [index[_category_key(value)] for value in observed], dtype=np.intp
                )
            except KeyError as error:
                raise ValueError(
                    f"observed category {error.args[0][1]!r} is not one this prediction "
                    f"declares: {list(self.categories)}"
                ) from None
            density = self.density_at(values)[np.arange(self.n_observations), columns]
        floored = float(floor)
        with np.errstate(divide="ignore"):
            scores = np.where(density > 0.0, np.log(density), floored)
        return protected_array(np.maximum(scores, floored), dtype=np.float64)

    def choice_prediction(self) -> CategoricalPrediction:
        """Marginalise the grid away and return the categorical prediction it implies.

        Rows are renormalised, because a :class:`CategoricalPrediction` must sum to one and
        a truncated grid does not. The renormalisation is exact only when
        :attr:`total_mass` is one; read that array before treating the result as the model's
        unconditional choice probabilities.
        """

        mass = np.asarray(self.category_mass, dtype=np.float64)
        total = np.sum(mass, axis=1, keepdims=True)
        probability = mass / total
        with np.errstate(divide="ignore"):
            linear_predictor = np.log(probability)
        return CategoricalPrediction(
            probability=probability,
            linear_predictor=linear_predictor,
            categories=self.categories or (),
            mode=self.mode,
        )

    def expected_outcome(self) -> NDArray[np.float64]:
        """Mass-weighted mean outcome per trial, conditional on the grid's support."""

        weighted = self.density * self.grid
        integral = np.trapezoid(weighted, self.grid, axis=-1)
        if self.categories is not None:
            integral = np.sum(integral, axis=-1)
        return protected_array(integral / self.total_mass, dtype=np.float64)

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> DensityPrediction:
        """Return a protected row subset on the same grid and category coordinate."""

        return DensityPrediction(
            grid=self.grid,
            density=self.density[indices],
            outcome=self.outcome,
            mode=self.mode,
            categories=self.categories,
        )


@runtime_checkable
class DensityBehaviourEstimator(BehaviourEstimator, Protocol):
    """An estimator that also predicts a density for a continuous scored outcome.

    It is a *widening* of :class:`~behavio.contracts.estimator.BehaviourEstimator`, not a
    replacement: ``predict`` still returns the discrete prediction every consumer already
    reads, and ``predict_density`` returns the continuous half that would otherwise be
    discarded. A model implementing both cannot quietly disagree with itself --
    :func:`behavio.adapters.estimator_conformance.check_behaviour_estimator` integrates the
    density and compares it against the choice probabilities.
    """

    @property
    def density_outcome(self) -> str: ...

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction: ...


def _category_key(value: Any) -> Any:
    """Hashable identity that keeps ``0``, ``0.0``, ``False`` and ``"0"`` apart."""

    scalar = value.item() if isinstance(value, np.generic) else value
    return (type(scalar), scalar)


__all__ = [
    "LOG_DENSITY_FLOOR",
    "DensityBehaviourEstimator",
    "DensityPrediction",
]
