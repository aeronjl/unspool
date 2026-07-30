"""The softmax likelihood, seen through one linear predictor cell per category.

:class:`MultinomialLikelihood` is the multinomial half of
:class:`~behavio.models.MultinomialLogit` written as the four operations
:class:`~behavio.contracts.compose.LinearPredictorLikelihood` asks for. Everything it needs
about a trial arrives in the linear predictor: a category that was not offered on a trial
reaches it as ``-inf`` in that cell, contributing exactly zero probability, zero gradient
and zero curvature, so nothing here has to know what availability is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import CategoricalPrediction, PredictionMode


@dataclass(frozen=True, slots=True)
class MultinomialLikelihood:
    """The softmax likelihood on a fixed, explicit category coordinate.

    The categories are carried because a categorical prediction is unreadable without its
    labels, not because the arithmetic needs them: only their number enters any formula.
    """

    categories: tuple[Any, ...]

    def __post_init__(self) -> None:
        categories = tuple(self.categories)
        if len(categories) < 2:
            raise ValueError("a multinomial likelihood needs at least two categories")
        object.__setattr__(self, "categories", categories)

    @property
    def n_categories(self) -> int:
        """The width of one row's linear predictor."""

        return len(self.categories)

    def prediction(
        self, linear_predictor: NDArray[np.float64], *, mode: PredictionMode
    ) -> CategoricalPrediction:
        """Return category probabilities alongside the logits that produced them."""

        return CategoricalPrediction(
            self._probabilities(linear_predictor), linear_predictor, self.categories, mode
        )

    def pointwise_log_prob(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed category without conditioning on any other row."""

        codes = _codes(outcomes, self.n_categories)
        scores = linear_predictor[np.arange(len(codes)), codes]
        scores = scores - logsumexp(linear_predictor, axis=1)
        return protected_array(scores, dtype=np.float64)

    def value_and_gradient(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the linear predictor."""

        codes = _codes(outcomes, self.n_categories)
        rows = np.arange(len(codes))
        normalizer = logsumexp(linear_predictor, axis=1)
        loss = -float(np.sum(linear_predictor[rows, codes] - normalizer))
        gradient = np.exp(linear_predictor - normalizer[:, None])
        gradient[rows, codes] -= 1.0
        return loss, np.asarray(gradient, dtype=np.float64)

    def curvature(self, linear_predictor: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each row's ``diag(p) - p p'`` block of the negative log likelihood.

        A scalar-predictor family returns one number per row here. A multinomial's cells
        are not independent -- a probability taken from one category is given to another --
        so the curvature is a matrix per row, and it is singular by construction: adding a
        constant to every cell of a row changes nothing.
        """

        probabilities = self._probabilities(linear_predictor)
        blocks = -probabilities[:, :, None] * probabilities[:, None, :]
        diagonal = np.arange(self.n_categories)
        blocks[:, diagonal, diagonal] += probabilities
        return np.asarray(blocks, dtype=np.float64)

    def _probabilities(self, linear_predictor: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            np.exp(linear_predictor - logsumexp(linear_predictor, axis=1)[:, None]),
            dtype=np.float64,
        )


def _codes(outcomes: NDArray[np.float64], n_categories: int) -> NDArray[np.intp]:
    """Read integral category codes out of the float outcome vector the contract carries."""

    codes = np.asarray(outcomes, dtype=np.intp)
    if not np.array_equal(codes, np.asarray(outcomes, dtype=np.float64)):
        raise ValueError("multinomial outcomes must be integral category codes")
    if codes.size and (codes.min() < 0 or codes.max() >= n_categories):
        raise ValueError("multinomial outcomes must index a declared category")
    return codes
