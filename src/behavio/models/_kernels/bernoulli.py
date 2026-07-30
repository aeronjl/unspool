"""The penalized Bernoulli likelihood and the session ordering every family shares.

``behavio.models.glm`` had become the de-facto utility module for seven sibling families
purely because it happened to be where these two functions were first written. Neither is
about generalized linear models: :func:`fit_bernoulli` fits any design matrix against a
binary outcome under a quadratic penalty, and :func:`ordered_session_indices` is a
statement about Behavio's trial-order contract, not about any one likelihood.

:data:`BERNOULLI` is the same likelihood again, exposed as the four operations
:class:`~behavio.contracts.compose.LinearPredictorLikelihood` asks for, so that a
combinator can write an objective over it without importing anything about GLMs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import FitResult, ModelPrediction, Prediction, PredictionMode
from behavio.models._kernels.penalised import fit_penalised_linear
from behavio.study import Study


@dataclass(frozen=True, slots=True)
class BernoulliLikelihood:
    """The logistic likelihood seen only through one linear predictor per row."""

    def prediction(
        self, linear_predictor: NDArray[np.float64], *, mode: PredictionMode
    ) -> ModelPrediction:
        """Return choice probabilities alongside the linear predictor that produced them."""

        return Prediction(
            probability=expit(linear_predictor),
            linear_predictor=linear_predictor,
            mode=mode,
        )

    def pointwise_log_prob(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed outcome without conditioning on any other row."""

        scores = outcomes * -np.logaddexp(0.0, -linear_predictor)
        scores += (1.0 - outcomes) * -np.logaddexp(0.0, linear_predictor)
        return protected_array(scores, dtype=np.float64)

    def value_and_gradient(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the linear predictor."""

        loss = np.logaddexp(0.0, linear_predictor).sum() - outcomes @ linear_predictor
        return float(loss), np.asarray(expit(linear_predictor) - outcomes, dtype=np.float64)

    def curvature(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Return the per-row Fisher weight ``p (1 - p)``.

        The logit is the canonical link, so expected and observed information agree and the
        observation is not read.
        """

        probabilities = expit(linear_predictor)
        return np.asarray(probabilities * (1.0 - probabilities), dtype=np.float64)


BERNOULLI = BernoulliLikelihood()
"""The single shared logistic observation model."""


def fit_bernoulli(
    *,
    model_name: str,
    model_signature: str,
    parameter_names: tuple[str, ...],
    design_matrix: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    penalty_matrix: NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
    coefficient_warning_threshold: float,
    offsets: NDArray[np.float64] | None = None,
    derived_estimates: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> FitResult:
    """Fit a quadratically penalized Bernoulli likelihood with deterministic L-BFGS-B.

    The arithmetic moved to :func:`~behavio.models._kernels.penalised.fit_penalised_linear`
    once a second family needed it. What is left here is the choice of likelihood, which is
    all that was ever Bernoulli about it -- and the operations are performed in the same
    order on the same doubles, so fits published before the move are reproduced exactly.
    """

    return fit_penalised_linear(
        model_name=model_name,
        model_signature=model_signature,
        parameter_names=parameter_names,
        design_matrix=design_matrix,
        outcomes=outcomes,
        penalty_matrix=penalty_matrix,
        likelihood=BERNOULLI,
        max_iterations=max_iterations,
        tolerance=tolerance,
        coefficient_warning_threshold=coefficient_warning_threshold,
        offsets=offsets,
        derived_estimates=derived_estimates,
    )


def ordered_session_indices(study: Study) -> tuple[tuple[int, ...], ...]:
    """Group source row indices by subject and session in chronological order."""

    sessions: dict[tuple[Any, Any], list[int]] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        session = _scalar(study["session"][index])
        sessions.setdefault((subject, session), []).append(index)
    return tuple(tuple(indices) for indices in sessions.values())


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
