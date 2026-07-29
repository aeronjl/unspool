"""A deterministic reimplementation of the IBL's own psychometric fit.

The International Brain Laboratory fits choice data with ``erf_psycho_2gammas`` from
``cortex-lab/psychofit``, minimising a binomial negative log-likelihood with unconstrained
Nelder-Mead and enforcing box constraints as a large penalty inside the objective. Two
different parameter boxes appear in the released code: one that decides training status and
one that produced the published figure values. Both are reproduced here.

The released routine draws four of its five optimiser restarts from an unseeded uniform
distribution, so the published pipeline is not reproducible run to run. This module keeps
the same restart schedule but derives it from an explicit seed, so a committed result can
be re-verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fmin
from scipy.special import erf

PENALTY = 1e7
DEFAULT_N_FITS = 5
DEFAULT_SEED = 20_210_101
MINIMUM_CONTRAST_LEVELS = 4
_THRESHOLD_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class PsychometricBox:
    """One of the IBL's two published parameter boxes for ``erf_psycho_2gammas``."""

    name: str
    threshold_start: float
    threshold_minimum: float
    threshold_maximum: float
    lapse_start: float
    lapse_maximum: float
    bias_start_is_contrast_mean: bool

    def start(self, contrasts: NDArray[np.float64]) -> NDArray[np.float64]:
        bias = float(np.mean(contrasts)) if self.bias_start_is_contrast_mean else 0.0
        return np.array(
            [bias, self.threshold_start, self.lapse_start, self.lapse_start], dtype=np.float64
        )

    def minimum(self, contrasts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [float(np.min(contrasts)), self.threshold_minimum, 0.0, 0.0], dtype=np.float64
        )

    def maximum(self, contrasts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [
                float(np.max(contrasts)),
                self.threshold_maximum,
                self.lapse_maximum,
                self.lapse_maximum,
            ],
            dtype=np.float64,
        )


#: ``ibllib.brainbox.behavior.training.compute_psychometric``: decides ``trained_1a``/``1b``.
CRITERION_BOX = PsychometricBox(
    name="ibllib-training-status",
    threshold_start=20.0,
    threshold_minimum=0.0,
    threshold_maximum=100.0,
    lapse_start=0.05,
    lapse_maximum=1.0,
    bias_start_is_contrast_mean=True,
)

#: ``paper_behavior_functions.fit_psychfunc``: produced the published figure values.
REPORTED_BOX = PsychometricBox(
    name="paper-behavior-fit-psychfunc",
    threshold_start=20.0,
    threshold_minimum=5.0,
    threshold_maximum=40.0,
    lapse_start=0.05,
    lapse_maximum=1.0,
    bias_start_is_contrast_mean=False,
)


def erf_psycho_2gammas(
    parameters: NDArray[np.float64], contrasts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return P(rightward choice) for the paper's error-function psychometric.

    ``parameters`` is ``[bias, threshold, lapse_low, lapse_high]`` with bias and threshold
    in percent contrast, matching ``psychofit.erf_psycho_2gammas`` exactly.
    """

    bias, threshold, lapse_low, lapse_high = (float(value) for value in parameters)
    scale = threshold if abs(threshold) > _THRESHOLD_FLOOR else _THRESHOLD_FLOOR
    span = 1.0 - lapse_low - lapse_high
    return lapse_low + span * (erf((contrasts - bias) / scale) + 1.0) / 2.0


def negative_log_likelihood(
    parameters: NDArray[np.float64],
    contrasts: NDArray[np.float64],
    n_trials: NDArray[np.float64],
    proportion_rightward: NDArray[np.float64],
    minimum: NDArray[np.float64],
    maximum: NDArray[np.float64],
) -> float:
    """Return the released binomial objective, with box constraints as a hard penalty."""

    if np.any(parameters < minimum) or np.any(parameters > maximum):
        return PENALTY
    probabilities = erf_psycho_2gammas(parameters, contrasts)
    if not np.all(np.isfinite(probabilities)):
        return PENALTY
    epsilon = float(np.finfo(float).eps)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return -float(
        np.sum(
            n_trials
            * (
                proportion_rightward * np.log(probabilities)
                + (1.0 - proportion_rightward) * np.log(1.0 - probabilities)
            )
        )
    )


def fit_psychometric(
    contrasts: NDArray[np.float64],
    n_trials: NDArray[np.float64],
    proportion_rightward: NDArray[np.float64],
    *,
    box: PsychometricBox,
    seed: int = DEFAULT_SEED,
    n_fits: int = DEFAULT_N_FITS,
) -> NDArray[np.float64]:
    """Fit ``erf_psycho_2gammas`` by the released restart schedule under a pinned seed.

    Returns ``[bias, threshold, lapse_low, lapse_high]``, or four NaNs when fewer than four
    distinct contrast levels were presented, matching ``fit_psychfunc``'s own guard.
    """

    contrasts = np.asarray(contrasts, dtype=np.float64)
    n_trials = np.asarray(n_trials, dtype=np.float64)
    proportion_rightward = np.asarray(proportion_rightward, dtype=np.float64)
    finite = np.isfinite(proportion_rightward)
    if int(np.count_nonzero(finite)) < MINIMUM_CONTRAST_LEVELS:
        return np.full(4, np.nan, dtype=np.float64)
    contrasts, n_trials = contrasts[finite], n_trials[finite]
    proportion_rightward = proportion_rightward[finite]

    minimum = box.minimum(contrasts)
    maximum = box.maximum(contrasts)
    generator = np.random.default_rng(seed)
    start = box.start(contrasts)

    best_parameters = np.full(4, np.nan, dtype=np.float64)
    best_objective = np.inf
    for _ in range(n_fits):
        estimate = np.asarray(
            fmin(
                negative_log_likelihood,
                start,
                args=(contrasts, n_trials, proportion_rightward, minimum, maximum),
                disp=False,
            ),
            dtype=np.float64,
        )
        objective = negative_log_likelihood(
            estimate, contrasts, n_trials, proportion_rightward, minimum, maximum
        )
        if objective < best_objective:
            best_objective, best_parameters = objective, estimate
        start = minimum + generator.random(minimum.size) * (maximum - minimum)
    return best_parameters


def contrast_summary(
    signed_contrast: NDArray[np.float64], rightward: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Collapse trials into the released ``[levels, counts, proportion rightward]`` matrix."""

    valid = np.isfinite(signed_contrast) & np.isfinite(rightward)
    levels = np.unique(signed_contrast[valid])
    counts = np.empty(levels.size, dtype=np.float64)
    proportions = np.empty(levels.size, dtype=np.float64)
    for index, level in enumerate(levels):
        selected = valid & (signed_contrast == level)
        counts[index] = float(np.count_nonzero(selected))
        proportions[index] = float(np.mean(rightward[selected]))
    return levels, counts, proportions
