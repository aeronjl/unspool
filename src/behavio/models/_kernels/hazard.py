"""Right-censored duration scoring, written once for every model whose outcome is a wait.

Does one hazard machinery serve patch leaving and response times?
-----------------------------------------------------------------
Partly, and the part it serves is exactly this module. Two things are involved in scoring a
duration and only one of them is shared.

*Not shared: where the density comes from.* A patch-leaving model is **hazard-first** -- the
animal is deciding, moment by moment, whether to go, so the instantaneous leaving rate is the
primitive and the density is derived from it. A response-time model is **density-first** --
the decision terminates when an accumulator first reaches a bound, so the first-passage
density is the primitive and its hazard is a description of it computed afterwards. Writing
one parametric-hazard family and asking a Wiener first-passage density to be expressed in it
would be a claim about drift diffusion that drift diffusion does not make.

*Shared: what censoring does to a score.* An observation that ended before the event is worth
:math:`\\log S(c)` and one that reached the event is worth :math:`\\log f(t)`, the gradient
follows the same selection, and the tolerance question -- *is a duration equal to its limit an
event or a censoring?* -- has one right answer that should not be decided twice. That is the
whole of this module, and it is the half a survival model for response times would import
rather than rewrite: :class:`~behavio.models.ddm.WienerDriftDiffusion` already tabulates
:math:`f`, and a censored variant of it needs the selection below and a survival function,
not a new hazard family.

Right-censoring only
--------------------
Left-truncation and interval censoring are deliberately absent. A behavioural session ends
while the animal is still in the patch, which is right-censoring; an animal that entered a
patch before recording began is a *different* observation whose likelihood is conditional
rather than marginal, and silently scoring it as if it were not would be the exact failure
this module exists to prevent. A study with that shape should say so; nothing here pretends
to handle it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.estimator import ModelDataError

__all__ = [
    "CENSORING_TOLERANCE",
    "CensoredDurations",
    "censored_log_scores",
    "censored_score_gradients",
    "read_censoring",
    "validated_durations",
]

#: Relative tolerance at which an observed duration counts as having reached its limit.
CENSORING_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class CensoredDurations:
    """One study's observed durations and which of them are right-censored."""

    times: NDArray[np.float64]
    censored: NDArray[np.bool_]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        censored = np.asarray(self.censored, dtype=np.bool_)
        if times.ndim != 1 or censored.shape != times.shape:
            raise ValueError("durations and their censoring flags must be equally sized")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "censored", censored)

    @property
    def n_rows(self) -> int:
        """The number of scored durations."""

        return len(self.times)

    @property
    def censored_fraction(self) -> float:
        """The share of rows whose event was never observed."""

        return 0.0 if not len(self.censored) else float(np.mean(self.censored))


def validated_durations(
    values: NDArray[np.float64], *, label: str, strictly_positive: bool = False
) -> NDArray[np.float64]:
    """Return finite, non-negative durations, or raise a readable model data error."""

    times = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ModelDataError(f"{label} must contain finite durations")
    if strictly_positive:
        if np.any(times <= 0.0):
            raise ModelDataError(f"{label} must be strictly positive")
    elif np.any(times < 0.0):
        raise ModelDataError(f"{label} must be non-negative")
    return times


def read_censoring(
    times: NDArray[np.float64],
    limits: NDArray[np.float64] | None,
    *,
    label: str,
) -> NDArray[np.bool_]:
    """Return which observed durations reached their declared observation limit.

    ``limits`` is the longest duration each row *could* have shown -- the time from patch
    entry to the end of the session, not a quantity anybody estimates. A row is censored when
    its observed duration reaches that limit, and a row that *exceeds* it is a contradiction
    in the study rather than a long observation, so it is refused.

    ``None`` means the model declares no censoring, which is a claim: every duration in the
    study ran to its event. The claim is not checkable from the durations alone -- that is
    what makes an undeclared censoring column a reporting matter rather than a validation one
    -- so the answer here is simply that nothing is censored.
    """

    observed = np.asarray(times, dtype=np.float64)
    if limits is None:
        return np.zeros(observed.shape, dtype=np.bool_)
    bounds = validated_durations(limits, label=label, strictly_positive=True)
    if bounds.shape != observed.shape:
        raise ModelDataError(f"{label} must contain one observation limit per row")
    slack = CENSORING_TOLERANCE * np.maximum(1.0, np.abs(bounds))
    if np.any(observed > bounds + slack):
        raise ModelDataError(
            f"a duration exceeds its {label} value, so the declared observation limit is not "
            "the limit that produced these data"
        )
    return np.asarray(observed >= bounds - slack, dtype=np.bool_)


def censored_log_scores(
    *,
    log_density: NDArray[np.float64],
    log_survival: NDArray[np.float64],
    censored: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return the log likelihood of each duration: a density, or a survival if censored.

    This is :math:`\\delta \\log f(t) + (1 - \\delta)\\log S(c)` written once. Both arguments
    are evaluated on every row rather than only where they are used, because both are closed
    form for the families that call this and branching in the arithmetic would cost more than
    it saved.
    """

    density = np.asarray(log_density, dtype=np.float64)
    survival = np.asarray(log_survival, dtype=np.float64)
    flags = np.asarray(censored, dtype=np.bool_)
    if density.shape != survival.shape or flags.shape != density.shape:
        raise ValueError("density, survival and censoring arrays must be equally sized")
    return np.asarray(np.where(flags, survival, density), dtype=np.float64)


def censored_score_gradients(
    *,
    density_gradient: NDArray[np.float64],
    survival_gradient: NDArray[np.float64],
    censored: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Select each row's gradient the same way :func:`censored_log_scores` selects its score.

    ``(rows, parameters)`` in and out. Keeping the selection here rather than at each call
    site is what stops a model from scoring a censored row through its survival function
    while differentiating it through its density, which converges quietly to a wrong answer.
    """

    density = np.asarray(density_gradient, dtype=np.float64)
    survival = np.asarray(survival_gradient, dtype=np.float64)
    flags = np.asarray(censored, dtype=np.bool_)
    if density.shape != survival.shape or density.ndim != 2:
        raise ValueError("gradients must be equally sized (rows, parameters) arrays")
    if flags.shape != (density.shape[0],):
        raise ValueError("censoring flags must contain one value per row")
    return np.asarray(np.where(flags[:, None], survival, density), dtype=np.float64)
