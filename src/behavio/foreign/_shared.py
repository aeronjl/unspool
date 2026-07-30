"""The parts of a foreign wrapper that stopped being one wrapper's business.

Three wrappers now live in :mod:`behavio.foreign`, and three is the first point at which a
shared helper can be justified by evidence rather than by anticipation. What follows is
therefore *only* what more than one wrapper already contains, written once:

- :func:`quiet_foreign_package`, which all three need. A wrapped package logs its own
  progress and warns about its own numerics, none of which is a finding about the fit; what
  *is* a finding is computed by Behavio and retained on the result.
- :class:`ForeignCurvature` and :func:`unknown_curvature`, which both *point-estimate*
  wrappers need. PyDDM reports an optimum and a loss; dynamax reports a parameter pytree.
  Neither reports an uncertainty, so both wrappers recover one from the curvature of the
  objective that produced the estimate, and both have to be able to say "not this time"
  without inventing a number.

What is deliberately *not* here is as much of the argument as what is. The Bambi wrapper
nominated four of its own helpers -- ``_rename_observation_dim``, ``_with_behavio_evidence``,
``_training_frame`` and ``_replace_groups`` -- for exactly this module, on the reasoning that
the next wrapper would want them. It did not: those four exist to repair an ``InferenceData``
that a *sampler* produced, and dynamax fits by expectation maximization, so it has no
posterior groups, no observation dimension to rename and no training frame to rebuild. They
still have one user each and they stay in :mod:`behavio.foreign.bambi` until they have two.
Lifting them here would have moved code without removing a duplicate, which is the failure
mode a shared module invites.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import ConvergenceStatus


def quiet_foreign_package(
    *loggers: str,
    categories: Sequence[type[Warning]] = (RuntimeWarning,),
    messages: Sequence[str] = (),
) -> Any:
    """Silence one foreign package's own logging and advisories for the duration of a call.

    ``loggers`` names the logger hierarchies to hold at :data:`logging.ERROR`; ``categories``
    the warning classes to ignore; ``messages`` any additional regular expressions matched
    against a warning's text, for the advisories a package emits under a category too broad
    to suppress wholesale.

    Both the logger levels and the warning filters are restored on exit, so this never
    changes what a caller sees outside the block. Nothing suppressed here is evidence about
    a fit: convergence, curvature, boundary contact and every pointwise score are computed
    by Behavio from the returned numbers and retained on the result.
    """

    return _QuietForeignPackage(tuple(loggers), tuple(categories), tuple(messages))


class _QuietForeignPackage:
    __slots__ = ("_catcher", "_categories", "_levels", "_loggers", "_messages")

    def __init__(
        self,
        loggers: tuple[str, ...],
        categories: tuple[type[Warning], ...],
        messages: tuple[str, ...],
    ) -> None:
        self._loggers = loggers
        self._categories = categories
        self._messages = messages
        self._levels: dict[str, int] = {}

    def __enter__(self) -> None:
        self._levels = {}
        for name in self._loggers:
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            logger.setLevel(logging.ERROR)
        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        for category in self._categories:
            warnings.simplefilter("ignore", category)
        for message in self._messages:
            warnings.filterwarnings("ignore", message=message)

    def __exit__(self, *exception: Any) -> None:
        self._catcher.__exit__(*exception)
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)


@dataclass(frozen=True, slots=True)
class ForeignCurvature:
    """What differencing or differentiating an objective at its optimum established.

    ``converged`` is ``True``/``False`` when a stationarity check ran, and
    :attr:`~behavio.contracts.ConvergenceStatus.UNREPORTED` when it could not run at all --
    the one state in which neither the foreign package nor the wrapper has an answer.

    ``estimated`` says whether :attr:`covariance` is a number the wrapper believes. When it
    is ``False`` the matrix is all-``NaN`` and :attr:`message` says why, which
    :meth:`behavio.contracts.FitResult.audit` records as a warning rather than a failure: a
    fit whose interval is unknown is still a fit, and saying so is different from quietly
    reporting a curvature that is not one.
    """

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    gradient_norm: float | None
    converged: bool | ConvergenceStatus
    estimated: bool
    message: str

    @property
    def status(self) -> int | None:
        """The integer convergence status a fit records, absent when there is no verdict."""

        if isinstance(self.converged, ConvergenceStatus):
            return None
        return 0 if self.converged else 1


def unknown_curvature(
    size: int,
    message: str,
    *,
    converged: bool | ConvergenceStatus,
    gradient_norm: float | None = None,
) -> ForeignCurvature:
    """Return a curvature record that declines to report a covariance, and says why."""

    return ForeignCurvature(
        covariance=protected_array(np.full((size, size), np.nan), dtype=np.float64),
        standard_errors=protected_array(np.full(size, np.nan), dtype=np.float64),
        gradient_norm=gradient_norm,
        converged=converged,
        estimated=False,
        message=message,
    )


def condition_number(matrix: NDArray[np.float64]) -> float | None:
    """Return a matrix's condition number, or ``None`` when there is nothing to condition."""

    if not np.all(np.isfinite(matrix)):
        return None
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:  # pragma: no cover - a singular matrix is already NaN
        return None


__all__ = [
    "ForeignCurvature",
    "condition_number",
    "quiet_foreign_package",
    "unknown_curvature",
]
