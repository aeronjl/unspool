"""Fixed temporal bases and their penalties, shared by every smooth model family.

Knots are part of a model's *specification*: they are chosen before fitting and appear in
the model signature and in every parameter name, so their validation has to be identical
everywhere. :func:`validated_knots` is that single rule; :func:`format_time` is the single
rendering used by signatures and parameter names alike, so a knot reads the same way in
both.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


def validated_knots(knots: Sequence[float]) -> tuple[float, ...]:
    """Return strictly increasing finite knots or raise the shared ``ValueError``.

    Every smooth family repeated this block verbatim, which meant three chances for the
    messages to drift apart while describing the same defect.
    """

    try:
        values = tuple(float(knot) for knot in knots)
    except (TypeError, ValueError):
        raise ValueError("knots must contain finite numbers") from None
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("knots must contain at least two finite values")
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError("knots must be strictly increasing")
    return values


def format_time(value: float) -> str:
    """Render a clock value the one way signatures and parameter names both use."""

    return np.format_float_positional(value, trim="-")


def linear_time_basis(
    times: Sequence[float] | NDArray[np.float64], knots: tuple[float, ...]
) -> NDArray[np.float64]:
    """Return piecewise linear interpolation weights on a fixed knot sequence."""

    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("temporal values must be a finite one-dimensional array")
    if np.any(values < knots[0]) or np.any(values > knots[-1]):
        raise ValueError(
            f"temporal values must lie within the fixed knot range [{knots[0]}, {knots[-1]}]"
        )

    basis = np.zeros((len(values), len(knots)), dtype=np.float64)
    knot_array = np.asarray(knots, dtype=np.float64)
    for row, value in enumerate(values):
        if value == knots[-1]:
            basis[row, -1] = 1.0
            continue
        right = int(np.searchsorted(knot_array, value, side="right"))
        left = right - 1
        span = knots[right] - knots[left]
        right_weight = (value - knots[left]) / span
        basis[row, left] = 1.0 - right_weight
        basis[row, right] = right_weight
    return basis


def roughness_matrix(knots: tuple[float, ...]) -> NDArray[np.float64]:
    """Return the spacing-scaled first-difference penalty over a knot sequence.

    This is the precision of a Gaussian random walk observed at the knots: a coefficient
    path pays for a jump in proportion to how little clock time it had to make it.
    """

    differences = np.zeros((len(knots) - 1, len(knots)), dtype=np.float64)
    for row, spacing in enumerate(np.diff(knots)):
        scale = 1.0 / np.sqrt(spacing)
        differences[row, row] = -scale
        differences[row, row + 1] = scale
    return differences.T @ differences


def knot_support_findings(
    knots: tuple[float, ...], times: NDArray[np.float64], clock: str
) -> tuple[str, ...]:
    """Describe knots that the observed clock cannot support.

    :func:`linear_time_basis` refuses times outside the knot range, but says nothing about
    the reverse omission: a knot placed outside the observed range is never interpolated
    towards from both sides, so the coefficient value it carries is extrapolation that the
    fit reports with the same confidence as an interpolated one. This function is the
    reverse comparison, reported rather than raised because an unsupported knot is a
    modelling decision, not a malformed study.
    """

    if len(times) == 0:
        return (f"clock {clock!r} has no observations, so no knot is supported by data",)
    low = float(np.min(times))
    high = float(np.max(times))
    findings: list[str] = []
    outside = [knot for knot in knots if knot < low or knot > high]
    if outside:
        rendered = ", ".join(format_time(knot) for knot in outside)
        findings.append(
            f"knots [{rendered}] lie outside the observed {clock} range "
            f"[{format_time(low)}, {format_time(high)}]; their coefficient values are "
            "extrapolation, not estimates"
        )
    empty = [
        (left, right)
        for left, right in pairwise(knots)
        if not np.any((times >= left) & (times <= right))
    ]
    if empty:
        rendered = ", ".join(
            f"[{format_time(left)}, {format_time(right)}]" for left, right in empty
        )
        findings.append(
            f"no observation falls in {clock} interval(s) {rendered}; the coefficient path "
            "is unconstrained across them"
        )
    return tuple(findings)
