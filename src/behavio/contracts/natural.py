"""The optional contract for a model's natural parameterisation.

``parameter_names`` is simultaneously three different coordinates: the one the optimizer
searches, the one recovery assesses coverage in, and the one a result is reported in.
Those genuinely differ. A positive scale parameter is estimated as ``log_width`` because
that is where its Wald interval is honest; a bounded rate is estimated as a logit for the
same reason. Neither is what anybody publishes.

Before this module every model that cared reconciled the two coordinates by hand, and four
incompatible conventions had accumulated. The contract here is deliberately small:

* :attr:`NaturalParameterisation.natural_names` -- the reporting coordinate's names.
* :meth:`NaturalParameterisation.to_natural` -- optimizer vector to natural mapping.
* :meth:`NaturalParameterisation.from_natural` -- natural mapping back to an optimizer
  mapping, which is what :meth:`~behavio.contracts.estimator.GenerativeBehaviourModel.simulate`
  and :func:`~behavio.recovery.run_parameter_recovery` take.
* :meth:`NaturalParameterisation.natural_jacobian` -- the derivative of the first with
  respect to its argument, which is all the delta method needs.

and two helpers, :func:`natural_covariance` and :func:`natural_quantities`, that propagate
an estimated covariance onto the reporting coordinate and express the result as
:class:`~behavio.contracts.estimator.DerivedQuantity` values. Both take the estimate vector
and covariance as arrays rather than a fit, so that a model can call them inside ``fit``
and attach the result to the fit result it is constructing.

It is entirely optional. A model whose estimated coordinate already *is* the reporting
coordinate declares nothing, satisfies nothing, and behaves exactly as it did before this
module existed.

What this contract deliberately does not do
-------------------------------------------
It does not produce intervals. A Wald interval formed on the natural scale can leave a
parameter's admissible range -- a lapse rate's upper bound can exceed one, a width's lower
bound can fall below zero -- and the repair is to form the interval where the estimate is
approximately normal and then map it, which requires knowing the map is monotone in that
coordinate. That is a model's own knowledge, so :func:`natural_quantities` reports a
delta-method standard error and leaves :attr:`DerivedQuantity.interval` absent. A model
that can form an honest interval attaches one itself.

It is also not :class:`behavio.inference.parameters.ParameterSpace`. A ``ParameterSpace`` is a
declarative, serialisable, element-by-element description in which every natural parameter
maps through one of three named transforms of exactly one optimizer coordinate. That is a
different and stricter thing: it cannot express ordered rating criteria built from
cumulative sums of positive gaps, which is exactly the reparameterisation the detection-
theory models need. A model may implement either, both, or neither.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.estimator import DerivedQuantity

__all__ = [
    "NaturalParameterisation",
    "natural_covariance",
    "natural_quantities",
]


@runtime_checkable
class NaturalParameterisation(Protocol):
    """A model that reports in a different coordinate from the one it estimates in."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The estimated coordinate, in the order the optimizer sees it."""
        ...

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reporting coordinate, in the order :meth:`natural_jacobian` rows use."""
        ...

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Map one estimated coordinate onto the reporting coordinate."""
        ...

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Map a complete natural mapping back onto the estimated coordinate."""
        ...

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return ``d natural / d estimated`` with one row per natural name."""
        ...


def natural_covariance(
    model: NaturalParameterisation,
    estimates: Sequence[float] | NDArray[np.floating[Any]],
    covariance: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Propagate an estimated covariance onto the reporting coordinate by the delta method.

    Returns ``J C J'`` for the Jacobian ``J`` of the natural map evaluated at ``estimates``
    and the estimated covariance ``C``. The result is a genuine covariance, not just its
    diagonal, because a reparameterisation almost always introduces correlation that a
    per-parameter standard error would hide.

    Arrays rather than a :class:`~behavio.contracts.estimator.FitResult` are taken so that
    a model can compute its derived quantities inside ``fit`` and pass them to the fit
    result it is about to construct.
    """

    names, jacobian, _ = _validated_jacobian(model, estimates)
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (jacobian.shape[1], jacobian.shape[1]):
        raise ValueError("covariance must be square over the estimated coordinate")
    propagated = jacobian @ matrix @ jacobian.T
    return np.asarray(propagated, dtype=np.float64).reshape(len(names), len(names))


def natural_quantities(
    model: NaturalParameterisation,
    estimates: Sequence[float] | NDArray[np.floating[Any]],
    covariance: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
) -> tuple[DerivedQuantity, ...]:
    """Return the reporting coordinate with delta-method standard errors.

    Every returned quantity carries a value and a standard error and no interval; see this
    module's docstring for why the interval is the model's own business.
    """

    names, _, vector = _validated_jacobian(model, estimates)
    values = model.to_natural(vector)
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError("to_natural must return exactly the declared natural names")
    propagated = natural_covariance(model, vector, covariance)
    errors = np.sqrt(np.maximum(np.diag(propagated), 0.0))
    return tuple(
        DerivedQuantity(
            name=name,
            value=float(values[name]),
            standard_error=float(error),
            description="natural coordinate, delta method from the estimated covariance",
        )
        for name, error in zip(names, errors, strict=True)
    )


def _validated_jacobian(
    model: NaturalParameterisation,
    estimates: Sequence[float] | NDArray[np.floating[Any]],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(model, NaturalParameterisation):
        raise TypeError("model must satisfy the NaturalParameterisation contract")
    names = tuple(model.natural_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("natural_names must be non-empty and unique")
    vector = np.asarray(estimates, dtype=np.float64)
    jacobian = np.asarray(model.natural_jacobian(vector), dtype=np.float64)
    expected = (len(names), len(model.parameter_names))
    if vector.shape != (expected[1],):
        raise ValueError(f"estimates must contain {expected[1]} values")
    if jacobian.shape != expected:
        raise ValueError(f"natural_jacobian must have shape {expected}, not {jacobian.shape}")
    return names, jacobian, vector
