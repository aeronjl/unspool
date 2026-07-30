"""The likelihood, basis, and curvature layer shared by Behavio's model families.

``behavio.models`` is a directory of *model families*. Everything a family needs but does
not own -- the two-boundary Wiener density, the penalized Bernoulli fit, the piecewise
linear temporal basis, the finite-difference curvature -- lives here instead, so that one
family never reaches into another family's private namespace to borrow it.

The modules are deliberately importable only from inside ``behavio.models``: they are an
implementation layer, not a second public surface. Nothing here validates a
:class:`~behavio.trials.Study` beyond what its own computation requires; column-level
contracts stay with the model that declares them.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
