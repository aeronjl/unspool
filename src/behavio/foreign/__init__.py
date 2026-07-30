"""Behavio estimators backed by third-party model implementations.

Behavio's defensible contribution is falsification -- prospective splitters that block
correctly, exact-design recovery, SBC, sensitivity, reliability, recovery gates, replayable
evidence -- and not another catalogue of likelihoods. Where a maintained package already
computes a density better than Behavio can hand-roll it, the right move is to wrap it and
put the falsification machinery around it. This package is where those wrappers live.

Every wrapper here:

- satisfies :class:`behavio.contracts.BehaviourEstimator` -- or, when the foreign package
  fits by sampling rather than by optimization,
  :class:`behavio.contracts.PosteriorBehaviourEstimator` -- and the generative counterpart
  when the package can simulate, so it flows through ``evaluate_splits``,
  ``compare_models``, ``run_parameter_recovery`` and ``describe()`` unchanged;
- keeps its dependency behind a per-model extra and names that extra in the error a user
  meets without it, so ``import behavio`` never requires a foreign solver;
- maps the foreign parameter names onto Behavio's explicitly, because recovery is worthless
  when the simulator and the fitter mean different things by one name;
- states its licence, because Behavio is MIT and several packages in this ecosystem are not.
  See ``docs/foreign-models.md`` for the compatibility and licence matrix.

The package sits *above* ``behavio.models`` in the layering, not beside ``behavio.adapters``.
A wrapped model is a model: it needs ``describe()``, the design algebra, and the response-time
contract, all of which sit at or above the model layer. The parts of a wrapper that are not
model-specific -- the sequence/row helper, the continuous-outcome prediction type, the
estimator conformance harness -- live in :mod:`behavio.adapters`, which stays below the
contracts' implementers and depends on no third-party package at all.

Importing this package imports no wrapper. Reach a wrapper by its own module::

    from behavio.foreign.pyddm import PyDDMDriftDiffusion
    from behavio.foreign.bambi import BambiRegression
"""

from behavio.foreign._optional import (
    BAMBI_EXTRA,
    BAMBI_SERIES,
    PYDDM_EXTRA,
    PYDDM_SERIES,
    ForeignPackageUnavailableError,
)

__all__ = [
    "BAMBI_EXTRA",
    "BAMBI_SERIES",
    "PYDDM_EXTRA",
    "PYDDM_SERIES",
    "ForeignPackageUnavailableError",
]
