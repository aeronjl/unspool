# Deterministic inference backends

An inference backend should change how an already-defined objective is searched, not what
the model means. Unspool therefore separates `OptimizationProblem` from
`OptimizationBackend`:

- the problem owns the parameter-space fingerprint, ordered optimizer coordinates,
  deterministic starts, negative log-likelihood, gradient availability, and MLE or MAP
  target;
- the backend owns the search algorithm and its numerical configuration; and
- `OptimizationRun` retains every attempted optimum before applying one deterministic
  selection rule.

Probabilistic samplers produce a different kind of evidence from an optimum. Their common
output contract is documented under [labelled posterior results](posterior-results.md):
natural-scale draws, predictive groups, pointwise likelihood, diagnostics, observed data,
and provenance with optional ArviZ/xarray interchange.

The first established full-posterior implementation is the optional
[PyMC hierarchical GLM backend](pymc-backend.md). It samples the existing fixed-scale
partial-pooling model while preserving its task, design matrix, likelihood, and prior
semantics.

## Run multistart maximum likelihood

```python
import numpy as np

from unspool import OptimizationProblem, ScipyMultistart


def negative_log_likelihood(vector):
    value, gradient = model.objective_and_gradient(study, vector)
    return value, gradient


problem = OptimizationProblem(
    parameter_space=model.parameter_space,
    objective=negative_log_likelihood,
    starts=(
        np.array([-1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
    ),
    has_gradient=True,
    objective_name="choice_negative_log_likelihood",
)

run = ScipyMultistart(
    max_iterations=1_000,
    function_tolerance=1e-9,
    gradient_tolerance=1e-9,
).run(problem)
```

The initial SciPy backend intentionally supports one well-understood path:
L-BFGS-B with analytic or SciPy numerical gradients. Adding a method name is not treated as
evidence that a method is suitable for behavioural models. Further optimizers enter as
separate tested backend configurations with recovery evidence.

`run.attempts` contains, for every declared start:

- the immutable start and final optimizer vectors;
- objective value, convergence flag, status, and raw message;
- iteration, function-evaluation, and gradient-evaluation counts; and
- final gradient norm.

Selection first considers finite attempts that reported convergence. If none converged, it
selects the best finite attempt but leaves its failed convergence visible. If every attempt
is non-finite, `selected` is `None`; the failed run remains serializable rather than being
converted into a fictional estimate.

The backend identifier and complete tolerances are stored separately from the problem
record. Both are immutable, JSON-safe, and carry the parameter-space fingerprint.

## MAP is explicit about its measure

```python
from unspool import ObjectiveTarget, PriorMeasure

map_problem = OptimizationProblem(
    parameter_space=model.parameter_space,
    objective=negative_log_likelihood,
    starts=starts,
    has_gradient=True,
    target=ObjectiveTarget.MAXIMUM_A_POSTERIORI,
    prior_measure=PriorMeasure.NATURAL,
)
```

MAP requires a prior for every free parameter. `NATURAL` maximizes the posterior density
declared on interpretable scientific parameters. `OPTIMIZER` additionally includes the
inverse-transform log Jacobian and therefore targets the density in optimizer coordinates.
Those modes need not have the same maximum; the choice is recorded because MAP is not
invariant to reparameterization.

Prior and Jacobian gradients are propagated analytically through identity, shifted-log,
and bounded-logit transforms. Value-only objectives remain valid and let SciPy compute a
numerical gradient.

## Current model binding

`BinaryQLearning.fit()` now constructs an `OptimizationProblem` and runs
`ScipyMultistart`. `QLearningFitResult.optimization_run` retains the complete common result
while its established restart arrays remain available for backwards compatibility and
the common fit audit. The two representations are checked against each other when the fit
result is created.

`export_fit()` includes a shared optimization run when a live result exposes one, so starts
and failed attempts survive the portable artifact boundary. Migrating the composable RL,
DDM, mixture, and GLM-HMM implementations to this contract is incremental compatibility
work, not a second inference design.

## Optional PyBADS backend

Install the optional extra when BADS is appropriate for a moderately expensive,
derivative-free likelihood:

```bash
python -m pip install "unspool[optimization]"
```

```python
from unspool import PyBADSMultistart

run = PyBADSMultistart(
    random_seed=42,
    max_iterations=200,
    max_function_evaluations=1_000,
    function_tolerance=1e-6,
).run(problem)
```

`PyBADSMultistart` maps optimizer hard and plausible bounds from the identical
`ParameterSpace`. Every free parameter must declare two finite plausible bounds and every
start must lie inside that plausible box; the adapter will not invent or silently expand
scientific search assumptions. Each start receives the deterministic seed
`random_seed + attempt_index`, and the legacy NumPy random state is restored after each
run because PyBADS currently seeds it globally.

The result maps PyBADS `x`, `fval`, iterations, function evaluations, success, and message
onto `OptimizationAttempt`. PyBADS 1.0.6 currently reports `success=True` even for its
documented limit terminations, so Unspool conservatively marks messages that reached the
maximum iteration or function-evaluation count as non-converged. They remain eligible as
the best finite fallback only when no attempt converged.

This initial adapter is for deterministic objectives. It records zero gradient evaluations
and leaves gradient norm unavailable; PyBADS' stochastic-objective protocol will require a
separate noise-aware problem contract rather than overloading the deterministic one.
PyBADS remains optional, so its dependencies do not constrain Unspool core or the SciPy
backend.

- [PyBADS API](https://acerbilab.github.io/pybads/api/classes/bads.html)
- [PyBADS JOSS paper](https://doi.org/10.21105/joss.05694)

## API

::: unspool.inference
    options:
      members:
        - ObjectiveTarget
        - PriorMeasure
        - OptimizationProblem
        - OptimizationAttempt
        - OptimizationRun
        - OptimizationBackend
        - ScipyMultistart
        - PyBADSMultistart
