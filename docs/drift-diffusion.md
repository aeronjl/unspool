# Joint choice and response-time modelling

`WienerDriftDiffusion` is Unspool's first model whose likelihood scores more than a
binary choice. It fits the joint first-passage density of choice and response time while
making the observation boundary, physical units, simulation approximation, and numerical
diagnostics explicit.

## Parameterization

For trial *t*, evidence follows a unit-diffusion Wiener process

\[
dX_t(s) = v_t\,ds + dW(s), \qquad
v_t = \beta_0 + x_t^\top\beta,
\]

between absorbing boundaries at zero and `boundary`. The process starts at
`starting_bias * boundary`; hitting the upper boundary produces choice one, and hitting
the lower boundary produces choice zero. Observed response time is first-passage decision
time plus `nondecision_time`.

Fixing diffusion variance to one identifies the scale. Drift may depend on named numeric
covariates, while boundary, starting bias, and non-decision time are shared across trials.
All public parameters use their natural scale:

- `drift.intercept` and `drift.<covariate>`;
- positive `boundary`;
- `starting_bias` strictly between zero and one;
- non-negative `nondecision_time`, expressed in seconds.

This is deliberately a first, stationary family. It does not yet model across-trial
variability, lapse/contaminant responses, collapsing bounds, history-dependent starting
points, or parameters that drift across learning.

## Response-time schema

Response time is an explicit typed observation rather than an anonymous covariate:

```python
from unspool import ResponseTimeSpec, ResponseTimeUnit, WienerDriftDiffusion

model = WienerDriftDiffusion(
    covariates=("stimulus",),
    response_time=ResponseTimeSpec(
        column="response_time_ms",
        unit=ResponseTimeUnit.MILLISECONDS,
    ),
)
```

`ResponseTimeSpec.read()` rejects absent, non-numeric, non-finite, zero, and negative
values and converts validated observations to canonical seconds. Simulation writes values
back in the declared source unit. Unit metadata are part of the model signature, preventing
a fit in milliseconds from being silently reused by a model configured for seconds.

The model declares

```python
scored_columns = ("choice", "response_time_ms")
```

because each pointwise score is a joint choice/response-time density. Unspool therefore
rejects direct likelihood ranking against a choice-only model. Such a ranking would compare
different observed events, not competing explanations of the same event.

## Fitting and simulation

Fitting evaluates paired small- and large-time expansions of the Wiener first-passage
density, uses deterministic bounded L-BFGS-B restarts, and estimates local uncertainty
from a numerical Hessian. `DriftDiffusionFitResult` retains every restart objective,
convergence flag, and optimizer message, plus the selected restart, minimum observed
response time, and number of observations assigned the finite log-density floor.

Simulation uses vectorized Euler-Maruyama paths and linearly interpolates each boundary
crossing within its final time step. `simulation_time_step` controls the accuracy/cost
tradeoff, and simulation fails visibly if a path exceeds `simulation_max_time`. The
analytic likelihood itself is not discretized.

```python
from unspool import WienerDriftDiffusion

model = WienerDriftDiffusion(covariates=("stimulus",), n_restarts=3)
truth = model.parameters_from_components(
    drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
    boundary=1.2,
    starting_bias=0.45,
    nondecision_time=0.25,
)
simulated = model.simulate(design, truth, seed=105)
fit = model.fit(simulated)
joint_log_density = model.pointwise_log_prob(simulated, fit)
choice_probability = model.predict(simulated, fit).probability
```

Prediction returns the analytic marginal probability of an upper-boundary choice. Pointwise
scoring returns the joint density. Response times at or below fitted non-decision time, and
extreme numerical underflow, receive a finite floor score and trigger retained diagnostics
when encountered during fitting.

## Interpretation boundary

The fitted components are conditional on this model and its scaling convention. In
particular, a drift coefficient is not a direct measurement of evidence quality unless the
covariate coding and all other structural assumptions are defensible. Boundary and
non-decision estimates can also absorb mismatch from fast guesses, lapses, motor delays,
and unmodelled across-trial variation.

Use the model prospectively only when response times are recorded on a common, documented
event definition. Inspect the minimum response time, likelihood-floor count, bound warnings,
restart agreement, and audit before interpretation. Add an explicit contaminant model or
pre-registered exclusion rule when the task contains anticipatory or timeout responses;
the likelihood floor is a numerical safeguard, not a contaminant account.

The mathematical parameterization and paired first-passage expansions follow
[Navarro and Fuss (2009)](https://doi.org/10.1016/j.jmp.2009.02.003). The relation between
accuracy and response-time distributions, and the standard cognitive interpretation of
drift, boundaries, starting point, and non-decision time, are reviewed by
[Ratcliff and McKoon (2008)](https://doi.org/10.1162/neco.2008.12-06-420).

## Recovery evidence

The [fixed-parameter recovery benchmark](../benchmarks/ddm_recovery/README.md) runs 20
repetitions at both 400 and 1,200 trials. All 40 fits pass audit, and RMSE decreases with
the larger design for every fitted parameter. This validates the implementation in one
specified regime; it does not establish universal identifiability or interval calibration.

Run the small executable example and full benchmark with:

```bash
uv run python examples/drift_diffusion.py
uv run python -m benchmarks.ddm_recovery.benchmark
```
