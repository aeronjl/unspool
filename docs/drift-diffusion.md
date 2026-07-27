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

This is deliberately the stationary member of the family. The base configuration does not model
across-trial Wiener-parameter variability, collapsing bounds, history-dependent starting
points, or parameters that drift across learning. `SmoothWienerDriftDiffusion` adds
fixed-knot longitudinal paths for selected drift coefficients, boundary, and starting bias;
see the [session-varying guide](smooth-ddm.md). An optional contaminant mixture provides a
narrow robustness account for responses outside the stationary decision process.
The [hierarchical trajectory guide](hierarchical-smooth-ddm.md) extends the longitudinal
family to partially pooled animal-specific paths.

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
response time, number of observations assigned the finite log-density floor, and fitted
posterior contaminant responsibilities when that component is configured.

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
scoring returns the joint density. In the base configuration, response times at or below
fitted non-decision time and extreme numerical underflow receive a finite floor score and
trigger retained diagnostics when encountered during fitting.

## Explicit response contaminants

`UniformResponseTimeContaminant` adds one normalized joint mixture component:

\[
p(y,t) = (1-\pi)f_{\mathrm{Wiener}}(y,t)
       + \pi\,\operatorname{Bernoulli}(y;q)\,
         \frac{\mathbb{1}[L \le t \le U]}{U-L}.
\]

The fitted `contaminant_probability` is \(\pi\). The response-time support \([L,U]\),
its unit in canonical seconds, the fixed contaminant choice probability \(q\), and the
mixture-probability bounds are model configuration. They are not estimated from the scored
session.

```python
from unspool import UniformResponseTimeContaminant, WienerDriftDiffusion

model = WienerDriftDiffusion(
    covariates=("stimulus",),
    contaminant=UniformResponseTimeContaminant(
        time_bounds=(0.05, 3.0),
        probability_bounds=(0.0, 0.2),
    ),
    nondecision_time_bounds=(0.1, 0.6),
)
```

A fixed non-decision-time search interval is required in this configuration. Otherwise the
fastest observed contaminant would constrain non-decision time before the mixture could
explain it. Both intervals should come from task timing, equipment limits, prior studies,
or a rule fitted only to training data.

`simulate_with_contaminants()` returns a `DriftDiffusionSimulation` whose latent Boolean
indicators are separate from its observed `Study`. `fit.posterior_contaminant_probability`
and `model.contaminant_responsibility(study, fit)` expose soft trial assignments. The
prediction API also marginalizes the contaminant choice process rather than returning the
Wiener choice probability alone.

## Interpretation boundary

The fitted components are conditional on this model and its scaling convention. In
particular, a drift coefficient is not a direct measurement of evidence quality unless the
covariate coding and all other structural assumptions are defensible. Boundary and
non-decision estimates can also absorb mismatch from fast guesses, lapses, motor delays,
and unmodelled across-trial variation.

Use the model prospectively only when response times are recorded on a common, documented
event definition. Inspect the minimum response time, likelihood-floor count, bound warnings,
restart agreement, and audit before interpretation. When using the mixture, inspect its
support, fitted weight, and responsibilities as model-dependent uncertainty. A high
responsibility is not an observed fact about a trial, and the uniform component is not a
mechanistic theory of distraction, guessing, anticipation, or timeout responses.

The mathematical parameterization and paired first-passage expansions follow
[Navarro and Fuss (2009)](https://doi.org/10.1016/j.jmp.2009.02.003). The relation between
accuracy and response-time distributions, and the standard cognitive interpretation of
drift, boundaries, starting point, and non-decision time, are reviewed by
[Ratcliff and McKoon (2008)](https://doi.org/10.1162/neco.2008.12-06-420).
The decision to represent contaminants explicitly rather than rely on unreported trimming
follows [Ratcliff and Tuerlinckx (2002)](https://doi.org/10.3758/BF03196302); Unspool's
fixed-support independent mixture is a deliberately simpler first contract.

## Recovery evidence

<figure class="doc-figure">
  <img src="../assets/ddm-recovery.svg" alt="Parameter root-mean-square error versus trial count for drift intercept, drift stimulus, boundary, starting bias, and nondecision time in the committed drift-diffusion recovery benchmark.">
  <figcaption><strong>Design-specific DDM recovery.</strong> Increasing the simulated trial count reduces RMSE for every fitted parameter. Lines summarize committed repetitions; they are not universal sample-size recommendations.</figcaption>
</figure>

The [fixed-parameter recovery benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ddm_recovery) runs 20
repetitions at both 400 and 1,200 trials. All 40 fits pass audit, and RMSE decreases with
the larger design for every fitted parameter. This validates the implementation in one
specified regime; it does not establish universal identifiability or interval calibration.

The [contaminant benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ddm_contaminants) compares robust and
naive Wiener fits on 20 matched designs with five-percent contamination. The robust model
has lower RMSE for every shared parameter and lower future-session joint log loss in all 20
repetitions. This supports the specified uniform mixture under matched simulation; it does
not show that real contaminants are uniform or independently chosen.

Run the small executable example and full benchmark with:

```bash
uv run python examples/drift_diffusion.py
uv run python examples/smooth_drift_diffusion.py
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python examples/contaminant_ddm.py
uv run python -m benchmarks.ddm_recovery.benchmark
uv run python -m benchmarks.ddm_contaminants.benchmark
uv run python -m benchmarks.smooth_ddm.benchmark
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
```
