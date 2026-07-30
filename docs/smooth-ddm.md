# Session-varying drift-diffusion trajectories

A drift-diffusion model whose parameters move across training is not a class of its own.
It is `WienerDriftDiffusion` passed through
[`smooth()`](composing-models.md), which replaces each named parameter with one value per
fixed knot of an explicit longitudinal clock and interpolates between them. This page is
about what that model *claims* — [Composing models](composing-models.md) is about the
combinator that builds it. The scientific question is whether evidence sensitivity or
response caution changes across training sessions.

The word *time-varying* has two distinct meanings in diffusion modelling. Here it means
that a trial's parameters depend on the trial's position in an external study clock. Each
individual decision still uses a constant-drift, constant-boundary Wiener process. This
model does not implement a drift or collapsing boundary that changes during one decision.

## Parameter paths

For trial *i* at longitudinal time \(u_i\),

\[
dX_i(s) = v_i(u_i)\,ds + dW(s), \qquad
v_i(u_i)=\beta_0(u_i)+x_i^\top\beta(u_i).
\]

Boundary separation \(a(u_i)\), relative starting bias \(z(u_i)\), non-decision time and —
when the mixture is configured — the contaminant weight may all vary across trials in the
same way. Every varying parameter is represented by natural-scale values at knots fixed
before fitting and is linearly interpolated between them. Its roughness contribution is

\[
\frac{\lambda}{2}\sum_{j=2}^{J}
\frac{(\theta_j-\theta_{j-1})^2}{u_j-u_{j-1}}.
\]

This time-scaled first-difference penalty is a random-walk regularizer, not evidence that
the true biological process is piecewise linear. `smoothness` is fixed configuration and
belongs inside training-only model selection when it is compared with other values.

`smooth()` puts every parameter on a path by default. A narrower scientific hypothesis
should say so explicitly with `parameters=`, which leaves the rest **stationary**: one
coordinate rather than one per knot, and no roughness penalty at all.

```python
from behavio import WienerDriftDiffusion
from behavio.compose import smooth

model = smooth(
    WienerDriftDiffusion(predictors=("stimulus",)),
    over="session_order",
    knots=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    parameters=("drift.stimulus", "boundary"),
    smoothness=10.0,
)
```

Parameter names are qualified with the clock and knot, coefficient-major and knot-minor:
`drift.stimulus[session_order=0]`, `drift.stimulus[session_order=1]`, and so on. A
parameter left out of `parameters=` keeps its own name and its single value.

Simulation parameters use one sequence per varying parameter and one scalar per stationary
parameter:

```python
truth = model.parameters_from_paths(
    {
        "drift.intercept": 0.1,
        "drift.stimulus": [0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
        "boundary": [1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }
)
study = model.simulate(design, truth, seed=302)
fit = model.fit(study)
trajectory = model.coefficient_trajectory(fit)
stimulus_path = trajectory.path("drift.stimulus")
```

`CoefficientTrajectory` retains its clock, evaluation times, named natural-scale
coefficients, and read-only values; `coefficient_trajectory(fit, times=...)` evaluates the
fitted paths anywhere inside the knot range. A composed model has no
`parameter_components()`: collapsing a path to one set of scalar components would hide the
longitudinal object being estimated.

### Non-decision time is now a declaration rather than a refusal

The deleted `SmoothWienerDriftDiffusion` held non-decision time stationary because its
admissible range is entangled with the fastest observed response. `smooth()` does not
special-case it, so it varies whenever it is named — and the `parameters=None` default names
everything. That is a real modelling choice, not a free improvement: the box that keeps
non-decision time admissible is computed once from the study's minimum response time and
applied at every knot, so a path is bounded by the fastest response in the *whole* study
rather than by the fastest response near each knot. Name the parameters you mean.

## Prospective meaning of a future knot

The complete knot basis must be declared before a prospective split is fitted, including
the held-out session's coordinate. A knot with no training observations is informed only
by the roughness penalty. With the first-difference penalty, its optimum carries the final
supported value forward. This is a transparent persistence forecast, not linear trend
extrapolation and not access to the held-out outcome.

Knots do not by themselves make an analysis prospective. Use `forward_session_splits`,
fit preprocessing only on training rows, and select `smoothness`, varying parameters, or
competing model families inside the outer training study. `evaluate_splits` enforces the
first part of that contract.

Paths are subject-specific by default. A multi-subject study is rejected unless
`shared_trajectory=True` explicitly asserts one common path on an aligned clock. That
opt-in is complete pooling, not hierarchical inference; for the partially pooled
alternative, wrap the smooth model in `hierarchical()`.

## Current boundary

A session-varying Wiener model still excludes:

- within-decision time-varying drift or collapsing boundaries — a different model, not a
  different setting;
- learned knots, automatic change points, or automatic trajectory interpretation;
- non-linear bases: paths are piecewise linear between declared knots;
- posterior uncertainty over the paths. `fit.standard_errors` are a local Gaussian
  approximation at the penalised optimum.

Two restrictions the hand-written class carried are gone, because they were properties of
that class rather than of the science. A contaminant mixture composes: it is
`mix(model, UniformResponseGuess(...))`, whose weight is an ordinary parameter, so a lapse
rate that grows or shrinks across training is `smooth(mix(model), ...)` and nothing new.
And per-parameter deviation scales are available through `hierarchical()` rather than being
a feature the smooth class had to grow.

The distinction between the two senses of *time-varying* is empirically important. Learning
studies have long treated diffusion components as possible loci of behavioral change, and a
recent multi-timescale analysis reported drift changes across days alongside boundary
changes within daily sessions. Those results motivate explicit clocks; they do not license
reading every fitted path as a unique cognitive mechanism. See
[Dutilh et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/37291102/)
and the earlier learning-focused discussion by
[Ditterich (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2493300/).

## Recovery evidence

The [session-varying Wiener benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/smooth_ddm) compares static
and smooth fits under stationary and changing truth. In 20 matched repetitions per regime,
the static model wins both training-path RMSE and final-session joint log loss under
stationarity; the smooth model wins both under the specified change trajectory. This is a
design-specific implementation check, not a universal change detector.

The composition is also checked against the class it replaced. Every fit, trajectory and
seeded simulation in `tests/test_compose_ddm.py` is compared with a stored reference
produced by `SmoothWienerDriftDiffusion` before it was deleted. Simulated choices, response
times and drawn random effects are bit-for-bit equal; the fitted estimates agree to within
`1e-5` rather than exactly, because the composed design contracts the same
products in a different order and a bounded quasi-Newton search amplifies that last-place
difference into a slightly different stopping point. At the deleted class's own optimum the
two objectives agree exactly.

For multi-animal partial pooling, see
[Partially pooled Wiener trajectories](hierarchical-smooth-ddm.md). That model estimates a
population path and shrunken animal deviations instead of silently sharing one trajectory.

Run the executable example and benchmark with:

```bash
uv run python examples/smooth_drift_diffusion.py
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python -m benchmarks.smooth_ddm.benchmark
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
```
