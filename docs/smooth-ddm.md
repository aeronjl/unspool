# Session-varying drift-diffusion trajectories

`SmoothWienerDriftDiffusion` extends the joint choice/response-time Wiener model along an
explicit longitudinal clock. It is designed for questions such as whether evidence
sensitivity or response caution changes across training sessions.

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

Boundary separation \(a(u_i)\) and relative starting bias \(z(u_i)\) may also vary across
trials. Non-decision time remains stationary in this first implementation. Every varying
parameter is represented by natural-scale values at knots fixed before fitting and is
linearly interpolated between them. Its roughness contribution is

\[
\frac{\lambda}{2}\sum_{j=2}^{J}
\frac{(\theta_j-\theta_{j-1})^2}{u_j-u_{j-1}}.
\]

This time-scaled first-difference penalty is a random-walk regularizer, not evidence that
the true biological process is piecewise linear. `smoothness` is fixed configuration and
belongs inside training-only model selection when it is compared with other values.

By default all drift coefficients, boundary, and starting bias vary. A narrower scientific
hypothesis should say so explicitly:

```python
from unspool import SmoothWienerDriftDiffusion

model = SmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    time="session_order",
    knots=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    varying_parameters=("drift.stimulus", "boundary"),
    smoothness=10.0,
)
```

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
fit = model.fit(study)
trajectory = model.parameter_trajectory(fit)
stimulus_path = trajectory.path("drift.stimulus")
```

`DriftDiffusionTrajectory` retains its clock, evaluation times, named natural-scale
parameters, and read-only values. The model deliberately rejects `parameter_components()`:
collapsing a path to one vector would hide the longitudinal object being estimated.

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
opt-in is complete pooling, not hierarchical inference.

## Current boundary

This first family intentionally excludes:

- session-varying non-decision time, whose admissible range is entangled with local minimum
  response times;
- contaminant mixtures, which need a trajectory-aware robustness contract;
- hierarchical pooling of animal-specific Wiener paths;
- within-decision time-varying drift or collapsing boundaries;
- learned knots, automatic change points, or automatic trajectory interpretation.

The distinction is empirically important. Learning studies have long treated diffusion
components as possible loci of behavioral change, and a recent multi-timescale analysis
reported drift changes across days alongside boundary changes within daily sessions. Those
results motivate explicit clocks; they do not license reading every fitted path as a
unique cognitive mechanism. See [Dutilh et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/37291102/)
and the earlier learning-focused discussion by
[Ditterich (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2493300/).

## Recovery evidence

The [session-varying Wiener benchmark](../benchmarks/smooth_ddm/README.md) compares static
and smooth fits under stationary and changing truth. In 20 matched repetitions per regime,
the static model wins both training-path RMSE and final-session joint log loss under
stationarity; the smooth model wins both under the specified change trajectory. This is a
design-specific implementation check, not a universal change detector.

Run the executable example and benchmark with:

```bash
uv run python examples/smooth_drift_diffusion.py
uv run python -m benchmarks.smooth_ddm.benchmark
```
