# Fixed-parameter drift-diffusion recovery

This benchmark tests the first joint choice/response-time family under two matched trial
counts. It is a parameter-recovery check for one exact generative regime, not evidence that
the model is identifiable for every task.

## Contract

- One subject completes four sessions with a standard-normal stimulus covariate.
- Drift is `0.2 + 1.2 * stimulus`.
- Boundary separation is `1.2`, relative starting bias is `0.45`, and non-decision time is
  `0.25` seconds.
- Diffusion variance is fixed to one for scale identification.
- Simulation uses Euler–Maruyama paths at `0.0001` seconds with linearly interpolated
  boundary-crossing times.
- Fitting uses the paired small-/large-time Wiener first-passage expansions described by
  Navarro and Fuss (2009), three deterministic bounded L-BFGS-B restarts, and local-Hessian
  uncertainty.
- Twenty repetitions are run at 400 and 1,200 trials. Every seed, estimate, standard error,
  optimizer message, and complete fit audit remains in `result.json`.

## Result

All 40 fits pass audit. Increasing the design from 400 to 1,200 trials reduces RMSE for
every parameter:

| Parameter | RMSE, 400 trials | RMSE, 1,200 trials |
| --- | ---: | ---: |
| Drift intercept | 0.12959 | 0.05838 |
| Stimulus drift | 0.11452 | 0.05699 |
| Boundary | 0.03261 | 0.01550 |
| Starting bias | 0.02049 | 0.00936 |
| Non-decision time | 0.00521 | 0.00246 |

Approximate 95% local-Hessian coverage ranges from 85% to 100% across the ten
parameter-by-design cells. Those 20-run proportions are descriptive and simulation is
discretized; neither should be read as a calibrated universal interval guarantee.

## Run

```bash
uv run python -m benchmarks.ddm_recovery.benchmark
```

The implementation follows the unit-diffusion Wiener parameterization and paired series in
[Navarro & Fuss (2009)](https://doi.org/10.1016/j.jmp.2009.02.003). The wider cognitive
interpretation and standard model components are reviewed by
[Ratcliff & McKoon (2008)](https://doi.org/10.1162/neco.2008.12-06-420).
