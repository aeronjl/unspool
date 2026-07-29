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
- Simulation uses Euler–Maruyama paths at `0.0001` seconds with an exact Brownian-bridge
  absorption test on every step whose endpoints both stay inside the corridor, and
  linearly interpolated crossing times for steps that end outside it.
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
| Drift intercept | 0.11887 | 0.05998 |
| Stimulus drift | 0.11367 | 0.05505 |
| Boundary | 0.03002 | 0.01335 |
| Starting bias | 0.01978 | 0.00798 |
| Non-decision time | 0.00475 | 0.00263 |

Boundary bias at 1,200 trials is `-0.00013`, against `+0.01181` before the Brownian-bridge
absorption test was added to the simulator. That residual was discretization overshoot in
the generator, not a fitting failure, and it is the clearest single indicator that the
simulated truth and the analytic likelihood now agree.

Approximate 95% local-Hessian coverage ranges from 90% to 100% across the ten
parameter-by-design cells. Those 20-run proportions are descriptive and simulation is
still discretized; neither should be read as a calibrated universal interval guarantee.

## Run

```bash
uv run python -m benchmarks.ddm_recovery.benchmark
```

The implementation follows the unit-diffusion Wiener parameterization and paired series in
[Navarro & Fuss (2009)](https://doi.org/10.1016/j.jmp.2009.02.003). The wider cognitive
interpretation and standard model components are reviewed by
[Ratcliff & McKoon (2008)](https://doi.org/10.1162/neco.2008.12-06-420).
