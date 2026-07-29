# Session-varying drift-diffusion trajectories

This benchmark asks whether a smooth Wiener model earns its extra longitudinal structure.
It compares the same static and smooth fits under stationary and changing truth, using a
held-out final session rather than in-sample likelihood.

## Contract

- One subject completes six sessions of 150 trials.
- The first five sessions are fitted and the sixth is scored prospectively.
- The knot basis for all six session coordinates is fixed before fitting.
- Stimulus drift and boundary may vary; intercept, starting bias, and non-decision time are
  stationary.
- The smooth fit uses a time-scaled first-difference penalty of `10.0`.
- Under stationarity, stimulus drift is `1.0` and boundary is `1.2`.
- Under change, stimulus drift rises from `0.3` to `1.9` while boundary falls from `1.55`
  to `0.95`.
- Twenty matched repetitions retain every seed, fit audit, trajectory error, and held-out
  joint choice/response-time log loss.

The future knot has no outcome data during fitting. Its first-difference penalty carries
the last supported value forward, providing a persistence forecast rather than trend
extrapolation.

## Result

| Truth | Method | Training path RMSE | Future joint log loss | Future wins |
| --- | --- | ---: | ---: | ---: |
| Stationary | Static | **0.04043** | **0.32273** | 10/20 |
| Stationary | Smooth | 0.09174 | 0.32539 | 10/20 |
| Changing | Static | 0.35798 | 0.09300 | 0/20 |
| Changing | Smooth | **0.09235** | **-0.30943** | 20/20 |

The stationary future-wins column is now an even 10/20 split. The static model still wins
on the mean score and on training-path RMSE, but a per-repetition win count near chance is
the expected signature of two candidates that agree under stationarity; it is not evidence
that the smooth family is free.

All 80 fits converge and all paired fits remain audit-eligible. A continuous density may
exceed one, so a mean negative joint log density can be negative; only matched differences
on the same choice/response-time event are interpreted.

The benchmark separates two claims that are easy to conflate. The smooth family recovers
and forecasts the specified change, while the stationary control does not manufacture a
smooth advantage. The exact paths, sample size, smoothness, and simulation approximation
are part of that claim. Real-data use still requires training-only structural selection
and comparison against other mechanisms.

## Run

```bash
uv run python -m benchmarks.smooth_ddm.benchmark
```
