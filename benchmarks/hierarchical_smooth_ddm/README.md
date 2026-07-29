# Hierarchical session-varying drift-diffusion trajectories

This benchmark asks when animal-specific Wiener trajectories are warranted. Four methods
compete under stationary identical animals, shared population change, and genuinely
individual change. The held-out target is always the fifth session.

## Contract

- Five animals complete five sessions of 70 trials; the first four sessions are fitted.
- Stimulus drift and boundary separation vary on fixed knots `(0, 2, 4)`.
- Drift intercept, starting bias, and non-decision time remain stationary.
- The hierarchical fit uses population smoothness `8.0`, subject scale `0.2`, and subject
  smoothness `8.0`, all fixed before fitting.
- Complete pooling, a shared smooth path, five independent smooth fits, and the hierarchical
  smooth model receive identical observed trials.
- Twenty repetitions per regime retain all seeds, audits, trajectory errors, and held-out
  joint choice/response-time scores.

The trajectory metric is subject-level RMSE for stimulus drift and boundary during the
four training sessions. The predictive metric is mean negative joint log density in the
fifth session for animals represented in training.

## Result

| Truth | Method | Subject-path RMSE | Future joint log loss |
| --- | --- | ---: | ---: |
| Stationary identical | Complete pooling | **0.03628** | **0.33537** |
|  | Shared smooth | 0.05856 | 0.34407 |
|  | Independent smooth | 0.11562 | 0.38958 |
|  | Hierarchical smooth | 0.08289 | 0.36130 |
| Shared change | Complete pooling | 0.24974 | 0.14554 |
|  | Shared smooth | **0.05632** | **-0.14631** |
|  | Independent smooth | 0.12686 | -0.10234 |
|  | Hierarchical smooth | 0.08234 | -0.13259 |
| Individual change | Complete pooling | 0.26789 | 0.21030 |
|  | Shared smooth | 0.11462 | -0.14894 |
|  | Independent smooth | 0.12029 | -0.03934 |
|  | Hierarchical smooth | **0.08836** | **-0.17299** |

All 480 fits converge and all 60 complete matched panels remain audit-eligible. A joint
continuous density can exceed one, so negative mean log loss is possible; only matched
differences for the same choice/response-time event are interpreted.

Adding the simulator's Brownian-bridge absorption test moved every held-out log loss down
by roughly `0.02` to `0.11` nats, because the earlier discretized generator produced
systematically slow trials that no candidate could predict. The complete ranking inside
each regime is unchanged, and so are the margins that carry the claim: the hierarchical
model still beats the shared-smooth control in the individual-change regime by `0.024`,
the same margin as before.

The negative controls matter. Hierarchical flexibility does not win when animals are
identical or when one shared trajectory is sufficient. It wins only in the specified
individual-change regime. The fixed subject scale is supplied from the design and is not
estimated, so this validates shrinkage and trajectory mechanics rather than
variance-component recovery.

## Run

```bash
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
```
