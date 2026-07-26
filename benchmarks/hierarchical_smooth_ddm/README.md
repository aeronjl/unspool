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
| Stationary identical | Complete pooling | **0.04202** | **0.36021** |
|  | Shared smooth | 0.06184 | 0.37107 |
|  | Independent smooth | 0.11030 | 0.40028 |
|  | Hierarchical smooth | 0.08292 | 0.38353 |
| Shared change | Complete pooling | 0.24919 | 0.19467 |
|  | Shared smooth | **0.06069** | **-0.07807** |
|  | Independent smooth | 0.12324 | -0.03744 |
|  | Hierarchical smooth | 0.08620 | -0.06826 |
| Individual change | Complete pooling | 0.26658 | 0.25563 |
|  | Shared smooth | 0.12117 | -0.03846 |
|  | Independent smooth | 0.11545 | 0.16654 |
|  | Hierarchical smooth | **0.09784** | **-0.06247** |

All 480 fits converge and all 60 complete matched panels remain audit-eligible. A joint
continuous density can exceed one, so negative mean log loss is possible; only matched
differences for the same choice/response-time event are interpreted.

The negative controls matter. Hierarchical flexibility does not win when animals are
identical or when one shared trajectory is sufficient. It wins only in the specified
individual-change regime. The fixed subject scale is supplied from the design and is not
estimated, so this validates shrinkage and trajectory mechanics rather than
variance-component recovery.

## Run

```bash
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
```
