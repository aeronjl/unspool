# Population and individual trajectory recovery

This factorial benchmark asks whether longitudinal models distinguish four scientifically
different structures rather than rewarding the most flexible model in every dataset:

- stationary, identical animals;
- stable differences between animals;
- one smooth trajectory shared by the population;
- genuinely animal-specific smooth trajectories.

```bash
uv run python -m benchmarks.trajectory_recovery.benchmark
```

Every regime uses the same 12-subject design, four 50-trial training sessions, one held-out
final session, and fixed knots at sessions 0, 2, and 4. Twenty matched repetitions compare
complete pooling, static partial pooling, shared smooth drift, independent smooth fits, and
hierarchical smooth drift. Hyperparameters are fixed before simulation and evaluation.

## Pinned result

The intended structural account wins in all four regimes under both realized subject-
trajectory RMSE and prospective final-session log loss:

| Generating structure | Recovered winner |
| --- | --- |
| Stationary, identical | Complete pooling |
| Stable individual differences | Static partial pooling |
| Shared population drift | Shared smooth model |
| Individual drift | Hierarchical smooth model |

All component fits converge. The hierarchical smooth model therefore earns its advantage
only when subjects actually change differently; it loses to simpler models when the truth
is stationary, statically heterogeneous, or shared drift.

The exact outputs are retained in [`result.json`](result.json). This benchmark uses linear
fixed-knot paths, one common subject shrinkage scale, fixed penalties, and subjects already
represented in training. It does not establish hyperparameter recovery, arbitrary curve
recovery, or prediction for a new animal.
