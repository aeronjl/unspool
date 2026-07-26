# Recovery near model-family boundaries

This benchmark asks how prospective model recovery changes when four generative families
approach limiting cases in which another explanation can imitate them. It repeats a
stronger reference regime and a boundary-near regime for each current model family:

- a static GLM with either clear or weak stimulus and history effects;
- a smooth GLM with either substantial or subtle coefficient drift;
- a GLM-HMM with either separated or overlapping emission states; and
- Q-learning with either faster, more deterministic or slower, noisier value updating.

All eight scenarios use the same 300-trial, five-session design and the same four
candidates. Each is simulated ten times with recorded child seeds, then evaluated at two
expanding forward-session origins. The comparison therefore changes generative parameters,
not validation data or candidate definitions.

```bash
uv run python -m benchmarks.weak_signal_recovery.benchmark
```

The exact machine-readable result is committed in [`result.json`](result.json). Use
`--repeats` to run a different repetition count and `--output PATH` to retain it.

## Verified result

Across 40 runs from the stronger reference regimes, the generating family is selected in
28 (`70.0%`; 95% Wilson interval `54.6–81.9%`). Across 40 boundary-near runs, it is selected
in 13 (`32.5%`; `20.1–48.0%`). Resolution is similar (`85.0%` versus `82.5%`), so the loss
is genuine off-diagonal confusion rather than only an increase in ties.

The scenario-level matrix exposes the direction of that confusion:

- subtle drift is selected as static in 6/10 runs and as smooth in 2/10;
- overlapping HMM states are selected as static in 6/10, smooth in 3/10, and HMM in 1/10;
- slow/noisy Q-learning is still selected as Q-learning in 7/10 runs, with 3 unresolved;
- weak stationary effects scatter across static, HMM, and Q-learning selections, with two
  unresolved runs.

No candidate-run cell has a failing audit. Warnings occur in 34.4% of cells and remain
visible in the artifact, including boundary estimates, ill-conditioned Hessians, and
restart-objective disagreement.

## Interpretation boundary

These are finite-simulation recovery frequencies for fixed parameter values, candidate
settings, folds, and seeds. The Wilson intervals describe Monte Carlo uncertainty for
these runs; they do not make the scenarios a random sample of all scientifically plausible
behaviours. The result supports a narrower claim: near explicit model-family limits,
prospective predictive selection often cannot recover the generating mechanism, and the
direction of confusion should be reported rather than hidden behind aggregate accuracy.
