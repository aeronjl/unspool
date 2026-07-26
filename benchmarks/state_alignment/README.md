# Latent-state alignment recovery

This benchmark tests whether GLM-HMM state recovery remains meaningful after arbitrary
label permutations and whether weakly distinguished states produce an explicit alignment
warning. It does not treat the model's canonical intercept ordering as recovered truth.

```bash
uv run python -m benchmarks.state_alignment.benchmark
```

Each of 20 repetitions simulates four 75-trial sessions from a persistent two-state HMM,
fits the observations with three deterministic restarts, and aligns outcome-filtered state
probabilities to the separately retained simulation truth. The assignment maximizes mean
posterior mass balanced across true states. Its uncertainty diagnostic compares the winning
assignment with the best alternative; a gap at or below `0.05` is marked ambiguous.
The filter excludes later outcomes from each state update, but its parameters are estimated
from the complete simulated study; this is conditional decoding recovery, not a prospective
parameter-fitting claim.

Two regimes share initial and transition probabilities and differ only in their emission
intercepts:

- **clear:** `(-2.0, 2.0)`;
- **overlapping:** `(-0.25, 0.25)`.

## Pinned result

All 40 fits converge. Clear states reach 91.85% mean aligned decoded accuracy and 88.06%
mean posterior accuracy, with no ambiguous assignments. Overlapping states reach 56.53%
decoded accuracy and 55.92% posterior accuracy; 35% of their assignments are ambiguous.
The mean best-versus-runner-up assignment gap falls from `0.7523` to `0.0922`.

Reversing every inferred-state column changes raw decoded accuracy from 91.85% to 8.15% in
the clear regime, while the aligned probabilities and all aligned metrics are unchanged in
every run. This is the distinction the benchmark is designed to enforce.

The exact output is retained in [`result.json`](result.json). Alignment uses known
simulation states and therefore belongs to recovery analysis only. It cannot validate
state identities in empirical data, and filtered recovery here does not imply cognitive
interpretability.
