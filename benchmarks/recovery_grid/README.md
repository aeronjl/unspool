# Matched four-family recovery grid

This synthetic benchmark asks whether the same prospective evaluation can distinguish
four explanations represented by Unspool's current reference models:

- stationary stimulus and choice-history effects;
- smoothly drifting coefficients;
- discrete switching among GLM emission states; and
- reward-driven Q-learning.

It is deliberately a first grid, not a universal recovery claim. Each family contributes
one explicit parameter regime, each cell has one repeat, and the two nested designs contain
five sessions with either 30 or 60 trials per session. Every candidate is refitted at each
expanding forward-session origin and selected by future-trial mean log probability.

```bash
uv run python -m benchmarks.recovery_grid.benchmark
```

The exact machine-readable output is committed in [`result.json`](result.json).

## Shared design

All scenarios receive the same session identities, stimulus stream, volatile two-action
reward environment, and a pre-generated nuisance reward stream. Static, smooth, and
GLM-HMM generators replace only choice, so the Q-learning candidate can still be tested
against those non-learning truths. The Q-learning generator replaces both choice and
reward with its action-contingent process. Within each simulated `Study`, every candidate
receives the same rows and validation folds; the nuisance reward keeps Q-learning a valid
competitor when another family generated the choices.

The sparse design is nested within the dense design: its first 30 trials of every session
are identical design rows. Generative randomness is independently and reproducibly derived
for each design cell and scenario run.

## Verified result

In the sparse design, the stationary and Q-learning truths are recovered, while smooth
drift is selected as static and switching states are selected as smooth drift. Overall
accuracy is therefore `2 / 4`. In the dense design all four generating families are
selected correctly.

Warnings are retained for candidate fits and summarized per cell. They remain eligible for
selection because a warning can limit uncertainty or latent-state interpretation without
making filtered prediction numerically unusable. Failing audits are retained but excluded
from selection. Unresolved ties remain an explicit fifth confusion-matrix column.

## Interpretation boundary

The improvement from 150 to 300 trials demonstrates design dependence for these exact
parameters, seeds, candidate settings, and folds. It does not establish a general sample-
size threshold, and perfect recovery in one dense run is not evidence of universal
identifiability. The next expansion should add repeats and scientifically plausible weak-
signal parameter regimes near the boundaries where these families imitate one another.
