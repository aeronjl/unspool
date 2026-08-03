# Session-dynamic GLM-HMM selection and recovery

This synthetic benchmark asks two separate questions on matched seven-session designs:

1. Can nested prospective selection choose the state count `K`, emission random-walk scale
   `sigma`, and transition concentration `alpha` without seeing the seventh-session choices?
2. Does that selected procedure forecast the seventh session better than stationary and
   observed-transition GLM-HMMs, a smooth observable drift model, and Q-learning—and does
   it recover the realized latent path when the dynamic model generated the data?

The two regimes share the same two-state baseline parameters. One retains them across all
sessions; the other draws emission paths and session transition matrices from the declared
dynamic priors. Rewards are random and choice-independent so Q-learning is an explicit
falsification competitor. The outer test is session 7. Sessions 1–6 alone feed the nested
selector, whose inner forward folds choose among the full `K × sigma × alpha` grid. The
selected specification is then refitted on sessions 1–6 and scored once on session 7.

Trajectory recovery is descriptive and separate from forecasting: after selection, the
chosen configuration is fitted to the complete simulated path and aligned to retained
truth. It is reported only for two-state selections, because coefficient-path RMSE is not
defined between different state dimensions.

Run the pinned experiment with:

```bash
uv run python -m benchmarks.session_dynamic_glm_hmm.benchmark
```

The committed `result.json` records every selected specification, prospective loss,
convergence outcome, and compatible truth-aligned recovery result, plus deterministic
environment provenance.

The pinned four-repetition result is deliberately negative on its strongest claim. Under
dynamic truth, the selector recovered the exact generating grid point in all four runs and
decoded latent states accurately, but the stationary GLM-HMM still had the lower mean
seventh-session log loss. Under stationary truth, the stationary model won three of four
individual runs, while one selected three-state fit shifted the four-run mean and did not
converge in its subsequent full-path partial stage. This benchmark therefore establishes
executable training-only selection and descriptive path recovery—not reliable prospective
discrimination of stationary and session-dynamic structure.
