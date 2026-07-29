# Chen 2021 restless-bandit literature recipe

This benchmark turns the open data from Chen et al., “Sex differences in learning from
exploration,” *eLife* 10:e69748 (2021), into a bounded Behavio worked study. It is a
**literature-shaped prospective analysis**, not a reproduction of the paper's sex-effect
claim. The source paper and its HMM motivate the task; the recipe asks a narrower package
question: how do common behavioural accounts forecast each animal's next session?

## Reproduce

```bash
uv run python -m benchmarks.chen2021_bandit.fetch_data
uv run python -m benchmarks.chen2021_bandit.benchmark
```

The fetcher uses the public Zenodo mirror of the
[Dryad dataset](https://doi.org/10.5061/dryad.z612jm6c0), verifies its SHA-256 digest, and
rejects unsafe ZIP paths before extraction. The ignored source data comprise 256 CSV files:
32 mice × 8 restless two-armed-bandit sessions. The small verified output is committed as
[`result.json`](result.json).

## Frozen design

- include all 32 mice and all eight sessions;
- retain the first 100 source rows in each animal-session, an outcome-blind compute cap;
- map source choices 1/2 to canonical actions 0/1 and keep the two drifting reward
  probabilities as the explicit simulation environment;
- fit on sessions 1–7 and score session 8 without refitting;
- weight each animal equally and obtain uncertainty by resampling animals;
- compare bias, perseveration, win–stay/lose–shift, and session-reset Q-learning;
- recover only win–stay/lose–shift versus Q-learning, because these are the two candidates
  here that generate coherent action-contingent rewards.

The source `state` column is retained as provenance only. It is the original authors'
HMM-derived quantity, not an observed ground-truth state and not a target in this recipe.

## Claim boundary

Reward-sensitive models forecast the held-out session better than bias and pure
perseveration. Q-learning has a slightly lower animal-balanced point loss than
win–stay/lose–shift, but their paired 95% animal-bootstrap interval includes zero. The
recovery matrix shows separation for two declared parameter regimes under this exact
design; it does not establish which mechanism generated the real animals.

Data are released under CC0. The source dataset DOI is
[`10.5061/dryad.z612jm6c0`](https://doi.org/10.5061/dryad.z612jm6c0), and the analysis
reference is [`10.7554/eLife.69748`](https://doi.org/10.7554/eLife.69748).

## Published parity

[`published_claims.json`](published_claims.json) checks the two cohort facts this recipe
does reproduce — 32 mice and eight sessions each — and records the paper's sex-difference
result as `waived` with a written rationale. The waiver is the point: the recipe never
reads subject sex and fits no exploration parameter, so the claim is machine-readably
unchecked rather than quietly implied by proximity to the citation.
