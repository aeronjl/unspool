# IBL 2021 prospective decision-model worked studies

This benchmark extends the checksum-pinned IBL endpoint-window cohort into two deliberately
bounded single-animal examples:

- a stationary Wiener drift-diffusion model that jointly scores choice and movement-onset
  response time in an untouched late-training session; and
- a training-only state-count selection followed by prospective GLM-HMM scoring in the
  same untouched session.

The selected animal is the lexicographically first eligible subject in the frozen manifest.
No choices, response times, or fitted results enter subject selection. The example is useful
for demonstrating a complete evidence path; one animal is not a population estimate.

Run it with:

```bash
uv run --extra ibl python -m benchmarks.ibl2021_decision_models.benchmark
```

The first run resolves about six checksum-pinned Parquet tables from the existing IBL
release cache. Response time is defined in physical seconds as first detected movement
minus go cue, following the IBL wheel-data convention. The committed result retains
eligibility counts, validation boundaries, fit audits, pointwise held-out evidence, state
selection, and model-dependent contaminant and latent-state probabilities.
