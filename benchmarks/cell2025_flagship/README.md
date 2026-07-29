# Cell 2025 flagship behavioural study

This benchmark is the full behavioural flagship for Behavio. Its
[`DESIGN.md`](DESIGN.md) freezes the questions, exclusions, clocks, candidates, validation
geometry, recovery requirements, and claim boundaries before fitting.

There are two deliberately separate result layers:

- a published-parity reproduction of behavioural results from Liebana, Laffere et al.
  (*Cell*, 2025); and
- a new historical-cohort-calibrated forecast of an animal's final five sessions from
  its first eight days.

The implementation retains the published trial and source-session identities, constructs
the declared 13-coordinate panel, evaluates six models with animal-balanced prospective
scores, and runs design-matched structural, path, and early-bias recovery. The existing
bounded Figure 1 reproduction remains in
[`benchmarks/cell2025`](../cell2025/README.md).

## Run the study

Fetch and verify the public behaviour table and the small released analysis artifacts:

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
```

Then execute the frozen study:

```bash
uv run python -m benchmarks.cell2025_flagship.benchmark
```

The command writes `result.json`. It is intentionally compute-heavy because every
candidate is refitted in six animal-level folds and the same experimental design is used
for repeated model and parameter recovery.

## Released-analysis compatibility

The published Gaussian-process/soft-DTW visualization is reproduced in a small isolated
compatibility environment rather than adding old numerical packages to Behavio's runtime:

```bash
uv venv --python 3.12 .venv-cell2025-release
uv pip install --python .venv-cell2025-release/bin/python \
  numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 \
  scikit-learn==1.5.1 tslearn==0.6.3 jax==0.4.34 jaxlib==0.4.34
.venv-cell2025-release/bin/python \
  -m benchmarks.cell2025_flagship.released_trajectory_clustering
```

The resulting semantic memberships are checked animal by animal against the released
CSV. The released Q-value pickle is decoded into safe, reviewable JSON and summarized as
a retrospective result; Behavio does not claim to have independently reoptimized that
115-minute fit.

The same isolated stack produces the audited Figure 1H/1J centroid artifact directly
from the checksum-pinned released GP fits:

```bash
.venv-cell2025-release/bin/python \
  -m benchmarks.cell2025_flagship.released_figure1hj
```

`figure1hj_trajectories.json` records the distinct panel contracts, both sets of
soft-DTW centroids, exact semantic memberships, and numerical versions. Its companion
`figure1hj_audit.json` traces the paper panel, released notebook cells, source checksums,
preserved geometry, intentional documentation changes, and the boundary that this is a
released-fit replay rather than an independent raw-trial GP refit.
