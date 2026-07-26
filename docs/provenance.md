# Provenance

Unspool is a new library, but its scientific questions and some future reference
implementations have a traceable history.

## Cell 2025 anchor

Liebana, Laffere et al., “Dopamine encodes deep network teaching signals for individual
learning trajectories,” *Cell* 188 (2025), 3789–3805.e33.

- Article DOI: <https://doi.org/10.1016/j.cell.2025.05.025>
- Public data: <https://doi.org/10.6084/m9.figshare.28877912>
- MIT-licensed analysis code: <https://doi.org/10.6084/m9.figshare.28877942>

The published archive is a research-analysis reference, not the package skeleton. Any
code adapted from it must retain its license and attribution and be covered by focused
tests in Unspool.

## Subsequent research programme

The private `latent-state-belief-models` research programme developed IBL loaders,
GLM-HMM variants, smooth-drift controls, RL-HMM experiments, recovery grids, and
subject/session-aware comparisons. Unspool will extract only concepts that belong in a
general library. Research-specific evidence machinery and broad thesis dependencies will
remain outside the package.

## Earlier public explorations

- [InfiniteIOHMM.jl](https://github.com/aeronjl/InfiniteIOHMM.jl)
- [daplearning](https://github.com/aeronjl/daplearning)
- [PyDAP](https://github.com/aeronjl/PyDAP)

These repositories establish intellectual lineage but are not production dependencies.

## Extraction rule

Every extracted component should record:

1. its scientific source;
2. whether code was adapted or independently reimplemented;
3. the original license and contributors where adaptation occurred;
4. a regression or recovery test showing what was preserved; and
5. any change in the allowed scientific interpretation.
