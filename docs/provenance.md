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

The first [Cell 2025 benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/cell2025) independently
reimplements one bounded Figure 1 result rather than copying the released notebook. It
pins the Figshare article version, archive file ID, inner member, byte size, and SHA-256;
records a verified numerical result; and keeps all paper-specific code outside
`src/unspool`. The data remain CC BY 4.0 and are fetched on demand rather than redistributed.

The subsequent [Cell behavioural flagship](https://github.com/aeronjl/unspool/tree/main/benchmarks/cell2025_flagship)
retains that independent reproduction and adds a separately named prospective estimand.
It also verifies the released trajectory membership in a pinned compatibility environment
and converts the released Q-value pickle into reviewable JSON without executing or
redistributing the raw artifact. Paper-specific fitting and compatibility code remains in
the benchmark; only the general historical-cohort splitter and comparison provenance enter
the library.

## Subsequent research programme

The private `latent-state-belief-models` research programme developed IBL loaders,
GLM-HMM variants, smooth-drift controls, RL-HMM experiments, recovery grids, and
subject/session-aware comparisons. Unspool will extract only concepts that belong in a
general library. Research-specific evidence machinery and broad thesis dependencies will
remain outside the package.

The public [IBL 2021 benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021) independently implements
a smaller, trial-outcome-blind extraction contract rather than copying that programme's
policy or modelling machinery. It pins 54 public `trials.table` datasets from the fixed
behaviour release, validates each source hash, and keeps all IBL-specific selection and
adaptation code outside `src/unspool`. The data remain CC BY 4.0 and are fetched on demand.

The [flagship prospective study](https://github.com/aeronjl/unspool/tree/main/benchmarks/flagship_longitudinal) composes
those two independently implemented adapters without copying either paper's analysis. It
records the source hashes again, aligns only the declared six-session analysis rank, and
retains source chronology so that the alignment remains reversible and auditable.

## NWB/DANDI interoperability anchor

The [NWB/DANDI benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/nwb_dandi_interoperability) uses published
Dandiset `000004`, version `0.220126.1852`, under CC BY 4.0. It pins one NWB asset by exact
path, asset ID, byte size, and SHA-256. No source file is redistributed: the benchmark
streams only selected trial-table datasets from the content-addressed public blob. Source
field names remain uninterpreted unless an explicit mapping declares otherwise.

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
