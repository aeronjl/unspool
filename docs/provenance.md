# Provenance

Behavio is a new library, but its scientific questions and some future reference
implementations have a traceable history.

## Two machine-readable contracts

Every committed `benchmarks/*/result.json` carries a `provenance` block written by
[`benchmarks/provenance.py`](https://github.com/aeronjl/behavio/blob/main/benchmarks/provenance.py)
at the moment the file is written:

```json
"provenance": {
  "schema_version": 1,
  "git_describe": "41b8cd3-dirty",
  "python": "3.12.13",
  "libraries": {"numpy": "2.3.5", "scipy": "1.18.0", "behavio": "0.1.0"}
}
```

The block deliberately records no wall-clock timestamp. A result that is regenerated on an
unchanged tree at unchanged library versions must be byte-identical, so a changed file
always means changed numbers rather than a changed date. A `-dirty` suffix means the
producing tree had uncommitted changes.

A benchmark that checks a value printed in someone else's paper also carries a
`published_claims.json` beside its result. Each claim names the paper, the accession and
member checksum of the input, the published value, the observed value, an explicit
tolerance with a written rationale, and a status of `pass`, `fail`, or `waived`. No claim
may remain `pending`. `tests/test_published_parity.py` walks every such file offline,
re-reads each observed value out of the committed result, and recomputes the comparison,
so a silent drift away from a published number fails the default test run. A comparison
that does not recover the published value is recorded as `fail` and retained; the ladder in
the [figure standard](reference/figure-standard.md) keeps a `failed-parity` label for
exactly that outcome.

## Cell 2025 anchor

Liebana, Laffere et al., “Dopamine encodes deep network teaching signals for individual
learning trajectories,” *Cell* 188 (2025), 3789–3805.e33.

- Article DOI: <https://doi.org/10.1016/j.cell.2025.05.025>
- Public data: <https://doi.org/10.6084/m9.figshare.28877912>
- MIT-licensed analysis code: <https://doi.org/10.6084/m9.figshare.28877942>

The published archive is a research-analysis reference, not the package skeleton. Any
code adapted from it must retain its license and attribution and be covered by focused
tests in Behavio.

The first [Cell 2025 benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/cell2025) independently
reimplements one bounded Figure 1 result rather than copying the released notebook. It
pins the Figshare article version, archive file ID, inner member, byte size, and SHA-256;
records a verified numerical result; and keeps all paper-specific code outside
`src/behavio`. The data remain CC BY 4.0 and are fetched on demand rather than redistributed.

The subsequent [Cell behavioural flagship](https://github.com/aeronjl/behavio/tree/main/benchmarks/cell2025_flagship)
retains that published-parity reproduction and adds a separately named prospective
estimand. It also verifies the released trajectory membership in a pinned compatibility
environment and converts the released Q-value pickle into reviewable JSON without executing or
redistributing the raw artifact. Paper-specific fitting and compatibility code remains in
the benchmark; only the general historical-cohort splitter and comparison provenance enter
the library.

## Subsequent research programme

The private `latent-state-belief-models` research programme developed IBL loaders,
GLM-HMM variants, smooth-drift controls, RL-HMM experiments, recovery grids, and
subject/session-aware comparisons. Behavio will extract only concepts that belong in a
general library. Research-specific evidence machinery and broad thesis dependencies will
remain outside the package.

The public [IBL 2021 benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021) independently implements
a smaller, trial-outcome-blind extraction contract rather than copying that programme's
policy or modelling machinery. It pins 54 public `trials.table` datasets from the fixed
behaviour release, validates each source hash, and keeps all IBL-specific selection and
adaptation code outside `src/behavio`. The data remain CC BY 4.0 and are fetched on demand.

The [flagship prospective study](https://github.com/aeronjl/behavio/tree/main/benchmarks/flagship_longitudinal) composes
those two independently implemented adapters without copying either paper's analysis. It
records the source hashes again, aligns only the declared six-session analysis rank, and
retains source chronology so that the alignment remains reversible and auditable.

## NWB/DANDI interoperability anchor

The [NWB/DANDI benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/nwb_dandi_interoperability) uses published
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
