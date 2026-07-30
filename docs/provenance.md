# Provenance

Behavio is a new library, but its scientific questions and some future reference
implementations have a traceable history.

## The library's own provenance: `environment/environment.json`

Every [evidence bundle](protocols/evidence-bundles.md) carries an environment record
written by `behavio.report.evidence_bundles.capture_environment` at the moment the bundle is built:

```json
{
  "python": {"implementation": "CPython", "version": "3.12.13"},
  "platform": {"system": "Linux", "machine": "x86_64"},
  "packages": {
    "behavio": "0.1.0", "numpy": "2.3.5", "scipy": "1.18.0",
    "arviz": "1.2.0", "arviz-stats": "1.2.0", "pymc": "6.1.0",
    "pytensor": "3.2.3", "pybads": "not installed",
    "h5py": "3.16.0", "matplotlib": "3.11.1", "one-api": "not installed",
    "pandas": "3.0.5", "pyarrow": "25.0.0", "pynwb": "not installed",
    "remfile": "not installed", "tables": "not installed",
    "xarray": "2026.7.0"
  },
  "source_control": {
    "system": "git",
    "available": true,
    "commit": "41b8cd3fbb2a0e5e0d1c8f9a6e4b2c7d3a5f1e09",
    "dirty": false
  }
}
```

Three properties are deliberate.

**Absence is recorded, not omitted.** Every extra declared in `pyproject.toml` appears in
`packages` whether or not it was installed — the posterior stack (`arviz`, `arviz-stats`,
`pymc`, `pytensor`), the optimization backend (`pybads`), the data adapters (`h5py`,
`one-api`, `pandas`, `pyarrow`, `pynwb`, `remfile`, `tables`, `xarray`), and the renderer
(`matplotlib`). A missing key cannot be told apart from a reader that forgot to look,
whereas `"arviz": "not installed"` is a positive claim about the run that produced the
numbers. These are exactly the libraries whose versions change results, so a bundle that
named only `numpy` and `scipy` would omit the parts most likely to explain a numerical
difference. `matplotlib` earns its place for a reason worth stating: figure bytes are
archived files, so they are content-addressed into the `bundle_id`, and a renderer upgrade
changes the identity of a bundle whose science did not move.

**A dirty tree says so.** `source_control` records the exact `HEAD` commit of the working
tree the bundle was built from, and whether that tree had uncommitted changes. A directory
that is not a repository, or a machine with no usable `git`, reports `available: false`
with a machine-readable `reason`; it never reports a commit it could not read. This is the
library-side counterpart of the `git_describe` field that `benchmarks/provenance.py` has
always recorded, and it closes the gap where the benchmarks had stronger provenance than
the library they exercise.

**There is no timestamp.** Like the benchmark stamp, the environment record carries no
wall-clock time and no filesystem path, so a re-run on an unchanged tree at unchanged
versions produces byte-identical bytes. The record is one of the archived files, so it
contributes to the bundle identity: a bundle built on a different interpreter, at different
library versions, or from a different commit has a different `bundle_id`. Nothing else does
— no protocol, cohort, plan, evaluation, recovery, or report fingerprint is computed from
the environment, so changing the environment record never disturbs the scientific identity
cross-checks that `replay_evidence_bundle` performs.

Pass `environment=` to `build_evidence_bundle` to record a description you captured
yourself; the default calls `capture_environment()` against the current directory.

## The declaration is checked against what ran

A content address is only provenance if the thing addressed is the thing that happened. A
protocol fingerprint covers a frozen `CandidateSpec` — an `implementation` path and a set
of `hyperparameters` — so an evaluation executed with a different estimator, or the same
estimator at a different regularization strength, would content-address a claim rather
than a result.

`run_protocol` and `run_nested_protocol` therefore verify every supplied estimator against
its frozen declaration before the first fit and refuse the run on a contradiction. The
declared string is never imported, so a frozen protocol stays data rather than becoming a
code-execution surface. Resolution goes instead through an
[`EstimatorRegistry`](extensions.md#local-registration), the same allowlist the command
line builds candidates from.

That registry is what makes the check *decidable*. A registration declares the class its
factory produces, so a declared implementation the registry knows is either verified or
contradicted — never shrugged at. A combinator registration additionally exposes the model
it wraps, so a candidate declared as `behavio.compose.hierarchical` with `base` and
`base.`-prefixed settings has its wrapped model checked too, rather than reporting every
`base.*` setting as a field that does not exist. Pass `registry=` to
`verify_candidate_declarations` to have your own registrations checked on the same terms.

Declarations the registry cannot speak to fall back to an import-free comparison of class
names against already-imported modules, and what that cannot decide is recorded rather
than assumed. A setting with no matching field, a value that is not a comparable JSON
scalar, an estimator that is not a dataclass, or a declared module the process never
imported all produce an *unverifiable* finding, which a bounded report discloses in its own
table. A fully verified study emits no such table and keeps a byte-identical report.

The same principle covers the data: an `ObservationSpec` declares a measurement type and a
permitted value set, and `materialize_protocol` refuses a cohort that does not satisfy
them. A bundle therefore cannot assert a data contract that was never tested.

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
