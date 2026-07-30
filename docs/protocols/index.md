# Study protocols

A `StudyProtocol` is the frozen scientific contract for one longitudinal modelling
study. It records the question, source identity, cohort, units, clocks, target estimand,
validation geometry, fixed candidates, uncertainty calculation, recovery gates, and the
claims the resulting evidence may and may not support.

The protocol is deliberately narrower than a workflow language. It contains JSON-safe
declarations, not arbitrary Python callbacks, search spaces, credentials, or executable
serialization. Python code supplies a source adapter, registered estimators, and a
registered splitter; those implementations are checked against the frozen declaration
before a fit can begin — the [data contract](#declarations-are-tested-not-annotated) when
the cohort is materialized, and the [candidate declaration](#the-model-that-ran-is-the-model-that-was-declared)
when the runner is handed its models.

<figure class="doc-figure" data-figure-kind="Conceptual">
  <img src="../assets/workflow-map.svg" alt="A scientific question passes through a frozen study contract, materialized cohort, audited execution plan, evaluation, recovery, and bounded report.">
  <figcaption><strong>The protocol boundary.</strong> Design choices become reviewable before outcomes from the scoring set can influence them.</figcaption>
</figure>

## What belongs in the declaration

| Section | What it fixes |
| --- | --- |
| `source` | Adapter, release, locator, checksum, and trial-addressable identity |
| `cohort` | Outcome-blind predicates, selection columns, and expected denominators |
| `units` | Experimental, repeated-measures, and aggregation units |
| `observations` | Outcome, predictor, and auxiliary columns, their measurement type, and their permitted values |
| `clocks` and `panel` | Longitudinal coordinate, scope, alignment, and completeness |
| `estimands` | Population, outcome, contrast, weighting, and unit of inference |
| `transforms` | Inputs, outputs, parameters, and training-only visibility |
| `validation` | Deployment geometry, prediction information, origin, and horizon |
| `candidates` | Closed candidate set and fixed hyperparameters |
| `selection` | Optional inner validation performed inside outer training data |
| `comparison` | Proper score, pairing, uncertainty, seed, multiplicity adjustment, and winner policy |
| `recovery` | Exact-design falsification requirements and the claims they gate |
| `reporting` | Required evidence, limitations, and prohibited claims |

## The shortest useful workflow

```python
from behavio import (
    compile_execution_plan,
    materialize_protocol,
    model_capabilities,
    run_protocol,
)

frozen = protocol.freeze()
materialized = materialize_protocol(frozen, source_study)
splits = splitter(materialized.study)
compiled = compile_execution_plan(
    materialized,
    splits,
    capabilities={name: model_capabilities(model) for name, model in models.items()},
)

if not compiled.plan.audit.passed:
    raise RuntimeError(compiled.plan.audit.errors)

evaluation = run_protocol(compiled, models)
```

This sequence is intentionally explicit. Materialization proves the cohort denominator;
compilation proves the row and information boundaries; only then does the runner fit and
score candidates.

## Declarations are tested, not annotated

`ObservationSpec` fixes a column's measurement type and, optionally, the exact values it
may take. Both are checked against the materialized cohort, so a protocol cannot assert a
data contract it was never held to:

```python
ObservationSpec(
    "choice", ObservationRole.OUTCOME, ObservationDataType.BINARY, allowed_values=(0, 1)
)
ObservationSpec(
    "response_time",
    ObservationRole.AUXILIARY,
    ObservationDataType.CONTINUOUS,
    allowed_values=(None,),
)
```

`data_type` is a closed vocabulary — `binary`, `categorical`, `continuous`, `count`,
`ordinal` — so it means something a materializer can test. The values are the wire format,
so every protocol serialized before the vocabulary was closed round-trips unchanged and
keeps its fingerprint.

Missing observations are ordinary behavioural data: omissions, aborted trials, and
unrecorded response times are not contract violations. An empty `allowed_values` declares
no value contract and never rejects a missing observation. A non-empty `allowed_values`
does constrain the column, and must name `None` before missing rows are permitted — the
same treatment `ChoiceSpec` gives omissions, which it retains only when they are declared.

`materialize_protocol` refuses a cohort that violates the contract and names the column,
the rule, the number of offending rows, and a sample of the offending values.
`validate_observation_contract(protocol, study)` returns the same findings without raising
when you want to review the complete set first.

## The model that ran is the model that was declared

A `CandidateSpec` fixes an `implementation` and its `hyperparameters` before any data is
seen. `run_protocol` and `run_nested_protocol` verify the estimator you supply against that
declaration before the first fit, so an evidence bundle records what happened rather than
what was claimed:

```python
from behavio import verify_candidate_declarations

for verification in verify_candidate_declarations(frozen, models):
    print(verification.candidate, verification.verified)
```

Identity is resolved from the supplied object — its type's import path, including the
public paths that already re-export it — and never by importing the declared string. A
frozen protocol is data; importing a module path out of it would turn a declaration into
code execution.

Because every model in the package is a frozen dataclass, each declared `Setting` is
compared against the corresponding field value, with floating-point tolerance and with
booleans kept distinct from `0` and `1`.

The two outcomes are kept apart deliberately:

- **Contradicted.** The supplied object is a different class, or a field's value differs
  from the frozen one. The run is refused. There is nothing to salvage: every prediction,
  score, and comparison would be content-addressed under a protocol that declares a
  different model.
- **Unverifiable.** A declared setting has no matching field, an observed value is not a
  comparable JSON scalar, the object is not a dataclass, or the declared module is not
  imported. These are retained rather than treated as satisfied, and a bounded report
  discloses them in an *Unverified declarations* table.

Identity is resolved through an [`EstimatorRegistry`](../extensions.md#local-registration),
defaulting to the package's built-in allowlist, and never by importing the declared string.
A registered implementation is therefore always decidable: the registration declares the
class its factory produces, so the supplied object either is an instance of it or is not.
Combinator registrations expose the model they wrap, so a candidate declared as
`behavio.compose.hierarchical` with `base.`-prefixed settings has its wrapped model checked
too. Only a name no registry knows falls back to the import-free class-name comparison, and
only that fallback can report *unverifiable* for an implementation. Pass `registry=` to
`verify_candidate_declarations` to make your own models decidable on the same terms.

## The declared score is executable, not descriptive

The common runner currently supports equal-unit paired comparison with log loss,
joint log loss, or Brier score. The declared rule controls inner selection, outer
evaluation, paired differences, uncertainty, ranking, serialization, and report labels.
Brier scoring is accepted only for one observed binary 0/1 outcome; multiple jointly
scored outcomes require joint log loss. Unsupported weighting, unpaired comparison, or
interval declarations are rejected when the protocol is constructed rather than being
silently approximated.

Nested procedures may deliberately use a different aggregation unit or proper score for
training-only selection than for the final outer estimand. The runner keeps those two
contracts separate: inner candidate evidence uses `selection`, while untouched outer
performance uses `comparison`.

## One fold loop, and how many contrasts read it

The runner does not implement fitting, prediction, or scoring. It adapts each compiled fold
to the [`ValidationFold`](../reference/contracts.md) contract and calls
`behavio.evaluation.evaluate_splits`, the same primitive `compare_models` calls, with
`FoldFailurePolicy.RETAIN`. Retaining a fold failure rather than raising on it is the one
thing a protocol run needs that an interactive call does not, and it is now a declared
option on the shared loop instead of a second implementation of it. A number produced under
a frozen protocol therefore equals the number `compare_models` produces from the same models
and folds, bit for bit.

`WinnerPolicy.INTERVAL_EXCLUDES_ZERO` reads the whole family of contrasts, not one at a
time. `K` eligible candidates produce `K(K-1)/2` simultaneous readings of the same interval
level, so a leader compared against four rivals has four chances to clear a threshold stated
for one. The winner rule therefore requires each contrast to survive a multiplicity
adjustment across the family at `1 - interval_level` — the rate the protocol already
declared, now controlled family-wise rather than per contrast. `Ranking.family` records the
family size, how many contrasts separated before adjustment, how many were expected to by
chance, the exact binomial probability of at least that many, and how many survived; and
`Ranking.reason` states all of it in prose. Every per-contrast interval and probability
remains unadjusted and fully retained. Correction can only remove a separation, never add
one, so this makes `UNRESOLVED` strictly more likely than the uncorrected rule did.

That adjustment is **declared, not chosen after the fact**. It is
`ComparisonSpec.multiplicity`, frozen with the rest of the design:

```python
ComparisonSpec(
    metric=ScoreMetric.LOG_LOSS,
    aggregation_unit="animal",
    weighting=AggregationWeighting.EQUAL_UNIT,
    interval_method="paired-unit-bootstrap",
    interval_level=0.95,
    bootstrap_repetitions=5_000,
    seed=2025,
    paired=True,
    winner_policy=WinnerPolicy.INTERVAL_EXCLUDES_ZERO,
    multiplicity=ComparisonMultiplicity.BENJAMINI_HOCHBERG,
)
```

Every other verdict-determining choice in this package is fixed before data is seen; an
adjustment that decides which candidate wins had no business being the one exception. The
default is `BENJAMINI_HOCHBERG`, which is exactly what the runner applied unconditionally
before the member existed, so adopting it changes no verdict — it writes down a behaviour
that was already happening. `BONFERRONI` controls the family-wise rather than the
false-discovery rate. `NONE` restores the uncorrected per-contrast reading and will name a
winner strictly more often; `Ranking.reason` then says so in as many words rather than
reporting an adjustment that did not run. There is no separate family error rate to
declare: it is `1 - interval_level`, the rate the protocol already fixed.

Adding the member moved the schema to `behavio.study-protocol/2`. Protocols recorded under
version 1 still load, keep their fingerprints, and are read as declaring Benjamini-Hochberg
— see [serialization and schema versions](lifecycle.md#serialization-and-schema-versions).

## Start from a complete study

The complete Cell declaration lives in
[`benchmarks/cell2025_protocol/benchmark.py`](https://github.com/aeronjl/behavio/blob/main/benchmarks/cell2025_protocol/benchmark.py).
It freezes a 30-animal, 73,042-trial historical-cohort forecast with six candidates and
three exact-design recovery gates. The IBL declaration in
[`benchmarks/ibl2021_protocol/benchmark.py`](https://github.com/aeronjl/behavio/blob/main/benchmarks/ibl2021_protocol/benchmark.py)
uses the same core types for a checksum-pinned 78-animal public cohort, ordinal session
alignment, and nested same-animal or held-out-laboratory selection.

Those two migrations are part of the contract tests: the abstraction must express both
studies without Cell- or IBL-specific branches in the protocol core.

## Continue

- [Freeze, amend, and advance a protocol](lifecycle.md)
- [Review the compiled boundary audit](auditing.md)
- [Run and inspect protocols from the command line](cli.md)
- [Build and replay an evidence bundle](evidence-bundles.md)
