# Study protocols

A `StudyProtocol` is the frozen scientific contract for one longitudinal modelling
study. It records the question, source identity, cohort, units, clocks, target estimand,
validation geometry, fixed candidates, uncertainty calculation, recovery gates, and the
claims the resulting evidence may and may not support.

The protocol is deliberately narrower than a workflow language. It contains JSON-safe
declarations, not arbitrary Python callbacks, search spaces, credentials, or executable
serialization. Python code supplies a source adapter, registered estimators, and a
registered splitter; the compiler checks those implementations against the frozen
declaration before a fit can begin.

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
| `observations` | Outcome, predictor, and auxiliary columns with data semantics |
| `clocks` and `panel` | Longitudinal coordinate, scope, alignment, and completeness |
| `estimands` | Population, outcome, contrast, weighting, and unit of inference |
| `transforms` | Inputs, outputs, parameters, and training-only visibility |
| `validation` | Deployment geometry, prediction information, origin, and horizon |
| `candidates` | Closed candidate set and fixed hyperparameters |
| `selection` | Optional inner validation performed inside outer training data |
| `comparison` | Proper score, pairing, uncertainty, seed, and winner policy |
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
