# Portable evidence bundles

An evidence bundle is the final, content-addressed record of a reported protocol. It is a
byte-deterministic ZIP archive whose manifest hashes every logical file. It carries enough
information to inspect the scientific boundary and reproduce the analysis against the
identified source, without redistributing the raw source trials.

## Bundle anatomy

```text
study-evidence.zip
├── bundle.json
├── protocol/
│   ├── protocol.json
│   └── amendments.json
├── environment/environment.json
├── source/source.json
├── cohort/manifest.json
├── execution/
│   ├── plan.json
│   └── folds.json
├── audits/
│   ├── protocol.json
│   └── fits.json
├── comparison/evaluation.json
├── predictions/pointwise.json
├── recovery/recovery.json        # when recovery was run
├── posterior/                    # when the study ran Bayesian inference
│   ├── convergence.json
│   ├── loo.json
│   ├── predictive.json
│   ├── calibration.json
│   ├── reliability.json
│   └── sensitivity.json
├── figures/
├── report/
│   ├── report.json
│   ├── report.md
│   └── items.json
└── reproduction/replay.json
```

The protocol, source identity, cohort and execution fingerprints, predictions, numerical
audits, comparisons, recovery assessment, rendered figures, and bounded report remain
separate artifacts. This makes it possible to identify what changed between two analyses
without loading a Python object graph.

`predictions/pointwise.json` and `audits/fits.json` are keyed candidate-by-candidate and
then fold-by-fold, under the name the fold declared through
[`ValidationFold.identifier`](../reference/contracts.md). That is the only key those maps
have, so `evaluate_splits` refuses a split set whose fold names collide rather than letting
one fold overwrite another in the archive. See
[every fold names itself](../validation.md#every-fold-names-itself) for the naming scheme.

## Archive the Bayesian half

`PosteriorEvidence` carries the posterior artefacts. Every slot is optional and an empty
slot writes no file, so a study that ran no Bayesian inference produces exactly the archive
it produced before these slots existed.

```python
from behavio import PosteriorEvidence, build_evidence_bundle

posterior = PosteriorEvidence(
    convergence={"hierarchical-glm": audit},
    loo={
        "hierarchical-glm/observation": observation_loo,
        "hierarchical-glm/subject": subject_loo,
    },
    predictive={"hierarchical-glm": predictive_audit},
    calibration={"hierarchical-glm": sbc_report},
    reliability={"learning-rate": reliability_report},
    sensitivity={"prior-width": sensitivity_report},
)

bundle = build_evidence_bundle(reported, figures=(figure,), posterior=posterior)
```

Each slot maps a label you choose to one report, serialized through that report's own
`to_dict`. `posterior/loo.json` is grouped by estimand rather than stored as one flat list:
leave-one-observation-out and leave-one-*block*-out ELPD answer different questions and are
never comparable, so the archive keeps them under separate, explicitly named groups that a
later reader cannot difference by accident. Two results that share a model and a blocking
are indistinguishable evidence and are refused outright.

```text
posterior/loo.json
{
  "leave-one-observation-out": {"block": null,      "results": {...}},
  "leave-one-subject-out":     {"block": "subject", "results": {...}}
}
```

## Schema versions

`bundle.json` declares the oldest published schema name that can express the bundle's
content, not the newest name the library knows:

| Name | Adds |
| --- | --- |
| `behavio.evidence-bundle/1` | the frequentist evidence |
| `behavio.evidence-bundle/2` | the optional `posterior/` slots |

A version-1 bundle is a valid version-2 bundle that happens to carry no posterior evidence,
so both names are accepted on read and neither is ever restamped — restamping a
content-addressed archive would invalidate its own identity. Stamping the minimum is what
keeps a frequentist bundle byte-identical across the version bump; a bundle carrying any
`posterior/` slot declares version 2. An unrecognised name is refused rather than guessed at.

## Build the final archive

```python
from pathlib import Path

from behavio import BundleFigure, build_evidence_bundle, write_evidence_bundle

figure = BundleFigure(
    name="forecast-comparison",
    filename="forecast.svg",
    content=Path("forecast.svg").read_bytes(),
)

bundle = build_evidence_bundle(reported, figures=(figure,))
write_evidence_bundle(bundle, "study-evidence.zip")

print(bundle.bundle_id)
```

Only a protocol in the `reported` state can produce a bundle. Every figure required by
the reporting declaration must be supplied. Unsupported executable serializations such as
pickle, joblib, NumPy object archives, or bytecode are rejected.

The writer uses fixed ZIP metadata, sorted paths, stored compression, canonical JSON, and
exclusive file creation. The same logical evidence therefore produces the same bytes and
bundle identity; an existing archive is never overwritten silently.

## Replay without executing code

```python
from behavio import read_evidence_bundle, replay_evidence_bundle

verified = read_evidence_bundle("study-evidence.zip")
replay = replay_evidence_bundle(verified)

print(replay.protocol_fingerprint)
print(replay.cohort_observations)
print(replay.ranking_status, replay.winner)
print(replay.blocked_claims)
```

Reading verifies safe paths, unique entries, declared sizes, SHA-256 digests, and the
bundle identity without extracting files. Replay then cross-checks protocol, cohort,
plan, evaluation, recovery, report, and lifecycle fingerprints. It does not unpickle a
fit or execute source code.

## Compare two analyses

```python
from behavio import compare_evidence_bundles

difference = compare_evidence_bundles("before.zip", "after.zip")
print(difference.same_protocol)
print(difference.changed_paths)
print(difference.left_winner, difference.right_winner)
print(difference.same_posterior_evidence)
print(difference.added_posterior_paths, difference.removed_posterior_paths)
print(difference.left_loo_estimands, difference.right_loo_estimands)
```

A comparison distinguishes a new scientific protocol from new evidence under the same
protocol. It also exposes changes to rankings and blocked claims, rather than reducing the
comparison to a checksum mismatch.

Comparing a bundle that carries posterior evidence against one that does not is allowed and
is reported explicitly: `same_posterior_evidence` is false, and the appearing or vanishing
slots and cross-validation estimands are named. A study that gained a Bayesian half is a
different claim, not a formatting change, so the difference is surfaced rather than absorbed
into the generic path lists.

## What is deliberately absent

- raw source trials and access credentials;
- mutable caches or machine-specific absolute paths;
- arbitrary estimator objects and executable serialization;
- a claim that the environment description alone recreates unavailable data;
- claims prohibited by the protocol, even when the point estimate is favorable.

Reproduction requires resolving the declared source release and checksum through an
appropriate adapter. If that source can no longer be obtained, the bundle remains an
auditable record of what was done but cannot make the data reappear.
