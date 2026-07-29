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
```

A comparison distinguishes a new scientific protocol from new evidence under the same
protocol. It also exposes changes to rankings and blocked claims, rather than reducing the
comparison to a checksum mismatch.

## What is deliberately absent

- raw source trials and access credentials;
- mutable caches or machine-specific absolute paths;
- arbitrary estimator objects and executable serialization;
- a claim that the environment description alone recreates unavailable data;
- claims prohibited by the protocol, even when the point estimate is favorable.

Reproduction requires resolving the declared source release and checksum through an
appropriate adapter. If that source can no longer be obtained, the bundle remains an
auditable record of what was done but cannot make the data reappear.
