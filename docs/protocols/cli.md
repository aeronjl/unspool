# Protocol command line

The `unspool` command exposes a deliberately small operational surface. It validates and
freezes canonical protocol JSON, executes registered built-in models and splitters, and
verifies final evidence bundles. It is not a scheduler and does not execute import paths
or arbitrary configuration code from a protocol.

## Validate or freeze

```bash
unspool protocol-validate protocol.json
unspool protocol-validate draft.json --freeze-out frozen.json
```

The command prints the schema version, state, and scientific fingerprint as JSON. The
freeze output must be a new path; existing files are never overwritten.

## Execute a registered protocol

```bash
unspool execute frozen.json study.json evaluation/
```

`study.json` is a JSON object with a `columns` mapping accepted by `Study`. The command
materializes the cohort, invokes the declared registered splitter, compiles and checks the
plan, instantiates the exact registered candidate set, and runs flat or nested evaluation.
The new output directory contains:

```text
evaluation/
├── protocol.json
├── cohort.json
├── plan.json
├── evaluation.json
└── snapshot.json
```

The snapshot contains fingerprints of every artifact and records whether required
recovery is still pending. This is an evaluation snapshot, not a final evidence bundle;
required recovery, bounded reporting, and figures must still be completed.

## Inspect and compare evidence

```bash
unspool inspect study-evidence.zip
unspool bundle-compare previous.zip current.zip
unspool report study-evidence.zip
unspool report study-evidence.zip --output report.md
```

All three commands verify the archive manifest and cross-artifact identities before
returning content. `bundle-compare` reports changed logical paths, whether the protocol
identity changed, and whether the scientific decision or blocked claims differ.

## Closed registries are intentional

Protocols may name only model and splitter implementations in Unspool's built-in CLI
registry. Unknown implementations fail with a clear error. Library users may supply
Python estimator objects directly to `run_protocol` or `run_nested_protocol`, but the
portable JSON file never becomes a route to import and execute arbitrary code.

This division keeps the CLI safe and reproducible while leaving the estimator contract
open to research extensions.
