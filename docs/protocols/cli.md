# Protocol command line

The `behavio` command exposes a deliberately small operational surface. It validates and
freezes canonical protocol JSON, executes registered built-in models and splitters, and
verifies final evidence bundles. It is not a scheduler and does not execute import paths
or arbitrary configuration code from a protocol.

## Validate or freeze

```bash
behavio protocol-validate protocol.json
behavio protocol-validate draft.json --freeze-out frozen.json
```

The command prints the schema version, state, and scientific fingerprint as JSON. The
freeze output must be a new path; existing files are never overwritten.

## Execute a registered protocol

```bash
behavio execute frozen.json trials.csv evaluation/
behavio execute frozen.json trials.parquet evaluation/
behavio execute frozen.json study.json evaluation/
```

The study argument is a real trial table: `.csv`, `.tsv`, or `.parquet`, or the original
`.json` object with a `columns` mapping accepted by `Study`. The format follows the suffix;
`--study-format {auto,json,csv,tsv,parquet}` declares it when a file is named something
else. CSV and TSV need no optional dependencies; Parquet needs `behavio[parquet]`.

Source columns rarely use Behavio's canonical names, and chronology is often absent:

```bash
behavio execute frozen.json trials.csv evaluation/ \
  --subject-column participant \
  --session-column visit \
  --session-order-from-column visit_date \
  --number-trials-by-row-order
```

`--session-order-from-column COLUMN` ranks each subject's sessions by a column that carries
time. `--session-order-from-appearance` claims instead that the rows were written in
chronological order. One or the other is *required* when the table has no `session_order`
column: the command never infers chronology, and the failure message lists every flag that
can supply it. Both choices are recorded on every trial, as is
`--number-trials-by-row-order`.

By default each trial also records its absolute source path, which makes the protocol
fingerprint machine-specific; `--omit-source-path` drops that column when a fingerprint
that is reproducible across machines matters more.

The command materializes the cohort, invokes the declared registered splitter, compiles and
checks the plan, instantiates the exact registered candidate set, and runs flat or nested
evaluation. The new output directory contains:

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
behavio inspect study-evidence.zip
behavio bundle-compare previous.zip current.zip
behavio report study-evidence.zip
behavio report study-evidence.zip --output report.md
```

All three commands verify the archive manifest and cross-artifact identities before
returning content. `bundle-compare` reports changed logical paths, whether the protocol
identity changed, and whether the scientific decision or blocked claims differ.

## Closed registries are intentional

Protocols may name only model and splitter implementations in Behavio's built-in CLI
registry. Unknown implementations fail with a clear error. Library users may supply
Python estimator objects directly to `run_protocol` or `run_nested_protocol`, but the
portable JSON file never becomes a route to import and execute arbitrary code.

This division keeps the CLI safe and reproducible while leaving the estimator contract
open to research extensions.
