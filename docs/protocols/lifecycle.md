# Freeze, amend, and advance

Protocol state is evidence state. A scientific declaration may be edited while it is a
draft, but the transition to `frozen` establishes the pre-evidence identity used by every
downstream artifact. Later states are appended only when the corresponding artifact
exists and supplies its content fingerprint.

```text
draft -> frozen -> materialized -> audited -> evaluated -> recovered -> reported
                                      |                         |
                                      +------> reported <-------+
```

`recovered` is required when the protocol declares required recovery analyses. A protocol
without recovery requirements may advance from `evaluated` directly to `reported`.

## Freeze before looking

```python
frozen = protocol.freeze()

print(frozen.state.value)  # frozen
print(frozen.fingerprint)  # scientific identity
print(frozen.canonical_json())  # complete portable record
```

The fingerprint covers the scientific declaration and amendment history. Lifecycle
events do not alter it, so the materialized cohort, execution plan, evaluation, recovery,
and report can all refer to one stable study identity.

Freezing is allowed only for a pristine draft. Materialization and fitting APIs reject
protocols in the wrong state.

## Amend transparently before evidence

If a frozen design must change before materialization, make an explicit amendment rather
than mutating or replacing the JSON silently:

```python
amended_draft = frozen.amend(
    identifier="amendment-001",
    reason="The registered source exposes the lab identifier as institution.",
    units=revised_units,
    validation=revised_validation,
)

assert amended_draft.amendments[-1].parent_fingerprint == frozen.fingerprint
assert amended_draft.amendments[-1].changed_sections == ("units", "validation")

amended = amended_draft.freeze()
```

Each amendment records its identifier, reason, parent fingerprint, and changed top-level
sections. Unchanged sections cannot be presented as amendments. Amendments are forbidden
after materialization begins because that would allow observed evidence to rewrite the
declared design.

## Evidence-backed transitions

Normal users do not call `advance` for compilation and evaluation: the materializer,
compiler, runner, recovery runner, and reporter do it after constructing their artifact.
The resulting lifecycle is still inspectable:

```python
for event in reported.protocol.lifecycle:
    print(event.from_state, event.to_state, event.artifact_fingerprint)
```

Every event identifies the exact artifact that justified it. Replaying an evidence bundle
cross-checks those identities rather than trusting a status label.

## Serialization and schema versions

```python
from behavio.protocol import protocol_from_json

encoded = frozen.canonical_json()
decoded = protocol_from_json(encoded)

assert decoded == frozen
assert decoded.fingerprint == frozen.fingerprint
```

The current schema identifier is embedded in every protocol. Unknown schema versions are
rejected rather than guessed. Canonical JSON sorts keys, uses deterministic separators,
and rejects non-finite or executable values, making fingerprints stable across processes.

The current version is `behavio.study-protocol/2`. Two superseded names are still read:

| Superseded name | What it lacks | How it is read |
| --- | --- | --- |
| `behavio.study-protocol/1` | `ComparisonSpec.multiplicity` | the adjustment its runner applied unconditionally, `benjamini-hochberg`, is supplied |
| `unspool.study-protocol/1` | nothing; the package was renamed | read as-is |

A protocol reconstructed from a superseded payload **keeps its recorded schema name and
its fingerprint**. A frozen protocol is content-addressed and its own freeze event quotes
that address, so writing a member its author never declared into the payload would change
the identity of a declaration nobody amended, and would invalidate every lifecycle event
recorded against it. `to_dict` therefore omits `multiplicity` from a version 1 record, and
`StudyProtocol` refuses to construct a version 1 protocol carrying any adjustment other
than the one its era applied — so nothing distinguishable is ever dropped.

`amend` stamps the current schema version on the new draft. An amendment is a new
declaration with a new fingerprint, linked to its parent by
`amendments[-1].parent_fingerprint`, so it is recorded under the schema it is written in
rather than inheriting a name that would deny it members that schema predates.

## What the lifecycle does not prove

A `reported` state proves that the required sequence and artifact identities are
internally consistent. It does not prove that a scientific claim is true, that a source
dataset is unbiased, or that an estimand transports beyond its declared population.
Those remain matters for design, diagnostics, recovery, and bounded interpretation.
