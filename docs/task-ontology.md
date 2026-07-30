# The task ontology

`TaskSpec` says what a column of *this* table means. The task ontology says what the
*experiment* was — which reusable family it belongs to, which variables it turns on, and
what its choices are called in a vocabulary shared with every other dataset in that family.

Two studies of the same experiment, one coding choice `-1`/`+1` and one coding it
`"left port"`/`"right port"`, produce two unrelated `TaskSpec`s and nothing connects them.
The ontology is the layer that connects them, and it does so by *producing* the
specification rather than sitting beside it.

!!! note "Optional by construction"

    Nothing on this page is required. `TaskSpec(choice=ChoiceSpec(options=(0, 1)))` is a
    complete task contract, and fitting a GLM to your own CSV never involves naming a
    family. Reach for the ontology when you need one analysis to span sources that were
    curated separately.

## Two layers, one direction

| | Structural layer | Named layer |
| --- | --- | --- |
| Where | `behavio.task.spec`, `behavio.task.response_times` | `behavio.task.vocabulary`, `behavio.task.ontology` |
| Says | which column holds the choice, how it is coded | what the choice *means*, across datasets |
| Example | `ChoiceSpec(options=("left port", "right port"))` | `ChoiceTerm.LEFT`, `ChoiceTerm.RIGHT` |
| Produced by | you, or by the named layer | a curator |

The arrow only runs one way. A declaration produces a `ChoiceSpec`, a `TaskSpec`, and a
tuple of `ObservationSpec`; nothing reads a specification back into a declaration. That is
what stops the two from becoming rival vocabularies, and it is why a term such as
`ObservationDataType.COUNT` is defined exactly once — in `behavio.task.vocabulary` — and
imported by `behavio.protocol.schema` rather than restated there.

## The controlled vocabulary

Every closed set is a `StrEnum`, and `CONTROLLED_VOCABULARIES` is the machine-readable
index of all of them:

```python
from behavio.task import CONTROLLED_VOCABULARIES

CONTROLLED_VOCABULARIES["choice_terms"]
# ('left', 'right', 'go', 'withhold', 'no-response', 'unknown')
```

Twelve sets are published: `modalities`, `species`, `choice_types`, `response_modalities`,
`evidence_types`, `feedback_types`, `curation_statuses`, `choice_terms`, `feedback_terms`,
`stimulus_sides`, `observation_roles` and `observation_data_types`. Adding a term means
adding an enum member; there is no second list.

Two members carry meaning that a bare label cannot:

- **`ChoiceTerm.NO_RESPONSE`** is a *retained* trial on which nothing was done. It becomes a
  `ChoiceSpec.omission_values` entry, so those trials stay in the denominator.
  `ChoiceTerm.WITHHOLD` is deliberately *not* an omission — withholding is the correct
  action in a go/no-go task and gets a coordinate of its own.
- **`ChoiceTerm.UNKNOWN`** records that a curator could not determine the choice. It is
  never a valid observation: a conversion that meets it fails, because a model fitted to
  that category would be fitting the curation process.

## Declaring a task family

A family is versioned and content-addressed exactly as a
[study protocol](protocols/index.md) is: `fingerprint` is the SHA-256 of its canonical
JSON, so a family that gained a variable is a different family and says so.

```python
from behavio.task import (
    CanonicalVariable,
    ChoiceTerm,
    ChoiceType,
    CurationStatus,
    Modality,
    ObservationDataType,
    ObservationRole,
    Provenance,
    Reference,
    TaskFamily,
)

family = TaskFamily(
    identifier="family.visual-2afc-contrast",
    name="Visual contrast discrimination",
    description="Subjects use contrast or its lateralisation to choose between alternatives.",
    modalities=(Modality.VISUAL,),
    canonical_variables=(
        CanonicalVariable(
            name="contrast",
            column="contrast",
            role=ObservationRole.PREDICTOR,
            data_type=ObservationDataType.CONTINUOUS,
            unit="percent contrast",
        ),
        CanonicalVariable(name="stimulus side"),
    ),
    choice_types=(ChoiceType.TWO_ALTERNATIVE,),
    choice_terms=(ChoiceTerm.LEFT, ChoiceTerm.RIGHT),
    curation_status=CurationStatus.ADAPTER_READY,
    references=(Reference(identifier="ref.ibl-2021", citation="IBL et al., eLife, 2021."),),
    provenance=Provenance(created="2026-04-24", updated="2026-07-30"),
)

family.fingerprint[:12]
task = family.task_spec()  # an ordinary TaskSpec
task.choice.options  # ('left', 'right')
task.predictors  # ('contrast',)
```

A canonical variable is either a **name** or a **binding**. `CanonicalVariable("stimulus
side")` is a curated claim that two experiments manipulate the same thing; that is real, and
it is not enough to fit anything. Adding `column`, `role` and `data_type` binds the name to
a measured column, which is what lets `task_spec()` build predictors and what lets
`observations_from_task_protocol()` build a column contract. An unbound variable serializes
as a plain string, so a record that carries only names round-trips unchanged.

## Declaring a task protocol

A `TaskProtocol` is one concrete realisation of a family. It is not a `StudyProtocol`: that
declares an *analysis* — cohort, estimands, candidates, comparison — whereas this declares
the *experiment the data came out of*.

The interesting field is `choice`:

```python
from behavio.task import ChoiceDeclaration, ChoiceTerm, ChoiceType, ResponseModality

ChoiceDeclaration(
    choice_type=ChoiceType.TWO_ALTERNATIVE,
    alternatives=("left port", "right port"),  # what the apparatus calls them
    terms=(ChoiceTerm.LEFT, ChoiceTerm.RIGHT),  # what they mean
    response_modalities=(ResponseModality.NOSE_POKE,),
    action_mapping="the rewarded port is the one on the stimulus side",
).choice_spec()
# ChoiceSpec(options=('left', 'right'), column='choice', omission_values=(), ...)
```

`alternatives` are the operational labels, because that is what the paper says.
`terms` is the parallel tuple naming what each label *means*, and it is the piece without
which no two protocols can be compared. A label that already spells a term needs no
declaration; anything else does, and until it is supplied `choice_spec()` refuses:

```text
OntologyError: cannot derive a ChoiceSpec: no controlled term is declared for
['direction A', 'direction B']. Add a `terms` entry naming what each alternative means.
```

That failure is the point. No rule recovers "left" from "direction A", and guessing would
put a fabricated meaning into every downstream comparison.

## Response-time origin

`ResponseTimeSpec` carries an `origin`: the event the clock started at, as the source
describes it.

```python
from behavio.task import ResponseTimeSpec

ResponseTimeSpec(origin="response_times - stimOn_times")
ResponseTimeSpec(column="rt_ms", unit="milliseconds", origin="seconds after go cue")
```

This is the one fact a trials table cannot carry and a downstream reader cannot recover. As
[the interoperability guide](interoperability.md) notes, an NWB `response_time` may be a
decision duration, a movement latency, or an absolute event timestamp, and no NWB field
distinguishes them — so two files that look identical can mean different things and a
pooled analysis across them is not comparing like with like.

It is free text on purpose. The event is a fact about one apparatus, not a member of any
closed set, and a controlled vocabulary that forced `"stimulus onset"` onto a
go-cue-locked measurement would be worse than silence. A hand-written `ResponseTimeSpec`
may leave it `None`, which records that nobody said; a declaration made through
`TaskProtocol` **must** supply it.

`RewardSpec` gained `units` for the same reason, and Behavio deliberately does not convert
between them: a microlitre and a millilitre of water are not interchangeable rewards, and
silently rescaling one into the other would make a pooled analysis look valid when it is
not.

## Canonical trials to a `Study`

A canonical trial is the interchange row: one trial, every term drawn from the vocabulary.
Turning a table of them into a `Study` looks like a rename and is not.

```python
from behavio.adapters import session_order_from_column, study_from_canonical_trials

study = study_from_canonical_trials(
    records,
    session_order=session_order_from_column("session_date"),
    protocol=protocol,
)
```

Four identity problems get resolved explicitly:

| Canonical record | `Study` | Resolution |
| --- | --- | --- |
| `subject_id` (optional) | `subject` (required) | refused if absent, never invented |
| `session_id` | `session` | renamed |
| `trial_index` | `trial` | renamed; must be non-negative |
| *nothing* | `session_order` | you name a derivation |

**Session chronology does not exist in a canonical record** — not as a date, not as an
ordinal, not anywhere. Behavio refuses to infer it, so the caller names one of the same
three rules the [table reader](interoperability.md) already uses:
`session_order_from_column(key)` (which looks the key up in `task_variables`, then
`source`), `session_order_from_explicit([...])`, or `session_order_from_appearance()`. The
rule that was applied is written to a `source_session_order_rule` column on every trial, so
a study whose chronology was derived can never be mistaken for one whose source recorded
it, and the adapter reports `SessionOrderPolicy.DERIVED`.

Supplying `protocol=` is what turns the conversion from a rename into a check. It verifies
that every row's `protocol_id`, choice term, stimulus modality, reward units and
response-time origin match what the protocol declares, and it refuses a `choice` of
`unknown`. Unit fields must also be constant across the conversion: two harmonisers writing
`uL` and `mL` for the same quantity is not one column, and the conversion says so rather
than pooling them.

`task_variables` are flattened into ordinary study columns when every row declares the same
keys. `source` is dropped: it is per-row provenance of arbitrary shape, which a trial table
cannot hold.

## One column contract, not two

`ObservationSpec` in a study protocol declares a column's role, measurement type and
allowed values. So does a bound canonical variable, and a `ChoiceSpec` declares the option
set a third time. `observations_from_task_protocol()` collapses that:

```python
from behavio.protocol.schema import observations_from_task_protocol

observations = observations_from_task_protocol(protocol)
```

The choice becomes a categorical outcome whose `allowed_values` are exactly the terms the
`ChoiceSpec` uses, omissions included; a response time becomes a continuous outcome
carrying the declared unit; a reward becomes a continuous auxiliary observation carrying
the declared reward units; and each bound variable becomes an observation with the role and
type its binding declared. The protocol compiler enforces that contract against materialized
data, so the option set a model validates against and the option set a protocol audits
against can no longer disagree.

## Emitting the schema

A vocabulary that can only be *accepted* is not owned. Behavio emits JSON Schema for all
three record types, with the enumerations read off the vocabulary rather than restated:

```python
from behavio.task import ONTOLOGY_JSON_SCHEMAS

ONTOLOGY_JSON_SCHEMAS["canonical_trial"]()
ONTOLOGY_JSON_SCHEMAS["task_family"]()
ONTOLOGY_JSON_SCHEMAS["protocol"]()
```

The documents follow the conventions of the generated schemas they are meant to replace —
`additionalProperties: false`, a title on every property, `anyOf` with an explicit null for
every optional scalar, `$defs` under their published names — so a consumer can drop them in.
Three differences are deliberate:

1. Fields with a controlled vocabulary are emitted as `enum` rather than as bare `string`.
   The vocabulary existed before; it was enforced by a separate validator the schema knew
   nothing about, so a record could be schema-valid and still name a modality that does not
   exist.
2. `canonical_variables` accepts an object as well as a string, which is how a variable is
   bound to a column.
3. A task protocol may declare `reward`, `response_time`, `variables` and `block_column`,
   and a choice may declare `terms` and `column` — the members that let a declaration
   produce a `TaskSpec`.

## Versioning

`ONTOLOGY_SCHEMA_VERSION` is `"0.2.0"`. Records written at `"0.1.0"` predate every member
Behavio added, so one that carries a `choice_terms`, a bound variable, or a protocol-level
`reward` is refused rather than silently upgraded:

```text
OntologyError: task family schema_version '0.1.0' predates ['choice_terms'];
record it under '0.2.0' to declare them
```

This is the discipline `StudyProtocol` already applies to its own superseded versions. A
declaration is content-addressed, so writing a member its author never declared would change
the identity of something nobody amended.
