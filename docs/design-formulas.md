# Design formulas

`DesignSpec` is composable but verbose. A formula is the notation for writing one down:

```python
from behavio import DesignSpec

design = DesignSpec.from_formula("choice ~ stimulus * phase + lag(choice, 1)")
```

That is the same object as the hand-built equivalent, term for term:

```python
from behavio import DesignSpec, HistoryTerm, InteractionTerm, NumericTerm

stimulus = NumericTerm("stimulus")
phase = NumericTerm("phase")
design = DesignSpec(
    terms=(
        stimulus,
        phase,
        InteractionTerm(stimulus, phase),
        HistoryTerm("choice", lags=(1,)),
    )
)
```

A formula is notation and nothing else. Every accepted form desugars onto a term that
[`behavio.design`](design-matrices.md) already has; the parser has no algebra of its own,
builds nothing `DesignSpec` cannot build, and cannot express a transformation the fixed
design terms would refuse. If a formula parses, the design it names is a design you could
have written by hand.

## Grammar

```text
formula     := [ response "~" ] sum
response    := column | column "|" column        # response_time | choice
sum         := ["+" | "-"] product (("+" | "-") product)*
product     := interaction ("*" interaction)*
interaction := atom (":" atom)*
atom        := "1" | "0" | column | call | "(" sum ")" | "(" sum "|" column ")"
call        := "numeric" "(" column [, kwargs] ")"
             | "C"       "(" column [, levels] [, kwargs] ")"
             | "scale"   "(" column [, kwargs] ")"
             | "lag"     "(" column [, lags] [, kwargs] ")"
             | "kernel"  "(" column [, weights] [, kwargs] ")"
column      := name | "`" any-characters "`"
```

Operators bind in the order `+`/`-` (loosest), then `*`, then `:` (tightest), matching the
convention every other formula language uses. `a * b:c` therefore expands over `{a, b:c}`,
not over `{a, b, c}`.

## Terms

| Formula | Design term |
| --- | --- |
| `stimulus` | `NumericTerm("stimulus")` |
| `numeric(stimulus, center=0.5, scale=2.0, name='z')` | `NumericTerm` with a declared affine transformation |
| `C(condition, ['train', 'probe'], reference='probe')` | `CategoricalTerm` with a fixed level set |
| `C(condition)` | `CategoricalTerm` whose levels are read off a **training** study |
| `scale(stimulus)` | `StandardizeTerm("stimulus").fit(training_study)` |
| `lag(choice, 1, 2)` | `HistoryTerm("choice", lags=(1, 2), coding="effect")` |
| `kernel(choice, [0.6, 0.3, 0.1])` | `HistoryKernelTerm` with fixed weights, effect coded |
| `a:b` | `InteractionTerm(a, b)` |
| `a * b` | `a + b + a:b` |
| `1` / `0` / `- 1` | the intercept, kept or suppressed |

Every call takes the keyword arguments its design term takes: `name` throughout,
`drop_reference` on `C()`, `ddof` on `scale()`, and `reset_by`, `coding` and `fill_value`
on `lag()` and `kernel()`. A column whose name is not a plain identifier is written
between backquotes: ``` `reaction time (s)` ```.

### History coding

`lag()` and `kernel()` default to `coding="effect"`, which is the one place the notation
does not simply inherit its design term's default. `HistoryTerm` itself defaults to
`coding="identity"`, but `BernoulliHistoryGLM(choice_lags=1)` has always built the
effect-coded -1/+1 column and has always called it `choice_lag_1`. A formula is what
users migrate to from that shorthand, so `lag(choice, 1)` builds the same column:

```python
DesignSpec.from_formula("choice ~ stimulus + lag(choice, 1)")
# the design BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1) builds
```

For the literal lagged value of a column that is not zero/one, ask for it:

```python
DesignSpec.from_formula("choice ~ lag(reward_magnitude, 1, coding='identity')")
```

The default cannot go wrong quietly. Effect coding rejects a column that is not zero/one,
and a formula that reached it by default says so and names the fix:

```text
DesignValidationError: effect-coded history requires zero/one values, and
'reward_magnitude' has others. A formula's lag() and kernel() default to coding='effect',
the -1/+1 coding that the choice_lags= shorthand has always built for a binary history
column. Write lag(reward_magnitude, 1, coding='identity') to lag 'reward_magnitude'
literally instead.
```

`describe(study)` prints the coding beside every history column, so two fits that both
report `choice_lag_1` can be told apart.

### Responses

The left-hand side names what a model will score, using the same two roles and the same
two names as [`TaskSpec`](task-contract.md):

```text
choice ~ stimulus                      # one categorical outcome
response_time | choice ~ stimulus      # a joint response time and choice
```

`Formula.response.outcome_columns` returns them in `TaskSpec.outcome_columns` order,
`(choice, response_time)`. The response is checked against the study but does not enter
the design matrix; it is the declaration of what the design is *for*.

The response is optional, so a formula can name a design alone:

```python
DesignSpec.from_formula("1 + stimulus + lag(choice, 1)")
```

## Learned coordinates stay inside the training fold

Two atoms estimate their coordinate from rows rather than declaring it: `scale(x)`, which
needs a centre and a scale, and `C(x)` without a level set, which needs a category
vocabulary. Both are exactly the kind of preprocessing that leaks a held-out fold into a
fit if it is estimated on the whole study.

The formula path makes that impossible to write by accident. A formula that contains one
cannot be built at all without naming a training study:

```python
DesignSpec.from_formula("choice ~ scale(stimulus)")
```

```text
FormulaError: scale(stimulus) estimates its coordinate from study rows, so it cannot be
built without naming a training fold; call Formula.fit(training_study) instead, or declare
the estimate in the formula at position 9
  choice ~ scale(stimulus)
           ^
```

The estimate is reached only through a call that says whose rows it came from:

```python
from behavio.formula import Formula

design = Formula.parse("choice ~ scale(stimulus)").fit(training_study)
```

`fit()` returns an ordinary fixed `DesignSpec`: the training centre and scale, or the
training level set, are frozen into it, so building the same design on the test fold
produces the same coordinate rather than a re-estimated one. This is the boundary
`StandardizeTerm.fit()` already draws and
[`fit_transform_split()`](clocks-and-transforms.md) enforces for clocks; the formula
notation inherits it instead of routing around it.

An inferred level set is *ordered*, not taken in row order, so the reference level of
`C(condition)` cannot change because the rows arrived in a different sequence. If the
observed categories cannot be ordered, the formula says so and asks you to declare them.

## Group terms are parsed, not honoured

`(1|subject)` and `(stimulus|subject)` parse into a structured `GroupTerm`:

```python
from behavio.formula import Formula

formula = Formula.parse("choice ~ stimulus + (stimulus | subject)")
group = formula.groups[0]

group.grouping  # 'subject'
group.to_design().feature_names  # ('intercept', 'stimulus')
```

`model_from_formula` honours it, by handing exactly those two things --- the grouping
column and the within-group design --- to
[`hierarchical()`](composing-models.md):

```python
from behavio import BernoulliHistoryGLM, model_from_formula

model = model_from_formula(
    "choice ~ stimulus + (1 + stimulus | subject)",
    BernoulliHistoryGLM(),
    scale=0.4,
)
model.varying_parameters  # ('intercept', 'stimulus')
```

A `DesignSpec` is still a single fixed matrix with no varying-effect representation, so
building one from a formula that declares a group term routes you to the combinator rather
than quietly dropping the declaration:

```text
FormulaError: the group term (1 | subject) declares an effect that varies by 'subject',
and a DesignSpec is one fixed matrix with nowhere to put it. Call
behavio.compose.model_from_formula(formula, model) to build the hierarchical model this
formula describes, or Formula.fixed_design() for the fixed part alone at position 20
  choice ~ stimulus + (1|subject)
                      ^
```

`Formula.fixed_design()` is the deliberate way to keep only the fixed half, and is what
`model_from_formula` uses once it has read `Formula.groups`.

## Round-tripping

Canonical rendering expands `*`, makes the intercept explicit, and drops duplicated terms:

```python
Formula.parse("choice ~ stimulus * phase").render()
# 'choice ~ 1 + stimulus + phase + stimulus:phase'
```

Rendering is a fixed point: parsing the canonical string returns an equal formula that
renders identically. The reverse direction also holds. A design built by hand describes
itself as a formula, and that description parses back into the same design:

```python
design = DesignSpec(terms=(NumericTerm("stimulus", scale=2.0), HistoryTerm("choice")))

design.describe()
# "1 + numeric(stimulus, scale=2.0) + lag(choice, 1, coding='identity')"
assert DesignSpec.from_formula(design.describe()) == design
```

`describe()` complements `signature`. `signature` is a complete fingerprint that never
fails but cannot be read back; `describe()` is a readable string that parses, and raises
if a third-party term has no formula spelling. `str(design)` prints the description and
falls back to a term's own signature where none exists, so it is always safe to print.

## Checking a formula against a study

Column references are checked before a fit rather than deep inside one:

```python
Formula.parse("choice ~ stimulis").validate(study)
```

```text
FormulaError: study has no column 'stimulis'; did you mean 'stimulus'? at position 9
  choice ~ stimulis
           ^
```

`validate()` checks the response columns, every term's source columns, a `lag()` reset
boundary, and a group term's grouping column. `Formula.fit()` calls it first, so a typo
never reaches an estimator.

Every error -- syntax, checking, or refusal -- carries `position` and `source` and prints
the offending character under a caret.

## What a formula cannot say

The notation covers the term algebra and stops there. It has no spelling for a
transformation `behavio.design` does not implement, and adding one would mean adding a
design term first. Known limits:

- **Varying effects.** `(1|subject)` parses but cannot be built, as above.
- **Nested and crossed grouping.** `(1|subject:session)` is not accepted; a `GroupTerm`
  groups by one column.
- **Splines and basis expansions.** Session-varying coefficients are a model-fitting
  concern in Behavio, not a design one, and belong to the smooth model families.
- **Third-party terms.** A term from outside the package can be put in a `DesignSpec` by
  hand, but has no formula spelling and so cannot be `describe()`d.

Models still take their own design arguments. `DesignSpec.from_formula()` builds the
design; it does not change how any model is configured.
