# The literature-recipe standard

A literature recipe is a small, executable scientific argument. It is not a notebook that
ends when an optimizer returns parameters, and it is not a claim of reproduction merely
because it uses a published dataset.

Every Unspool recipe should let a reader answer: What entered the analysis? What was
predicted? What was learned only from training data? Which alternatives could have won?
What failed? Which claim remains justified?

## Required structure

### 1. Source and scope

Name the paper, public data release, retrieval route, license or access boundary, file
identity, and exact part of the published analysis being adapted. Distinguish:

- **exact reproduction:** the released inputs and specification reconstruct a reported
  result;
- **independent reproduction:** the same scientific quantity is recomputed from source
  trials;
- **literature-shaped analysis:** a new bounded question uses the paper's task or model
  structure; and
- **demonstration:** synthetic or small data illustrate the API without an empirical
  claim.

### 2. Runtime profile

Give an expected runtime class and required optional dependencies before the first code
block:

| Profile | Intended use |
| --- | --- |
| Quick | Documentation check; seconds to roughly two minutes |
| Standard | Complete worked analysis on a laptop |
| Benchmark | Repeated recovery or full public cohort; archived result supplied |

Hardware-sensitive times should be approximate. Never make a hidden cached artifact look
like a fresh fit.

### 3. Experimental unit and cohort

State the independent unit, source denominator, eligibility rule, exclusions, final
denominator, and whether the cohort rule uses outcomes. A row count is not a substitute
for an animal or participant count. Failed fits and unavailable sessions remain visible.

### 4. Task and observed event

Construct and validate `TaskSpec` before a model. Declare choice labels, omissions,
available actions, reward semantics, response-time origin and units, predictors, blocks,
episodes, and reset boundaries. Name `scored_columns` explicitly.

### 5. Estimand and deployment boundary

Write the scientific question in prospective language. Examples:

- predict a represented animal's next complete session;
- predict a new animal from the fitted population;
- predict joint choice and movement-onset latency in a later session;
- estimate a descriptive trajectory over an explicitly retrospective cohort.

Select folds that correspond to that use. If the analysis is retrospective, say so rather
than borrowing forecasting language.

### 6. Candidate set and selection

Explain why every candidate is present, especially simple ones. Models must score the same
observed event on the same test rows. State which tuning decisions—smoothness, state count,
priors, features—are selected inside training data and which are frozen a priori.

### 7. Diagnostics and uncertainty

Retain numerical audits, optimizer restarts or posterior diagnostics, predictive checks,
scored observations, and the uncertainty unit. Do not discard a candidate because its fit
failed without reporting that failure in the candidate denominator.

### 8. Recovery and sensitivity

Simulate the actual trial/session/subject geometry over scientifically relevant parameter
regimes. Parameter recovery supports parameter interpretation; model recovery supports
distinguishing the declared candidate families. Add exact-refit sensitivity for defensible
analysis choices and simulation-based calibration when validating a Bayesian inference
implementation.

### 9. Figures as evidence objects

Every empirical figure must identify its source artifact, unit, denominator, and supported
claim in the [figure-provenance register](../reference/figure-provenance.md). Conceptual
figures must say that their values are schematic. Prefer panels that expose observations,
predictions, calibration, fit diagnostics, and recovery rather than ornamental summaries.

### 10. Result and claim limits

Lead with the bounded result, including negative or unresolved evidence. End with a short
list of claims the workflow does not support. A recipe should make it difficult to confuse
prediction with mechanism, fit convergence with identifiability, or one public dataset
with population generality.

## Minimal executable skeleton

```python
from unspool import (
    ChoiceSpec,
    TaskSpec,
    compare_models,
    forward_session_splits,
)

task = TaskSpec(
    choice=ChoiceSpec(options=(0, 1)),
    predictors=("stimulus",),
)
validation = task.validate(study)
task.validate_model(baseline)
task.validate_model(candidate)
splits = forward_session_splits(study, min_train_sessions=3)

comparison = compare_models(
    {
        "baseline": baseline,
        "mechanistic": candidate,
    },
    study,
    splits,
)
```

The exact API around a recipe may be richer—frozen protocol, nested selection, recovery,
or evidence bundle—but it must retain this order: validate the task, declare the split,
fit all candidates only on training data, and compare matched held-out observations.

## Current recipe audit

| Recipe | Classification | Observed event | Deployment boundary | Main remaining boundary |
| --- | --- | --- | --- | --- |
| Cell 2025 flagship | Independent reproduction plus new prospective analysis | Binary choice | Later sessions in a completed historical cohort | Full neural and clustering claims excluded |
| IBL trajectories | Literature-shaped descriptive analysis | Session accuracy | Outcome-blind endpoint windows | Not an unbiased population learning curve |
| IBL prospective selection | Literature-shaped prospective analysis | Binary choice | Later session and held-out lab | Fixed empirical labs, plug-in population prediction |
| IBL choice/RT | Literature-shaped prospective analysis | Choice + movement-onset RT | Untouched later session | One selected animal; negative robustification result retained |
| Ashwood GLM-HMM | Literature-shaped prospective analysis | Binary choice | Training-only state selection, untouched later session | Near-tied selection and warning-level fit constrain state claims |
| Model recovery design | Synthetic benchmark | Binary choice | Future session under known generators | Evidence is conditional on tested regimes |

The current suite covers psychometric/history, longitudinal comparison, GLM-HMM, and
perceptual DDM workflows on public Cell or IBL data. A genuinely public bandit dataset and
task adapter are still required for the canonical RL recipe; the synthetic RL example is
an API and recovery demonstration, not a substitute empirical result.

## Contribution checklist

A proposed recipe is ready for review when:

- all retrieval inputs are public or a small non-redistributable fixture is clearly
  separated;
- a quick structural test runs in CI;
- expensive benchmark outputs are deterministic, schema-checked, and committed with their
  generating command;
- every displayed number can be traced to a data or benchmark artifact;
- code uses public Unspool interfaces rather than internal helpers; and
- the title and conclusion accurately identify reproduction, adaptation, or demonstration.

For package integration, continue to [Extend Unspool](../extensions.md). For choosing a
family, use [Choose a model by the claim](../model-choice-guide.md).
