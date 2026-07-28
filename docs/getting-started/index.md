# Choose a workflow

Start with the claim and the deployment boundary, not the most elaborate model.

If this is your first visit, use the [installation guide](installation.md) and run the
[first prospective analysis](first-analysis.md). It produces a protected future-session
score, an animal-level figure, and fit audits from one complete script.

[Install Unspool](installation.md){ .md-button }
[Run the first analysis](first-analysis.md){ .md-button .md-button--primary }

<figure class="doc-figure">
  <img src="../assets/workflow-map.svg" alt="Four routes from a scientific question through a validation boundary to a bounded result: describing change, predicting later sessions, comparing explanations, and testing identifiability.">
  <figcaption><strong>Workflow map.</strong> The intended generalization target determines the split and evidence object before it determines the model family. This is a conceptual contract diagram.</figcaption>
</figure>

## Start from the work you need to do

| I need to… | Begin here | You should leave with… |
| --- | --- | --- |
| Bring a trial table into Unspool | [Longitudinal study contract](../data-contract.md) | explicit subject, session, trial, and chronology columns |
| Forecast genuinely later behaviour | [Prospective validation](../validation.md) | a leakage-safe split and held-out score |
| Compare scientific explanations | [Model-choice guide](../model-choice-guide.md) | a matched candidate set and declared alternatives |
| Tune or select models | [Nested comparison](../comparison.md) | selection contained inside each training boundary |
| Test whether the design can identify a claim | [Recovery design](../tutorials/model-recovery-design.md) | parameter and model-recovery evidence |
| Freeze an analysis before fitting | [Study protocols](../protocols/index.md) | a validated, auditable scientific declaration |
| Read IBL, NWB, or DANDI data | [Data interoperability](../interoperability.md) | canonical trials plus source provenance |
| Rework an existing analysis | [Migration guides](../migration-guides.md) | an explicit map from familiar tooling to Unspool |
| Extend the model catalogue | [Extension guide](../extensions.md) | a tested estimator or adapter boundary |

## The minimum analysis path

### 1. Preserve identity and chronology

Map the four required columns into a [`Study`](../data-contract.md). Source order and
additional columns are retained; Unspool does not infer chronology from filenames or row
position.

### 2. Declare observations independently of a model

A [`TaskSpec`](../task-contract.md) states choices, omissions, predictors, rewards,
response times, blocks, and episodes. This prevents model-specific preprocessing from
quietly changing the scientific denominator.

### 3. Protect the intended future

Choose a splitter for later sessions, unseen animals, held-out laboratories, or a combined
boundary. Learned clocks, scalers, landmarks, and hyperparameters must be fitted again
inside each training fold.

### 4. Compare matched candidates

Use [`compare_models`](../comparison.md) for a predeclared candidate set, or
`nested_select_model` when model structure or regularization is learned from data. Keep
simple observable competitors next to drift, latent-state, RL, or DDM accounts.

### 5. Bound the interpretation

Fit audits establish numerical credibility. Prospective scores establish out-of-sample
performance. Recovery establishes what this design can distinguish. None substitutes for
the others.

## Continue with public evidence

The [worked studies](../tutorials/index.md) carry published scientific questions through
cohort definition, modelling, validation, figures, and bounded interpretation. The common
[recipe standard](../tutorials/recipe-contract.md) labels exact reproduction, independent
reproduction, literature-shaped analysis, and synthetic demonstration so visual
similarity is never mistaken for numerical parity.

[Browse worked studies](../tutorials/index.md){ .md-button .md-button--primary }
[Read the API map](../reference/index.md){ .md-button }
