# Python API map

The package is one small top-level namespace plus a named area per scientific
responsibility, and this reference mirrors that tree. Start with a guide or a worked example
when deciding what to do; use these pages when you need exact signatures, return types, and
public attributes.

## What lives at the top level

`behavio` itself is a curated golden path of about a hundred names: the ones a first
analysis cannot be written without, plus the surface the package promises to third parties.
It is deliberately **not** a re-export of everything — see
[API reorganisation](../api-reorganisation.md) for the boundary and the argument behind it.

```python
from behavio import Study, TaskSpec, ChoiceSpec, compare_models, forward_session_splits
```

Everything else is public, documented, and reached at `behavio.<area>.<name>`.

## The areas

| Area | Contains | Reference | Guide |
| --- | --- | --- | --- |
| `behavio.trials` | `Study`, the trial-level data contract | [Trials and tasks](study-and-task.md) | [Data contract](../data-contract.md) |
| `behavio.task` | task observation contracts, response times | [Trials and tasks](study-and-task.md) | [Task contract](../task-contract.md) |
| `behavio.observed` | pose, ethograms, covariates, device clocks, interval policies, trialization | [Observed behaviour](observed-behaviour.md) | [Observed behaviour](../observed-behaviour.md) |
| `behavio.time` | learning-time clocks, landmark clocks, fold-fitted transforms | [Time and design](time-and-design.md) | [Clocks and transforms](../clocks-and-transforms.md) |
| `behavio.design` | fixed design-matrix terms and the formula notation | [Time and design](time-and-design.md) | [Design matrices](../design-matrices.md) |
| `behavio.models` | the model catalogue and the shared estimator contract | [Choice](choice-models.md), [latent and RL](latent-and-rl-models.md), [drift diffusion](drift-diffusion-models.md) | [Model cards](../model-cards.md) |
| `behavio.compose` | the `smooth` / `hierarchical` / `mix` combinators | [Choice models](choice-models.md) | [Composing models](../composing-models.md) |
| `behavio.inference` | parameter spaces, priors, optimizer backends | [Inference and registry](inference.md) | [Inference backends](../inference-backends.md) |
| `behavio.posterior` | labelled draws, convergence, predictive checks, PSIS-LOO, SBC, sensitivity, reliability | [Posterior](posterior.md), [diagnostics](diagnostics.md) | [Posterior results](../posterior-results.md) |
| `behavio.evaluate` | validation splits and the fold-evaluation loop | [Evaluate and compare](validation-and-comparison.md) | [Prospective validation](../validation.md) |
| `behavio.compare` | paired model comparison, nested selection, parameter-trajectory shapes | [Evaluate and compare](validation-and-comparison.md) | [Model comparison](../comparison.md) |
| `behavio.recovery` | parameter recovery and model recovery | [Recovery](recovery.md) | [Recovery design](../model-recovery.md) |
| `behavio.protocol` | the frozen declaration, its compiler, its runner, exact-design recovery | [Schema](protocol.md), [execution](execution.md) | [Protocol authoring](../protocols/index.md) |
| `behavio.report` | bounded reports, evidence bundles, fit artifacts | [Report](evidence-bundles.md) | [Evidence bundles](../protocols/evidence-bundles.md) |
| `behavio.adapters` | CSV/TSV/Parquet tables, NWB, DANDI, IBL ONE, adapter and estimator conformance, trial sequences, continuous-outcome predictions | [Data adapters](data-adapters.md) | [Interoperability](../interoperability.md) |
| `behavio.foreign` | estimators backed by third-party model packages, each behind its own extra | [Wrapped models](foreign-models.md) | [Compatibility and licences](../foreign-models.md) |
| `behavio.contracts` | every protocol a downstream package implements, at one address | [Extension contracts](contracts.md) | [Extend Behavio](../extensions.md) |
| `behavio.plot` | SBC bands, Pareto-`k`, ELPD differences, predictive checks, recovery, calibration, convergence | [Plotting](plots.md) | [Figure standard](figure-standard.md) |
| `behavio.diagnostics` | the fit audit, `audit_fit` | [Diagnostics](diagnostics.md) | [Fit diagnostics](../diagnostics.md) |
| `behavio.registry` | named estimator registration for protocols and the CLI | [Inference and registry](inference.md) | [Extend Behavio](../extensions.md) |

Optional data-source, optimization, and probabilistic dependencies are required only when
their corresponding APIs are used.
