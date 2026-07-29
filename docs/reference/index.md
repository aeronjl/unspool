# Python API map

The reference is organized by scientific responsibility rather than as one alphabetical
dump of the root namespace. Start with a guide or worked example when deciding what to do;
use these pages when you need exact signatures, return types, and public attributes.

## Core analysis

| API area | Contains | Guide |
| --- | --- | --- |
| [Studies and tasks](study-and-task.md) | `Study`, `ChoiceSpec`, `TaskSpec`, response-time declarations | [Data contract](../data-contract.md) |
| [Clocks, transforms, and design](time-and-design.md) | explicit time coordinates, fold-fitted transforms, labelled design matrices | [Clocks and transforms](../clocks-and-transforms.md) |
| [Validation and comparison](validation-and-comparison.md) | session-, subject-, and lab-aware splitters, evaluation, paired model comparison | [Prospective validation](../validation.md) |

## Model families

| API area | Contains | Guide |
| --- | --- | --- |
| [Observable choice models](choice-models.md) | baselines, static and smooth GLMs, partial pooling, multinomial choice | [Model cards](../model-cards.md) |
| [Latent-state and reinforcement-learning models](latent-and-rl-models.md) | GLM-HMMs, Q-learning, composable RL agents | [Model-choice guide](../model-choice-guide.md) |
| [Drift-diffusion models](drift-diffusion-models.md) | static, smooth, hierarchical, and contaminant DDMs | [Drift diffusion](../drift-diffusion.md) |

## Evidence and computation

| API area | Contains | Guide |
| --- | --- | --- |
| [Recovery](recovery.md) | parameter, model, exact-protocol, and trajectory-shape recovery | [Recovery design](../model-recovery.md) |
| [Diagnostics and sensitivity](diagnostics.md) | fit audits, posterior checks, SBC, sensitivity, reliability, PSIS-LOO | [Fit diagnostics](../diagnostics.md) |
| [Inference and parameters](inference.md) | parameter spaces, deterministic optimizers, estimator registration | [Inference backends](../inference-backends.md) |
| [Posterior results](posterior.md) | labelled draws, PyMC backend, posterior interchange | [Posterior results](../posterior-results.md) |

## Protocols and interoperability

| API area | Contains | Guide |
| --- | --- | --- |
| [Protocol schema](protocol.md) | immutable scientific declarations and lifecycle | [Protocol authoring](../protocols/index.md) |
| [Protocol execution](execution.md) | compilation, materialization, running, exact-design recovery | [Compile and audit](../protocols/auditing.md) |
| [Reporting and evidence](evidence-bundles.md) | bounded reports and content-addressed evidence bundles | [Evidence bundles](../protocols/evidence-bundles.md) |
| [Data adapters](data-adapters.md) | NWB, DANDI, IBL ONE, and fit-artifact interchange | [Interoperability](../interoperability.md) |

The stable public import surface remains `behavio`. Optional data-source, optimization,
and probabilistic dependencies are required only when their corresponding APIs are used.
