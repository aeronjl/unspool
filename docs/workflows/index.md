# Analysis workflows

Behavio workflows are organized around the evidence a claim needs, not around a sequence
of model classes.

| Scientific task | Primary workflow | Evidence object |
| --- | --- | --- |
| Forecast later sessions | [Prospective validation](../validation.md) | fold-level predictions and scores |
| Compare candidate explanations | [Model comparison](../comparison.md) | paired, unit-balanced differences |
| Tune a candidate without leakage | [Nested selection](../comparison.md#training-only-nested-selection) | untouched outer-fold performance |
| Test identifiability | [Parameter and model recovery](../model-recovery.md) | design-specific recovery report |
| Freeze a complete study | [Study protocols](../protocols/index.md) | compiled declaration and audit |
| Carry claims into an archive | [Evidence bundles](../protocols/evidence-bundles.md) | content-addressed evidence record |

Numerical convergence, future prediction, and recovery answer different questions. A
complete workflow keeps all three visible and ends with the narrowest interpretation they
jointly support.

[Run the first workflow](../getting-started/first-analysis.md){ .md-button .md-button--primary }
[Choose a model](../model-choice-guide.md){ .md-button }
