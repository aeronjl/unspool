# Fit artifacts and extension registries

`fit_model()` keeps the complete live, model-specific result. `export_fit()` adds a
portable common view that binds that result to its task semantics, complete input data,
package version, numerical diagnostics, and normalized audit.

```python
from unspool import export_fit, fit_artifact_from_json

fitted = fit_model(model, study, task=task)
artifact = export_fit(fitted, study)

text = artifact.canonical_json()
restored = fit_artifact_from_json(text)
assert restored.fingerprint == artifact.fingerprint
```

The `unspool.fit-artifact/1` schema contains:

- model name, configuration signature, and concrete result type;
- the choice, omission, availability, reward, response-time, predictor, block, and episode
  declarations from `TaskSpec`;
- trial and subject counts, source column names, and a SHA-256 identity of the complete
  study content without embedding the raw table;
- labelled estimates, standard errors, and covariance, using JSON `null` rather than
  non-standard `NaN` or infinity;
- raw common optimizer diagnostics and the normalized fit audit;
- the Unspool package version that produced the record.

Models implementing `ParameterSpaceProvider` additionally retain the complete
content-addressed [parameter-space declaration](parameter-spaces.md). Their parameter
records distinguish optimizer names and estimates from natural names and estimates rather
than implying that transformed values are directly interpretable. Covariance and standard
errors remain on the optimizer coordinate in this schema.

The artifact is deliberately non-executable. Reading it validates data, but does not
import a class, call a factory, or reconstruct a fitted Python object. Specialized live
results can contain richer state trajectories, restart arrays, and model-specific
uncertainty; these remain on `fitted.result`. The common artifact identifies the concrete
result type instead of pretending that the initial schema is a lossless posterior format.
Labelled posterior and predictive groups belong to the ArviZ/xarray interchange work in
0.23.

## External estimator packages

An extension can expose models without modifying Unspool core:

```python
from unspool import EstimatorRegistry
from my_package import make_model

registry = EstimatorRegistry()
registry.add(
    "my-behaviour-model",
    make_model,
    provider="my-package",
    version="1.2.0",
)

model = registry.create("my-behaviour-model", {"n_states": 3})
print(registry.manifest())
```

Each factory receives a read-only configuration mapping. Its result is checked against
the public `BehaviourEstimator` contract, and its `model_name` must match the registered
name. Duplicate registration and silent identity drift are errors.

Registries are deliberately instance-scoped rather than process-global. A workflow must
choose the exact implementations in scope, and `manifest()` records their provider and
version without serializing arbitrary callables. This prevents an import in one notebook
cell from invisibly changing a later scientific run.
