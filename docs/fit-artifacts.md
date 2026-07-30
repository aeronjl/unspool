# Fit artifacts and extension registries

`fit_model()` keeps the complete live, model-specific result. `export_fit()` adds a
portable common view that binds that result to its task semantics, complete input data,
package version, numerical diagnostics, and normalized audit.

```python
from behavio import export_fit
from behavio.report import fit_artifact_from_json

fitted = fit_model(model, study, task=task)
artifact = export_fit(fitted, study)

text = artifact.canonical_json()
restored = fit_artifact_from_json(text)
assert restored.fingerprint == artifact.fingerprint
```

The `behavio.fit-artifact/1` schema contains:

- model name, configuration signature, and concrete result type;
- the choice, omission, availability, reward, response-time, predictor, block, and episode
  declarations from `TaskSpec`;
- trial and subject counts, source column names, and a SHA-256 identity of the complete
  study content without embedding the raw table;
- labelled estimates, standard errors, and covariance, using JSON `null` rather than
  non-standard `NaN` or infinity;
- raw common optimizer diagnostics and the normalized fit audit;
- the Behavio package version that produced the record.

Models implementing `ParameterSpaceProvider` additionally retain the complete
content-addressed [parameter-space declaration](parameter-spaces.md). Their parameter
records distinguish optimizer names and estimates from natural names and estimates rather
than implying that transformed values are directly interpretable. Covariance and standard
errors remain on the optimizer coordinate in this schema.

When a live result exposes an `OptimizationRun`, the artifact also retains its complete
[multistart evidence](inference-backends.md): backend configuration, deterministic starts,
all attempted estimates and messages, evaluation counts, and the selected index. Failed
attempts are not reduced to a convergence rate.

The artifact is deliberately non-executable. Reading it validates data, but does not
import a class, call a factory, or reconstruct a fitted Python object. Specialized live
results can contain richer state trajectories, restart arrays, and model-specific
uncertainty; these remain on `fitted.result`. The common artifact identifies the concrete
result type instead of pretending that the initial schema is a lossless posterior format.
Labelled posterior and predictive groups use the separate
[posterior-result contract](posterior-results.md) and ArviZ/xarray interchange layer.

## External estimator packages

An extension can expose models without modifying Behavio core:

```python
from behavio import builtin_estimator_registry
from my_package import MyModel

registry = builtin_estimator_registry()
registry.add(
    "my_package.MyModel",
    MyModel,
    provider="my-package",
    version="1.2.0",
    produces=MyModel,
    model_name="my-behaviour-model",
)

model = registry.create("my_package.MyModel", {"n_states": 3})
print(registry.manifest())
```

Each factory receives the declared settings as keyword arguments. Its result is checked
against the declared `produces` class and, when `model_name` is declared, against that
stable name too. Duplicate registration and silent identity drift are errors.

The registration name is the string a frozen protocol declares as its `implementation`,
and this registry is the allowlist a protocol run resolves through — the same one
`builtin_estimator_registry()` fills with the package's own models. Registries are
deliberately instance-scoped rather than process-global. A workflow must choose the exact
implementations in scope, and `manifest()` records their provider and version without
serializing arbitrary callables. This prevents an import in one notebook cell from
invisibly changing a later scientific run. See
[local registration](extensions.md#local-registration) for combinators and for what
declaring `produces` buys at declaration-verification time.
