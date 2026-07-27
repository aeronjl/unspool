# Parameter spaces and inference backends

A behavioural parameter has at least two identities. Its **natural coordinate** is the
quantity a scientist interprets and simulates, such as a learning rate in $(0,1)$. Its
**optimizer coordinate** is the unconstrained or numerically bounded value presented to a
fitting algorithm, such as the logit of that learning rate. Treating those names as
interchangeable makes recovery plots, priors, and backend comparisons surprisingly easy
to misread.

`ParameterSpace` is Unspool's backend-neutral declaration of that boundary. It fixes the
meaning of a model before SciPy, PyBADS, PyDDM, or a probabilistic-programming backend is
chosen.

## Declare one semantic object

```python
from unspool import (
    ParameterRole,
    ParameterSpace,
    ParameterSpec,
    ParameterTransform,
    PriorSpec,
)

space = ParameterSpace(
    (
        ParameterSpec(
            name="learning_rate",
            optimizer_name="learning_rate_logit",
            transform=ParameterTransform.BOUNDED_LOGIT,
            bounds=(0.0, 1.0),
            plausible_bounds=(0.05, 0.95),
            prior=PriorSpec.beta(2.0, 2.0),
        ),
        ParameterSpec(
            name="inverse_temperature",
            optimizer_name="inverse_temperature_log",
            transform=ParameterTransform.LOG,
            bounds=(0.0, None),
            plausible_bounds=(0.1, 10.0),
            prior=PriorSpec.half_normal(5.0),
        ),
        ParameterSpec(
            name="lapse_rate",
            role=ParameterRole.FIXED,
            bounds=(0.0, 0.2),
            fixed_value=0.01,
        ),
    )
)
```

The ordered declaration records:

- stable natural and optimizer names;
- identity, shifted-log, or bounded-logit transforms;
- hard scientific bounds on the natural coordinate;
- plausible natural bounds for search algorithms such as PyBADS;
- optional numerical safeguards on the optimizer coordinate;
- fixed or free status and the exact fixed value; and
- normalized normal, half-normal, beta, or uniform priors on the natural coordinate.

Natural hard bounds and optimizer safeguards are deliberately separate. A finite range
used to keep one numerical optimizer stable must not silently become a claim that the
scientific parameter cannot exist outside that range.

## Encode, decode, and audit coordinates

```python
natural = {
    "learning_rate": 0.25,
    "inverse_temperature": 4.0,
    "lapse_rate": 0.01,
}

optimizer_vector = space.encode(natural)
assert tuple(space.decode(optimizer_vector)) == space.natural_names

print(space.optimizer_names)
print(space.optimizer_bounds)
print(space.optimizer_plausible_bounds)
print(space.fingerprint)
```

Mappings must contain exactly the declared names. Values outside their domain, changed
fixed values, duplicate names, and incomplete priors fail before fitting. Returned vectors
and mappings are read-only, declaration order is stable, and the complete schema has a
canonical JSON representation and SHA-256 fingerprint.

`log_prior(natural)` evaluates priors on the scientific scale. If a backend represents a
density in transformed coordinates, it can add
`log_abs_det_inverse_jacobian(optimizer_vector)`. These are separate calls because a MAP
estimate depends on its parameterization: hiding the Jacobian choice inside a transform
would make two backends appear equivalent when they are not.

## Current model integration

`BinaryQLearning.parameter_space` is the first complete consumer. Its legacy
`parameter_names` remain the optimizer names so existing fits and recovery reports do not
break, while `parameter_space.natural_names` exposes the scientific learning rate,
inverse temperature, choice bias, and perseveration coordinates. The model's encoding,
decoding, and L-BFGS-B bounds now all come from that one declaration.

```python
fit = model.fit(study)
natural_estimates = model.parameter_space.decode(fit.estimates)
```

When a model exposes `ParameterSpaceProvider`, `export_fit()` adds the complete parameter
space and fingerprint to its portable diagnostics. Each fitted free parameter retains its
optimizer estimate alongside its natural name, natural estimate, and transform. Covariance
and standard errors remain explicitly on the optimizer coordinate until the result-
conversion contract lands; they are not relabelled as natural-scale uncertainty.

## Adapter boundary

| Backend lane | Parameter-space responsibility | Backend responsibility |
| --- | --- | --- |
| SciPy / PyBADS | names, transforms, hard and plausible bounds, fixed values, priors | search strategy, restart results, objective evaluations |
| PyDDM | stable task inputs and natural DDM parameter declarations | generalized DDM solver and likelihood |
| HSSM / PyMC | natural names, bounds, priors, fixed/free roles | hierarchical graph, sampling, posterior draws |
| ArviZ / xarray | coordinate labels and parameter-space fingerprint | labelled posterior, predictive, likelihood, and diagnostic groups |

The shared [deterministic inference contract](inference-backends.md), SciPy multistart
implementation, and optional PyBADS adapter consume this parameter space directly. The
ArviZ result adapter now retains labelled groups and the fingerprint when one exists. The
first [PyMC backend](pymc-backend.md) preserves the older hierarchical GLM's explicit
fixed-scale prior semantics, but that model does not yet expose `ParameterSpaceProvider`,
so its posterior result correctly leaves the parameter-space fingerprint unset rather than
manufacturing one. PyDDM and HSSM adapters remain deferred ecosystem work. Optional
dependencies do not become part of Unspool core.

- [PyBADS API](https://acerbilab.github.io/pybads/api/classes/bads.html)
- [PyDDM documentation](https://pyddm.readthedocs.io/)
- [HSSM documentation](https://lnccbrown.github.io/HSSM/)
- [ArviZ data schema](https://python.arviz.org/en/stable/schema/schema.html)

## API

::: unspool.parameters
    options:
      members:
        - ParameterRole
        - ParameterTransform
        - PriorFamily
        - PriorSpec
        - ParameterSpec
        - ParameterSpace
        - ParameterSpaceProvider
        - parameter_space_from_dict
        - parameter_space_from_json
