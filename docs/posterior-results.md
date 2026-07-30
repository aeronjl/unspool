# Labelled posterior results

Behavio keeps probabilistic inference outputs behind a small, backend-neutral contract.
The contract is intentionally about **scientific evidence**, not a particular sampler:

- posterior draws in constrained, natural parameter coordinates;
- posterior-predictive draws on the observations used for fitting;
- out-of-sample predictions on separately labelled trials;
- pointwise log likelihood for established Bayesian diagnostics;
- per-draw sampler diagnostics;
- observed and constant data needed to interpret the fit; and
- model, backend, and parameter-space provenance.

This follows the [ArviZ schema](https://python.arviz.org/en/stable/schema/schema.html), whose
purpose is labelled, reproducible interoperability across inference backends. Behavio uses
the same conceptual groups and the reserved `chain` and `draw` sample dimensions. It does
not make xarray or ArviZ a core dependency.

## Why this layer exists

A PyMC, HSSM, NumPyro, or Stan result can already be analysed with ArviZ. Behavio's added
responsibility is to preserve the behavioural meaning around that result:

1. which model and task were fitted;
2. which natural parameters the samples represent;
3. which subjects, sessions, and trials label every axis;
4. which observations each pointwise likelihood scores; and
5. which parameter-space definition and inference library produced the draws.

Without that boundary, changing a sampler can accidentally change a transform, observation
denominator, trial order, or prediction target while leaving a plausible-looking trace.

## Core objects

`PosteriorVariable` is an immutable NumPy array with one name for every axis and an explicit
coordinate for every dimension. `PosteriorGroup` collects related variables.
`PosteriorResult` validates relations between those groups and retains scientific identity.

```python
import numpy as np

from behavio import PosteriorResult
from behavio.posterior import PosteriorGroup, PosteriorVariable

chain = np.arange(4)
draw = np.arange(1_000)
subject = np.array(["mouse-a", "mouse-b"])

learning_rate = PosteriorVariable(
    name="learning_rate",
    values=np.random.default_rng(4).beta(2, 5, size=(4, 1_000, 2)),
    dims=("chain", "draw", "subject"),
    coords={"chain": chain, "draw": draw, "subject": subject},
)

result = PosteriorResult(
    model_name="binary-q-learning",
    model_signature="sha256:...",
    inference_library="PyMC",
    inference_library_version="5.x",
    parameter_names=("learning_rate",),
    groups=(PosteriorGroup("posterior", (learning_rate,)),),
    parameter_space_fingerprint="sha256:...",
)
```

Values and coordinates are copied and made read-only. A sampled variable must lead with
`chain` and `draw`; every sampled group must use the same coordinates. Behavio currently
requires per-draw `sample_stats` rather than backend constants in that group. Unsampled
`observed_data`, `constant_data`, and `predictions_constant_data` must not contain sample
dimensions.

The `parameter_names` tuple distinguishes declared natural model parameters from any other
posterior variables or latent quantities. Every declared parameter must occur in the
`posterior` group. Transformed sampler coordinates belong in `unconstrained_posterior`, not
under a misleading natural parameter name.

## ArviZ export and import

Install only when this interchange is needed:

```bash
pip install "behavio[probabilistic]"
```

Then export without changing the core result:

```python
arviz_data = result.to_arviz()
```

On Python 3.12 and later, the current ArviZ API returns an `xarray.DataTree`. Python 3.11
uses the supported ArviZ 0.22-0.23 `InferenceData` representation. Behavio tests both paths
and presents the same public contract. The current `from_dict` API converts nested labelled
groups into a DataTree while preserving named dimensions and coordinates; see the
[ArviZ conversion reference](https://python.arviz.org/en/stable/api/generated/arviz.from_dict.html).

An existing backend-produced ArviZ object can enter the contract explicitly:

```python
from behavio.posterior import posterior_result_from_arviz

result = posterior_result_from_arviz(
    arviz_data,
    model_name="binary-q-learning",
    model_signature="sha256:...",
    inference_library="PyMC",
    inference_library_version="5.x",
    parameter_names=("learning_rate",),
    parameter_space_fingerprint="sha256:...",
)
```

Model and backend identity are required at import. Behavio does not infer provenance from a
Python object's class name or silently treat every posterior variable as a model parameter.

## Diagnose a retained posterior

When ArviZ is installed, the common result can be audited without knowing which backend
created it:

```python
from behavio import audit_posterior

audit = audit_posterior(result)
print(audit.status, audit.issue_codes)
```

The audit keeps rank-normalized $\widehat R$, bulk/tail ESS, HMC warning counts when
available, explicit thresholds, and labelled failure targets. See
[posterior convergence diagnostics](posterior-diagnostics.md).

When the backend also retained a pointwise `log_likelihood`, estimate observation-level
predictive fit and retain its influential-observation diagnostics:

```python
from behavio import psis_loo

loo = psis_loo(result)
print(loo.elpd_loo, loo.issue_codes)
```

See [PSIS-LOO predictive evaluation](psis-loo.md), including why this diagnostic does not
replace future-session validation.

For model criticism, compare observed behavioural summaries with the complete replicated
reference distribution using [posterior-predictive checks](posterior-predictive-checks.md).
To test the joint prior-simulation and inference implementation over repeated synthetic
studies, use [simulation-based calibration](simulation-based-calibration.md). It consumes
the same labelled posterior contract without making a sampler part of Behavio core.
To compare those same labelled parameters across defensible prior, preprocessing, or
backend refits, use [analysis sensitivity](sensitivity-analysis.md).
When two comparable occasions estimate the same individual-level target, extract labelled
subject means for [test-retest reliability](test-retest-reliability.md).

## What this does not claim

The container does not fit a model or turn approximate samples into a calibrated posterior.
The common audit checks standard sampling diagnostics, but a pass is not evidence of model
adequacy or inferential calibration. Posterior-predictive checks, simulation-based
calibration, PSIS-LOO, and sensitivity analyses are separate diagnostic procedures built
on top of these retained groups.
