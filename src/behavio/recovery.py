"""Design-specific parameter-recovery experiments.

Interval coverage
-----------------
``coverage_95`` is only meaningful alongside the interval that produced it, so every
report declares one :data:`WALD_INTERVAL` or :data:`POSTERIOR_QUANTILE_INTERVAL` kind and
every summary repeats it. A maximum-likelihood fit keeps the Wald interval
``estimate +/- 1.96 * standard_error`` built from the numerical-Hessian pseudo-inverse. A
sampled fit uses the equal-tailed 95% posterior quantile interval taken directly from the
draws, which is the interval that model actually reports; reusing a Wald interval there
would substitute a normal approximation to a posterior the sampler already characterized
exactly. The two are never pooled: one call recovers one model and therefore produces one
interval kind.

Failed runs are evidence
------------------------
A simulation or refit that raises is retained as a failed run with non-finite estimates
and an ``ERROR`` audit issue rather than aborting the experiment, matching
``behavio.runner``. Its audit fails, so it is excluded from every estimation summary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.posterior import (
    AnyGenerativeBehaviourModel,
    GenerativePosteriorBehaviourModel,
    PosteriorCentre,
    any_model_capabilities,
    is_posterior_estimator,
    posterior_draw_matrix,
    posterior_parameter_columns,
)
from behavio.diagnostics import (
    AuditSeverity,
    FitAudit,
    FitAuditPolicy,
    FitAuditStatus,
    FitDiagnostics,
    FitIssue,
)
from behavio.evaluation import PosteriorFoldPolicy
from behavio.models.base import (
    FitResult,
    GenerativeBehaviourModel,
)
from behavio.posterior import PosteriorResult
from behavio.posterior_diagnostics import PosteriorAudit, PosteriorAuditStatus, audit_posterior
from behavio.study import Study

#: Coverage computed from ``estimate +/- 1.96 * standard_error``.
WALD_INTERVAL = "wald"

#: Coverage computed from the equal-tailed 95% quantiles of the posterior draws.
POSTERIOR_QUANTILE_INTERVAL = "posterior-quantile"

#: Issue code attached to a recovery run whose simulation or refit raised.
RUN_FAILURE_CODE = "recovery_run_failed"

_NORMAL_95 = 1.959963984540054
_COVERAGE_QUANTILES = (0.025, 0.975)


@dataclass(frozen=True, slots=True)
class ParameterRecoverySummary:
    """Recovery metrics for one parameter across audit-eligible fits.

    ``interval_kind`` names the interval behind ``coverage_95`` so a Wald number and a
    posterior-quantile number can never be read as the same quantity.
    """

    parameter: str
    bias: float
    rmse: float
    correlation: float
    coverage_95: float
    n_successful: int
    n_with_uncertainty: int
    interval_kind: str = WALD_INTERVAL


@dataclass(frozen=True, slots=True)
class ParameterRecoveryReport:
    """Raw results and summaries tied to a particular study design."""

    model_name: str
    model_signature: str
    parameter_names: tuple[str, ...]
    true_values: NDArray[np.float64]
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    converged: NDArray[np.bool_]
    messages: tuple[str, ...]
    audits: tuple[FitAudit, ...]
    seeds: NDArray[np.uint64]
    n_trials: int
    n_subjects: int
    repeats: int
    root_seed: int
    interval_kind: str = WALD_INTERVAL
    interval_lower: NDArray[np.float64] | None = None
    interval_upper: NDArray[np.float64] | None = None
    posterior_audits: tuple[PosteriorAudit | None, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        true_values = protected_array(self.true_values, dtype=np.float64)
        estimates = protected_array(self.estimates, dtype=np.float64)
        standard_errors = protected_array(self.standard_errors, dtype=np.float64)
        converged = protected_array(self.converged, dtype=np.bool_)
        seeds = protected_array(self.seeds, dtype=np.uint64)
        audits = tuple(self.audits)
        expected_shape = (len(self.messages), len(names))
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be non-empty and unique")
        if true_values.shape != expected_shape:
            raise ValueError("true_values must have one row per recovery run")
        if estimates.shape != expected_shape or standard_errors.shape != expected_shape:
            raise ValueError("estimate arrays must match true_values")
        if converged.shape != (len(self.messages),) or seeds.shape != (len(self.messages),):
            raise ValueError("run metadata must have one value per recovery run")
        if len(audits) != len(self.messages):
            raise ValueError("every recovery run must retain one fit audit")
        if any(
            audit.model_name != self.model_name or audit.model_signature != self.model_signature
            for audit in audits
        ):
            raise ValueError("recovery audits must match the report model")
        if self.n_trials < 1 or self.n_subjects < 1 or self.repeats < 1:
            raise ValueError("design counts and repeats must be positive")
        if self.interval_kind not in (WALD_INTERVAL, POSTERIOR_QUANTILE_INTERVAL):
            raise ValueError(
                f"interval_kind must be {WALD_INTERVAL!r} or {POSTERIOR_QUANTILE_INTERVAL!r}"
            )
        posterior_audits = tuple(self.posterior_audits)
        if self.interval_kind == POSTERIOR_QUANTILE_INTERVAL:
            if self.interval_lower is None or self.interval_upper is None:
                raise ValueError("a posterior-quantile report must retain its interval bounds")
            lower = protected_array(self.interval_lower, dtype=np.float64)
            upper = protected_array(self.interval_upper, dtype=np.float64)
            if lower.shape != expected_shape or upper.shape != expected_shape:
                raise ValueError("interval bounds must match the estimate arrays")
            finite = np.isfinite(lower) & np.isfinite(upper)
            if np.any(lower[finite] > upper[finite]):
                raise ValueError("interval lower bounds must not exceed their upper bounds")
            if len(posterior_audits) != len(self.messages):
                raise ValueError("every sampled recovery run must retain one posterior audit")
            object.__setattr__(self, "interval_lower", lower)
            object.__setattr__(self, "interval_upper", upper)
        elif self.interval_lower is not None or self.interval_upper is not None:
            raise ValueError("a Wald report derives its interval from the standard errors")
        elif posterior_audits:
            raise ValueError("a Wald report has no posterior audits to retain")
        object.__setattr__(self, "posterior_audits", posterior_audits)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "true_values", true_values)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "audits", audits)
        object.__setattr__(self, "seeds", seeds)

    @property
    def n_runs(self) -> int:
        return len(self.messages)

    @property
    def convergence_rate(self) -> float:
        return float(np.mean(self.converged))

    @property
    def audit_pass_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.PASS for audit in self.audits]))

    @property
    def audit_warning_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.WARNING for audit in self.audits]))

    @property
    def audit_failure_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.FAIL for audit in self.audits]))

    def summary(self) -> tuple[ParameterRecoverySummary, ...]:
        """Summarize bias, RMSE, association, and ``interval_kind`` coverage.

        Warnings remain eligible; failed audits do not enter estimation summaries. Wald
        coverage uses only eligible runs with finite, non-negative standard errors;
        posterior-quantile coverage uses only eligible runs with finite retained bounds.
        Either way the denominator is reported separately as ``n_with_uncertainty``.
        """

        summaries: list[ParameterRecoverySummary] = []
        audit_eligible = np.asarray(
            [audit.status is not FitAuditStatus.FAIL for audit in self.audits],
            dtype=np.bool_,
        )
        for column, name in enumerate(self.parameter_names):
            valid = audit_eligible & np.isfinite(self.estimates[:, column])
            n_successful = int(np.sum(valid))
            if n_successful:
                truth = self.true_values[valid, column]
                estimate = self.estimates[valid, column]
                error = estimate - truth
                bias = float(np.mean(error))
                rmse = float(np.sqrt(np.mean(error**2)))
            else:
                truth = np.array([], dtype=np.float64)
                estimate = np.array([], dtype=np.float64)
                bias = rmse = float("nan")
            if self.interval_kind == POSTERIOR_QUANTILE_INTERVAL:
                assert self.interval_lower is not None and self.interval_upper is not None
                lower = self.interval_lower[:, column]
                upper = self.interval_upper[:, column]
                uncertainty_valid = valid & np.isfinite(lower) & np.isfinite(upper)
                n_with_uncertainty = int(np.sum(uncertainty_valid))
                if n_with_uncertainty:
                    contained = self.true_values[uncertainty_valid, column]
                    covered = (lower[uncertainty_valid] <= contained) & (
                        contained <= upper[uncertainty_valid]
                    )
                    coverage = float(np.mean(covered))
                else:
                    coverage = float("nan")
            else:
                uncertainty_valid = valid & np.isfinite(self.standard_errors[:, column])
                uncertainty_valid &= self.standard_errors[:, column] >= 0
                n_with_uncertainty = int(np.sum(uncertainty_valid))
                if n_with_uncertainty:
                    uncertainty_error = (
                        self.estimates[uncertainty_valid, column]
                        - self.true_values[uncertainty_valid, column]
                    )
                    covered = np.abs(uncertainty_error) <= (
                        _NORMAL_95 * self.standard_errors[uncertainty_valid, column]
                    )
                    coverage = float(np.mean(covered))
                else:
                    coverage = float("nan")
            truth_scale = max(1.0, float(np.max(np.abs(truth), initial=0.0)))
            estimate_scale = max(1.0, float(np.max(np.abs(estimate), initial=0.0)))
            truth_varies = n_successful >= 2 and np.ptp(truth) > 1e-12 * truth_scale
            estimate_varies = n_successful >= 2 and np.ptp(estimate) > 1e-12 * estimate_scale
            if n_successful >= 2 and truth_varies and estimate_varies:
                correlation = float(np.corrcoef(truth, estimate)[0, 1])
            else:
                correlation = float("nan")
            summaries.append(
                ParameterRecoverySummary(
                    parameter=name,
                    bias=bias,
                    rmse=rmse,
                    correlation=correlation,
                    coverage_95=coverage,
                    n_successful=n_successful,
                    n_with_uncertainty=n_with_uncertainty,
                    interval_kind=self.interval_kind,
                )
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record of every run and summary denominator.

        ``coverage_interval``, per-summary ``interval_kind``, and the per-run ``interval``
        and ``posterior_audit`` entries appear only for a posterior-derived report, so a
        maximum-likelihood record is exactly what it was before sampled models existed.
        """

        summaries = self.summary()
        sampled = self.interval_kind == POSTERIOR_QUANTILE_INTERVAL
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "parameter_names": list(self.parameter_names),
            "design": {
                "n_trials": self.n_trials,
                "n_subjects": self.n_subjects,
            },
            "repeats": self.repeats,
            "root_seed": self.root_seed,
            "n_runs": self.n_runs,
            "convergence_rate": self.convergence_rate,
            "audit_rates": {
                "pass": self.audit_pass_rate,
                "warning": self.audit_warning_rate,
                "fail": self.audit_failure_rate,
            },
            "summary": [
                {
                    "parameter": summary.parameter,
                    "bias": _json_float(summary.bias),
                    "rmse": _json_float(summary.rmse),
                    "correlation": _json_float(summary.correlation),
                    "coverage_95": _json_float(summary.coverage_95),
                    "n_successful": summary.n_successful,
                    "n_with_uncertainty": summary.n_with_uncertainty,
                    **({"interval_kind": summary.interval_kind} if sampled else {}),
                }
                for summary in summaries
            ],
            "runs": [
                {
                    "seed": int(self.seeds[index]),
                    "truth": {
                        name: float(self.true_values[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "estimate": {
                        name: _json_float(self.estimates[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "standard_error": {
                        name: _json_float(self.standard_errors[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "converged": bool(self.converged[index]),
                    "message": self.messages[index],
                    "fit_audit": _json_safe(self.audits[index].to_dict()),
                    **(self._run_posterior_payload(index) if sampled else {}),
                }
                for index in range(self.n_runs)
            ],
        }
        if sampled:
            payload["coverage_interval"] = self.interval_kind
        return payload

    def _run_posterior_payload(self, index: int) -> dict[str, Any]:
        assert self.interval_lower is not None and self.interval_upper is not None
        audit = self.posterior_audits[index]
        return {
            "interval": {
                name: [
                    _json_float(self.interval_lower[index, column]),
                    _json_float(self.interval_upper[index, column]),
                ]
                for column, name in enumerate(self.parameter_names)
            },
            "posterior_audit": None if audit is None else _json_safe(audit.to_dict()),
        }


def run_parameter_recovery(
    model: AnyGenerativeBehaviourModel,
    design: Study,
    parameter_sets: Sequence[Mapping[str, float]],
    *,
    repeats: int = 1,
    seed: int,
    audit_policy: FitAuditPolicy | None = None,
    posterior_policy: PosteriorFoldPolicy | None = None,
) -> ParameterRecoveryReport:
    """Simulate and refit explicit parameter sets under one observed design.

    Every parameter set is repeated ``repeats`` times. The report retains the design size,
    random seeds, convergence flags, and optimizer or projection messages so recovery is
    never detached from the conditions under which it was assessed. A run whose simulation
    or refit raises is retained with non-finite estimates and a failing audit instead of
    aborting the experiment.

    A :class:`~behavio.contracts.posterior.GenerativePosteriorBehaviourModel` is sampled
    rather than optimized. Its posterior is audited, projected through ``point_summary``,
    and read back through the model's declared ``posterior_parameter_labels``, because the
    projected coordinate names (``beta[coefficient='stimulus']``) are not the simulator's
    scalar parameter names. Coverage then uses the posterior quantile interval and the
    report declares :data:`POSTERIOR_QUANTILE_INTERVAL`.
    """

    sampled = is_posterior_estimator(model)
    if sampled:
        if not isinstance(model, GenerativePosteriorBehaviourModel):
            raise TypeError(
                "model must satisfy the GenerativeBehaviourModel or "
                "GenerativePosteriorBehaviourModel contract"
            )
    elif not isinstance(model, GenerativeBehaviourModel):
        raise TypeError("model must satisfy the GenerativeBehaviourModel contract")
    any_model_capabilities(model)
    if posterior_policy is not None:
        if not isinstance(posterior_policy, PosteriorFoldPolicy):
            raise TypeError("posterior_policy must be a PosteriorFoldPolicy")
        if not sampled:
            raise ValueError(
                "posterior_policy applies only to a sampled generative model; "
                f"model {model.model_name!r} is fitted by optimization"
            )
    fold_policy = PosteriorFoldPolicy() if posterior_policy is None else posterior_policy
    parameter_sets = tuple(parameter_sets)
    if not parameter_sets:
        raise ValueError("parameter_sets must not be empty")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if audit_policy is not None and not isinstance(audit_policy, FitAuditPolicy):
        raise TypeError("audit_policy must be a FitAuditPolicy")

    validated_parameters: list[dict[str, float]] = []
    expected = set(model.parameter_names)
    for index, parameters in enumerate(parameter_sets):
        if not isinstance(parameters, Mapping):
            raise TypeError(f"parameter set {index} must be a mapping")
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                f"parameter set {index} must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            values = {name: float(parameters[name]) for name in model.parameter_names}
        except (TypeError, ValueError):
            raise ValueError(f"parameter set {index} must contain finite numeric values") from None
        if not np.all(np.isfinite(tuple(values.values()))):
            raise ValueError(f"parameter set {index} must contain finite numeric values")
        validated_parameters.append(values)

    n_runs = len(parameter_sets) * repeats
    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    true_values = np.empty((n_runs, len(model.parameter_names)), dtype=np.float64)
    estimates = np.empty_like(true_values)
    standard_errors = np.empty_like(true_values)
    interval_lower = np.empty_like(true_values)
    interval_upper = np.empty_like(true_values)
    converged = np.empty(n_runs, dtype=np.bool_)
    seeds = np.empty(n_runs, dtype=np.uint64)
    messages: list[str] = []
    audits: list[FitAudit] = []
    posterior_audits: list[PosteriorAudit | None] = []

    run = 0
    for parameters in validated_parameters:
        for _ in range(repeats):
            child_seed = int(child_sequences[run].generate_state(1, dtype=np.uint64)[0])
            outcome = _recover_once(
                model,
                design,
                parameters,
                seed=child_seed,
                sampled=sampled,
                audit_policy=audit_policy,
                fold_policy=fold_policy,
            )
            true_values[run] = [parameters[name] for name in model.parameter_names]
            estimates[run] = outcome.estimates
            standard_errors[run] = outcome.standard_errors
            interval_lower[run] = outcome.interval_lower
            interval_upper[run] = outcome.interval_upper
            converged[run] = outcome.converged
            seeds[run] = child_seed
            messages.append(outcome.message)
            audits.append(outcome.audit)
            posterior_audits.append(outcome.posterior_audit)
            run += 1

    return ParameterRecoveryReport(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        true_values=true_values,
        estimates=estimates,
        standard_errors=standard_errors,
        converged=converged,
        messages=tuple(messages),
        audits=tuple(audits),
        seeds=seeds,
        n_trials=len(design),
        n_subjects=len(design.subjects),
        repeats=repeats,
        root_seed=seed,
        interval_kind=POSTERIOR_QUANTILE_INTERVAL if sampled else WALD_INTERVAL,
        interval_lower=interval_lower if sampled else None,
        interval_upper=interval_upper if sampled else None,
        posterior_audits=tuple(posterior_audits) if sampled else (),
    )


@dataclass(frozen=True, slots=True)
class _RecoveryRun:
    """One completed or failed simulate-and-refit cycle."""

    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    interval_lower: NDArray[np.float64]
    interval_upper: NDArray[np.float64]
    converged: bool
    message: str
    audit: FitAudit
    posterior_audit: PosteriorAudit | None


def _recover_once(
    model: AnyGenerativeBehaviourModel,
    design: Study,
    parameters: Mapping[str, float],
    *,
    seed: int,
    sampled: bool,
    audit_policy: FitAuditPolicy | None,
    fold_policy: PosteriorFoldPolicy,
) -> _RecoveryRun:
    """Simulate, refit, and summarize one run, retaining any failure as evidence."""

    posterior = None
    convergence = None
    try:
        simulated = model.simulate(design, parameters, seed=seed)
        if sampled:
            posterior = model.sample(simulated)
            convergence = audit_posterior(posterior, policy=fold_policy.audit_policy)
            fit = model.point_summary(
                posterior,
                converged=convergence.status is not PosteriorAuditStatus.FAIL,
                centre=PosteriorCentre(fold_policy.centre),
            )
        else:
            fit = model.fit(simulated)
    except Exception as error:  # a failed recovery run is scientific evidence
        return _failed_run(model, max(1, len(design)), error)
    # Contract violations are deliberately outside the capture: a model that misreports its
    # identity or cannot name its own posterior coordinates is a defect to fix, not a run
    # to average over.
    if sampled:
        assert posterior is not None and convergence is not None
        return _sampled_run(model, simulated, posterior, convergence, fit, audit_policy)
    return _optimized_run(model, simulated, fit, audit_policy)


def _optimized_run(
    model: GenerativeBehaviourModel,
    simulated: Study,
    fit: FitResult,
    audit_policy: FitAuditPolicy | None,
) -> _RecoveryRun:
    if not isinstance(fit, FitResult):
        raise TypeError("model.fit must return a FitResult")
    if (
        fit.model_name != model.model_name
        or fit.model_signature != model.signature
        or fit.parameter_names != model.parameter_names
    ):
        raise ValueError("fit result does not match the recovery model")
    if fit.n_observations != len(simulated):
        raise ValueError("fit result n_observations must equal the simulated-study length")
    absent = np.full(len(model.parameter_names), np.nan, dtype=np.float64)
    return _RecoveryRun(
        estimates=np.asarray(fit.estimates, dtype=np.float64),
        standard_errors=np.asarray(fit.standard_errors, dtype=np.float64),
        interval_lower=absent,
        interval_upper=absent,
        converged=bool(fit.diagnostics.converged),
        message=fit.diagnostics.message,
        audit=fit.audit(policy=audit_policy),
        posterior_audit=None,
    )


def _sampled_run(
    model: GenerativePosteriorBehaviourModel,
    simulated: Study,
    posterior: PosteriorResult,
    convergence: PosteriorAudit,
    fit: FitResult,
    audit_policy: FitAuditPolicy | None,
) -> _RecoveryRun:
    is_converged = convergence.status is not PosteriorAuditStatus.FAIL
    if not isinstance(fit, FitResult):
        raise TypeError("model.point_summary must return a FitResult")
    if fit.model_name != model.model_name or fit.model_signature != model.signature:
        raise ValueError("fit result does not match the recovery model")
    if fit.n_observations != len(simulated):
        raise ValueError("fit result n_observations must equal the simulated-study length")
    if fit.diagnostics.converged is not is_converged:
        raise ValueError(
            "model.point_summary must record the convergence verdict it was given; the "
            "posterior convergence audit, not the model, decides whether a run is usable"
        )
    summary_columns = posterior_parameter_columns(model, fit.parameter_names)
    draw_names, draws = posterior_draw_matrix(posterior)
    draw_columns = posterior_parameter_columns(model, draw_names)
    quantiles = np.quantile(draws[:, draw_columns], _COVERAGE_QUANTILES, axis=0)
    return _RecoveryRun(
        estimates=np.asarray(fit.estimates, dtype=np.float64)[list(summary_columns)],
        standard_errors=np.asarray(fit.standard_errors, dtype=np.float64)[list(summary_columns)],
        interval_lower=np.asarray(quantiles[0], dtype=np.float64),
        interval_upper=np.asarray(quantiles[1], dtype=np.float64),
        converged=is_converged,
        message=fit.diagnostics.message,
        audit=fit.audit(policy=audit_policy),
        posterior_audit=convergence,
    )


def _failed_run(
    model: AnyGenerativeBehaviourModel,
    n_observations: int,
    error: Exception,
) -> _RecoveryRun:
    message = f"{type(error).__name__}: {error}"
    absent = np.full(len(model.parameter_names), np.nan, dtype=np.float64)
    return _RecoveryRun(
        estimates=absent,
        standard_errors=absent,
        interval_lower=absent,
        interval_upper=absent,
        converged=False,
        message=message,
        audit=FitAudit(
            model_name=model.model_name,
            model_signature=model.signature,
            n_observations=n_observations,
            numerical=FitDiagnostics(
                converged=False,
                optimizer="unavailable",
                status=-1,
                message=message,
                n_iterations=None,
                objective=None,
                gradient_norm=None,
                hessian_condition=None,
                boundary_estimate=None,
            ),
            issues=(
                FitIssue(
                    code=RUN_FAILURE_CODE,
                    severity=AuditSeverity.ERROR,
                    message=f"the recovery run raised before producing a fit: {message}",
                ),
            ),
        ),
        posterior_audit=None,
    )


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _json_float(float(value))
    return value
