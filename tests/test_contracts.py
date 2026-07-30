"""The extension surface lives at one address, and the package has no import cycle."""

from __future__ import annotations

import ast
import graphlib
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import behavio.contracts as contracts
from behavio import BernoulliHistoryGLM, ChoiceSpec, PosteriorResult, Study, TaskSpec
from behavio._internal.arrays import protected_array
from behavio.contracts.audit import AuditSeverity, FitAuditStatus, FitDiagnostics
from behavio.contracts.estimator import FitResult, PredictionMode, fit_auditor
from behavio.contracts.posterior import (
    DesignGenerativeBehaviourModel,
    GenerativePosteriorBehaviourModel,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    bind_to_design,
    is_design_generative,
    is_posterior_estimator,
    posterior_log_predictive_density,
    posterior_model_capabilities,
    posterior_parameter_columns,
    posterior_point_summary,
)
from behavio.diagnostics import audit_fit
from behavio.posterior import PosteriorGroup, PosteriorVariable
from behavio.posterior.result import PosteriorError

PACKAGE = Path(behavio_root := Path(__file__).parents[1] / "src" / "behavio")

# Every contract name and the friendly implementation-side home that must keep re-exporting
# it. A contract is declared once in ``behavio.contracts`` and surfaced where the author who
# implements it is already reading; these pairs are that promise, not a compatibility shim.
CONTRACT_RE_EXPORT_HOMES = {
    "behavio.models.base": (
        "BehaviourEstimator",
        "BehaviourModel",
        "CategoricalBehaviourEstimator",
        "CategoricalPrediction",
        "DensityBehaviourEstimator",
        "DensityPrediction",
        "FitDiagnostics",
        "FitResult",
        "GenerativeBehaviourModel",
        "ModelCapabilities",
        "ModelDataError",
        "ModelPrediction",
        "Prediction",
        "PredictionMode",
        "UnsupportedPredictionMode",
        "model_capabilities",
    ),
    "behavio.diagnostics": (
        "AuditSeverity",
        "FitAudit",
        "FitAuditPolicy",
        "FitAuditStatus",
        "FitDiagnostics",
        "FitIssue",
        "LatentStateAudit",
        "LatentStateFit",
        "MultistartFit",
        "RestartAudit",
    ),
    "behavio.inference.optimize": ("ObjectiveTarget", "OptimizationBackend", "PriorMeasure"),
    "behavio.inference.parameters": ("ParameterSpaceProvider",),
    "behavio.posterior.predictive": ("PredictiveDiscrepancy", "PredictiveTail"),
    "behavio.time.transforms": (
        "FittedStudyTransform",
        "StudyTransform",
        "TransformProvenance",
    ),
    "behavio.evaluate.splits": ("EvaluationFold",),
    "behavio.models": (
        "BehaviourEstimator",
        "DensityBehaviourEstimator",
        "DensityPrediction",
        "FitDiagnostics",
        "FitResult",
        "model_capabilities",
    ),
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE.parent).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_level_behavio_imports(path: Path) -> set[str]:
    """Return the behavio modules imported at runtime when ``path`` is executed.

    Function-local imports and ``if TYPE_CHECKING`` blocks are excluded: neither runs at
    module import time, so neither can produce an import cycle.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                continue
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "behavio" or node.module.startswith("behavio."):
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "behavio" or alias.name.startswith("behavio."):
                        found.add(alias.name)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    visit([child])
                else:
                    for grandchild in ast.walk(child):
                        if isinstance(grandchild, ast.stmt):
                            visit([grandchild])

    visit(tree.body)
    return found


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _import_graph() -> dict[str, set[str]]:
    modules = {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}
    return {
        name: {edge for edge in _module_level_behavio_imports(path) if edge in modules}
        for name, path in modules.items()
    }


# --------------------------------------------------------------------------------------
# 3. The cycle is broken.
# --------------------------------------------------------------------------------------


def test_the_package_has_no_module_level_import_cycle() -> None:
    graph = _import_graph()

    # graphlib raises CycleError and names the offending modules.
    graphlib.TopologicalSorter(graph).prepare()


#: The package's layering, lowest first. Every module belongs to exactly one layer, matched by
#: the longest declared prefix, and **every module-level import must land in the same layer or
#: a lower one**. That is the invariant this file exists to keep: a reorganisation that groups
#: modules into readable areas is only worth anything if the areas stay ordered, and an
#: accidental upward import is the one defect that a passing test suite will not otherwise
#: reveal until something downstream cannot be imported at all.
#:
#: Two entries name a single module rather than an area, and both are deliberate.
#: ``behavio.trials`` and ``behavio.posterior.result`` are pure data containers with no
#: intra-package imports of their own; ``behavio.contracts`` names both of them in the
#: signatures it declares, so they must sit *below* the contracts package even though one of
#: them lives inside ``behavio.posterior``, whose other modules sit above it. A package may
#: span layers; a module may not.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("behavio._internal",),
    ("behavio.trials", "behavio.posterior.result"),
    ("behavio.contracts",),
    (
        "behavio.adapters",
        "behavio.design",
        "behavio.diagnostics",
        "behavio.inference",
        "behavio.observed",
        "behavio.posterior",
        "behavio.task",
        "behavio.time",
    ),
    ("behavio.models",),
    #: Wrappers around third-party model packages. Above ``behavio.models`` because a wrapped
    #: model is a model: it needs ``describe()``, the design algebra and the response-time
    #: contract, all of which live at or above the model layer. The parts of a wrapper that
    #: name no third-party package -- the sequence/row helper, the continuous-outcome
    #: prediction type, the estimator conformance harness -- stay in ``behavio.adapters``.
    ("behavio.foreign",),
    ("behavio.compose",),
    ("behavio.registry",),
    ("behavio.evaluate",),
    ("behavio.compare",),
    ("behavio.protocol", "behavio.recovery"),
    ("behavio.pymc_backend", "behavio.report"),
    ("behavio.plot",),
    ("behavio.cli",),
    ("behavio",),
)


def _layer_of(module: str) -> int:
    best, best_length = None, -1
    for index, prefixes in enumerate(LAYERS):
        for prefix in prefixes:
            if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > best_length:
                best, best_length = index, len(prefix)
    assert best is not None, f"{module} belongs to no declared layer"
    return best


def test_every_module_belongs_to_exactly_one_declared_layer() -> None:
    """A new top-level area must be placed in the order, not merely created."""

    for module in _import_graph():
        _layer_of(module)


def test_no_module_imports_a_module_from_a_higher_layer() -> None:
    violations = [
        (module, edge, _layer_of(module), _layer_of(edge))
        for module, edges in _import_graph().items()
        for edge in sorted(edges)
        if _layer_of(edge) > _layer_of(module)
    ]

    assert not violations, "\n".join(
        f"{module} (layer {low}) imports {edge} (layer {high})"
        for module, edge, low, high in violations
    )


def test_no_area_imports_an_area_that_imports_it_back() -> None:
    """The module graph being acyclic is not enough; the *areas* must be ordered too.

    Two modules in different packages can depend on each other without forming a module
    cycle, and the result still cannot be read as layers and still breaks as soon as either
    package's ``__init__`` imports the other's module. Modules with no intra-package imports
    at all are excluded, because they cannot participate in a cycle and grouping them into an
    area would invent one: ``behavio.posterior.result`` is named by ``behavio.contracts``, but
    it imports nothing, so ``contracts`` and ``posterior`` are not mutually dependent.
    """

    graph = _import_graph()
    leaves = {module for module, edges in graph.items() if not edges and module != "behavio"}

    def area(module: str) -> str:
        parts = module.split(".")
        return ".".join(parts[:2]) if len(parts) > 1 else module

    areas: dict[str, set[str]] = {}
    for module, edges in graph.items():
        if module == "behavio" or module in leaves:
            continue
        source = area(module)
        areas.setdefault(source, set()).update(
            {area(edge) for edge in edges if edge not in leaves and edge != "behavio"} - {source}
        )

    # graphlib raises CycleError and names the offending areas.
    graphlib.TopologicalSorter(areas).prepare()


def test_contracts_never_depends_on_the_modules_that_implement_it() -> None:
    graph = _import_graph()
    implementers = {
        "behavio",
        "behavio.diagnostics",
        "behavio.inference.optimize",
        "behavio.models",
        "behavio.models.base",
        "behavio.inference.parameters",
        "behavio.posterior.diagnostics",
        "behavio.posterior.predictive",
        "behavio.time.transforms",
        "behavio.evaluate.splits",
    }

    for module, edges in graph.items():
        if module.startswith("behavio.contracts") or module.startswith("behavio._internal"):
            assert not (edges & implementers), f"{module} imports {sorted(edges & implementers)}"


def test_fit_result_audit_dispatches_without_a_function_local_import() -> None:
    source = (PACKAGE / "contracts" / "estimator.py").read_text(encoding="utf-8")
    assert "from behavio.diagnostics" not in source
    assert fit_auditor() is audit_fit


def test_fit_result_audit_still_works_from_its_documented_public_path() -> None:
    fit = _mle_fit()

    audit = fit.audit()

    assert audit.status is FitAuditStatus.PASS
    assert audit.model_name == "demo"
    assert audit.to_dict()["status"] == "pass"


# --------------------------------------------------------------------------------------
# 1 and 2. Re-homing preserved every import path and every identity.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("module_name", "names"), sorted(CONTRACT_RE_EXPORT_HOMES.items()))
def test_every_friendly_home_re_exports_the_one_contract_object(
    module_name: str, names: tuple[str, ...]
) -> None:
    module = __import__(module_name, fromlist=["__name__"])

    for name in names:
        assert hasattr(module, name), f"{module_name}.{name} disappeared"
        assert getattr(module, name) is getattr(contracts, name), (
            f"{module_name}.{name} is not the contracts object"
        )


def test_contracts_exports_exactly_what_it_declares() -> None:
    for name in contracts.__all__:
        assert hasattr(contracts, name)
    # The isort-style order ruff's RUF022 enforces: constants, then everything else in
    # ASCII order, which puts classes before functions. Asserting the same order here
    # rather than a plain ``sorted`` keeps the test and the linter from disagreeing about
    # where a name like ``LOG_DENSITY_FLOOR`` belongs.
    assert sorted(contracts.__all__, key=lambda name: (not name.isupper(), name)) == list(
        contracts.__all__
    )


def test_protected_array_moved_but_the_old_private_alias_still_works() -> None:
    from behavio.models.base import _protected_array

    assert _protected_array is protected_array
    values = protected_array([1.0, 2.0], dtype=np.float64)
    assert not values.flags.writeable
    assert not values.base.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 3.0


def test_validation_fold_is_a_declared_protocol_every_first_party_split_satisfies() -> None:
    from behavio.contracts.fold import EvaluationFold
    from behavio.evaluate.splits import (
        CohortSplit,
        HistoricalCohortForecastSplit,
        PopulationForecastSplit,
        PopulationSplit,
        Split,
        cohort_forward_session_splits,
        forward_session_splits,
    )

    assert EvaluationFold is contracts.EvaluationFold
    study = _study()
    for split in forward_session_splits(study):
        assert isinstance(split, EvaluationFold)
        assert isinstance(split, Split)
    for split in cohort_forward_session_splits(study):
        assert isinstance(split, EvaluationFold)
        assert isinstance(split, CohortSplit)
    for declared in (
        PopulationSplit,
        PopulationForecastSplit,
        HistoricalCohortForecastSplit,
    ):
        # ``__protocol_attrs__`` is a CPython 3.12 implementation detail and
        # ``typing.get_protocol_members`` only arrives in 3.13, so the declared members are
        # read off the protocol itself; the package supports 3.11.
        members = {name for name in dir(EvaluationFold) if not name.startswith("_")}
        assert members
        assert members <= set(declared.__dataclass_fields__) | set(dir(declared))
    assert not isinstance(object(), EvaluationFold)


def test_first_party_models_still_satisfy_the_re_homed_estimator_protocol() -> None:
    model = BernoulliHistoryGLM(choice_lags=1)

    assert isinstance(model, contracts.BehaviourEstimator)
    assert isinstance(model, contracts.GenerativeBehaviourModel)
    capabilities = contracts.model_capabilities(model)
    assert capabilities.can_simulate
    assert PredictionMode.FILTERED in capabilities.prediction_modes


# --------------------------------------------------------------------------------------
# 5. The posterior estimator protocol and its point-summary projection.
# --------------------------------------------------------------------------------------


class _SampledGLM:
    """Minimal structural implementation of the sampled-estimator contract."""

    model_name = "sampled-demo"
    signature = "sampled-demo[v1]"
    scored_columns = ("choice",)
    supported_prediction_modes = (PredictionMode.FILTERED,)
    parameter_names = ("intercept", "slope")
    # The simulator names scalars; the projection names posterior coordinates. The
    # correspondence is declared, never guessed.
    posterior_parameter_labels: ClassVar[dict[str, str]] = {
        "intercept": "beta[coefficient='intercept']",
        "slope": "beta[coefficient='stimulus']",
    }

    def sample(self, study: Study) -> PosteriorResult:
        return _posterior()

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Any:
        raise NotImplementedError

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Any:
        raise NotImplementedError

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    def simulate(self, design: Study, parameters: Any, *, seed: Any) -> Study:
        raise NotImplementedError


def test_a_sampled_model_satisfies_the_posterior_estimator_contract() -> None:
    model = _SampledGLM()

    assert isinstance(model, PosteriorBehaviourEstimator)
    assert isinstance(model, GenerativePosteriorBehaviourModel)
    assert not isinstance(BernoulliHistoryGLM(choice_lags=1), PosteriorBehaviourEstimator)

    capabilities = posterior_model_capabilities(model)
    assert capabilities.scored_columns == ("choice",)
    assert capabilities.can_simulate
    assert capabilities.can_recover_parameters
    assert capabilities == contracts.ModelCapabilities(
        scored_columns=("choice",),
        prediction_modes=(PredictionMode.FILTERED,),
        can_simulate=True,
        can_recover_parameters=True,
        is_sampled=True,
    )


def test_posterior_model_capabilities_rejects_a_non_conforming_object() -> None:
    with pytest.raises(TypeError, match="PosteriorBehaviourEstimator"):
        posterior_model_capabilities(object())  # type: ignore[arg-type]


class _DesignGenerativeGLM:
    """A sampler whose parameter vector is a fact about the design, not the specification.

    ``parameter_names`` and ``posterior_parameter_labels`` are deliberately absent -- how
    many coordinates ``(1|subject)`` has is a fact about the data -- so this model must not
    claim the generative contract; the property would be a promise it breaks on the first
    design it meets. ``bind`` is the whole of the extra surface.
    """

    model_name = "design-generative-demo"
    signature = "design-generative-demo[v1]"
    scored_columns = ("choice",)
    supported_prediction_modes = (PredictionMode.FILTERED,)

    def sample(self, study: Study) -> PosteriorResult:
        return _posterior()

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Any:
        raise NotImplementedError

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Any:
        raise NotImplementedError

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    def bind(self, design: Study) -> _SampledGLM:
        del design
        return _SampledGLM()


class _EmptyBinder(_DesignGenerativeGLM):
    def bind(self, design: Study) -> Any:
        del design
        return object()


def test_a_model_can_be_generative_relative_to_a_design() -> None:
    model = _DesignGenerativeGLM()
    design = _study()

    assert isinstance(model, PosteriorBehaviourEstimator)
    assert not isinstance(model, GenerativePosteriorBehaviourModel)
    assert isinstance(model, DesignGenerativeBehaviourModel)
    assert is_design_generative(model)
    # An already-generative model answers False even though every generative model could
    # in principle grow a ``bind``: there is nothing left for a design to supply.
    assert not is_design_generative(_SampledGLM())

    capabilities = posterior_model_capabilities(model)
    assert not capabilities.can_simulate
    assert capabilities.can_bind_design
    assert capabilities.can_recover_parameters

    bound = bind_to_design(model, design)

    assert isinstance(bound, GenerativePosteriorBehaviourModel)
    assert bind_to_design(bound, design) is bound


def test_binding_checks_its_result_rather_than_trusting_it() -> None:
    with pytest.raises(TypeError, match="not a generative model"):
        bind_to_design(_EmptyBinder(), _study())
    with pytest.raises(TypeError, match="DesignGenerativeBehaviourModel"):
        bind_to_design(object(), _study())  # type: ignore[arg-type]


def test_a_capability_record_cannot_claim_both_kinds_of_simulation() -> None:
    with pytest.raises(ValueError, match="either in the abstract or relative to a design"):
        contracts.ModelCapabilities(
            scored_columns=("choice",),
            prediction_modes=(PredictionMode.FILTERED,),
            can_simulate=True,
            can_recover_parameters=True,
            can_bind_design=True,
        )
    with pytest.raises(ValueError, match="parameter recovery requires simulation"):
        contracts.ModelCapabilities(
            scored_columns=("choice",),
            prediction_modes=(PredictionMode.FILTERED,),
            can_simulate=False,
            can_recover_parameters=True,
        )


def test_the_declared_parameter_mapping_is_what_makes_recovery_well_defined() -> None:
    model = _SampledGLM()
    projected = posterior_point_summary(_posterior(), converged=True)

    # The two vocabularies genuinely differ, so the mapping is not decoration.
    assert projected.parameter_names != model.parameter_names
    assert posterior_parameter_columns(model, projected.parameter_names) == (0, 1)

    class _Mislabelled(_SampledGLM):
        posterior_parameter_labels: ClassVar[dict[str, str]] = {
            "intercept": "beta",
            "slope": "gamma",
        }

    with pytest.raises(ValueError, match="does not round-trip"):
        posterior_parameter_columns(_Mislabelled(), projected.parameter_names)

    class _Incomplete(_SampledGLM):
        posterior_parameter_labels: ClassVar[dict[str, str]] = {
            "intercept": "beta[coefficient='intercept']",
        }

    with pytest.raises(ValueError, match="every simulator parameter"):
        posterior_model_capabilities(_Incomplete())


def test_the_predictive_density_helper_is_the_draw_average_not_the_plug_in() -> None:
    per_draw = np.log(np.asarray([[0.2, 0.9], [0.8, 0.1]]))

    density = posterior_log_predictive_density(per_draw)

    np.testing.assert_allclose(density, np.log([0.5, 0.5]))
    assert np.all(density >= per_draw.mean(axis=0))


def test_capability_dispatch_covers_both_estimator_contracts() -> None:
    sampled = _SampledGLM()
    optimized = BernoulliHistoryGLM(choice_lags=1)

    assert is_posterior_estimator(sampled)
    assert not is_posterior_estimator(optimized)
    assert any_model_capabilities(sampled) == posterior_model_capabilities(sampled)
    assert any_model_capabilities(optimized) == contracts.model_capabilities(optimized)


def test_the_point_summary_projection_reports_only_measured_quantities() -> None:
    result = _posterior()

    fit = posterior_point_summary(result, converged=True)

    assert isinstance(fit, FitResult)
    assert fit.model_name == result.model_name
    assert fit.model_signature == result.model_signature
    assert fit.parameter_names == (
        "beta[coefficient='intercept']",
        "beta[coefficient='stimulus']",
    )
    draws = np.asarray(result["posterior"]["beta"].values).reshape(-1, 2)
    np.testing.assert_allclose(fit.estimates, draws.mean(axis=0))
    np.testing.assert_allclose(fit.standard_errors, draws.std(axis=0, ddof=1))
    np.testing.assert_allclose(fit.covariance, np.cov(draws, rowvar=False, ddof=1))
    assert fit.n_observations == 7
    assert fit.diagnostics.converged is True
    assert fit.diagnostics.optimizer == "test-sampler/1"
    # Nothing optimizer-shaped is fabricated.
    assert fit.diagnostics.n_iterations is None
    assert fit.diagnostics.objective is None
    assert fit.diagnostics.gradient_norm is None
    assert fit.diagnostics.hessian_condition is None
    assert fit.diagnostics.boundary_estimate is None


def test_a_projected_posterior_audits_cleanly_instead_of_warning_about_absent_diagnostics() -> None:
    fit = posterior_point_summary(_posterior(), converged=True)

    audit = fit.audit()

    assert audit.status is FitAuditStatus.PASS
    assert audit.issue_codes == ()
    assert audit.to_dict()["numerical"]["gradient_norm"] is None


def test_a_non_converged_projection_still_fails_the_audit() -> None:
    fit = posterior_point_summary(_posterior(), converged=False)

    audit = fit.audit()

    assert audit.status is FitAuditStatus.FAIL
    assert "optimizer_nonconvergence" in audit.issue_codes


def test_non_finite_optimizer_diagnostics_are_still_reported() -> None:
    """Absent is not the same as non-finite: existing MLE behaviour is unchanged."""

    fit = _mle_fit(gradient_norm=np.nan, hessian_condition=np.nan, objective=np.nan)

    audit = audit_fit(fit)

    assert set(audit.issue_codes) >= {
        "nonfinite_gradient",
        "nonfinite_hessian_condition",
        "nonfinite_objective",
    }
    assert audit.status is FitAuditStatus.FAIL
    assert any(issue.severity is AuditSeverity.ERROR for issue in audit.issues)


def test_the_projection_supports_a_median_centre_and_an_explicit_row_count() -> None:
    result = _posterior()

    fit = posterior_point_summary(
        result, converged=True, centre=PosteriorCentre.MEDIAN, n_observations=3
    )

    draws = np.asarray(result["posterior"]["beta"].values).reshape(-1, 2)
    np.testing.assert_allclose(fit.estimates, np.median(draws, axis=0))
    assert fit.n_observations == 3


def test_the_projection_refuses_to_guess_a_row_count() -> None:
    result = _posterior(with_log_likelihood=False)

    with pytest.raises(PosteriorError, match="n_observations"):
        posterior_point_summary(result, converged=True)


def test_the_projection_requires_an_explicit_convergence_verdict() -> None:
    with pytest.raises(TypeError):
        posterior_point_summary(_posterior())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="boolean"):
        posterior_point_summary(_posterior(), converged="yes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _mle_fit(
    *,
    gradient_norm: float = 1e-6,
    hessian_condition: float = 10.0,
    objective: float = 12.5,
) -> FitResult:
    return FitResult(
        model_name="demo",
        model_signature="demo[v1]",
        parameter_names=("a", "b"),
        estimates=np.array([0.5, -0.25]),
        standard_errors=np.array([0.1, 0.2]),
        covariance=np.eye(2) * 0.01,
        n_observations=32,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="scipy",
            status=0,
            message="converged",
            n_iterations=17,
            objective=objective,
            gradient_norm=gradient_norm,
            hessian_condition=hessian_condition,
            boundary_estimate=False,
        ),
    )


def _posterior(*, with_log_likelihood: bool = True) -> PosteriorResult:
    generator = np.random.default_rng(11)
    chains, draws, trials = 2, 40, 7
    coords = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
        "coefficient": np.asarray(["intercept", "stimulus"]),
        "trial": np.arange(trials),
    }
    groups = [
        PosteriorGroup(
            "posterior",
            (
                PosteriorVariable(
                    "beta",
                    generator.normal(size=(chains, draws, 2)),
                    ("chain", "draw", "coefficient"),
                    {
                        "chain": coords["chain"],
                        "draw": coords["draw"],
                        "coefficient": coords["coefficient"],
                    },
                ),
            ),
        )
    ]
    if with_log_likelihood:
        groups.append(
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        -np.abs(generator.normal(size=(chains, draws, trials))),
                        ("chain", "draw", "trial"),
                        {
                            "chain": coords["chain"],
                            "draw": coords["draw"],
                            "trial": coords["trial"],
                        },
                    ),
                ),
            )
        )
    return PosteriorResult(
        model_name="test-model",
        model_signature="test-signature",
        inference_library="test-sampler",
        inference_library_version="1",
        parameter_names=("beta",),
        groups=tuple(groups),
    )


def _study() -> Study:
    rows = 24
    subjects = ["m1"] * (rows // 2) + ["m2"] * (rows // 2)
    sessions = []
    orders = []
    trials = []
    for _ in range(2):
        for session in range(1, 4):
            for trial in range(1, 5):
                sessions.append(f"s{session}")
                orders.append(session)
                trials.append(trial)
    generator = np.random.default_rng(3)
    study = Study(
        {
            "subject": subjects,
            "session": sessions,
            "session_order": orders,
            "trial": trials,
            "choice": generator.integers(0, 2, size=rows),
            "stimulus": generator.normal(size=rows),
        }
    )
    TaskSpec(choice=ChoiceSpec(options=(0, 1))).validate(study)
    return study


# --------------------------------------------------------------------------------------
# Derived quantities, the natural parameterisation, and the multistart contract.
#
# These three widenings share one rule: a model that declares nothing behaves exactly as it
# did before they existed. Every test below either exercises a declaration or asserts that
# absence of one is inert.
# --------------------------------------------------------------------------------------


def test_a_fit_declaring_no_derived_quantities_is_unchanged() -> None:
    fit = _mle_fit()

    assert fit.derived == ()
    assert dict(fit.derived_values) == {}
    assert dict(fit.derived_quantities) == {}
    assert dict(fit.parameters) == {"a": 0.5, "b": -0.25}
    assert fit.audit().status is FitAuditStatus.PASS
    with pytest.raises(KeyError, match="declares no derived quantity"):
        fit.derived_value("threshold")


def test_a_derived_quantity_carries_a_value_and_an_optional_uncertainty() -> None:
    bare = contracts.DerivedQuantity("m_ratio", 0.8)
    full = contracts.DerivedQuantity(
        "threshold",
        -4.0,
        standard_error=0.5,
        interval=(-5.0, -3.0),
        interval_level=0.95,
        description="stimulus level at the curve midpoint",
    )

    assert bare.standard_error is None and bare.interval is None
    assert full.to_dict()["interval"] == [-5.0, -3.0]
    assert bare.to_dict()["interval_level"] is None
    with pytest.raises(ValueError, match="requires a level"):
        contracts.DerivedQuantity("threshold", 1.0, interval=(0.0, 2.0))
    with pytest.raises(ValueError, match="interval_level requires an interval"):
        contracts.DerivedQuantity("threshold", 1.0, interval_level=0.95)
    with pytest.raises(ValueError, match="bounds must be ordered"):
        contracts.DerivedQuantity("threshold", 1.0, interval=(2.0, 0.0), interval_level=0.95)
    with pytest.raises(ValueError, match="non-negative"):
        contracts.DerivedQuantity("threshold", 1.0, standard_error=-0.1)


def test_derived_quantity_names_must_be_unique_within_one_fit() -> None:
    with pytest.raises(ValueError, match="unique within one fit"):
        FitResult(
            model_name="demo",
            model_signature="demo[v1]",
            parameter_names=("a",),
            estimates=np.array([1.0]),
            standard_errors=np.array([0.1]),
            covariance=np.eye(1),
            n_observations=8,
            diagnostics=_mle_fit().diagnostics,
            derived=(
                contracts.DerivedQuantity("threshold", 1.0),
                contracts.DerivedQuantity("threshold", 2.0),
            ),
        )


def test_derived_quantities_reach_a_consumer_typed_on_plain_fit_result() -> None:
    from behavio.models import PsychometricFunction

    model = PsychometricFunction(stimulus="stimulus", outcome="choice")
    fit: FitResult = model.fit(_psychometric_study())

    assert set(fit.derived_values) >= {"threshold", "width", "guess_rate", "lapse_rate"}
    threshold = fit.derived_quantities["threshold"]
    assert threshold.interval_level == 0.95
    assert threshold.interval is not None and threshold.interval[0] <= threshold.value
    # The estimated coordinate is unchanged: derived quantities are additional, never a
    # replacement for the numbers the optimizer actually searched.
    assert fit.parameter_names == ("threshold", "log_width", "guess_logit", "lapse_logit")


def test_the_natural_contract_is_optional_and_a_declining_model_is_inert() -> None:
    plain = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)

    assert not isinstance(plain, contracts.NaturalParameterisation)
    with pytest.raises(TypeError, match="NaturalParameterisation"):
        contracts.natural_quantities(plain, np.zeros(2), np.eye(2))  # type: ignore[arg-type]


def test_the_natural_jacobian_propagates_uncertainty_off_the_optimizer_scale() -> None:
    from behavio.models import UnequalVarianceSDT

    model = UnequalVarianceSDT()
    assert isinstance(model, contracts.NaturalParameterisation)
    coordinate = np.asarray(
        list(
            model.parameters_from_components(
                signal_mean=1.2, signal_sd=1.3, criteria=[-1.0, -0.3, 0.2, 0.9, 1.5]
            ).values()
        )
    )

    natural = model.to_natural(coordinate)
    assert natural["signal_sd"] == pytest.approx(1.3)
    assert natural["criterion_3"] == pytest.approx(0.2)
    # A round trip returns the coordinate the optimizer searched.
    encoded = model.from_natural(natural)
    assert np.allclose(list(encoded.values()), coordinate)

    analytic = model.natural_jacobian(coordinate)
    numeric = np.zeros_like(analytic)
    step = 1e-6
    for column in range(coordinate.size):
        shifted = coordinate.copy()
        shifted[column] += step
        forward = model.to_natural(shifted)
        shifted[column] -= 2 * step
        backward = model.to_natural(shifted)
        numeric[:, column] = [
            (forward[name] - backward[name]) / (2 * step) for name in model.natural_names
        ]
    assert np.allclose(analytic, numeric, atol=1e-5)

    covariance = np.eye(coordinate.size) * 0.04
    quantities = contracts.natural_quantities(model, coordinate, covariance)
    assert tuple(item.name for item in quantities) == model.natural_names
    assert all(item.standard_error is not None and item.interval is None for item in quantities)

    # The propagated covariance is a full matrix, not a diagonal: the cumulative-sum
    # construction correlates neighbouring criteria even from independent gaps.
    propagated = contracts.natural_covariance(model, coordinate, covariance)
    assert propagated.shape == (len(model.natural_names), len(model.natural_names))
    assert np.allclose(propagated, propagated.T)
    assert propagated[3, 4] != 0.0
    assert np.allclose([item.standard_error for item in quantities], np.sqrt(np.diag(propagated)))


def test_the_multistart_contract_is_what_produces_a_restart_audit() -> None:
    from behavio.models import PsychometricFunction

    fit = PsychometricFunction(stimulus="stimulus", outcome="choice").fit(_psychometric_study())

    assert isinstance(fit, contracts.MultistartFit)
    assert not isinstance(_mle_fit(), contracts.MultistartFit)
    audit = contracts.RestartAudit.from_fit(fit)
    assert audit is not None
    assert audit.n_restarts == len(fit.restart_objectives)
    assert audit.selected_restart == fit.selected_restart
    assert contracts.RestartAudit.from_fit(_mle_fit()) is None
    assert contracts.LatentStateAudit.from_fit(fit) is None
    # The public constructor is exactly what ``audit_fit`` now dispatches to.
    assert fit.audit().restarts == audit


def test_a_closed_form_estimator_records_a_procedure_not_an_optimizer() -> None:
    """A procedure that solves rather than searches leaves every search field absent.

    ``converged=True, status=0`` used to be written here, and the comment that defended it
    said the quiet part: it was a claim made *so that the audit would not complain*, not a
    measurement. Both fields are now ``None``, which is what "there was nothing to
    converge" actually looks like in a record.
    """

    diagnostics = contracts.FitDiagnostics.closed_form(
        procedure="closed-form z-transform",
        message="two-by-two table",
        objective=12.0,
        boundary_estimate=False,
    )

    assert diagnostics.optimizer == "closed-form z-transform"
    assert diagnostics.n_iterations is None
    assert diagnostics.gradient_norm is None
    assert diagnostics.hessian_condition is None
    assert diagnostics.converged is None
    assert diagnostics.status is None
    assert diagnostics.convergence is contracts.ConvergenceStatus.INAPPLICABLE
    assert not diagnostics.failed_to_converge
    with pytest.raises(ValueError, match="non-empty name"):
        contracts.FitDiagnostics.closed_form(procedure="", message="x")


def test_the_audit_separates_non_convergence_from_an_inapplicable_question() -> None:
    """Three fits, three different answers, and only one of them is an issue."""

    searched = _mle_fit()
    failed = replace(
        searched,
        diagnostics=replace(searched.diagnostics, converged=False, status=1),
    )
    solved = replace(
        searched,
        diagnostics=contracts.FitDiagnostics.closed_form(
            procedure="closed-form z-transform", message="two-by-two table", objective=1.0
        ),
    )

    assert searched.audit().convergence is contracts.ConvergenceStatus.CONVERGED
    assert failed.audit().convergence is contracts.ConvergenceStatus.NOT_CONVERGED
    assert solved.audit().convergence is contracts.ConvergenceStatus.INAPPLICABLE

    assert "optimizer_nonconvergence" not in searched.audit().issue_codes
    assert "optimizer_nonconvergence" in failed.audit().issue_codes
    # The distinction that used to be impossible: an absent flag is not a failed one.
    assert "optimizer_nonconvergence" not in solved.audit().issue_codes
    assert failed.audit().status is contracts.FitAuditStatus.FAIL
    assert solved.audit().status is contracts.FitAuditStatus.PASS
    assert solved.audit().to_dict()["convergence"] == "inapplicable"


def test_a_fitter_that_searched_and_said_nothing_is_neither_a_success_nor_a_failure() -> None:
    """The fourth value, and the three claims it replaces.

    A wrapper around a third-party fitter whose stopping rule is private lands here every
    time. ``True`` would claim a success nobody measured; ``False`` would condemn the fit
    to an audit ``FAIL`` and evict it from every comparison; ``None`` would assert that no
    search happened, which is the one thing that is certainly false.
    """

    searched = _mle_fit()
    silent = replace(
        searched,
        diagnostics=replace(
            searched.diagnostics,
            converged=contracts.ConvergenceStatus.UNREPORTED,
            status=None,
            message="the fitter reported no convergence flag",
        ),
    )

    diagnostics = silent.diagnostics
    assert diagnostics.convergence is contracts.ConvergenceStatus.UNREPORTED
    # Not a failure: the question every consumer actually asks answers "no".
    assert not diagnostics.failed_to_converge

    audit = silent.audit()
    assert audit.convergence is contracts.ConvergenceStatus.UNREPORTED
    assert audit.status is contracts.FitAuditStatus.WARNING
    assert "optimizer_convergence_unreported" in audit.issue_codes
    assert "optimizer_nonconvergence" not in audit.issue_codes
    assert audit.to_dict()["convergence"] == "unreported"
    assert audit.to_dict()["numerical"]["converged"] == "unreported"


def test_the_fourth_value_is_the_only_status_a_fit_may_spell_as_an_enum() -> None:
    """One state, one spelling. The other three are written True, False and None."""

    for status in (
        contracts.ConvergenceStatus.CONVERGED,
        contracts.ConvergenceStatus.NOT_CONVERGED,
        contracts.ConvergenceStatus.INAPPLICABLE,
    ):
        with pytest.raises(ValueError, match="only ConvergenceStatus a fit records"):
            _diagnostics(converged=status, status=None)
    with pytest.raises(ValueError, match="reported no convergence verdict"):
        _diagnostics(converged=contracts.ConvergenceStatus.UNREPORTED, status=0)
    with pytest.raises(ValueError, match="boolean, None, or ConvergenceStatus"):
        _diagnostics(converged="maybe", status=None)


def _diagnostics(*, converged: Any, status: int | None) -> contracts.FitDiagnostics:
    return contracts.FitDiagnostics(
        converged=converged,
        optimizer="foreign",
        status=status,
        message="x",
        n_iterations=None,
        objective=None,
        gradient_norm=None,
        hessian_condition=None,
        boundary_estimate=None,
    )


def test_a_procedure_with_no_convergence_question_cannot_report_a_status() -> None:
    with pytest.raises(ValueError, match="convergence status"):
        contracts.FitDiagnostics(
            converged=None,
            optimizer="closed-form z-transform",
            status=0,
            message="x",
            n_iterations=None,
            objective=None,
            gradient_norm=None,
            hessian_condition=None,
            boundary_estimate=None,
        )
    with pytest.raises(ValueError, match="name the procedure"):
        contracts.FitDiagnostics(
            converged=True,
            optimizer="",
            status=0,
            message="x",
            n_iterations=None,
            objective=None,
            gradient_norm=None,
            hessian_condition=None,
            boundary_estimate=None,
        )


def test_required_task_columns_is_part_of_the_contract_and_is_surfaced() -> None:
    from behavio.models import EqualVarianceSDT

    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)
    assert isinstance(model, contracts.BehaviourEstimator)
    assert contracts.model_capabilities(model).required_task_columns == ("stimulus",)
    assert contracts.model_task_columns(EqualVarianceSDT()) == ("signal",)

    class _Silent:
        """An estimator whose likelihood reads nothing but the column it scores."""

        model_name = "silent"
        signature = "silent[v1]"
        scored_columns = ("choice",)
        required_task_columns = ()
        supported_prediction_modes = (PredictionMode.FILTERED,)

        def fit(self, study: Study) -> FitResult:  # pragma: no cover - never called
            raise NotImplementedError

        def predict(self, study: Study, fit: FitResult, *, mode: Any = None) -> Any:
            raise NotImplementedError  # pragma: no cover - never called

        def pointwise_log_prob(self, study: Study, fit: FitResult, *, mode: Any = None) -> Any:
            raise NotImplementedError  # pragma: no cover - never called

    silent = _Silent()
    assert isinstance(silent, contracts.BehaviourEstimator)
    assert contracts.model_capabilities(silent).required_task_columns == ()
    with pytest.raises(ValueError, match="tuple of column names"):
        contracts.validate_required_task_columns("stimulus")
    with pytest.raises(ValueError, match="must be unique"):
        contracts.validate_required_task_columns(("stimulus", "stimulus"))
    with pytest.raises(ValueError, match="repeat a scored column"):
        contracts.ModelCapabilities(
            scored_columns=("choice",),
            prediction_modes=(PredictionMode.FILTERED,),
            can_simulate=False,
            can_recover_parameters=False,
            required_task_columns=("choice",),
        )


def _psychometric_study() -> Study:
    levels = np.linspace(-2.0, 2.0, 9)
    stimulus = np.repeat(levels, 12)
    generator = np.random.default_rng(4)
    probability = 0.05 + 0.9 / (1.0 + np.exp(-(stimulus - 0.2) / 0.5))
    rows = stimulus.size
    return Study(
        {
            "subject": ["m1"] * rows,
            "session": ["s1"] * rows,
            "session_order": [1] * rows,
            "trial": list(range(1, rows + 1)),
            "stimulus": stimulus,
            "choice": generator.binomial(1, probability).astype(np.int8),
        }
    )


def _joint_prediction() -> contracts.CategoricalPrediction:
    probability = np.asarray([[0.4, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
    return contracts.CategoricalPrediction(
        probability=probability,
        linear_predictor=np.log(probability),
        categories=(("no", 2), ("no", 1), ("yes", 1), ("yes", 2)),
        mode=PredictionMode.FILTERED,
        category_factors=("response", "confidence"),
    )


def test_a_composite_category_names_a_joint_cell_and_can_be_marginalised() -> None:
    """Tuple categories replace the stringly typed joint labels a model used to emit.

    ``_prediction_category`` admitted only scalars, so a model scoring a joint observation
    had to flatten a cell into a string like ``"no-2"`` and every caller wanting one
    margin had to parse it back apart. ``scored_columns`` already said the observation was
    joint; only the labels were not.
    """

    prediction = _joint_prediction()
    response = prediction.marginal("response")

    assert prediction.is_composite
    assert prediction.categories[0] == ("no", 2)
    assert prediction.factor_levels("response") == ("no", "yes")
    assert prediction.factor_levels("confidence") == (2, 1)
    assert response.categories == ("no", "yes")
    assert response.category_factors is None
    assert np.allclose(response.probability, [[0.5, 0.5], [0.3, 0.7]])
    assert np.allclose(response.linear_predictor, np.log(response.probability))
    assert np.allclose(prediction.marginal("confidence").probability, [[0.7, 0.3], [0.5, 0.5]])
    # A row subset keeps the factorisation, or the margin would be lost on every fold.
    assert prediction.take([1]).category_factors == ("response", "confidence")
    assert prediction.take([1]).categories == prediction.categories


def test_a_composite_category_must_be_labelled_and_uniformly_shaped() -> None:
    """An unlabelled tuple is no more readable than the string it replaces."""

    probability = np.asarray([[0.5, 0.5]])
    predictor = np.log(probability)

    def build(categories: Any, factors: Any = None) -> contracts.CategoricalPrediction:
        return contracts.CategoricalPrediction(
            probability=probability,
            linear_predictor=predictor,
            categories=categories,
            mode=PredictionMode.FILTERED,
            category_factors=factors,
        )

    with pytest.raises(ValueError, match="require declared category_factors"):
        build((("no", 1), ("yes", 1)))
    with pytest.raises(ValueError, match="all scalar or all composite"):
        build((("no", 1), "yes"), ("response", "confidence"))
    with pytest.raises(ValueError, match="same factors"):
        build((("no", 1), ("yes", 1, 2)), ("response", "confidence"))
    with pytest.raises(ValueError, match="every position"):
        build((("no", 1), ("yes", 1)), ("response",))
    with pytest.raises(ValueError, match="must be unique"):
        build((("no", 1), ("yes", 1)), ("response", "response"))
    with pytest.raises(ValueError, match="requires composite"):
        build(("no", "yes"), ("response",))
    with pytest.raises(ValueError, match="at least one factor"):
        build(((), ("yes", 1)), ("response", "confidence"))
    with pytest.raises(ValueError, match="finite scalar"):
        build((("no", float("nan")), ("yes", 1)), ("response", "confidence"))


def test_composite_categories_keep_the_scalar_identity_rule_inside_a_tuple() -> None:
    """``0`` and ``False`` stay distinct at every depth, as they already did at the top.

    Uniqueness has always been decided on ``(type, value)`` rather than on ``==``, so a
    scalar ``0`` and a scalar ``False`` are two categories. Hashing a tuple directly would
    silently drop that rule one level down, because ``(0, 1) == (False, 1)`` and the two
    hash alike; the check therefore recurses into the tuple.
    """

    probability = np.asarray([[0.5, 0.5]])
    distinct = contracts.CategoricalPrediction(
        probability=probability,
        linear_predictor=np.log(probability),
        categories=((0, 1), (False, 1)),
        mode=PredictionMode.FILTERED,
        category_factors=("response", "confidence"),
    )

    assert distinct.factor_levels("response") == (0, False)
    with pytest.raises(ValueError, match="categories must be unique"):
        contracts.CategoricalPrediction(
            probability=probability,
            linear_predictor=np.log(probability),
            categories=((0, 1), (0, 1)),
            mode=PredictionMode.FILTERED,
            category_factors=("response", "confidence"),
        )
