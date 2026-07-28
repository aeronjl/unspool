"""Public-surface regression for the completed 0.21--0.24 package boundary."""

from __future__ import annotations

from pathlib import Path

import unspool

ROOT = Path(__file__).parents[1]

GOLDEN_PATH = {
    "Study",
    "ChoiceSpec",
    "RewardSpec",
    "ResponseTimeSpec",
    "TaskSpec",
    "fit_model",
    "compare_models",
    "cohort_forward_session_splits",
    "nested_select_model",
    "run_parameter_recovery",
    "run_model_recovery",
    "StudyProtocol",
    "compile_execution_plan",
    "run_protocol",
    "build_evidence_bundle",
    "EstimatorRegistry",
    "export_fit",
}

MODEL_CATALOGUE = {
    "BiasOnly",
    "Psychometric",
    "LapsePsychometric",
    "Perseveration",
    "WinStayLoseShift",
    "BernoulliHistoryGLM",
    "SmoothBernoulliHistoryGLM",
    "HierarchicalBernoulliHistoryGLM",
    "HierarchicalSmoothBernoulliHistoryGLM",
    "BernoulliGLMHMM",
    "BinaryQLearning",
    "BinaryRLAgent",
    "MultinomialLogit",
    "WienerDriftDiffusion",
    "SmoothWienerDriftDiffusion",
    "HierarchicalSmoothWienerDriftDiffusion",
}

INTEROPERABILITY_AND_EVIDENCE = {
    "ParameterSpace",
    "ScipyMultistart",
    "PyBADSMultistart",
    "PosteriorResult",
    "PyMCHierarchicalGLMBackend",
    "audit_posterior",
    "posterior_predictive_check",
    "psis_loo",
    "run_simulation_based_calibration",
    "run_sensitivity_analysis",
    "assess_test_retest_reliability",
    "generate_bounded_report",
    "write_evidence_bundle",
}


def test_completed_release_surface_is_public_and_explicit() -> None:
    promised = GOLDEN_PATH | MODEL_CATALOGUE | INTEROPERABILITY_AND_EVIDENCE

    assert promised <= set(unspool.__all__)
    assert all(hasattr(unspool, name) for name in promised)


def test_release_orientation_documents_are_in_the_strict_site_navigation() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    promised_pages = (
        "getting-started/index.md",
        "getting-started/installation.md",
        "getting-started/first-analysis.md",
        "model-choice-guide.md",
        "model-cards.md",
        "tutorials/recipe-contract.md",
        "tutorials/cell2025-figure1gi-reproduction.md",
        "tutorials/cell2025-figure1hj-reproduction.md",
        "tutorials/chen2021-bandit.md",
        "migration-guides.md",
        "extensions.md",
        "reference/validation-and-comparison.md",
        "reference/figure-standard.md",
        "roadmap.md",
    )

    for page in promised_pages:
        assert page in navigation
