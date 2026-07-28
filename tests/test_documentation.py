"""Structural checks for the versioned documentation figures."""

from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "docs" / "assets"
EXPECTED_FIGURES = {
    "cell2025-strategy.svg",
    "cell2025-forecast.svg",
    "cell2025-qvalue-response-time.svg",
    "cell2025-recovery.svg",
    "cell2025-trajectories.svg",
    "chen2021-bandit.svg",
    "choice-model-evidence-atlas.svg",
    "clock-boundary.svg",
    "ddm-recovery.svg",
    "diagnostic-layers.svg",
    "first-analysis.svg",
    "hierarchical-pooling.svg",
    "ibl-learning-trajectories.svg",
    "ibl-choice-response-time.svg",
    "ibl-glmhmm-states.svg",
    "ibl-prospective-selection.svg",
    "interoperability-pipeline.svg",
    "model-atlas.svg",
    "model-choice-workflow.svg",
    "model-recovery-matrix.svg",
    "nested-selection.svg",
    "reliability-agreement.svg",
    "sbc-workflow.svg",
    "sensitivity-specification.svg",
    "trajectory-components.svg",
    "validation-geometry.svg",
    "validation-splits.svg",
    "workflow-map.svg",
}
MODEL_CARD_CLASSES = {
    "BernoulliGLMHMM",
    "BernoulliHistoryGLM",
    "BiasOnly",
    "BinaryQLearning",
    "BinaryRLAgent",
    "HierarchicalBernoulliHistoryGLM",
    "HierarchicalSmoothBernoulliHistoryGLM",
    "HierarchicalSmoothWienerDriftDiffusion",
    "LapsePsychometric",
    "MultinomialLogit",
    "Perseveration",
    "Psychometric",
    "SmoothBernoulliHistoryGLM",
    "SmoothWienerDriftDiffusion",
    "WienerDriftDiffusion",
    "WinStayLoseShift",
}


def test_documentation_figures_are_versioned_valid_svgs() -> None:
    actual = {path.name for path in ASSETS.glob("*.svg")}

    assert actual == EXPECTED_FIGURES
    for figure in sorted(ASSETS.glob("*.svg")):
        root = ElementTree.parse(figure).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert figure.stat().st_size > 1_000


def test_figure_provenance_register_covers_every_figure() -> None:
    register = (ROOT / "docs" / "reference" / "figure-provenance.md").read_text()

    for name in EXPECTED_FIGURES:
        assert f"`{name}`" in register


def test_model_cards_cover_every_first_party_family() -> None:
    cards = (ROOT / "docs" / "model-cards.md").read_text()

    for name in MODEL_CARD_CLASSES:
        assert f"`{name}`" in cards


def test_first_analysis_is_executable(tmp_path: Path) -> None:
    """Keep the visible quickstart on the tested public API when plotting is installed."""

    import runpy
    import sys

    import pytest

    pytest.importorskip("matplotlib")

    example = ROOT / "examples" / "first_analysis.py"
    output = tmp_path / "first-analysis.svg"
    original_argv = sys.argv
    try:
        sys.argv = [str(example), str(output)]
        runpy.run_path(str(example), run_name="__main__")
    finally:
        sys.argv = original_argv

    assert output.is_file()
    assert output.stat().st_size > 1_000
