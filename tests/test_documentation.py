"""Structural checks for the versioned documentation figures."""

from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "docs" / "assets"
EXPECTED_FIGURES = {
    "cell2025-strategy.svg",
    "choice-model-evidence-atlas.svg",
    "clock-boundary.svg",
    "ddm-recovery.svg",
    "diagnostic-layers.svg",
    "hierarchical-pooling.svg",
    "ibl-learning-trajectories.svg",
    "ibl-choice-response-time.svg",
    "ibl-glmhmm-states.svg",
    "ibl-prospective-selection.svg",
    "interoperability-pipeline.svg",
    "model-atlas.svg",
    "model-recovery-matrix.svg",
    "nested-selection.svg",
    "trajectory-components.svg",
    "validation-geometry.svg",
    "validation-splits.svg",
    "workflow-map.svg",
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
