"""Structural checks for the versioned documentation figures."""

import json
import re
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "docs" / "assets"
EXPECTED_FIGURES = {
    "behavior-ecosystem-v0.1.svg",
    "interval-policy-pipeline-v0.1.svg",
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
    "EqualVarianceSDT",
    "HierarchicalBernoulliHistoryGLM",
    "HierarchicalSmoothBernoulliHistoryGLM",
    "HierarchicalSmoothWienerDriftDiffusion",
    "LapsePsychometric",
    "MetaSDT",
    "MultinomialLogit",
    "Perseveration",
    "Psychometric",
    "PsychometricFunction",
    "SmoothBernoulliHistoryGLM",
    "SmoothWienerDriftDiffusion",
    "UnequalVarianceSDT",
    "WienerDriftDiffusion",
    "WinStayLoseShift",
}
EVIDENCE_LADDER = (
    "published-parity",
    "independent-analysis",
    "released-replay",
    "literature-shaped",
    "demonstration",
)
RETAINED_OUTCOMES = ("failed-parity",)
NON_COMPARATIVE_CLASSES = ("mixed-evidence", "synthetic-benchmark", "conceptual")
FIGURE_CLASSES = {
    "published-parity": "Published parity",
    "independent-analysis": "Independent analysis",
    "released-replay": "Released replay",
    "literature-shaped": "Literature-shaped",
    "demonstration": "Demonstration",
    "failed-parity": "Failed parity",
    "mixed-evidence": "Mixed evidence",
    "synthetic-benchmark": "Synthetic benchmark",
    "conceptual": "Conceptual",
}
MANIFEST_FIELDS = {
    "classification",
    "generator",
    "source",
    "unit",
    "denominator",
    "estimand",
    "claim",
    "published_target",
}


def _figure_manifest_payload() -> dict[str, object]:
    path = ROOT / "docs" / "reference" / "figure-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    return payload


def _figure_manifest() -> dict[str, dict[str, str]]:
    return _figure_manifest_payload()["figures"]


def test_evidence_ladder_ranks_published_parity_above_released_replay() -> None:
    """A replay of released artifacts is a lower evidential bar than a published comparison."""

    payload = _figure_manifest_payload()
    ladder = payload["evidence_ladder"]

    assert tuple(ladder) == EVIDENCE_LADDER
    assert ladder.index("published-parity") < ladder.index("independent-analysis")
    assert ladder.index("independent-analysis") < ladder.index("released-replay")
    assert ladder.index("released-replay") < ladder.index("literature-shaped")
    assert tuple(payload["retained_outcomes"]) == RETAINED_OUTCOMES
    assert tuple(payload["non_comparative_classifications"]) == NON_COMPARATIVE_CLASSES
    assert set(ladder) | set(RETAINED_OUTCOMES) | set(NON_COMPARATIVE_CLASSES) == set(
        FIGURE_CLASSES
    )


def test_documentation_figures_are_versioned_valid_svgs() -> None:
    actual = {path.name for path in ASSETS.glob("*.svg") if not path.stem.endswith(" 2")}

    assert actual == EXPECTED_FIGURES
    for figure in (ASSETS / name for name in sorted(EXPECTED_FIGURES)):
        root = ElementTree.parse(figure).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert figure.stat().st_size > 1_000


def test_documentation_figures_use_searchable_sans_serif_text() -> None:
    """Keep scientific typography modern, legible, and accessible in SVG output."""

    for figure in (ASSETS / name for name in sorted(EXPECTED_FIGURES)):
        root = ElementTree.parse(figure).getroot()
        text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
        assert text_nodes, f"{figure.name} converted all text to paths"
        for node in text_nodes:
            style = node.attrib.get("style", "")
            descendant_styles = " ".join(child.attrib.get("style", "") for child in node.iter())
            families = re.findall(r"font-family:\s*([^;]+)", descendant_styles)
            assert families, f"{figure.name} has text without an explicit font family"
            assert all("sans" in family.lower() for family in families), (
                f"{figure.name} uses non-sans typography: {families}"
            )
            size = re.search(r"font-size:\s*([0-9.]+)px", style)
            if size:
                assert float(size.group(1)) >= 7.0, (
                    f"{figure.name} has base text below 7 pt: {size.group(1)}"
                )


def test_figure_manifest_is_complete_and_typed() -> None:
    manifest = _figure_manifest()

    assert set(manifest) == EXPECTED_FIGURES
    for name, record in manifest.items():
        assert set(record) == MANIFEST_FIELDS, name
        assert record["classification"] in FIGURE_CLASSES, name
        assert all(isinstance(value, str) and value.strip() for value in record.values()), name


def test_rendered_figure_cards_match_the_manifest() -> None:
    """Require visible classification, alternative text, and metadata-bearing captions."""

    manifest = _figure_manifest()
    documents = sorted((ROOT / "docs").rglob("*.md"))
    documents = [
        path
        for path in documents
        if not path.name.endswith(" 2.md") and path.name != "figure-standard.md"
    ]
    blocks: list[tuple[Path, str, str]] = []
    for document in documents:
        source = document.read_text(encoding="utf-8")
        for match in re.finditer(r"<figure\b(?P<attrs>.*?)>(?P<body>.*?)</figure>", source, re.S):
            if "doc-figure" in match.group("attrs"):
                blocks.append((document, match.group("attrs"), match.group("body")))

    assert blocks
    displayed: set[str] = set()
    for document, attrs, body in blocks:
        kind = re.search(r'data-figure-kind="([^"]+)"', attrs)
        assert kind, f"{document} has an unclassified figure"
        image = re.search(r'<img\b[^>]*src="[^"]*/([^/"]+\.svg)"[^>]*>', body, re.S)
        assert image, f"{document} has a figure without an SVG image"
        name = image.group(1)
        displayed.add(name)
        assert name in manifest, f"{document} displays unregistered {name}"
        expected_kind = FIGURE_CLASSES[manifest[name]["classification"]]
        assert kind.group(1) == expected_kind, f"{document}: {name} classification mismatch"
        alt = re.search(r'<img\b[^>]*alt="([^"]+)"', body, re.S)
        assert alt and len(alt.group(1).strip()) >= 40, f"{document}: {name} needs useful alt text"
        caption = re.search(r"<figcaption>(.*?)</figcaption>", body, re.S)
        assert caption, f"{document}: {name} needs a caption"
        if manifest[name]["classification"] != "conceptual":
            assert expected_kind.lower() in caption.group(1).lower(), (
                f"{document}: {name} caption must name its evidence class"
            )
            assert 'class="doc-figure__meta"' in caption.group(1), (
                f"{document}: {name} needs visible unit and estimand metadata"
            )

    assert displayed <= EXPECTED_FIGURES


def test_figure_provenance_register_covers_every_figure() -> None:
    register = (ROOT / "docs" / "reference" / "figure-provenance.md").read_text()

    for name in EXPECTED_FIGURES:
        assert f"`{name}`" in register


def test_cell_figure1gi_asset_and_chapter_name_both_published_panels() -> None:
    chapter = (ROOT / "docs" / "tutorials" / "cell2025-figure1gi-reproduction.md").read_text(
        encoding="utf-8"
    )
    root = ElementTree.parse(ASSETS / "cell2025-strategy.svg").getroot()
    figure_text = " ".join("".join(node.itertext()) for node in root.iter())
    text_labels = {
        "".join(node.itertext()).strip()
        for node in root.findall(".//{http://www.w3.org/2000/svg}text")
    }

    assert "Figure 1G and 1I" in chapter
    assert "final-five-**paper-day** window" in chapter
    assert 'class="doc-figure__full-resolution"' in chapter
    assert "Published parity with Cell Figure 1G and 1I" in figure_text
    assert {"G", "I"} <= text_labels
    assert "Early bias (days 4-8)" in figure_text
    assert "DejaVu Sans" in (ASSETS / "cell2025-strategy.svg").read_text(encoding="utf-8")


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
