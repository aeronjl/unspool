"""Numerical and documentation contract for the Cell 2025 Figure 1H/1J replay."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "cell2025_flagship"
ASSET = ROOT / "docs" / "assets" / "cell2025-trajectories.svg"


def _load(name: str) -> dict[str, object]:
    return json.loads((BENCHMARK / name).read_text(encoding="utf-8"))


def test_figure1hj_audit_freezes_distinct_panel_contracts() -> None:
    audit = _load("figure1hj_audit.json")

    assert audit["schema_version"] == 1
    assert audit["paper"]["pdf_page"] == 3
    assert audit["released_analysis"]["commit"] == ("2faa4680d5e9c0d6a9df516e3dede8c641e39a72")
    assert set(audit["panels"]) == {"1H", "1J"}
    assert audit["panels"]["1H"]["x"] == "Released interpolated session coordinate"
    assert audit["panels"]["1J"]["x"] == ("Released Gaussian-process right psychometric slope")
    assert audit["panels"]["1J"]["x_limits"] == [-0.35, 1.02]
    assert "not independently refit" in audit["unspool_display"]["claim_boundary"]


def test_exact_stack_replay_matches_all_released_memberships() -> None:
    audit = _load("figure1hj_audit.json")
    replay = _load("figure1hj_trajectories.json")

    assert replay["schema_version"] == 1
    assert (
        replay["sources"]["trajectory_sha256"]
        == audit["released_analysis"]["trajectory_source_sha256"]
    )
    assert (
        replay["sources"]["cluster_sha256"] == audit["released_analysis"]["cluster_source_sha256"]
    )
    assert replay["environment"] == {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "scikit_learn": "1.5.1",
        "scipy": "1.13.1",
        "tslearn": "0.6.3",
    }
    assert replay["released_membership_validation"] == {
        "exact_semantic_membership_match": True,
        "n_subjects": 30,
    }
    assert len(replay["subject_order"]) == len(set(replay["subject_order"])) == 30
    assert {key: len(value) for key, value in replay["memberships"].items()} == {
        "left": 9,
        "balanced": 10,
        "right": 11,
    }
    assert set(replay["semantic_label_by_subject"]) == set(replay["subject_order"])


def test_replayed_centroids_have_frozen_shapes_and_numerical_anchors() -> None:
    replay = _load("figure1hj_trajectories.json")

    for family in ("slope_difference_centers", "right_left_centers"):
        assert set(replay[family]) == {"left", "balanced", "right"}
        for center in replay[family].values():
            values = np.asarray(center)
            assert values.shape == (100, 2)
            assert np.isfinite(values).all()

    expected_figure1j_endpoints = {
        "left": (-0.08408902806625532, 0.5377240888885612),
        "balanced": (0.20402849667932665, 0.32377348389015653),
        "right": (0.5119504183881796, -0.04829796991837141),
    }
    for semantic, expected in expected_figure1j_endpoints.items():
        assert np.allclose(
            replay["right_left_centers"][semantic][-1],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_figure1hj_asset_and_chapter_publish_both_panels() -> None:
    chapter = (ROOT / "docs" / "tutorials" / "cell2025-figure1hj-reproduction.md").read_text(
        encoding="utf-8"
    )
    root = ElementTree.parse(ASSET).getroot()
    figure_text = " ".join("".join(node.itertext()) for node in root.iter())
    text_labels = {
        "".join(node.itertext()).strip()
        for node in root.findall(".//{http://www.w3.org/2000/svg}text")
    }

    assert "Figure 1H and 1J" in chapter
    assert "not** Figure 1J centroids" in chapter
    assert 'class="doc-figure__full-resolution"' in chapter
    assert "Exact replay of released Cell Figure 1H and 1J" in figure_text
    assert {"H", "J"} <= text_labels
    assert "R-L psychometric slope" in figure_text
    assert "Right psychometric slope" in figure_text
    assert "DejaVu Sans" in ASSET.read_text(encoding="utf-8")
