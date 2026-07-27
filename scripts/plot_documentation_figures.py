"""Generate conceptual diagrams and figures from frozen benchmark evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CELL_DATA = (
    ROOT
    / "benchmarks"
    / "cell2025"
    / "data"
    / "long_term_learning_dataset_preprocessed_behaviour_all.csv"
)
DEFAULT_OUTPUT = ROOT / "docs" / "assets"
INDIGO = "#26345e"
BLUE = "#4f6d9a"
AMBER = "#c57928"
TEAL = "#2d7f78"
INK = "#202534"
MUTED = "#778092"
LIGHT = "#e8ebf2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-data", type=Path, default=DEFAULT_CELL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-cell",
        action="store_true",
        help="generate figures that use committed JSON without the downloaded Cell table",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    _plot_validation_geometry(args.output_dir / "validation-geometry.svg")
    _plot_workflow_map(args.output_dir / "workflow-map.svg")
    _plot_clock_boundary(args.output_dir / "clock-boundary.svg")
    _plot_validation_splits(args.output_dir / "validation-splits.svg")
    _plot_model_atlas(args.output_dir / "model-atlas.svg")
    _plot_nested_selection(args.output_dir / "nested-selection.svg")
    _plot_diagnostic_layers(args.output_dir / "diagnostic-layers.svg")
    _plot_interoperability(args.output_dir / "interoperability-pipeline.svg")
    _plot_hierarchical_pooling(args.output_dir / "hierarchical-pooling.svg")
    _plot_ddm_recovery(args.output_dir / "ddm-recovery.svg")
    _plot_trajectory_components(args.output_dir / "trajectory-components.svg")
    _plot_choice_model_evidence_atlas(args.output_dir / "choice-model-evidence-atlas.svg")
    _plot_ibl_choice_rt(args.output_dir / "ibl-choice-response-time.svg")
    _plot_ibl_glm_hmm(args.output_dir / "ibl-glmhmm-states.svg")
    if not args.skip_cell:
        if not args.cell_data.exists():
            raise FileNotFoundError(
                f"Cell table not found at {args.cell_data}; run "
                "`uv run python -m benchmarks.cell2025.fetch_data` or pass --skip-cell"
            )
        _plot_cell_strategy(args.cell_data, args.output_dir / "cell2025-strategy.svg")
    _plot_ibl_trajectories(args.output_dir / "ibl-learning-trajectories.svg")
    _plot_ibl_selection(args.output_dir / "ibl-prospective-selection.svg")
    _plot_recovery_matrix(args.output_dir / "model-recovery-matrix.svg")
    print(args.output_dir)


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "axes.edgecolor": "#c5cad5",
            "axes.labelcolor": INK,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "unspool-documentation-v1",
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str = "",
    *,
    color: str = INDIGO,
    fill: str = "#f3f5fa",
) -> None:
    left, bottom = xy
    width, height = size
    axis.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.25,
        )
    )
    axis.text(
        left + width / 2,
        bottom + height * 0.63,
        title,
        ha="center",
        va="center",
        color=color,
        weight="bold",
    )
    if subtitle:
        axis.text(
            left + width / 2,
            bottom + height * 0.29,
            subtitle,
            ha="center",
            va="center",
            color=INK,
            fontsize=7.5,
        )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    stop: tuple[float, float],
    *,
    color: str = MUTED,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            stop,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=color,
        )
    )


def _plot_workflow_map(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 4.8))
    axis.set(xlim=(0, 10.8), ylim=(0, 4.8))
    axis.axis("off")
    rows = (
        (3.65, "Describe change", "explicit clock", "smooth trajectory", BLUE),
        (2.65, "Predict later", "forward split", "future score", AMBER),
        (1.65, "Compare accounts", "nested selection", "paired evidence", TEAL),
        (0.65, "Test identifiability", "simulate design", "recovery matrix", INDIGO),
    )
    _box(axis, (0.25, 1.75), (1.75, 1.25), "Scientific question", "start with the claim")
    for y, question, boundary, result, color in rows:
        _box(axis, (2.55, y), (2.05, 0.68), question, color=color, fill="white")
        _arrow(axis, (4.63, y + 0.34), (5.18, y + 0.34))
        _box(axis, (5.2, y), (2.05, 0.68), boundary, color=color, fill="#f7f8fb")
        _arrow(axis, (7.28, y + 0.34), (7.83, y + 0.34))
        _box(axis, (7.85, y), (2.15, 0.68), result, color=color, fill="#fffaf3")
        _arrow(axis, (2.03, 2.37), (2.52, y + 0.34), color=color)
    axis.text(
        5.4,
        4.55,
        "Choose the validation boundary before choosing the model family",
        ha="center",
        color=INDIGO,
        weight="bold",
        fontsize=11,
    )
    axis.text(10.45, 2.38, "bounded\nclaim", ha="center", va="center", color=AMBER, weight="bold")
    _save(figure, path)


def _plot_clock_boundary(path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 5.3), gridspec_kw={"height_ratios": (1, 1.15)})
    top, bottom = axes
    for axis in axes:
        axis.set_xlim(-0.5, 9.5)
        axis.axis("off")
    top.set_ylim(-0.2, 2.2)
    clock_rows = (
        (1.65, "Session order", np.arange(10), BLUE),
        (1.0, "Cumulative trials", np.array([0, 1, 2, 4, 5, 7, 8, 8.5, 9, 9.3]), TEAL),
        (0.35, "Elapsed time", np.array([0, 0.4, 1.1, 1.8, 3.4, 4.0, 6.4, 7.1, 8.8, 9.2]), AMBER),
    )
    for y, label, values, color in clock_rows:
        top.plot(values, np.full(10, y), "o-", color=color, linewidth=1.8, markersize=5)
        top.text(-0.45, y, label, ha="right", va="center", color=color, weight="bold")
    top.set_title("The same observations imply different scientific clocks", loc="left")
    bottom.set_ylim(-0.25, 2.5)
    for index in range(10):
        color = BLUE if index < 6 else AMBER
        bottom.add_patch(
            Rectangle(
                (index - 0.38, 0.8), 0.76, 0.7, facecolor=color, alpha=0.85, edgecolor="white"
            )
        )
        bottom.text(
            index, 1.15, str(index + 1), ha="center", va="center", color="white", weight="bold"
        )
    bottom.axvline(5.5, color=INK, linewidth=1.2, linestyle="--")
    bottom.text(2.5, 1.9, "TRAIN", ha="center", color=BLUE, weight="bold")
    bottom.text(7.5, 1.9, "FUTURE TEST", ha="center", color=AMBER, weight="bold")
    _box(
        bottom,
        (0.25, -0.05),
        (3.3, 0.48),
        "Fit landmark and transform",
        "training outcomes only",
        color=BLUE,
    )
    _arrow(bottom, (3.65, 0.2), (5.05, 0.2), color=BLUE)
    _box(
        bottom,
        (5.15, -0.05),
        (3.75, 0.48),
        "Apply unchanged to future",
        "no refit · no recentering",
        color=AMBER,
        fill="#fff8ef",
    )
    bottom.set_title("Outcome-learned clocks stay inside each training fold", loc="left")
    _save(figure, path)


def _plot_validation_splits(path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    titles = (
        "Forward session · represented animal",
        "Whole-session holdout",
        "Population holdout · unseen animal",
        "Held-out lab + future session",
    )
    for axis, title in zip(axes.ravel(), titles, strict=True):
        axis.set(xlim=(-0.7, 6.3), ylim=(-0.8, 3.8), xticks=range(6), yticks=range(3))
        axis.set_yticklabels(("Animal A", "Animal B", "Animal C"), fontsize=7.5)
        axis.set_xticklabels(("S1", "S2", "S3", "S4", "S5", "S6"), fontsize=7.5)
        axis.set_title(title, fontsize=9.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for row in range(3):
        for column in range(6):
            axes[0, 0].add_patch(
                Rectangle(
                    (column - 0.38, row - 0.3),
                    0.76,
                    0.6,
                    facecolor=BLUE if column < 4 else AMBER if column == 4 else LIGHT,
                    edgecolor="white",
                )
            )
            axes[0, 1].add_patch(
                Rectangle(
                    (column - 0.38, row - 0.3),
                    0.76,
                    0.6,
                    facecolor=AMBER if (row == 1 and column == 3) else BLUE,
                    edgecolor="white",
                )
            )
            axes[1, 0].add_patch(
                Rectangle(
                    (column - 0.38, row - 0.3),
                    0.76,
                    0.6,
                    facecolor=AMBER if row == 2 else BLUE,
                    edgecolor="white",
                )
            )
            axes[1, 1].add_patch(
                Rectangle(
                    (column - 0.38, row - 0.3),
                    0.76,
                    0.6,
                    facecolor=AMBER
                    if row == 2 and column == 5
                    else LIGHT
                    if row == 2
                    else BLUE
                    if column < 5
                    else LIGHT,
                    edgecolor="white",
                )
            )
    for axis in axes.ravel():
        axis.scatter([], [], marker="s", s=70, color=BLUE, label="training")
        axis.scatter([], [], marker="s", s=70, color=AMBER, label="test")
        axis.scatter([], [], marker="s", s=70, color=LIGHT, label="not targeted")
        axis.legend(
            frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18)
        )
    figure.suptitle("Validation geometry determines what generalization means", weight="semibold")
    _save(figure, path, tight=False)


def _plot_model_atlas(path: Path) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(12, 3.4), sharex=True, sharey=True)
    x = np.linspace(0, 1, 120)
    curves = (
        ("Static GLM", np.full_like(x, 0.52), "fixed effects"),
        ("Smooth drift", 0.25 + 0.55 / (1 + np.exp(-8 * (x - 0.5))), "continuous change"),
        ("GLM-HMM", np.where(x < 0.32, 0.25, np.where(x < 0.68, 0.72, 0.42)), "discrete regimes"),
        ("Q-learning", 0.2 + 0.55 * (1 - np.exp(-4 * x)), "value updating"),
        ("Drift diffusion", 0.28 + 0.28 * np.sin(np.pi * x), "choice + response time"),
    )
    for index, (axis, (title, values, subtitle)) in enumerate(zip(axes, curves, strict=True)):
        color = (INDIGO, BLUE, TEAL, AMBER, "#8a5d83")[index]
        axis.plot(x, values, color=color, linewidth=2.6)
        axis.fill_between(x, 0.15, values, color=color, alpha=0.08)
        axis.set(
            xlim=(0, 1), ylim=(0.1, 0.9), xticks=(0, 1), xticklabels=("early", "late"), title=title
        )
        axis.text(
            0.5,
            0.06,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            color=color,
            fontsize=7.5,
            weight="bold",
        )
        axis.grid(axis="y", color="#eef0f5")
        if index:
            axis.tick_params(labelleft=False)
    axes[0].set_ylabel("Illustrative latent decision tendency")
    figure.suptitle(
        "Different model families encode different explanations of change", weight="semibold"
    )
    _save(figure, path)


def _plot_nested_selection(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.set(xlim=(0, 11), ylim=(0, 4.2))
    axis.axis("off")
    _box(axis, (0.25, 1.35), (1.7, 1.25), "Outer training", "future test absent", color=BLUE)
    _arrow(axis, (2.0, 1.98), (2.55, 1.98))
    for y, label in ((2.75, "inner fold 1"), (1.75, "inner fold 2"), (0.75, "inner fold 3")):
        _box(axis, (2.65, y), (1.55, 0.58), label, "earlier → later", color=TEAL, fill="white")
        _arrow(axis, (4.25, y + 0.29), (4.75, y + 0.29))
    _box(
        axis, (4.85, 1.25), (1.8, 1.45), "Candidate ledger", "static · smooth 1 · 3 · 9", color=TEAL
    )
    _arrow(axis, (6.7, 1.98), (7.25, 1.98))
    _box(axis, (7.35, 1.35), (1.45, 1.25), "Select + refit", "training only", color=BLUE)
    _arrow(axis, (8.85, 1.98), (9.35, 1.98), color=AMBER)
    _box(axis, (9.45, 1.35), (1.3, 1.25), "Outer test", "opened once", color=AMBER, fill="#fff8ef")
    axis.plot([2.25, 9.05], [3.72, 3.72], color=AMBER, linewidth=1.5)
    axis.text(
        5.65,
        3.92,
        "No outer outcome enters candidate or hyperparameter selection",
        ha="center",
        color=AMBER,
        weight="bold",
    )
    axis.text(
        5.65,
        0.22,
        "The final score evaluates the complete selection procedure",
        ha="center",
        color=INDIGO,
        weight="bold",
    )
    _save(figure, path)


def _plot_diagnostic_layers(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    axis.set(xlim=(0, 9.6), ylim=(0, 5.2))
    axis.axis("off")
    layers = (
        (
            0.55,
            8.5,
            "Numerical fit",
            "convergence · gradients · boundaries · restarts",
            "Can this fitted object be used?",
            MUTED,
        ),
        (
            1.55,
            7.2,
            "Prospective prediction",
            "held-out log loss · calibration · aggregation",
            "Does it forecast the declared future?",
            BLUE,
        ),
        (
            2.55,
            5.9,
            "Parameter recovery",
            "bias · RMSE · interval coverage",
            "Can this design estimate the parameters?",
            TEAL,
        ),
        (
            3.55,
            4.6,
            "Model recovery",
            "confusion matrix · unresolved rate",
            "Can this design distinguish explanations?",
            AMBER,
        ),
    )
    for bottom, width, title, evidence, question, color in layers:
        left = (9.6 - width) / 2
        _box(axis, (left, bottom), (width, 0.72), title, evidence, color=color, fill="white")
        axis.text(8.95, bottom + 0.36, question, ha="right", va="center", fontsize=7.5, color=color)
    axis.text(4.8, 4.78, "Interpretation", ha="center", color=INDIGO, weight="bold", fontsize=11)
    axis.text(
        4.8,
        0.18,
        "Passing one layer never certifies the layers above it",
        ha="center",
        color=AMBER,
        weight="bold",
    )
    _save(figure, path)


def _plot_interoperability(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.set(xlim=(0, 11), ylim=(0, 4.5))
    axis.axis("off")
    sources = ((3.45, "Dataframe"), (2.55, "IBL ONE"), (1.65, "NWB"), (0.75, "DANDI"))
    for y, label in sources:
        _box(axis, (0.25, y), (1.55, 0.58), label, color=BLUE, fill="white")
        _arrow(axis, (1.85, y + 0.29), (2.45, 2.25), color=BLUE)
    _box(
        axis,
        (2.55, 1.5),
        (2.15, 1.5),
        "Study contract",
        "subject · session · trial\nsession order · source rows",
        color=INDIGO,
    )
    _arrow(axis, (4.75, 2.25), (5.35, 2.25))
    _box(
        axis,
        (5.45, 1.5),
        (2.05, 1.5),
        "Explicit semantics",
        "column map · units · clock\nchecksum · release identity",
        color=TEAL,
    )
    _arrow(axis, (7.55, 2.25), (8.15, 2.25))
    _box(
        axis,
        (8.25, 1.5),
        (2.35, 1.5),
        "Prospective analysis",
        "same model contract\nsame fold boundaries",
        color=AMBER,
        fill="#fff8ef",
    )
    axis.text(
        5.5,
        4.18,
        "Formats differ; scientific identity and chronology do not become implicit",
        ha="center",
        color=INDIGO,
        weight="bold",
    )
    axis.text(
        5.5,
        0.25,
        "Adapters translate source structure—they do not invent scientific meaning",
        ha="center",
        color=AMBER,
        weight="bold",
    )
    _save(figure, path)


def _plot_hierarchical_pooling(path: Path) -> None:
    payload = _load("hierarchical_glm")
    regimes = payload["regimes"]
    scales = np.asarray([float(value) for value in regimes], dtype=float)
    methods = ("complete_pooling", "partial_pooling", "independent")
    labels = ("Complete pooling", "Partial pooling", "Independent")
    colors = (MUTED, INDIGO, AMBER)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for method, label, color in zip(methods, labels, colors, strict=True):
        rmse = [
            regimes[str(scale)]["methods"][method]["mean_subject_coefficient_rmse"]
            for scale in scales
        ]
        loss = [
            regimes[str(scale)]["methods"][method]["mean_prospective_log_loss"] for scale in scales
        ]
        axes[0].plot(scales, rmse, "o-", color=color, linewidth=2, label=label)
        axes[1].plot(scales, loss, "o-", color=color, linewidth=2, label=label)
    axes[0].set(
        title="Individual coefficient recovery",
        xlabel="True between-animal scale",
        ylabel="Mean subject coefficient RMSE",
    )
    axes[1].set(
        title="Future-session prediction",
        xlabel="True between-animal scale",
        ylabel="Mean prospective log loss",
    )
    for axis in axes:
        axis.grid(color="#edf0f5")
        axis.legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "Partial pooling adapts between population and individual estimates", weight="semibold"
    )
    _save(figure, path)


def _plot_ddm_recovery(path: Path) -> None:
    payload = _load("ddm_recovery")
    designs = payload["designs"]
    names = next(iter(designs.values()))["parameter_names"]
    trial_counts = sorted(int(key.split("_")[0]) for key in designs)
    errors = {name: [] for name in names}
    for count in trial_counts:
        runs = designs[f"{count}_trials"]["runs"]
        for name in names:
            errors[name].append(
                float(
                    np.sqrt(
                        np.mean([(run["estimate"][name] - run["truth"][name]) ** 2 for run in runs])
                    )
                )
            )
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.75, len(names)))
    for name, color in zip(names, colors, strict=True):
        axis.plot(
            trial_counts,
            errors[name],
            "o-",
            color=color,
            linewidth=1.8,
            label=name.replace("drift.", ""),
        )
    axis.set(
        xlabel="Trials in the recovery design",
        ylabel="Parameter RMSE",
        title="More trials improve every recovered DDM parameter",
    )
    axis.set_xscale("log")
    axis.grid(color="#edf0f5")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    _save(figure, path)


def _plot_trajectory_components(path: Path) -> None:
    payload = _load("trajectory_shapes")
    x = np.linspace(0, 1, 100)
    trajectories = (
        ("Reference", -0.5 + x, INDIGO),
        ("Level shift", 1.5 + x, BLUE),
        ("Amplitude shift", -1 + 2 * x, AMBER),
        ("Shape change", 0.65 * np.sin(2 * np.pi * x), TEAL),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.3, 4.3))
    for label, values, color in trajectories:
        axes[0].plot(x, values, color=color, linewidth=2.3, label=label)
    axes[0].axhline(0, color=LIGHT)
    axes[0].set(
        title="Four generating trajectory components",
        xlabel="Normalized learning time",
        ylabel="Outcome trajectory",
    )
    axes[0].legend(frameon=False, fontsize=7.5)
    metrics = payload["mean_pairwise_metrics"]
    pairs = ("reference__level_shift", "reference__amplitude_shift", "reference__shape_change")
    labels = ("Level", "Amplitude", "Shape")
    values = np.asarray(
        [
            [
                metrics[pair]["absolute_level_difference"],
                metrics[pair]["absolute_amplitude_difference"],
                metrics[pair]["shape_distance"],
            ]
            for pair in pairs
        ]
    )
    image = axes[1].imshow(
        values,
        cmap=mpl.colors.LinearSegmentedColormap.from_list("components", ["#f5f6f9", INDIGO]),
        aspect="auto",
    )
    for row in range(3):
        for column in range(3):
            axes[1].text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if values[row, column] > 1 else INK,
                weight="bold",
            )
    axes[1].set(
        xticks=range(3),
        xticklabels=("Level metric", "Amplitude metric", "Shape metric"),
        yticks=range(3),
        yticklabels=labels,
        title="Reference contrasts isolate different geometry",
    )
    axes[1].tick_params(axis="x", rotation=25)
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle("Trajectory geometry separates level, amplitude, and shape", weight="semibold")
    _save(figure, path)


def _plot_validation_geometry(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 3.15))
    axis.set(xlim=(0, 11), ylim=(0, 3.15))
    axis.axis("off")
    boxes = (
        (0.2, "Observed process", "trials · sessions · animals"),
        (2.4, "Explicit time", "order · exposure · landmarks"),
        (4.6, "Training only", "transforms · fits · selection"),
        (6.8, "Prospective test", "future session · animal · lab"),
        (9.0, "Bounded claim", "audit · recovery · uncertainty"),
    )
    for index, (left, title, subtitle) in enumerate(boxes):
        color = AMBER if index == 3 else INDIGO
        patch = FancyBboxPatch(
            (left, 1.0),
            1.8,
            1.15,
            boxstyle="round,pad=0.08,rounding_size=0.09",
            facecolor="#fff8ef" if index == 3 else "#f3f5fa",
            edgecolor=color,
            linewidth=1.35,
        )
        axis.add_patch(patch)
        axis.text(left + 0.9, 1.72, title, ha="center", va="center", weight="bold", color=color)
        axis.text(left + 0.9, 1.35, subtitle, ha="center", va="center", fontsize=8, color=INK)
        if index < len(boxes) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (left + 1.85, 1.57),
                    (left + 2.35, 1.57),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.1,
                    color=MUTED,
                )
            )
    axis.text(
        5.5,
        2.72,
        "The future is absent while the analysis is learned",
        ha="center",
        color=AMBER,
        weight="bold",
    )
    axis.plot([4.7, 8.5], [2.52, 2.52], color=AMBER, linewidth=1.4)
    axis.text(1.1, 0.47, "history retained", ha="center", color=BLUE)
    axis.text(7.7, 0.47, "outcomes opened once", ha="center", color=AMBER)
    axis.text(9.9, 0.47, "failures remain visible", ha="center", color=TEAL)
    _save(figure, path)


def _plot_cell_strategy(data_path: Path, path: Path) -> None:
    from benchmarks.cell2025.benchmark import calculate_session_metrics, load_study

    study = load_study(data_path)
    rows_by_subject: dict[str, list[Any]] = defaultdict(list)
    for row in calculate_session_metrics(study):
        rows_by_subject[row.subject].append(row)
    early_bias: list[float] = []
    late_slope: list[float] = []
    for rows in rows_by_subject.values():
        rows.sort(key=lambda row: row.session_order)
        maximum = rows[-1].session_order
        early = [row for row in rows if 3 < row.session_order <= 8]
        late = [row for row in rows if maximum - 5 < row.session_order <= maximum]
        early_bias.append(float(np.mean([row.zero_bias for row in early])))
        late_slope.append(float(np.mean([row.right_slope - row.left_slope for row in late])))
    x = np.asarray(early_bias)
    y = np.asarray(late_slope)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()) - 0.01, float(x.max()) + 0.01, 100)
    correlation = float(np.corrcoef(x, y)[0, 1])
    figure, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.scatter(x, y, s=42, color=INDIGO, edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
    axis.plot(line_x, intercept + slope * line_x, color=AMBER, linewidth=2.1)
    axis.axhline(0, color=LIGHT, linewidth=1, zorder=0)
    axis.axvline(0, color=LIGHT, linewidth=1, zorder=0)
    axis.set(
        xlabel="Early zero-contrast bias (days 4-8)",
        ylabel="Late right-minus-left psychometric slope",
        title="Early strategy predicts late strategy across animals",
    )
    axis.text(
        0.04,
        0.93,
        f"30 mice  ·  r = {correlation:.3f}  ·  p = 2.04 x 10^-5",
        transform=axis.transAxes,
        color=AMBER,
        weight="bold",
    )
    axis.grid(color="#edf0f5", linewidth=0.8)
    _save(figure, path)


def _plot_ibl_trajectories(path: Path) -> None:
    payload = _load("ibl2021_replicated")
    summaries = payload["trajectory_comparison"]["group_summaries"]
    positions = np.asarray(payload["clock"]["grid"], dtype=float)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.85, len(summaries)))
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    weighted_sum = np.zeros(len(positions), dtype=float)
    total_subjects = 0
    for color, summary in zip(colors, summaries, strict=True):
        values = np.asarray(summary["mean_values"], dtype=float)
        subjects = len(summary["subjects"])
        weighted_sum += subjects * values
        total_subjects += subjects
        label = str(summary["group"]).replace("lab", "")
        axis.plot(positions, values, color=color, linewidth=1.25, alpha=0.58, label=label)
    population = weighted_sum / total_subjects
    axis.plot(
        positions, population, color=INDIGO, linewidth=3.2, marker="o", label="78-animal mean"
    )
    axis.axvspan(2.5, 5.25, color="#fff3e3", alpha=0.8, zorder=-2)
    axis.text(1, 1.015, "early windows", ha="center", color=BLUE)
    axis.text(4, 1.015, "late pre-transition windows", ha="center", color=AMBER)
    axis.set(
        xlim=(-0.2, 5.2),
        ylim=(0.32, 1.04),
        xlabel="Outcome-blind endpoint-window position",
        ylabel="Mean easy-trial accuracy",
        title="Learning trajectories vary across labs but improve in every animal",
        xticks=positions,
    )
    axis.grid(axis="y", color="#edf0f5", linewidth=0.8)
    axis.legend(ncol=2, frameon=False, fontsize=7.5, loc="lower right")
    _save(figure, path)


def _plot_ibl_selection(path: Path) -> None:
    fixed = _load("ibl2021_prospective")
    nested = _load("ibl2021_nested_selection")
    static_rows = fixed["within_subject_future_session"]["models"]["static_partial_pooling"][
        "unit_scores"
    ]
    drift_rows = fixed["within_subject_future_session"]["models"]["hierarchical_smooth_drift"][
        "unit_scores"
    ]
    selected_rows = nested["within_subject_future_session"]["subject_scores"]
    static = {row["unit"]: float(row["log_loss"]) for row in static_rows}
    drift = {row["unit"]: float(row["log_loss"]) for row in drift_rows}
    subjects = list(static)
    values = np.asarray(
        [
            [static[subject], drift[subject], selected_rows[subject]["log_loss"]]
            for subject in subjects
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    left, right = axes
    for row in values:
        left.plot((0, 1, 2), row, color="#c9ced9", linewidth=0.65, alpha=0.5)
    colors = (MUTED, BLUE, AMBER)
    for index, color in enumerate(colors):
        left.scatter(
            np.full(len(subjects), index),
            values[:, index],
            s=12,
            color=color,
            alpha=0.45,
            edgecolor="none",
        )
        left.scatter(
            index, np.mean(values[:, index]), s=85, color=color, edgecolor="white", zorder=5
        )
    left.set(
        xticks=(0, 1, 2),
        xticklabels=("Static", "Drift\nsmoothness 3", "Selected\nsmoothness 9"),
        ylabel="Future-session subject log loss",
        title="Untouched position 5 · represented animals",
    )
    left.grid(axis="y", color="#edf0f5", linewidth=0.8)
    candidates = ("static", "drift_smoothness_1", "drift_smoothness_3", "drift_smoothness_9")
    folds = nested["held_out_lab_future_session"]["folds"]
    inner = np.asarray(
        [
            [
                float(fold["inner_candidates"][candidate]["subject_balanced_log_loss"])
                for candidate in candidates
            ]
            for fold in folds
        ]
    )
    for row in inner:
        right.plot(range(4), row, color="#adb5c5", linewidth=0.9, alpha=0.65)
    right.plot(range(4), np.mean(inner, axis=0), color=INDIGO, linewidth=2.8, marker="o")
    right.scatter(3, np.mean(inner, axis=0)[3], s=105, color=AMBER, edgecolor="white", zorder=5)
    right.set(
        xticks=range(4),
        xticklabels=("Static", "Smooth 1", "Smooth 3", "Smooth 9"),
        ylabel="Inner subject-balanced log loss",
        title="Training-only selection · nine outer lab folds",
    )
    right.grid(axis="y", color="#edf0f5", linewidth=0.8)
    right.text(
        0.98,
        0.94,
        "smoothness 9 selected 9 / 9",
        transform=right.transAxes,
        ha="right",
        color=AMBER,
        weight="bold",
    )
    figure.suptitle(
        "Selection stays inside training data; outer outcomes score the procedure",
        weight="semibold",
    )
    _save(figure, path)


def _plot_recovery_matrix(path: Path) -> None:
    payload = _load("recovery_grid")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.15), constrained_layout=True)
    labels = ("Static", "Smooth", "GLM-HMM", "Q-learning", "Unresolved")
    truth = ("Static", "Smooth", "GLM-HMM", "Q-learning")
    for axis, design in zip(axes, payload["designs"], strict=True):
        counts = np.asarray(design["confusion"]["counts"], dtype=float)
        axis.imshow(
            counts,
            cmap=mpl.colors.LinearSegmentedColormap.from_list("unspool", ["#f5f6f9", INDIGO]),
            vmin=0,
            vmax=1,
        )
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                value = int(counts[row, column])
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value else MUTED,
                    weight="bold",
                )
        title = (
            f"{design['design'].title()} design · {design['n_trials']} trials\n"
            f"{design['resolved_accuracy']:.0%} recovered"
        )
        axis.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(len(truth)),
            yticklabels=truth,
            xlabel="Selected family",
            ylabel="Generating family",
            title=title,
        )
        axis.tick_params(axis="x", rotation=35)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("Model recovery is a property of the design", weight="semibold")
    _save(figure, path, tight=False)


def _plot_choice_model_evidence_atlas(path: Path) -> None:
    """Build one reproducible, end-to-end evidence panel for each choice-model family."""

    from benchmarks.recovery_grid.benchmark import build_design, experiment

    scenarios, candidates = experiment()
    recovery = json.loads(
        (ROOT / "benchmarks" / "recovery_grid" / "result.json").read_text(encoding="utf-8")
    )
    dense = next(cell for cell in recovery["designs"] if cell["design"] == "dense")
    figure, axes = plt.subplots(4, 5, figsize=(13.2, 10.2), constrained_layout=True)
    labels = ("Static GLM", "Smooth GLM", "GLM-HMM", "Q-learning")
    colors = (INDIGO, BLUE, TEAL, AMBER)
    design = build_design(trials_per_session=60)

    for row, (scenario, (key, model), label, color) in enumerate(
        zip(scenarios, candidates.items(), labels, colors, strict=True)
    ):
        seed = int(dense["run_seeds"][row])
        study = scenario.generator.simulate(design, scenario.parameters, seed=seed)
        train = study.take(np.flatnonzero(study["session_order"] < 4))
        test = study.take(np.flatnonzero(study["session_order"] == 4))
        fit = model.fit(train)
        prediction = model.predict(test, fit).probability
        outcomes = np.asarray(study["choice"], dtype=float)
        test_outcomes = np.asarray(test["choice"], dtype=float)

        observed = axes[row, 0]
        session_means = [
            outcomes[np.asarray(study["session_order"]) == session].mean() for session in range(5)
        ]
        observed.plot(range(5), session_means, "o-", color=color, linewidth=2)
        observed.axvspan(3.5, 4.5, color=AMBER, alpha=0.12)
        observed.set(ylim=(0, 1), xticks=range(5), ylabel=label)

        structure = axes[row, 1]
        _plot_fitted_structure(structure, key, model, study, fit, color)

        forecast = axes[row, 2]
        bin_edges = np.linspace(0, len(test_outcomes), 7, dtype=int)
        centers = np.arange(6)
        predicted_bins = [prediction[left:right].mean() for left, right in pairwise(bin_edges)]
        observed_bins = [test_outcomes[left:right].mean() for left, right in pairwise(bin_edges)]
        forecast.plot(centers, predicted_bins, "o-", color=color, label="predicted")
        forecast.scatter(centers, observed_bins, color=INK, marker="x", label="observed")
        forecast.set(ylim=(0, 1), xticks=(0, 5), xticklabels=("early", "late"))

        residual = axes[row, 3]
        quantiles = np.quantile(prediction, np.linspace(0, 1, 5))
        groups = np.clip(np.digitize(prediction, quantiles[1:-1]), 0, 3)
        mean_prediction = np.asarray([prediction[groups == group].mean() for group in range(4)])
        mean_outcome = np.asarray([test_outcomes[groups == group].mean() for group in range(4)])
        residual.axhline(0, color=MUTED, linewidth=1)
        residual.scatter(mean_prediction, mean_outcome - mean_prediction, color=color, s=28)
        residual.set(xlim=(0, 1), ylim=(-0.45, 0.45))

        recovery_axis = axes[row, 4]
        scores = np.asarray(dense["mean_log_probabilities"][row])
        bars = recovery_axis.bar(
            range(4), scores, color=[color if index == row else LIGHT for index in range(4)]
        )
        bars[row].set_edgecolor(INK)
        recovery_axis.set_xticks(range(4), ("S", "D", "H", "Q"))
        recovery_axis.set_ylim(min(scores) - 0.08, max(scores) + 0.04)
        recovery_axis.text(
            row, scores[row], "truth", ha="center", va="bottom", fontsize=7, color=color
        )

        for axis in axes[row]:
            axis.grid(color="#edf0f5", linewidth=0.7)
            axis.tick_params(labelsize=7)

    titles = (
        "Observed choice rate",
        "Fitted model structure",
        "Untouched session 5",
        "Calibration residual",
        "Dense-design recovery",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
    axes[0, 2].legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[-1, 0].set_xlabel("Session")
    axes[-1, 2].set_xlabel("Future-session trial bin")
    axes[-1, 3].set_xlabel("Predicted choice probability")
    axes[-1, 3].set_ylabel("Observed - predicted")
    axes[-1, 4].set_xlabel("Candidate: static · drift · HMM · Q")
    figure.suptitle("A model claim requires a complete chain of evidence", weight="semibold")
    _save(figure, path, tight=False)


def _plot_ibl_choice_rt(path: Path) -> None:
    result = _load("ibl2021_decision_models")
    ddm = result["ddm"]
    heldout = ddm["heldout"]
    observed_rt = np.asarray(heldout["response_time_seconds"], dtype=float)
    predictive_rt = np.asarray(heldout["predictive_response_time_seconds"], dtype=float)
    responsibilities = np.asarray(heldout["contaminant_responsibility"], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    response_axis, accuracy_axis, responsibility_axis, evidence_axis = axes.ravel()

    for values, label, color in (
        (observed_rt, "observed", INK),
        (predictive_rt, "one predictive draw", AMBER),
    ):
        ordered = np.sort(values)
        response_axis.step(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            where="post",
            label=label,
            color=color,
            linewidth=1.8,
        )
    response_axis.set(
        xscale="log",
        xlim=(0.045, 3.2),
        xlabel="Movement-onset response time (s)",
        ylabel="Cumulative fraction",
        title="Untouched-session response times",
    )
    response_axis.legend(frameon=False, fontsize=8)

    accuracy = heldout["conditional_accuracy"]
    contrast = np.asarray([row["absolute_contrast"] for row in accuracy], dtype=float)
    observed = np.asarray([row["observed_accuracy"] for row in accuracy], dtype=float)
    predicted = np.asarray([row["predicted_accuracy"] for row in accuracy], dtype=float)
    accuracy_axis.plot(contrast, predicted, "o-", color=BLUE, label="robust DDM")
    accuracy_axis.scatter(contrast, observed, marker="x", s=45, color=INK, label="observed")
    accuracy_axis.set(
        xscale="log",
        ylim=(0.45, 1.03),
        xlabel="Absolute contrast",
        ylabel="Conditional accuracy",
        title="Choice implication of the joint fit",
    )
    accuracy_axis.legend(frameon=False, fontsize=8)

    responsibility_axis.scatter(
        observed_rt,
        responsibilities,
        c=np.abs(np.asarray(heldout["stimulus"], dtype=float)),
        cmap="viridis",
        s=24,
        alpha=0.8,
        edgecolor="none",
    )
    responsibility_axis.set(
        xscale="log",
        xlim=(0.045, 3.2),
        ylim=(-0.04, 1.04),
        xlabel="Observed response time (s)",
        ylabel="Posterior responsibility",
        title="Model-dependent contaminant probability",
    )

    scores = (
        ddm["naive"]["mean_test_joint_log_density"],
        ddm["robust"]["mean_test_joint_log_density"],
    )
    bars = evidence_axis.bar((0, 1), scores, color=(BLUE, AMBER), width=0.62)
    evidence_axis.axhline(0, color=MUTED, linewidth=1)
    evidence_axis.set(
        xticks=(0, 1),
        xticklabels=("Naive", "Contaminant-aware"),
        ylabel="Mean joint log density / trial",
        title="Prospective evidence · position 5",
    )
    for bar, score in zip(bars, scores, strict=True):
        evidence_axis.text(
            bar.get_x() + bar.get_width() / 2,
            score - 0.012,
            f"{score:.3f}",
            ha="center",
            va="top",
            color="white",
            weight="bold",
        )
    evidence_axis.text(
        0.5,
        min(scores) - 0.055,
        f"robust - naive = {ddm['robust']['improvement_over_naive']:+.3f}",
        ha="center",
        color=INK,
        fontsize=8,
    )
    figure.suptitle(
        "Joint choice and response-time evidence for one outcome-blind IBL subject",
        weight="semibold",
    )
    _save(figure, path, tight=False)


def _plot_ibl_glm_hmm(path: Path) -> None:
    result = _load("ibl2021_decision_models")["glm_hmm"]
    selected = result["selected_glm_hmm"]
    probability = np.asarray(result["heldout"]["filtered_state_probability"], dtype=float)
    coefficients = np.asarray(selected["emission_coefficients"], dtype=float)
    transition = np.asarray(selected["transition_matrix"], dtype=float)
    n_states = probability.shape[1]

    figure = plt.figure(figsize=(11.8, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0))
    state_axis = figure.add_subplot(grid[0, :2])
    coefficient_axis = figure.add_subplot(grid[0, 2])
    transition_axis = figure.add_subplot(grid[1, 0])
    selection_axis = figure.add_subplot(grid[1, 1])
    evidence_axis = figure.add_subplot(grid[1, 2])

    state_colors = (BLUE, AMBER, TEAL, INDIGO)
    for state in range(n_states):
        state_axis.plot(
            np.arange(len(probability)),
            probability[:, state],
            color=state_colors[state],
            linewidth=1.5,
            label=f"state {state + 1}",
        )
    state_axis.set(
        xlim=(0, len(probability) - 1),
        ylim=(-0.03, 1.03),
        xlabel="Trial in untouched position 5",
        ylabel="Filtered probability",
        title="Model-dependent state path",
    )
    state_axis.legend(frameon=False, ncol=n_states, fontsize=7.5, loc="lower center")

    coefficient_limit = float(np.max(np.abs(coefficients)))
    coefficient_image = coefficient_axis.imshow(
        coefficients,
        cmap="coolwarm",
        aspect="auto",
        vmin=-coefficient_limit,
        vmax=coefficient_limit,
    )
    coefficient_axis.set(
        xticks=range(coefficients.shape[1]),
        xticklabels=selected["coefficient_names"],
        yticks=range(n_states),
        yticklabels=[f"state {state + 1}" for state in range(n_states)],
        title="Emission coefficients",
    )
    coefficient_axis.tick_params(axis="x", rotation=30, labelsize=7.5)
    figure.colorbar(coefficient_image, ax=coefficient_axis, shrink=0.75, label="logit weight")

    transition_image = transition_axis.imshow(
        transition,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    transition_axis.set(
        xticks=range(n_states),
        xticklabels=range(1, n_states + 1),
        yticks=range(n_states),
        yticklabels=range(1, n_states + 1),
        xlabel="To state",
        ylabel="From state",
        title="Refitted transition matrix",
    )
    figure.colorbar(transition_image, ax=transition_axis, shrink=0.75, label="probability")

    candidates = result["selection"]["candidates"]
    candidate_states = [row["n_states"] for row in candidates]
    candidate_losses = [row["mean_selection_log_loss"] for row in candidates]
    selection_axis.plot(candidate_states, candidate_losses, "o-", color=BLUE, linewidth=1.8)
    selected_states = result["selection"]["selected_states"]
    selected_index = candidate_states.index(selected_states)
    selection_axis.scatter(
        [selected_states],
        [candidate_losses[selected_index]],
        s=80,
        facecolor=AMBER,
        edgecolor=INK,
        zorder=3,
        label="selected",
    )
    selection_axis.set(
        xticks=candidate_states,
        xlabel="Candidate states",
        ylabel="Mean log loss",
        title="Inner position-4 selection",
    )
    selection_axis.legend(frameon=False, fontsize=8)

    outer_losses = (
        result["static_glm"]["mean_test_log_loss"],
        selected["mean_test_log_loss"],
    )
    bars = evidence_axis.bar((0, 1), outer_losses, color=(MUTED, TEAL), width=0.62)
    evidence_axis.set(
        xticks=(0, 1),
        xticklabels=("Static GLM", f"{selected_states}-state\nGLM-HMM"),
        ylabel="Mean choice log loss",
        title="Outer position-5 evidence",
    )
    for bar, loss in zip(bars, outer_losses, strict=True):
        evidence_axis.text(
            bar.get_x() + bar.get_width() / 2,
            loss - 0.025,
            f"{loss:.3f}",
            ha="center",
            va="top",
            color="white",
            weight="bold",
        )
    figure.suptitle(
        "Nested prospective GLM-HMM evidence for one outcome-blind IBL subject",
        weight="semibold",
    )
    _save(figure, path, tight=False)


def _plot_fitted_structure(
    axis: plt.Axes,
    key: str,
    model: Any,
    study: Any,
    fit: Any,
    color: str,
) -> None:
    if key == "static":
        axis.bar(range(len(fit.estimates)), fit.estimates, color=color)
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.set_xticks(range(3), ("bias", "stim.", "history"))
        return
    if key == "smooth":
        paths = fit.estimates.reshape(3, 5)
        axis.plot(range(5), paths[1], "o-", color=color, label="stimulus")
        axis.plot(range(5), paths[0], "o--", color=MUTED, label="bias")
        axis.axhline(0, color=LIGHT)
        axis.set_xticks(range(5))
        return
    if key == "hmm":
        probability = model.state_probabilities(study, fit).filtered[:, 1]
        axis.plot(np.arange(len(probability)), probability, color=color, linewidth=1)
        axis.set(ylim=(0, 1), xticks=(0, len(probability) - 1), xticklabels=("first", "last"))
        return
    trajectory = model.value_trajectory(study, fit)
    value_difference = trajectory.pre_choice[:, 1] - trajectory.pre_choice[:, 0]
    axis.plot(np.arange(len(value_difference)), value_difference, color=color, linewidth=1)
    axis.axhline(0, color=MUTED, linewidth=1)
    axis.set(xticks=(0, len(value_difference) - 1), xticklabels=("first", "last"))


def _load(name: str) -> dict[str, Any]:
    path = ROOT / "benchmarks" / name / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save(figure: plt.Figure, path: Path, *, tight: bool = True) -> None:
    if tight:
        figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    # Matplotlib writes spaces at the end of multiline SVG path commands. Normalizing
    # them keeps generated assets reviewable and makes `git diff --check` meaningful.
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
