"""Reproduce the bounded longitudinal-behaviour result in Cell 2025 Figure 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from benchmarks.cell2025.fetch_data import (
    ARCHIVE_MEMBER,
    FIGSHARE_ARTICLE_DOI,
    FIGSHARE_FILE_ID,
    MEMBER_SHA256,
    sha256,
)
from unspool import Study

EXPECTED = {
    "n_trials": 192_238,
    "n_subjects": 30,
    "n_source_sessions": 950,
    "n_sessions": 949,
    "early_late_bias_r": -0.527_639_075_183_207,
    "early_late_bias_p": 0.002_731_238_151_759_02,
    "early_bias_late_slope_r": 0.694_789_656_448_098,
    "early_bias_late_slope_p": 2.042_901_310_054_89e-05,
    "first_session_accuracy": 0.517_336_981_366_178,
    "last_session_accuracy": 0.758_025_064_538_576,
}


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    sum_squared_deviation: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_squared_deviation += delta * (value - self.mean)

    @property
    def sample_standard_deviation(self) -> float:
        if self.count < 2:
            return math.nan
        return math.sqrt(self.sum_squared_deviation / (self.count - 1))


@dataclass(frozen=True)
class SessionMetrics:
    subject: str
    session_order: int
    left_slope: float
    right_slope: float
    zero_bias: float
    accuracy: float


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    source_doi: str
    source_file_id: int
    source_member: str
    source_member_sha256: str
    n_trials: int
    n_subjects: int
    n_source_sessions: int
    n_sessions: int
    early_late_bias_r: float
    early_late_bias_p: float
    early_bias_late_slope_r: float
    early_bias_late_slope_p: float
    first_session_accuracy: float
    last_session_accuracy: float
    accuracy_change: float
    contract_passed: bool


def load_study(path: Path) -> Study:
    """Apply the released trial exclusions and map the source table to ``Study``."""

    reaction_time_stats: dict[tuple[str, int], RunningStats] = defaultdict(RunningStats)
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["choice"] == "NoGo":
                continue
            subject = row["expRef"][-6:]
            try:
                session_order = _integer(row["sessionNum"])
                reaction_time = _reaction_time(row)
            except ValueError:
                continue
            reaction_time_stats[(subject, session_order)].update(reaction_time)
            sessions_by_subject[subject].add(row["expRef"])

    chronological_order = {
        (subject, session): order
        for subject, sessions in sessions_by_subject.items()
        for order, session in enumerate(sorted(sessions))
    }

    columns: dict[str, list[Any]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "source_trial": [],
        "session_order": [],
        "paper_session_order": [],
        "choice": [],
        "reward": [],
        "stimulus_side": [],
        "signed_contrast": [],
        "left_contrast": [],
        "right_contrast": [],
        "response_time": [],
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["choice"] == "NoGo":
                continue
            subject = row["expRef"][-6:]
            try:
                session_order = _integer(row["sessionNum"])
                reaction_time = _reaction_time(row)
            except ValueError:
                continue
            rt_stats = reaction_time_stats[(subject, session_order)]
            rt_zscore = (reaction_time - rt_stats.mean) / rt_stats.sample_standard_deviation
            if not rt_zscore < 2.0:
                continue
            if _integer(row["repeatNumber"]) != 1:
                continue
            if _integer(row["isShapedMouse"]) != 0:
                continue
            if _integer(row["isExpertMouse"]) != 1:
                continue

            contrast = float(row["contrastRight"]) - float(row["contrastLeft"])
            columns["subject"].append(subject)
            columns["session"].append(row["expRef"])
            source_trial = _integer(row["trialNumber"])
            columns["trial"].append(source_trial)
            columns["source_trial"].append(source_trial)
            columns["session_order"].append(chronological_order[(subject, row["expRef"])])
            columns["paper_session_order"].append(session_order)
            columns["choice"].append(int(row["choice"] == "Right"))
            columns["reward"].append(int(row["feedback"] == "Rewarded"))
            columns["stimulus_side"].append(int(np.sign(contrast)))
            columns["signed_contrast"].append(contrast)
            columns["left_contrast"].append(min(contrast, 0.0))
            columns["right_contrast"].append(max(contrast, 0.0))
            columns["response_time"].append(reaction_time)
    return Study.from_columns(columns)


def calculate_session_metrics(study: Study) -> list[SessionMetrics]:
    """Calculate the paper's session bias, slopes, and non-zero-stimulus accuracy."""

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in study.chronological_indices():
        subject = str(study["subject"][index])
        session_order = int(study["paper_session_order"][index])
        if "ALK" not in str(study["session"][index]) and "MMM" not in str(study["session"][index]):
            grouped[(subject, session_order)].append(int(index))

    eligible_subjects = {subject for subject, session_order in grouped if session_order < 3}
    rows: list[SessionMetrics] = []
    previous_slopes: tuple[float, float, float] | None = None
    for (subject, session_order), indices in sorted(grouped.items()):
        if subject not in eligible_subjects:
            continue
        positions = np.asarray(indices, dtype=np.intp)
        side = study["stimulus_side"][positions]
        choice = study["choice"][positions].astype(float)
        reward = study["reward"][positions].astype(float)

        p_left = float(np.mean(choice[side == -1]))
        p_zero = float(np.mean(choice[side == 0]))
        p_right = float(np.mean(choice[side == 1]))
        if len(indices) < 70 or int(np.count_nonzero(side == 0)) < 10:
            if session_order == 1:
                slopes = (0.0, 0.0, 0.45 * (p_right + p_left) + 0.1 * p_zero - 0.5)
            elif previous_slopes is None:
                slopes = (math.nan, math.nan, math.nan)
            else:
                slopes = previous_slopes
        else:
            slopes = (p_zero - p_left, p_right - p_zero, p_zero - 0.5)
        previous_slopes = slopes

        nonzero_reward = reward[side != 0]
        accuracy = float(np.mean(nonzero_reward))
        if len(nonzero_reward) < 10:
            accuracy = 0.5
        rows.append(
            SessionMetrics(
                subject=subject,
                session_order=session_order,
                left_slope=slopes[0],
                right_slope=slopes[1],
                zero_bias=slopes[2],
                accuracy=accuracy,
            )
        )
    return rows


def run(path: Path, *, check: bool = True) -> BenchmarkResult:
    """Run the Figure 1 reproduction and optionally enforce its numerical contract."""

    observed_sha256 = sha256(path)
    if observed_sha256 != MEMBER_SHA256:
        raise ValueError(
            f"input checksum mismatch: observed {observed_sha256}, expected {MEMBER_SHA256}"
        )
    study = load_study(path)
    session_rows = calculate_session_metrics(study)
    by_subject: dict[str, list[SessionMetrics]] = defaultdict(list)
    for row in session_rows:
        by_subject[row.subject].append(row)

    early_bias: list[float] = []
    late_bias: list[float] = []
    late_slope_difference: list[float] = []
    first_accuracy: list[float] = []
    last_accuracy: list[float] = []
    for rows in by_subject.values():
        rows.sort(key=lambda row: row.session_order)
        maximum = rows[-1].session_order
        early = [row for row in rows if 3 < row.session_order <= 8]
        late = [row for row in rows if maximum - 5 < row.session_order <= maximum]
        early_bias.append(float(np.mean([row.zero_bias for row in early])))
        late_bias.append(float(np.mean([row.zero_bias for row in late])))
        late_slope_difference.append(
            float(np.mean([row.right_slope - row.left_slope for row in late]))
        )
        first_accuracy.append(rows[0].accuracy)
        last_accuracy.append(rows[-1].accuracy)

    bias_correlation = stats.pearsonr(early_bias, late_bias)
    slope_correlation = stats.pearsonr(early_bias, late_slope_difference)
    eligible_subjects = set(by_subject)
    source_sessions = {
        (str(subject), str(session))
        for subject, session in zip(study["subject"], study["session"], strict=True)
        if str(subject) in eligible_subjects
    }
    values = {
        "n_trials": len(study),
        "n_subjects": len(by_subject),
        "n_source_sessions": len(source_sessions),
        "n_sessions": len(session_rows),
        "early_late_bias_r": float(bias_correlation.statistic),
        "early_late_bias_p": float(bias_correlation.pvalue),
        "early_bias_late_slope_r": float(slope_correlation.statistic),
        "early_bias_late_slope_p": float(slope_correlation.pvalue),
        "first_session_accuracy": float(np.mean(first_accuracy)),
        "last_session_accuracy": float(np.mean(last_accuracy)),
    }
    passed = contract_matches(values)
    if check and not passed:
        differences = {
            key: {"observed": values[key], "expected": expected}
            for key, expected in EXPECTED.items()
            if not _matches(values[key], expected)
        }
        raise AssertionError(f"Cell 2025 reproduction contract failed: {differences}")
    return BenchmarkResult(
        benchmark="Liebana, Laffere et al. (2025), Cell, Figure 1G/I",
        source_doi=FIGSHARE_ARTICLE_DOI,
        source_file_id=FIGSHARE_FILE_ID,
        source_member=ARCHIVE_MEMBER,
        source_member_sha256=observed_sha256,
        accuracy_change=values["last_session_accuracy"] - values["first_session_accuracy"],
        contract_passed=passed,
        **values,
    )


def contract_matches(values: dict[str, int | float]) -> bool:
    return all(_matches(values[key], expected) for key, expected in EXPECTED.items())


def _matches(observed: int | float, expected: int | float) -> bool:
    if isinstance(expected, int):
        return observed == expected
    return math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-12)


def _reaction_time(row: dict[str, str]) -> float:
    return float(row["choiceCompleteTime"]) - float(row["stimulusOnsetTime"])


def _integer(value: str) -> int:
    return int(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="checksum-pinned Cell 2025 behaviour CSV")
    parser.add_argument("--no-check", action="store_true", help="report without enforcing contract")
    args = parser.parse_args()
    result = run(args.data.resolve(), check=not args.no_check)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
