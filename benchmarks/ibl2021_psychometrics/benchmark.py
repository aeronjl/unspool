"""Replicate published psychometric and training-duration values from IBL et al. (2021).

Every quantity, cohort rule and tolerance computed here was fixed in ``PROTOCOL.md`` before
the analysis was run. The gate is ``contract_passed``: true only when every non-waived
published value is recovered inside its pre-declared tolerance.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio import Study
from behavio.adapters import IBLONETrialSource, read_ibl_one_sessions
from benchmarks.ibl2021_psychometrics.fetch_data import (
    DEFAULT_CACHE,
    INSTITUTIONS,
    LICENCE,
    PUBLIC_ALYX_URL,
    RELEASE_TAG,
    SOURCE_DOI,
    load_manifest,
    open_one,
)
from benchmarks.ibl2021_psychometrics.psychometric import (
    CRITERION_BOX,
    DEFAULT_SEED,
    REPORTED_BOX,
    contrast_summary,
    fit_psychometric,
)
from benchmarks.provenance import render

BENCHMARK = "International Brain Laboratory et al. (2021), eLife 10:e63711"

TRIAL_COLUMNS = (
    "contrastLeft",
    "contrastRight",
    "choice",
    "feedbackType",
    "probabilityLeft",
    "response_times",
    "stimOn_times",
)

WINDOW = 3
EASY_CONTRAST = 50.0
SEED_SENSITIVITY_SEEDS = (20_210_101, 1, 7, 1_234, 99_991)

#: ``trained_1a`` and ``trained_1b`` from the paper's Methods and ``ibllib``.
CRITERIA = {
    "trained_1a": {
        "min_trials": 200,
        "min_perf_easy": 0.80,
        "max_abs_bias": 16.0,
        "max_threshold": 19.0,
        "max_lapse": 0.2,
        "max_median_rt": None,
    },
    "trained_1b": {
        "min_trials": 400,
        "min_perf_easy": 0.90,
        "max_abs_bias": 10.0,
        "max_threshold": 20.0,
        "max_lapse": 0.1,
        "max_median_rt": 2.0,
    },
}

#: Published values, dispersions and the tolerances frozen in ``PROTOCOL.md``.
PUBLISHED = {
    "threshold_during_training_pct": {"value": 17.8, "sd": 11.7, "n": 140, "tolerance": 1.938},
    "easy_trial_error_pct_at_proficiency": {"value": 9.5, "sd": 3.6, "n": 7, "tolerance": 2.667},
    "threshold_pct_at_proficiency": {"value": 14.3, "sd": 3.8, "n": 7, "tolerance": 2.815},
    "training_days_to_proficiency": {"value": 18.4, "sd": 13.0, "n": 140, "tolerance": 2.153},
    "training_kilotrials_to_proficiency": {"value": 10.8, "sd": 8.6, "n": 140, "tolerance": 1.425},
}
#: Non-gating strict bands for the two claims whose published ``n`` the paper reports
#: inconsistently with its own figure panels. See PROTOCOL.md.
STRICT_BANDS = {
    "easy_trial_error_pct_at_proficiency": 0.596,
    "threshold_pct_at_proficiency": 0.630,
}
PUBLISHED_N_SUBJECTS = 140
N_SUBJECTS_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Per-session quantities the trained criterion and the published figures need."""

    subject: str
    institution: str
    session: str
    date: str
    session_order: int
    n_trials: int
    n_easy: int
    n_easy_correct: int
    perf_easy: float
    median_rt_zero_contrast: float
    signed_contrast: NDArray[np.float64] = field(repr=False)
    rightward: NDArray[np.float64] = field(repr=False)
    has_twelve_percent: bool = False


@dataclass(frozen=True, slots=True)
class SubjectSummary:
    """One subject's replicated contribution to the published cohort means."""

    subject: str
    institution: str
    criterion: str
    proficiency_index: int
    training_days: int
    training_trials: int
    easy_trial_error_pct: float
    threshold_pct_at_proficiency: float
    threshold_pct_during_training: float
    n_training_threshold_sessions: int


def sources_from_manifest(manifest: dict[str, Any]) -> tuple[IBLONETrialSource, ...]:
    """Build exact, hash-checked adapter sources in committed manifest order."""

    return tuple(
        IBLONETrialSource(
            session_id=str(row["session"]),
            dataset_id=str(row["dataset_id"]),
            dataset_path=str(row["dataset_path"]),
            file_size=int(row["file_size"]),
            md5=str(row["md5"]),
            release_tag=str(manifest["release_tag"]),
            session_order=int(row["session_order"]),
            subject=str(row["subject"]),
            session=str(row["session"]),
            lab=str(row["lab"]),
            columns=TRIAL_COLUMNS,
            source_columns={
                "institution": str(row["institution"]),
                "session_date": str(row["date"]),
                "task_protocol": str(row["task_protocol"]),
            },
            alyx_url=str(manifest["public_alyx_url"]),
        )
        for row in manifest["sessions"]
    )


def load_study(cache_directory: Path = DEFAULT_CACHE) -> Study:
    """Load every pinned trial table through the shared ONE adapter."""

    manifest = load_manifest()
    return read_ibl_one_sessions(sources_from_manifest(manifest), client=open_one(cache_directory))


def rightward_choice_sign(study: Study) -> int:
    """Derive, rather than assume, which sign of IBL's ``choice`` means 'rightward'.

    The adapter deliberately preserves IBL's native ``-1/0/+1`` coding. On unambiguous
    trials -- non-zero contrast, rewarded -- the choice sign that co-occurs with a
    right-side stimulus is the rightward code. Both candidate signs are counted and the
    mapping must be overwhelming, otherwise the convention is not safe to use.
    """

    choice = np.asarray(study["choice"], dtype=np.float64)
    feedback = np.asarray(study["feedbackType"], dtype=np.float64)
    signed = _signed_contrast(study)
    correct = (feedback == 1.0) & (signed != 0.0) & np.isfinite(signed) & (choice != 0.0)
    right_stimulus = correct & (signed > 0.0)
    negative = int(np.count_nonzero(right_stimulus & (choice < 0.0)))
    positive = int(np.count_nonzero(right_stimulus & (choice > 0.0)))
    total = negative + positive
    if total == 0:
        raise ValueError("no rewarded non-zero-contrast trials to derive the choice coding")
    if max(negative, positive) / total < 0.99:
        raise ValueError(
            f"IBL choice coding is not consistent: {negative} negative and {positive} "
            "positive choices on rewarded right-stimulus trials"
        )
    return -1 if negative > positive else 1


def _signed_contrast(study: Study) -> NDArray[np.float64]:
    left = np.nan_to_num(np.asarray(study["contrastLeft"], dtype=np.float64), nan=0.0)
    right = np.nan_to_num(np.asarray(study["contrastRight"], dtype=np.float64), nan=0.0)
    return (right - left) * 100.0


def summarize_sessions(study: Study, *, rightward_sign: int) -> list[SessionSummary]:
    """Reduce every retained session to the quantities the replication needs."""

    signed = _signed_contrast(study)
    choice = np.asarray(study["choice"], dtype=np.float64)
    feedback = np.asarray(study["feedbackType"], dtype=np.float64)
    response = np.asarray(study["response_times"], dtype=np.float64)
    stimulus_on = np.asarray(study["stimOn_times"], dtype=np.float64)

    rightward = np.where(choice == 0.0, np.nan, (choice == rightward_sign).astype(np.float64))
    reaction_time = response - stimulus_on

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(len(study)):
        grouped[str(study["session"][index])].append(index)

    summaries: list[SessionSummary] = []
    for session, indices in grouped.items():
        positions = np.asarray(indices, dtype=np.intp)
        first = positions[0]
        easy = np.abs(signed[positions]) >= EASY_CONTRAST
        easy_feedback = feedback[positions][easy]
        zero = signed[positions] == 0.0
        zero_rt = reaction_time[positions][zero]
        zero_rt = zero_rt[np.isfinite(zero_rt)]
        summaries.append(
            SessionSummary(
                subject=str(study["subject"][first]),
                institution=str(study["institution"][first]),
                session=session,
                date=str(study["session_date"][first]),
                session_order=int(study["session_order"][first]),
                n_trials=int(positions.size),
                n_easy=int(easy_feedback.size),
                n_easy_correct=int(np.count_nonzero(easy_feedback == 1.0)),
                perf_easy=(
                    float(np.mean(easy_feedback == 1.0)) if easy_feedback.size else float("nan")
                ),
                median_rt_zero_contrast=(
                    float(np.median(zero_rt)) if zero_rt.size else float("nan")
                ),
                signed_contrast=signed[positions],
                rightward=rightward[positions],
                has_twelve_percent=bool(np.any(np.isclose(np.abs(signed[positions]), 12.5))),
            )
        )
    summaries.sort(key=lambda row: (row.subject, row.session_order))
    return summaries


def _fit_window(sessions: list[SessionSummary], box: Any, seed: int) -> NDArray[np.float64]:
    signed = np.concatenate([row.signed_contrast for row in sessions])
    rightward = np.concatenate([row.rightward for row in sessions])
    levels, counts, proportions = contrast_summary(signed, rightward)
    return fit_psychometric(levels, counts, proportions, box=box, seed=seed)


def _criterion_met(
    name: str, parameters: NDArray[np.float64], window: list[SessionSummary]
) -> bool:
    rule = CRITERIA[name]
    if not np.all(np.isfinite(parameters)):
        return False
    bias, threshold, lapse_low, lapse_high = (float(value) for value in parameters)
    if not (abs(bias) < rule["max_abs_bias"] and threshold < rule["max_threshold"]):
        return False
    if not (lapse_low < rule["max_lapse"] and lapse_high < rule["max_lapse"]):
        return False
    if any(row.n_trials <= rule["min_trials"] for row in window):
        return False
    if any(not (row.perf_easy > rule["min_perf_easy"]) for row in window):
        return False
    if rule["max_median_rt"] is not None:
        pooled = np.concatenate([row.signed_contrast for row in window])
        if not np.any(pooled == 0.0):
            return False
        median = float(np.nanmedian([row.median_rt_zero_contrast for row in window]))
        if not (math.isfinite(median) and median < rule["max_median_rt"]):
            return False
    return True


def find_proficiency(
    sessions: list[SessionSummary], *, seed: int = DEFAULT_SEED
) -> tuple[int, str] | None:
    """Return the index of the last session of the earliest qualifying triplet."""

    for end in range(WINDOW - 1, len(sessions)):
        window = sessions[end - WINDOW + 1 : end + 1]
        parameters = _fit_window(window, CRITERION_BOX, seed)
        for name in ("trained_1a", "trained_1b"):
            if _criterion_met(name, parameters, window):
                return end, name
    return None


def _summarize_subject(ordered: list[SessionSummary], *, seed: int) -> SubjectSummary | None:
    """Apply the trained criterion to one subject and compute its published quantities."""

    found = find_proficiency(ordered, seed=seed)
    if found is None:
        return None
    end, criterion = found
    window = ordered[end - WINDOW + 1 : end + 1]
    span = ordered[: end + 1]

    easy_total = sum(row.n_easy for row in window)
    easy_correct = sum(row.n_easy_correct for row in window)
    easy_error = 100.0 * (1.0 - easy_correct / easy_total) if easy_total else float("nan")

    proficiency_parameters = _fit_window(window, REPORTED_BOX, seed)

    first_twelve = next((index for index, row in enumerate(span) if row.has_twelve_percent), None)
    thresholds: list[float] = []
    if first_twelve is not None:
        for row in span[first_twelve:]:
            levels, counts, proportions = contrast_summary(row.signed_contrast, row.rightward)
            parameters = fit_psychometric(levels, counts, proportions, box=REPORTED_BOX, seed=seed)
            if np.isfinite(parameters[1]):
                thresholds.append(float(parameters[1]))

    return SubjectSummary(
        subject=ordered[0].subject,
        institution=window[0].institution,
        criterion=criterion,
        proficiency_index=end,
        training_days=len({row.date for row in span}),
        training_trials=int(sum(row.n_trials for row in span)),
        easy_trial_error_pct=easy_error,
        threshold_pct_at_proficiency=float(proficiency_parameters[1]),
        threshold_pct_during_training=(float(np.mean(thresholds)) if thresholds else float("nan")),
        n_training_threshold_sessions=len(thresholds),
    )


def group_by_subject(sessions: list[SessionSummary]) -> list[list[SessionSummary]]:
    """Return each subject's sessions in chronological order, in stable subject order."""

    by_subject: dict[str, list[SessionSummary]] = defaultdict(list)
    for row in sessions:
        by_subject[row.subject].append(row)
    return [
        sorted(by_subject[subject], key=lambda row: row.session_order)
        for subject in sorted(by_subject)
    ]


def summarize_subjects(
    sessions: list[SessionSummary], *, seed: int = DEFAULT_SEED, workers: int | None = None
) -> list[SubjectSummary]:
    """Apply the trained criterion to every subject, in parallel over subjects.

    Subjects are independent and every fit is seeded, so the parallel and serial results
    are identical.
    """

    grouped = group_by_subject(sessions)
    if workers == 1:
        results = [_summarize_subject(ordered, seed=seed) for ordered in grouped]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(partial(_summarize_subject, seed=seed), grouped))
    return [row for row in results if row is not None]


def seed_sensitivity(sessions: list[SessionSummary], summaries: list[SubjectSummary]) -> float:
    """Return the spread of the proficiency-threshold mean across alternative restart seeds.

    The released fitter draws four of its five optimiser restarts from an unseeded uniform
    distribution. Pinning a seed makes this benchmark reproducible, so this diagnostic
    measures what that pinning is worth: the proficiency windows are held fixed and only the
    reported fit is repeated under each seed. It is reported, never gated.
    """

    proficient = {summary.subject for summary in summaries}
    grouped = [group for group in group_by_subject(sessions) if group[0].subject in proficient]
    windows = [
        ordered[summary.proficiency_index - WINDOW + 1 : summary.proficiency_index + 1]
        for ordered, summary in zip(
            grouped, sorted(summaries, key=lambda summary: summary.subject), strict=True
        )
    ]
    means = [
        _mean([float(_fit_window(window, REPORTED_BOX, seed)[1]) for window in windows])
        for seed in SEED_SENSITIVITY_SEEDS
    ]
    return float(np.max(means) - np.min(means))


def diagnose_non_proficient(sessions: list[SessionSummary], proficient: set[str]) -> dict[str, int]:
    """Explain why the cohort claim falls short, without changing the criterion.

    A mouse that never clears the behavioural gates -- per-session trial count and easy-trial
    performance -- in any three-session window did not learn the basic task, and no
    psychometric implementation would recruit it. A mouse that clears those gates and still
    fails is a borderline case where a different fit might disagree. Separating the two
    bounds how much of the shortfall could possibly be implementation rather than data.
    """

    short = 0
    behavioural = 0
    psychometric_only = 0
    for ordered in group_by_subject(sessions):
        if ordered[0].subject in proficient:
            continue
        if len(ordered) < WINDOW:
            short += 1
            continue
        cleared = any(
            all(row.n_trials > CRITERIA["trained_1a"]["min_trials"] for row in window)
            and all(row.perf_easy > CRITERIA["trained_1a"]["min_perf_easy"] for row in window)
            for window in (
                ordered[start : start + WINDOW] for start in range(len(ordered) - WINDOW + 1)
            )
        )
        psychometric_only += int(cleared)
        behavioural += int(not cleared)
    return {
        "n_non_proficient_with_too_few_sessions": short,
        "n_non_proficient_failing_behavioural_gates": behavioural,
        "n_non_proficient_failing_only_psychometric_bounds": psychometric_only,
    }


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _sd(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def observed_values(summaries: list[SubjectSummary]) -> dict[str, float]:
    """Return the six reproduced quantities the contract compares."""

    return {
        "threshold_during_training_pct": _mean(
            [row.threshold_pct_during_training for row in summaries]
        ),
        "easy_trial_error_pct_at_proficiency": _mean(
            [row.easy_trial_error_pct for row in summaries]
        ),
        "threshold_pct_at_proficiency": _mean(
            [row.threshold_pct_at_proficiency for row in summaries]
        ),
        "training_days_to_proficiency": _mean([float(row.training_days) for row in summaries]),
        "training_kilotrials_to_proficiency": _mean(
            [row.training_trials / 1000.0 for row in summaries]
        ),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The committed replication result and its pre-declared gate."""

    benchmark: str
    source_doi: str
    release_tag: str
    public_alyx_url: str
    licence: str
    manifest_sha256: str
    psychometric_seed: int
    rightward_choice_code: int
    n_manifest_subjects: int
    n_manifest_sessions: int
    n_trials: int
    n_proficient_subjects: int
    n_institutions: int
    n_trained_1a: int
    n_trained_1b: int
    proficiency_rate: float
    n_non_proficient_with_too_few_sessions: int
    n_non_proficient_failing_behavioural_gates: int
    n_non_proficient_failing_only_psychometric_bounds: int
    threshold_during_training_pct: float
    easy_trial_error_pct_at_proficiency: float
    threshold_pct_at_proficiency: float
    training_days_to_proficiency: float
    training_kilotrials_to_proficiency: float
    threshold_during_training_pct_sd: float
    easy_trial_error_pct_at_proficiency_sd: float
    threshold_pct_at_proficiency_sd: float
    training_days_to_proficiency_sd: float
    training_kilotrials_to_proficiency_sd: float
    claim_status: dict[str, str]
    strict_band_status: dict[str, str]
    threshold_seed_sensitivity_pct: float
    contract_passed: bool
    classification: str


def _claim_statuses(values: dict[str, float], n_subjects: int) -> dict[str, str]:
    statuses = {
        name: (
            "pass"
            if math.isfinite(values[name])
            and abs(values[name] - published["value"]) <= published["tolerance"]
            else "fail"
        )
        for name, published in PUBLISHED.items()
    }
    statuses["n_proficient_subjects"] = (
        "pass"
        if abs(n_subjects - PUBLISHED_N_SUBJECTS)
        <= N_SUBJECTS_RELATIVE_TOLERANCE * PUBLISHED_N_SUBJECTS
        else "fail"
    )
    return statuses


def run(
    cache_directory: Path = DEFAULT_CACHE,
    *,
    seed: int = DEFAULT_SEED,
) -> BenchmarkResult:
    """Run the full replication and return its committed result."""

    manifest = load_manifest()
    study = load_study(cache_directory)
    rightward_sign = rightward_choice_sign(study)
    sessions = summarize_sessions(study, rightward_sign=rightward_sign)
    summaries = summarize_subjects(sessions, seed=seed)
    values = observed_values(summaries)

    institutions = {row.institution for row in summaries}
    if not institutions <= set(INSTITUTIONS):
        raise ValueError(
            f"assembled institutions outside the paper's seven: {sorted(institutions)}"
        )

    sensitivity = seed_sensitivity(sessions, summaries)

    diagnosis = diagnose_non_proficient(sessions, {row.subject for row in summaries})
    statuses = _claim_statuses(values, len(summaries))
    strict = {
        name: (
            "pass"
            if math.isfinite(values[name]) and abs(values[name] - PUBLISHED[name]["value"]) <= band
            else "fail"
        )
        for name, band in STRICT_BANDS.items()
    }
    passed = all(status == "pass" for status in statuses.values())
    return BenchmarkResult(
        benchmark=BENCHMARK,
        source_doi=SOURCE_DOI,
        release_tag=RELEASE_TAG,
        public_alyx_url=PUBLIC_ALYX_URL,
        licence=LICENCE,
        manifest_sha256=str(manifest["sessions_sha256"]),
        psychometric_seed=seed,
        rightward_choice_code=rightward_sign,
        n_manifest_subjects=int(manifest["n_subjects"]),
        n_manifest_sessions=int(manifest["n_sessions"]),
        n_trials=len(study),
        n_proficient_subjects=len(summaries),
        n_institutions=len({row.institution for row in summaries}),
        n_trained_1a=sum(1 for row in summaries if row.criterion == "trained_1a"),
        n_trained_1b=sum(1 for row in summaries if row.criterion == "trained_1b"),
        proficiency_rate=len(summaries) / int(manifest["n_subjects"]),
        **diagnosis,
        threshold_during_training_pct=values["threshold_during_training_pct"],
        easy_trial_error_pct_at_proficiency=values["easy_trial_error_pct_at_proficiency"],
        threshold_pct_at_proficiency=values["threshold_pct_at_proficiency"],
        training_days_to_proficiency=values["training_days_to_proficiency"],
        training_kilotrials_to_proficiency=values["training_kilotrials_to_proficiency"],
        threshold_during_training_pct_sd=_sd(
            [row.threshold_pct_during_training for row in summaries]
        ),
        easy_trial_error_pct_at_proficiency_sd=_sd([row.easy_trial_error_pct for row in summaries]),
        threshold_pct_at_proficiency_sd=_sd(
            [row.threshold_pct_at_proficiency for row in summaries]
        ),
        training_days_to_proficiency_sd=_sd([float(row.training_days) for row in summaries]),
        training_kilotrials_to_proficiency_sd=_sd(
            [row.training_trials / 1000.0 for row in summaries]
        ),
        claim_status=statuses,
        strict_band_status=strict,
        threshold_seed_sensitivity_pct=sensitivity,
        contract_passed=passed,
        classification="published-parity" if passed else "failed-parity",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    args = parser.parse_args()
    result = run(args.cache)
    rendered = render(asdict(result), libraries=("one-api",))
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
