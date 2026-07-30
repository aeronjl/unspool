"""Bounded replication of Ashwood et al. (2022), Nature Neuroscience 25:201-212.

The paper is the canonical GLM-HMM analysis of mouse perceptual decision-making. Behavio
ships a Bernoulli GLM-HMM, so the replication asks a direct question: on the exact public
data, with the paper's declared covariates and prior scale, does Behavio's implementation
recover the numbers the paper printed?

What is reproduced and what is not is fixed in ``PROTOCOL.md``, written before any of these
numbers existed, and machine-readable in ``published_claims.json``. The short version: the
cohort construction is reproduced exactly; the per-animal three-state fits are reproduced
with a documented set of substitutions; the pooled global fit, the violation-trial
observation mask and the full K-selection sweep across all 37 animals are out of scope, and
say so in writing.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    MixtureModel,
    Psychometric,
    Study,
    UniformChoiceGuess,
    mix,
)
from behavio.models import PredictionMode
from benchmarks.ashwood2022_glmhmm.fetch_data import (
    ARCHIVE_LICENCE,
    ARCHIVE_SHA256,
    FIGSHARE_ARTICLE_DOI,
    FIGSHARE_FILE_ID,
    digest,
)
from benchmarks.provenance import render

# ---------------------------------------------------------------------------
# Frozen analysis constants. Every value here is justified in PROTOCOL.md.
# ---------------------------------------------------------------------------

#: Ashwood's session filter: only sessions whose probabilityLeft takes exactly these values.
BIAS_BLOCK_PROBABILITIES = (0.2, 0.5, 0.8)
#: Ashwood models only the unbiased sub-block of each session.
UNBIASED_PROBABILITY = 0.5
#: A session is dropped if the unbiased block holds this many no-response trials or more.
MAXIMUM_SESSION_VIOLATIONS = 10
#: An animal is retained only with at least this many bias-block sessions.
MINIMUM_SESSIONS_PER_ANIMAL = 30
#: The animal plotted throughout the paper's Figures 2 and 3. The article calls it only "an
#: example mouse"; the identifier comes from the reference implementation, which branches on
#: ``animal == "CSHL_008"`` when drawing those figures.
EXAMPLE_ANIMAL = "CSHL_008"

#: Ashwood's M = 4 inputs are bias, stimulus, previous choice and win-stay/lose-switch.
COVARIATES = ("stimulus", "wsls")
CHOICE_LAGS = 1
#: The paper's Gaussian prior on GLM weights has standard deviation sigma = 2. Behavio's
#: ``l2`` enters the objective as 0.5 * l2 * ||w||^2, so l2 = 1 / sigma^2.
PRIOR_SIGMA = 2.0
L2 = 1.0 / PRIOR_SIGMA**2
#: Behavio canonicalizes latent labels by a named coefficient; the paper's "engaged" state
#: is the one with the largest stimulus weight, so ordering by stimulus is the right axis.
LABEL_BY = "stimulus"
N_RESTARTS = 2
N_FOLDS = 5
FOLD_SEED = 65
STATE_COUNTS = (1, 2, 3, 4, 5)
PAPER_STATE_COUNT = 3

#: Values printed in the paper, restated here for the numerical gate. Every entry is a
#: number the article itself prints; the location of each is recorded in PROTOCOL.md.
#:
#: The paper's choice of three states is deliberately absent. It selects three states for
#: "all subsequent analyses" on plateau and parsimony grounds and never claims the
#: three-state model beats the four- and five-state models, so an arg-max over K is not a
#: published value and is reported without being asserted.
PUBLISHED = {
    "n_animals": 37,
    "n_sessions": 2017,
    "n_source_trials": 181_530,
    "example_n_trials": 5040,
    "example_n_sessions": 56,
    "example_state_accuracies": (0.90, 0.60, 0.58),
    "median_dwell_times": (24.0, 13.0, 12.0),
    "median_engaged_occupancy": 0.69,
    "bits_per_trial_over_lapse": 0.09,
    "bits_per_trial_over_single_state": 0.13,
}

TRIALS_PER_UNBIASED_BLOCK = 90


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One retained unbiased block, in the coding Ashwood's preprocessing produces."""

    animal: str
    lab: str
    session: str
    date: str
    number: str
    signed_contrast: NDArray[np.float64]
    choice: NDArray[np.int64]
    reward: NDArray[np.float64]

    @property
    def n_source_trials(self) -> int:
        return len(self.choice)

    @property
    def n_violations(self) -> int:
        return int(np.count_nonzero(self.choice < 0))


@dataclass(frozen=True, slots=True)
class Cohort:
    """The 37-animal panel, with the counts that identify it."""

    sessions: tuple[SessionRecord, ...]
    animals: tuple[str, ...]
    stimulus_mean: float
    stimulus_scale: float

    @property
    def n_source_trials(self) -> int:
        return sum(record.n_source_trials for record in self.sessions)

    @property
    def n_violations(self) -> int:
        return sum(record.n_violations for record in self.sessions)


def _archive_sessions(archive: zipfile.ZipFile) -> tuple[str, ...]:
    marker = "/alf/_ibl_trials.choice.npy"
    return tuple(
        sorted({name[: -len(marker)] for name in archive.namelist() if name.endswith(marker)})
    )


def _dataset(archive: zipfile.ZipFile, session: str, name: str) -> NDArray[Any]:
    payload = archive.read(f"{session}/alf/_ibl_trials.{name}.npy")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _remap_choice(raw: NDArray[Any]) -> NDArray[np.int64]:
    """Map IBL's clockwise/counter-clockwise/no-response coding onto 0/1/-1.

    IBL codes a clockwise wheel turn (the correct response to a left stimulus) as +1, a
    counter-clockwise turn as -1, and a no-response trial as 0. Ashwood remaps these to
    left = 0, right = 1, violation = -1, so that the modelled Bernoulli outcome is
    "chose right".
    """

    remapped = np.full(len(raw), -1, dtype=np.int64)
    remapped[raw == 1] = 0
    remapped[raw == -1] = 1
    return remapped


def load_cohort(path: Path) -> Cohort:
    """Rebuild Ashwood's 37-animal panel directly from the pinned public archive.

    The three filters are Ashwood's, in his order: keep sessions whose ``probabilityLeft``
    takes exactly the three bias-block values; keep an animal only with at least thirty such
    sessions; then drop any session whose unbiased sub-block holds ten or more no-response
    trials.
    """

    with zipfile.ZipFile(path) as archive:
        candidates: dict[str, list[str]] = defaultdict(list)
        for session in _archive_sessions(archive):
            probabilities = np.unique(_dataset(archive, session, "probabilityLeft"))
            if probabilities.shape == (3,) and np.array_equal(
                probabilities, np.asarray(BIAS_BLOCK_PROBABILITIES)
            ):
                candidates[session.split("Subjects/")[1].split("/")[0]].append(session)

        animals = tuple(
            sorted(
                animal
                for animal, sessions in candidates.items()
                if len(sessions) >= MINIMUM_SESSIONS_PER_ANIMAL
            )
        )
        records: list[SessionRecord] = []
        for animal in animals:
            for session in sorted(candidates[animal]):
                probabilities = _dataset(archive, session, "probabilityLeft")
                positions = np.flatnonzero(probabilities == UNBIASED_PROBABILITY)
                choice = _remap_choice(_dataset(archive, session, "choice")[positions])
                if int(np.count_nonzero(choice < 0)) >= MAXIMUM_SESSION_VIOLATIONS:
                    continue
                left = np.nan_to_num(_dataset(archive, session, "contrastLeft")[positions])
                right = np.nan_to_num(_dataset(archive, session, "contrastRight")[positions])
                relative = session.split("Subjects/")[1]
                lab = session.split("/Subjects/")[0].rsplit("/", 1)[-1]
                _, date, number = relative.split("/")
                records.append(
                    SessionRecord(
                        animal=animal,
                        lab=lab,
                        session=relative.replace("/", "-"),
                        date=date,
                        number=number,
                        signed_contrast=np.asarray(right - left, dtype=np.float64),
                        choice=choice,
                        reward=np.asarray(
                            _dataset(archive, session, "feedbackType")[positions],
                            dtype=np.float64,
                        ),
                    )
                )

    contrasts = np.concatenate([record.signed_contrast for record in records])
    return Cohort(
        sessions=tuple(records),
        animals=animals,
        stimulus_mean=float(np.mean(contrasts)),
        stimulus_scale=float(np.std(contrasts)),
    )


def animal_study(cohort: Cohort, animal: str) -> Study:
    """Build one animal's ``Study``, dropping no-response trials.

    Ashwood keeps violation trials in the sequence and replaces their emission likelihood
    with one. Behavio's GLM-HMM has no observation mask, so the violation rows are removed
    instead. Because the two history regressors are constructed from the *retained* choices,
    dropping a violation reproduces Ashwood's own rule of carrying the previous non-violation
    choice forward. The residual difference is the first trial of each session, where Ashwood
    seeds the history from the trial's own choice and Behavio uses zero.
    """

    columns: dict[str, list[Any]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "lab": [],
        "stimulus": [],
        "signed_contrast": [],
        "wsls": [],
        "choice": [],
    }
    records = [record for record in cohort.sessions if record.animal == animal]
    for order, record in enumerate(sorted(records, key=lambda item: (item.date, item.number))):
        previous_choice: int | None = None
        previous_reward = 0.0
        for position, outcome in enumerate(record.choice.tolist()):
            if outcome < 0:
                continue
            if previous_choice is None:
                wsls = 0.0
            else:
                wsls = previous_reward * (2.0 * previous_choice - 1.0)
            columns["subject"].append(animal)
            columns["session"].append(record.session)
            columns["trial"].append(position)
            columns["session_order"].append(order)
            columns["lab"].append(record.lab)
            contrast = float(record.signed_contrast[position])
            columns["signed_contrast"].append(contrast)
            columns["stimulus"].append((contrast - cohort.stimulus_mean) / cohort.stimulus_scale)
            columns["wsls"].append(wsls)
            columns["choice"].append(int(outcome))
            previous_choice = int(outcome)
            previous_reward = float(record.reward[position])
    return Study.from_columns(columns)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def glm_hmm(n_states: int) -> BernoulliGLMHMM:
    """Return the frozen GLM-HMM configuration for a given state count."""

    return BernoulliGLMHMM(
        predictors=COVARIATES,
        choice_lags=CHOICE_LAGS,
        n_states=n_states,
        n_restarts=N_RESTARTS,
        l2=L2,
        label_by=LABEL_BY,
    )


def single_state_glm() -> BernoulliHistoryGLM:
    """Ashwood's K = 1 comparison model: the same GLM without latent states."""

    return BernoulliHistoryGLM(predictors=COVARIATES, choice_lags=CHOICE_LAGS, l2=L2)


def lapse_model() -> MixtureModel:
    """Behavio's nearest available stand-in for the paper's classic lapse model."""

    return mix(
        Psychometric(stimulus="stimulus", l2=L2),
        UniformChoiceGuess(),
        weight_bounds=(0.0, 0.2),
    )


def paper_state_order(coefficients: NDArray[np.float64]) -> tuple[int, ...]:
    """Order three fitted states as engaged, biased-left, biased-right.

    This is Ashwood's ``calculate_state_permutation``: the engaged state is the one with the
    largest stimulus weight, and of the remaining two the biased-left state is the one with
    the smaller bias (intercept) weight, because a smaller bias favours the left choice.
    """

    stimulus_index = 1 + COVARIATES.index("stimulus")
    engaged = int(np.argmax(coefficients[:, stimulus_index]))
    remaining = [state for state in range(len(coefficients)) if state != engaged]
    remaining.sort(key=lambda state: float(coefficients[state, 0]))
    return (engaged, *remaining)


def session_folds(
    study: Study, *, n_folds: int = N_FOLDS, seed: int = FOLD_SEED
) -> tuple[NDArray[np.intp], ...]:
    """Assign whole sessions to folds, reproducing Ashwood's randomized allocation.

    The paper randomizes the allocation of *sessions* to folds for each animal. That is an
    interpolation design rather than a prospective one: a held-out session may precede a
    training session in time. It is used here because the published number being checked was
    produced under it, and the choice is recorded in ``PROTOCOL.md`` rather than smoothed
    over.
    """

    sessions = np.asarray([str(value) for value in study["session"]])
    unique = np.unique(sessions)
    generator = np.random.default_rng(seed)
    assignments = np.repeat(np.arange(n_folds), math.ceil(len(unique) / n_folds))
    assignments = generator.permutation(assignments)[: len(unique)]
    lookup = dict(zip(unique.tolist(), assignments.tolist(), strict=True))
    membership = np.asarray([lookup[session] for session in sessions])
    return tuple(np.flatnonzero(membership == fold).astype(np.intp) for fold in range(n_folds))


def bits_per_trial(
    log_probability: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    baseline_rate: float,
) -> float:
    """Test log-likelihood above a Bernoulli null, in bits per trial.

    The paper normalizes held-out log-likelihood against a coin-flip model whose success
    probability is the training-set fraction of rightward choices, scores the *observed*
    held-out choices under it, then divides the difference by the number of test trials and
    converts to bits. This is Ashwood's ``calculate_baseline_test_ll``.
    """

    scores = np.asarray(log_probability, dtype=np.float64)
    observed = np.asarray(outcomes, dtype=np.float64)
    if scores.shape != observed.shape:
        raise ValueError("log probabilities and outcomes must align")
    null = float(
        observed.sum() * math.log(baseline_rate)
        + (len(observed) - observed.sum()) * math.log1p(-baseline_rate)
    )
    return float((scores.sum() - null) / (len(scores) * math.log(2.0)))


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossValidationScore:
    """Held-out bits per trial for one candidate, averaged over folds."""

    model: str
    n_states: int
    fold_bits_per_trial: tuple[float, ...]
    mean_bits_per_trial: float
    converged_folds: int


@dataclass(frozen=True, slots=True)
class AnimalFit:
    """A single animal's three-state fit, in the paper's state order."""

    animal: str
    n_trials: int
    n_sessions: int
    dwell_times: tuple[float, ...]
    occupancy: tuple[float, ...]
    hard_occupancy: tuple[float, ...]
    emission_coefficients: tuple[tuple[float, ...], ...]
    converged: bool
    label_ambiguous: bool
    low_occupancy: bool


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Everything the parity contract reads, plus the diagnostics behind it."""

    benchmark: str
    source_doi: str
    source_file_id: int
    source_sha256: str
    source_licence: str
    reference_implementation: str
    n_animals: int
    n_sessions: int
    n_source_trials: int
    n_violation_trials: int
    n_modelled_trials: int
    example_animal: str
    example_n_trials: int
    example_n_sessions: int
    selected_state_count: int
    bits_per_trial_by_state_count: dict[str, float]
    bits_per_trial_gain_beyond_three_states: float
    example_state_accuracies: tuple[float | None, ...]
    example_state_trial_counts: tuple[int, ...]
    example_overall_accuracy: float | None
    bits_per_trial_over_lapse: float
    bits_per_trial_over_single_state: float
    median_dwell_times: tuple[float, ...]
    median_engaged_occupancy: float
    median_engaged_hard_occupancy: float
    cross_validation: tuple[dict[str, Any], ...]
    animal_fits: tuple[dict[str, Any], ...]
    n_unconverged_animal_fits: int
    classification: str
    claim_classification: dict[str, str]
    contract_passed: bool
    contract_failures: tuple[str, ...] = field(default_factory=tuple)


def _candidates() -> tuple[tuple[str, int], ...]:
    return (("glm", 1), ("lapse", 1), *(("glm-hmm", k) for k in STATE_COUNTS if k > 1))


def _build_candidate(name: str, n_states: int) -> Any:
    if name == "glm":
        return single_state_glm()
    if name == "lapse":
        return lapse_model()
    return glm_hmm(n_states)


def _payload(study: Study) -> dict[str, NDArray[Any]]:
    """Return a picklable copy of a study's columns.

    ``Study`` wraps its columns in a ``MappingProxyType`` so callers cannot mutate them,
    which also makes it unpicklable. Worker processes therefore receive plain arrays and
    rebuild the study, paying the validation cost once per task.
    """

    return {name: np.array(study[name]) for name in study.columns}


def _score_fold(
    task: tuple[str, int, dict[str, NDArray[Any]], NDArray[np.intp]],
) -> tuple[str, int, float, bool]:
    """Fit one candidate on one training fold and score the held-out sessions."""

    name, n_states, columns, test = task
    study = Study.from_columns(columns)
    model = _build_candidate(name, n_states)
    mask = np.ones(len(study), dtype=bool)
    mask[test] = False
    training = study.take(np.flatnonzero(mask).astype(np.intp))
    fit = model.fit(training)
    baseline = float(np.mean(training["choice"]))
    held_out = study.take(test)
    log_probability = model.pointwise_log_prob(held_out, fit, mode=PredictionMode.FILTERED)
    score = bits_per_trial(log_probability, held_out["choice"], baseline)
    return name, n_states, score, bool(fit.diagnostics.converged)


def cross_validate_example_animal(
    study: Study,
    *,
    workers: int = 1,
) -> tuple[CrossValidationScore, ...]:
    """Score every candidate state count and the lapse model on held-out sessions.

    Folds and candidates are independent, so they are distributed over processes. The work
    each process does is unchanged, and results are reassembled in declaration order.
    """

    folds = session_folds(study)
    columns = _payload(study)
    tasks = [(name, n_states, columns, test) for name, n_states in _candidates() for test in folds]
    outcomes = list(_map(_score_fold, tasks, workers=workers, stage="cross-validation"))

    scores: list[CrossValidationScore] = []
    for name, n_states in _candidates():
        selected = [row for row in outcomes if row[0] == name and row[1] == n_states]
        fold_scores = [row[2] for row in selected]
        scores.append(
            CrossValidationScore(
                model=name,
                n_states=n_states,
                fold_bits_per_trial=tuple(fold_scores),
                mean_bits_per_trial=float(np.mean(fold_scores)),
                converged_folds=sum(int(row[3]) for row in selected),
            )
        )
    return tuple(scores)


def _map(function: Any, tasks: list[Any], *, workers: int, stage: str = "") -> list[Any]:
    """Run independent fits, optionally across processes, reporting progress to stderr.

    Progress goes to stderr only. Nothing here reaches ``result.json``, so a committed
    result stays byte-identical across re-runs at different worker counts.
    """

    total = len(tasks)
    started = time.monotonic()
    results: list[Any] = []

    def note() -> None:
        elapsed = time.monotonic() - started
        print(
            f"[{stage}] {len(results)}/{total} after {elapsed:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    if workers <= 1 or total <= 1:
        for task in tasks:
            results.append(function(task))
            note()
        return results
    with ProcessPoolExecutor(max_workers=min(workers, total)) as pool:
        for outcome in pool.map(function, tasks):
            results.append(outcome)
            note()
    return results


def state_conditioned_accuracy(
    study: Study,
    model: BernoulliGLMHMM,
    fit: Any,
    order: tuple[int, ...],
    *,
    threshold: float = 0.9,
) -> tuple[tuple[float | None, int], ...]:
    """Accuracy on non-zero-contrast trials that a state explains with probability >= 0.9.

    Ashwood conditions on the smoothed marginal posterior. Behavio publishes filtered and
    one-step-ahead predictive state probabilities but not smoothed ones, so the filtered
    distribution is used and the substitution is declared in ``PROTOCOL.md``. A state that
    never reaches the threshold reports ``None`` and a count of zero rather than a number
    computed from nothing.
    """

    probabilities = model.state_probabilities(study, fit).filtered
    contrast = np.asarray(study["signed_contrast"], dtype=np.float64)
    choice = np.asarray(study["choice"], dtype=np.int64)
    correct = (np.sign(contrast) + 1.0) / 2.0
    informative = contrast != 0.0
    accuracies: list[tuple[float | None, int]] = []
    for state in order:
        selected = informative & (probabilities[:, state] >= threshold)
        count = int(np.count_nonzero(selected))
        value = float(np.mean(choice[selected] == correct[selected])) if count else None
        accuracies.append((value, count))
    return tuple(accuracies)


def fit_animal(
    study: Study,
    animal: str,
) -> tuple[AnimalFit, BernoulliGLMHMM, Any, tuple[int, ...]]:
    """Fit the paper's three-state model to one animal and summarize it in paper order."""

    model = glm_hmm(PAPER_STATE_COUNT)
    fit = model.fit(study)
    components = model.parameter_components(fit)
    order = paper_state_order(components.emission_coefficients)
    transition = np.asarray(components.transition_matrix)
    dwell = tuple(1.0 / (1.0 - float(transition[state, state])) for state in order)
    occupancy = tuple(float(fit.state_occupancy[state]) for state in order)
    filtered = model.state_probabilities(study, fit).filtered
    labels = np.argmax(filtered, axis=1)
    hard = tuple(float(np.mean(labels == state)) for state in order)
    return (
        AnimalFit(
            animal=animal,
            n_trials=len(study),
            n_sessions=len(np.unique(np.asarray([str(value) for value in study["session"]]))),
            dwell_times=dwell,
            occupancy=occupancy,
            hard_occupancy=hard,
            emission_coefficients=tuple(
                tuple(float(value) for value in components.emission_coefficients[state])
                for state in order
            ),
            converged=bool(fit.diagnostics.converged),
            label_ambiguous=bool(fit.label_ambiguous),
            low_occupancy=bool(fit.low_occupancy),
        ),
        model,
        fit,
        order,
    )


def _fit_animal_task(
    task: tuple[str, dict[str, NDArray[Any]]],
) -> tuple[AnimalFit, tuple[tuple[float | None, int], ...] | None, float | None]:
    """Fit one animal and, for the example animal, also score its per-state accuracy."""

    animal, columns = task
    study = Study.from_columns(columns)
    summary, model, fit, order = fit_animal(study, animal)
    if animal != EXAMPLE_ANIMAL:
        return summary, None, None
    contrast = np.asarray(study["signed_contrast"], dtype=np.float64)
    choice = np.asarray(study["choice"], dtype=np.int64)
    informative = contrast != 0.0
    overall = float(np.mean(choice[informative] == (np.sign(contrast[informative]) + 1.0) / 2.0))
    return summary, state_conditioned_accuracy(study, model, fit, order), overall


def run(path: Path, *, animals: int | None = None, workers: int = 1) -> BenchmarkResult:
    """Execute the whole replication and return its stamped result."""

    observed_sha256 = digest(path)
    if observed_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            f"input checksum mismatch: observed {observed_sha256}, expected {ARCHIVE_SHA256}"
        )
    cohort = load_cohort(path)
    studies = {animal: animal_study(cohort, animal) for animal in cohort.animals}

    example = studies[EXAMPLE_ANIMAL]
    scores = cross_validate_example_animal(example, workers=workers)
    ranked = {(score.model, score.n_states): score.mean_bits_per_trial for score in scores}
    hmm_scores = {
        n_states: ranked[("glm-hmm", n_states)] for n_states in STATE_COUNTS if n_states > 1
    }
    hmm_scores[1] = ranked[("glm", 1)]
    selected = max(hmm_scores, key=lambda key: hmm_scores[key])
    plateau_gain = hmm_scores[max(STATE_COUNTS)] - hmm_scores[PAPER_STATE_COUNT]
    over_lapse = hmm_scores[PAPER_STATE_COUNT] - ranked[("lapse", 1)]
    over_single = hmm_scores[PAPER_STATE_COUNT] - hmm_scores[1]

    selection = cohort.animals if animals is None else cohort.animals[:animals]
    outcomes = _map(
        _fit_animal_task,
        [(animal, _payload(studies[animal])) for animal in selection],
        workers=workers,
        stage="per-animal fits",
    )
    fits = [row[0] for row in outcomes]
    example_accuracies: tuple[float | None, ...] = ()
    example_counts: tuple[int, ...] = ()
    example_overall: float | None = None
    for _, accuracies, overall in outcomes:
        if accuracies is not None and overall is not None:
            example_accuracies = tuple(value for value, _ in accuracies)
            example_counts = tuple(count for _, count in accuracies)
            example_overall = overall

    dwell = np.asarray([summary.dwell_times for summary in fits], dtype=np.float64)
    occupancy = np.asarray([summary.occupancy for summary in fits], dtype=np.float64)
    hard_occupancy = np.asarray([summary.hard_occupancy for summary in fits], dtype=np.float64)

    values: dict[str, Any] = {
        "n_animals": len(cohort.animals),
        "n_sessions": len(cohort.sessions),
        "n_source_trials": cohort.n_source_trials,
        "n_violation_trials": cohort.n_violations,
        "n_modelled_trials": sum(len(study) for study in studies.values()),
        "example_animal": EXAMPLE_ANIMAL,
        "example_n_trials": len(example),
        "example_n_sessions": len(np.unique(np.asarray([str(v) for v in example["session"]]))),
        "selected_state_count": int(selected),
        "bits_per_trial_by_state_count": {str(k): hmm_scores[k] for k in sorted(hmm_scores)},
        "bits_per_trial_gain_beyond_three_states": float(plateau_gain),
        "example_state_accuracies": example_accuracies,
        "example_state_trial_counts": example_counts,
        "example_overall_accuracy": example_overall,
        "bits_per_trial_over_lapse": float(over_lapse),
        "bits_per_trial_over_single_state": float(over_single),
        "median_dwell_times": tuple(float(value) for value in np.median(dwell, axis=0)),
        "median_engaged_occupancy": float(np.median(occupancy[:, 0])),
        "median_engaged_hard_occupancy": float(np.median(hard_occupancy[:, 0])),
    }
    failures = contract_failures(values)
    return BenchmarkResult(
        classification=FAILED_CLASSIFICATION if failures else PASSED_CLASSIFICATION,
        claim_classification=claim_classification(values),
        benchmark="Ashwood et al. (2022), Nature Neuroscience, Figures 2-4",
        source_doi=FIGSHARE_ARTICLE_DOI,
        source_file_id=FIGSHARE_FILE_ID,
        source_sha256=observed_sha256,
        source_licence=ARCHIVE_LICENCE,
        reference_implementation="github.com/zashwood/glm-hmm (a fork of Linderman's ssm)",
        cross_validation=tuple(asdict(score) for score in scores),
        animal_fits=tuple(asdict(summary) for summary in fits),
        n_unconverged_animal_fits=sum(1 for summary in fits if not summary.converged),
        contract_passed=not failures,
        contract_failures=failures,
        **values,
    )


#: Each in-scope claim, as it is identified in ``published_claims.json``, paired with the
#: result field it reads and the tolerance declared for it in PROTOCOL.md section 9.
CLAIM_CHECKS: tuple[tuple[str, str, Any, str, float], ...] = (
    ("n_animals", "n_animals", None, "exact", 0.0),
    ("n_sessions", "n_sessions", None, "exact", 0.0),
    ("n_source_trials", "n_source_trials", None, "exact", 0.0),
    ("example_mouse_n_trials", "example_n_trials", None, "exact", 0.0),
    ("example_mouse_n_sessions", "example_n_sessions", None, "exact", 0.0),
    ("example_engaged_accuracy", "example_state_accuracies", 0, "absolute", 0.05),
    ("example_biased_left_accuracy", "example_state_accuracies", 1, "absolute", 0.05),
    ("example_biased_right_accuracy", "example_state_accuracies", 2, "absolute", 0.05),
    ("median_engaged_dwell_time", "median_dwell_times", 0, "relative", 0.25),
    ("median_biased_left_dwell_time", "median_dwell_times", 1, "relative", 0.25),
    ("median_biased_right_dwell_time", "median_dwell_times", 2, "relative", 0.25),
    ("median_engaged_occupancy", "median_engaged_occupancy", None, "absolute", 0.05),
    ("bits_per_trial_over_lapse", "bits_per_trial_over_lapse", None, "absolute", 0.02),
    (
        "bits_per_trial_over_single_state",
        "bits_per_trial_over_single_state",
        None,
        "absolute",
        0.02,
    ),
)

PASSED_CLASSIFICATION = "published-parity"
FAILED_CLASSIFICATION = "failed-parity"


def claim_outcomes(values: dict[str, Any]) -> dict[str, bool]:
    """Return, for every in-scope published value, whether it landed inside its band."""

    outcomes: dict[str, bool] = {}
    for identifier, key, index, kind, tolerance in CLAIM_CHECKS:
        published = PUBLISHED[key]
        observed = values[key]
        if index is not None:
            published = published[index]
            observed = observed[index] if index < len(observed) else None
        if kind == "exact":
            outcomes[identifier] = observed == published
        elif kind == "relative":
            outcomes[identifier] = _close(observed, published, tolerance * abs(float(published)))
        else:
            outcomes[identifier] = _close(observed, published, tolerance)
    return outcomes


def claim_classification(values: dict[str, Any]) -> dict[str, str]:
    """Classify each in-scope claim on the repository's evidence ladder."""

    return {
        identifier: PASSED_CLASSIFICATION if passed else FAILED_CLASSIFICATION
        for identifier, passed in claim_outcomes(values).items()
    }


def contract_failures(values: dict[str, Any]) -> tuple[str, ...]:
    """Return the identifiers of every in-scope published value that did not reproduce."""

    return tuple(identifier for identifier, passed in claim_outcomes(values).items() if not passed)


def _close(observed: float | None, published: float, tolerance: float) -> bool:
    if observed is None or not np.isfinite(observed):
        return False
    return abs(float(observed) - float(published)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("data") / "ibl-behavior-data-Dec2019.zip",
        help="checksum-pinned IBL behaviour archive",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    parser.add_argument(
        "--animals",
        type=int,
        help="fit only the first N animals; for smoke tests, never for a committed result",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="processes used for the independent per-animal and per-fold fits",
    )
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args.data.resolve(), animals=args.animals, workers=args.workers)
    rendered = render(asdict(result))
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # Wall-clock time is reported but never committed: a stamped result stays byte-identical
    # across re-runs on an unchanged tree, so a real numerical drift cannot hide behind it.
    print(f"elapsed: {time.monotonic() - started:.1f}s ({args.workers} workers)", file=sys.stderr)


if __name__ == "__main__":
    main()
