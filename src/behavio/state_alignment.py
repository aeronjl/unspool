"""Permutation-invariant recovery diagnostics for simulated latent states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from behavio.models.base import _protected_array


@dataclass(frozen=True, slots=True)
class LatentStateAlignment:
    """One truth-aware assignment of inferred states to simulated reference states.

    ``reference_to_inferred[reference_state]`` gives the original inferred-state column
    assigned to that reference state. The aligned probability matrix is in reference-state
    order. Assignment scores are balanced across reference states so a common state cannot
    hide failure to recover a rare one.
    """

    reference_to_inferred: tuple[int, ...]
    reference_counts: NDArray[np.int64]
    confusion_matrix: NDArray[np.float64]
    aligned_probabilities: NDArray[np.float64]
    aligned_labels: NDArray[np.int64]
    best_score: float
    runner_up_score: float
    score_gap: float
    posterior_accuracy: float
    decoded_accuracy: float
    ambiguity_tolerance: float
    ambiguous: bool

    def __post_init__(self) -> None:
        permutation = tuple(self.reference_to_inferred)
        counts = _protected_array(self.reference_counts, dtype=np.int64)
        confusion = _protected_array(self.confusion_matrix, dtype=np.float64)
        probabilities = _protected_array(self.aligned_probabilities, dtype=np.float64)
        labels = _protected_array(self.aligned_labels, dtype=np.int64)
        n_states = len(permutation)
        if n_states < 2 or sorted(permutation) != list(range(n_states)):
            raise ValueError("reference_to_inferred must permute at least two states")
        if counts.shape != (n_states,) or np.any(counts < 0):
            raise ValueError("reference_counts must contain one non-negative count per state")
        if confusion.shape != (n_states, n_states):
            raise ValueError("confusion_matrix must contain one row and column per state")
        if probabilities.ndim != 2 or probabilities.shape[1] != n_states:
            raise ValueError("aligned_probabilities must contain one column per state")
        if labels.shape != (len(probabilities),) or int(np.sum(counts)) != len(probabilities):
            raise ValueError("aligned labels, probabilities, and reference counts must align")
        if not np.all(np.isfinite(confusion)) or np.any((confusion < 0) | (confusion > 1)):
            raise ValueError("confusion_matrix must contain finite probabilities")
        expected_row_sums = (counts > 0).astype(np.float64)
        if not np.allclose(confusion.sum(axis=1), expected_row_sums, atol=1e-8):
            raise ValueError("observed confusion rows must sum to one and missing rows to zero")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("aligned_probabilities must be finite and non-negative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("aligned_probabilities must sum to one")
        if np.any((labels < 0) | (labels >= n_states)):
            raise ValueError("aligned_labels must contain valid reference-state labels")
        scores = (
            self.best_score,
            self.runner_up_score,
            self.score_gap,
            self.posterior_accuracy,
            self.decoded_accuracy,
        )
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in scores):
            raise ValueError("alignment scores must be finite probabilities")
        if self.runner_up_score > self.best_score + 1e-12:
            raise ValueError("runner_up_score cannot exceed best_score")
        if not np.isclose(self.score_gap, self.best_score - self.runner_up_score, atol=1e-12):
            raise ValueError("score_gap must equal best_score minus runner_up_score")
        if not np.isfinite(self.ambiguity_tolerance) or not 0 <= self.ambiguity_tolerance <= 1:
            raise ValueError("ambiguity_tolerance must lie between zero and one")
        expected_ambiguity = bool(np.any(counts == 0) or self.score_gap <= self.ambiguity_tolerance)
        if self.ambiguous is not expected_ambiguity:
            raise ValueError("ambiguous must reflect missing states and the configured score gap")
        object.__setattr__(self, "reference_to_inferred", permutation)
        object.__setattr__(self, "reference_counts", counts)
        object.__setattr__(self, "confusion_matrix", confusion)
        object.__setattr__(self, "aligned_probabilities", probabilities)
        object.__setattr__(self, "aligned_labels", labels)

    @property
    def inferred_to_reference(self) -> tuple[int, ...]:
        """Return the inverse permutation for mapping inferred labels to reference labels."""

        inverse = np.empty(len(self.reference_to_inferred), dtype=np.int64)
        inverse[np.asarray(self.reference_to_inferred, dtype=np.intp)] = np.arange(len(inverse))
        return tuple(int(value) for value in inverse)

    @property
    def missing_reference_states(self) -> tuple[int, ...]:
        """Return reference states absent from the simulated realization."""

        return tuple(int(state) for state in np.flatnonzero(self.reference_counts == 0))


def align_latent_states(
    reference_states: NDArray[np.integer] | list[int] | tuple[int, ...],
    inferred_probabilities: NDArray[np.floating],
    *,
    ambiguity_tolerance: float = 0.05,
) -> LatentStateAlignment:
    """Align inferred state columns to known simulation truth without using label order.

    The assignment maximizes mean posterior mass on the matching inferred state, averaged
    equally across reference states. The runner-up is the best assignment that excludes at
    least one edge from the winning assignment. Missing reference states and score gaps at
    or below ``ambiguity_tolerance`` are marked ambiguous.
    """

    reference = np.asarray(reference_states)
    probabilities = np.asarray(inferred_probabilities, dtype=np.float64)
    if reference.ndim != 1 or probabilities.ndim != 2:
        raise ValueError(
            "reference_states and inferred_probabilities must be one- and two-dimensional"
        )
    if len(reference) != len(probabilities):
        raise ValueError("reference_states and inferred_probabilities must have one row per trial")
    if not len(reference):
        raise ValueError("state alignment requires at least one trial")
    if probabilities.shape[1] < 2:
        raise ValueError("inferred_probabilities must contain at least two states")
    if not np.issubdtype(reference.dtype, np.integer):
        raise ValueError("reference_states must contain integer labels")
    n_states = probabilities.shape[1]
    reference = reference.astype(np.int64, copy=False)
    if np.any((reference < 0) | (reference >= n_states)):
        raise ValueError("reference_states must be valid inferred-state labels")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("inferred_probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("inferred_probabilities must sum to one")
    if not np.isfinite(ambiguity_tolerance) or not 0 <= ambiguity_tolerance <= 1:
        raise ValueError("ambiguity_tolerance must lie between zero and one")

    counts = np.bincount(reference, minlength=n_states).astype(np.int64)
    confusion = np.zeros((n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        if counts[state]:
            confusion[state] = np.mean(probabilities[reference == state], axis=0)

    rows, columns = linear_sum_assignment(confusion, maximize=True)
    mapping = np.empty(n_states, dtype=np.int64)
    mapping[rows] = columns
    best_score = _assignment_score(confusion, mapping)
    runner_up_score = _runner_up_score(confusion, mapping)
    score_gap = max(0.0, best_score - runner_up_score)
    aligned_probabilities = probabilities[:, mapping]
    inferred_to_reference = np.empty(n_states, dtype=np.int64)
    inferred_to_reference[mapping] = np.arange(n_states)
    aligned_labels = inferred_to_reference[np.argmax(probabilities, axis=1)]
    posterior_accuracy = float(np.mean(aligned_probabilities[np.arange(len(reference)), reference]))
    decoded_accuracy = float(np.mean(aligned_labels == reference))
    ambiguous = bool(np.any(counts == 0) or score_gap <= ambiguity_tolerance)
    return LatentStateAlignment(
        reference_to_inferred=tuple(int(value) for value in mapping),
        reference_counts=counts,
        confusion_matrix=confusion,
        aligned_probabilities=aligned_probabilities,
        aligned_labels=aligned_labels,
        best_score=best_score,
        runner_up_score=runner_up_score,
        score_gap=score_gap,
        posterior_accuracy=posterior_accuracy,
        decoded_accuracy=decoded_accuracy,
        ambiguity_tolerance=float(ambiguity_tolerance),
        ambiguous=ambiguous,
    )


def _assignment_score(matrix: NDArray[np.float64], mapping: NDArray[np.int64]) -> float:
    return float(np.mean(matrix[np.arange(len(mapping)), mapping]))


def _runner_up_score(
    matrix: NDArray[np.float64],
    best_mapping: NDArray[np.int64],
) -> float:
    alternatives: list[float] = []
    for row, column in enumerate(best_mapping):
        constrained = matrix.copy()
        constrained[row, column] = -np.inf
        rows, columns = linear_sum_assignment(constrained, maximize=True)
        mapping = np.empty(len(best_mapping), dtype=np.int64)
        mapping[rows] = columns
        alternatives.append(_assignment_score(matrix, mapping))
    return max(alternatives)
