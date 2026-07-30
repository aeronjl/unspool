import numpy as np
import pytest

from behavio.models import LatentStateAlignment, align_latent_states


def test_alignment_recovers_swapped_probability_columns() -> None:
    reference = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )

    result = align_latent_states(reference, probabilities)

    assert isinstance(result, LatentStateAlignment)
    assert result.reference_to_inferred == (1, 0)
    assert result.inferred_to_reference == (1, 0)
    np.testing.assert_allclose(result.confusion_matrix, [[0.15, 0.85], [0.85, 0.15]])
    np.testing.assert_allclose(result.aligned_probabilities, probabilities[:, ::-1])
    np.testing.assert_array_equal(result.aligned_labels, reference)
    assert result.best_score == pytest.approx(0.85)
    assert result.runner_up_score == pytest.approx(0.15)
    assert result.score_gap == pytest.approx(0.70)
    assert result.posterior_accuracy == pytest.approx(0.85)
    assert result.decoded_accuracy == 1.0
    assert result.ambiguous is False


def test_alignment_metrics_are_invariant_to_inferred_label_permutation() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )
    inferred_permutation = np.array([2, 0, 1])

    original = align_latent_states(reference, probabilities)
    permuted = align_latent_states(reference, probabilities[:, inferred_permutation])

    np.testing.assert_allclose(original.aligned_probabilities, permuted.aligned_probabilities)
    np.testing.assert_array_equal(original.aligned_labels, permuted.aligned_labels)
    assert original.best_score == pytest.approx(permuted.best_score)
    assert original.runner_up_score == pytest.approx(permuted.runner_up_score)
    assert original.posterior_accuracy == pytest.approx(permuted.posterior_accuracy)
    assert original.decoded_accuracy == pytest.approx(permuted.decoded_accuracy)
    assert original.reference_to_inferred != permuted.reference_to_inferred


def test_near_tied_alignment_is_explicitly_ambiguous() -> None:
    reference = np.array([0, 0, 1, 1])
    probabilities = np.full((4, 2), 0.5)

    result = align_latent_states(reference, probabilities, ambiguity_tolerance=0.01)

    assert result.best_score == pytest.approx(0.5)
    assert result.runner_up_score == pytest.approx(0.5)
    assert result.score_gap == pytest.approx(0.0)
    assert result.ambiguous is True


def test_missing_reference_state_is_retained_as_alignment_failure() -> None:
    reference = np.array([0, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])

    result = align_latent_states(reference, probabilities)

    assert result.reference_counts.tolist() == [3, 0]
    assert result.missing_reference_states == (1,)
    assert result.ambiguous is True


def test_alignment_arrays_are_read_only() -> None:
    result = align_latent_states(
        np.array([0, 1]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    )

    with pytest.raises(ValueError, match="read-only"):
        result.aligned_probabilities[0, 0] = 0.5


@pytest.mark.parametrize(
    ("reference", "probabilities", "message"),
    [
        (np.array([[0, 1]]), np.array([[0.8, 0.2], [0.2, 0.8]]), "one- and two"),
        (np.array([0]), np.array([[0.8, 0.2], [0.2, 0.8]]), "one row per trial"),
        (np.array([0.0, 1.0]), np.array([[0.8, 0.2], [0.2, 0.8]]), "integer"),
        (np.array([0, 2]), np.array([[0.8, 0.2], [0.2, 0.8]]), "valid"),
        (np.array([0, 1]), np.array([[0.8, 0.3], [0.2, 0.8]]), "sum to one"),
    ],
)
def test_alignment_inputs_are_validated(
    reference: np.ndarray,
    probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        align_latent_states(reference, probabilities)


@pytest.mark.parametrize("tolerance", [-0.1, 1.1, np.nan])
def test_alignment_tolerance_is_validated(tolerance: float) -> None:
    with pytest.raises(ValueError, match="ambiguity_tolerance"):
        align_latent_states(
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            ambiguity_tolerance=tolerance,
        )
