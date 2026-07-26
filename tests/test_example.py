import runpy
from pathlib import Path

from pytest import CaptureFixture


def test_static_glm_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "static_glm.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Prospective session folds" in output
    assert "Design-specific parameter recovery" in output
    assert "convergence=100%" in output


def test_smooth_glm_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "smooth_glm.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Prospective stationary-versus-smooth comparison" in output
    assert "Fitted stimulus trajectory" in output
    assert "Smooth-path parameter recovery" in output
    assert "convergence=100%" in output


def test_model_recovery_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "model_recovery.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Prospective static-versus-smooth model recovery" in output
    assert "unresolved" in output
    assert "resolution rate: 83.3%" in output
    assert "accuracy among resolved runs: 100.0%" in output


def test_temporal_transforms_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "temporal_transforms.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Explicit design clocks" in output
    assert "Training-fold landmark" in output
    assert "learned values: {'mouse-1': 4.0}" in output
    assert "fit trials:     8" in output
    assert "test-relative:  [4.0, 5.0, 6.0, 7.0]" in output


def test_within_session_validation_example_runs_end_to_end(
    capsys: CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "within_session_validation.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Prospective within-session rolling origins" in output
    assert "origin=session-2:19, fit=100, context=20" in output
    assert "origin=session-2:29, fit=110, context=30" in output
    assert "test=(20, 21, 22, 23, 24)" in output


def test_glm_hmm_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "glm_hmm.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Fixed-transition Bernoulli GLM-HMM" in output
    assert "restart objectives" in output
    assert "label ambiguous:    False" in output
    assert "fit audit:          pass []" in output
    assert "Prospective competing explanations" in output
    assert " GLM-HMM: log-loss=" in output


def test_q_learning_example_runs_end_to_end(capsys: CaptureFixture[str]) -> None:
    example = Path(__file__).parents[1] / "examples" / "q_learning.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Session-reset binary Q-learning" in output
    assert "learning rate:" in output
    assert "restart objectives:" in output
    assert "fit audit:           pass []" in output
    assert "Prospective competing explanations" in output
    assert "Q-learning: log-loss=" in output


def test_population_validation_example_exposes_held_out_units(
    capsys: CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "population_validation.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Leave-subject-out folds" in output
    assert "held out=mouse-a" in output
    assert "Leave-lab-out folds" in output
    assert "held out=lab-1" in output
    assert "test=('mouse-a', 'mouse-b')" in output


def test_hierarchical_glm_example_exposes_population_policy(
    capsys: CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "hierarchical_glm.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Hierarchical Bernoulli GLM with estimated scale" in output
    assert "converged: True" in output
    assert "subject scale (estimated):" in output
    assert "scale at boundary: False" in output
    assert "unseen-subject policy: population-mean-plugin" in output
    assert "mouse-a fitted: True" in output
    assert "new-mouse fitted: False" in output
