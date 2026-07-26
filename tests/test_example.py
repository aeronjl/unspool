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
