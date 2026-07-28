"""A complete first Unspool analysis: future-session model comparison."""

# --8<-- [start:analysis]
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from unspool import (
    BiasOnly,
    ChoiceSpec,
    Psychometric,
    Study,
    TaskSpec,
    cohort_forward_session_splits,
    compare_models,
)


def make_example_study() -> Study:
    """Simulate a small multi-animal study with a real stimulus effect."""

    rng = np.random.default_rng(2026)
    subjects = tuple(f"mouse-{index + 1}" for index in range(6))
    n_sessions = 5
    trials_per_session = 40
    n_rows = len(subjects) * n_sessions * trials_per_session
    subject = np.repeat(subjects, n_sessions * trials_per_session)
    session_order = np.tile(
        np.repeat(np.arange(n_sessions), trials_per_session),
        len(subjects),
    )
    design = Study.from_columns(
        {
            "subject": subject,
            "session": [
                f"{animal}-session-{order + 1}"
                for animal, order in zip(subject, session_order, strict=True)
            ],
            "trial": list(range(trials_per_session)) * len(subjects) * n_sessions,
            "session_order": session_order,
            "stimulus": rng.normal(size=n_rows),
        }
    )
    generator = Psychometric(stimulus="stimulus")
    return generator.simulate(
        design,
        {"intercept": -0.2, "stimulus": 1.15},
        seed=2027,
    )


def run_analysis(output: Path = Path("first-analysis.svg")):
    """Validate, forecast, compare, and plot one complete analysis."""

    mpl.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "unspool-first-analysis-v1"})
    study = make_example_study()

    # Declare the observations before fitting a model.
    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        predictors=("stimulus",),
    )
    denominators = task.validate(study)

    # The last session is never visible during fitting.
    splits = cohort_forward_session_splits(study, min_train_sessions=4)
    models = {
        "bias only": BiasOnly(),
        "stimulus": Psychometric(stimulus="stimulus", l2=0.02),
    }
    report = compare_models(
        models,
        study,
        splits,
        bootstrap_resamples=2_000,
        bootstrap_seed=2028,
    )

    print(f"{denominators.n_trials} trials · {len(study.subjects)} animals")
    print(f"winner: {report.winner}")
    for result in report.model_results:
        interval = result.unit_balanced_log_loss_interval
        print(
            f"{result.name:>10}: log loss {result.unit_balanced_log_loss:.3f} "
            f"[{interval.lower:.3f}, {interval.upper:.3f}] · "
            f"audit {result.audit_status.value}"
        )

    # Positive values mean the stimulus model forecast the held-out session better.
    bias = report.result_for("bias only")
    stimulus = report.result_for("stimulus")
    improvement = bias.unit_log_losses - stimulus.unit_log_losses

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.axhline(0.0, color="#778092", linewidth=1.0)
    axis.bar(bias.aggregation_units, improvement, color="#c57928")
    axis.set(
        xlabel="Animal",
        ylabel="Held-out log-loss improvement",
        title="Does stimulus sensitivity improve the future-session forecast?",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    print(f"figure: {output}")
    return report


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("first-analysis.svg")
    run_analysis(destination)
# --8<-- [end:analysis]
