"""Inspect complete-subject and complete-lab validation folds."""

from __future__ import annotations

from behavio import Study
from behavio.evaluate import leave_one_lab_out_splits, leave_one_subject_out_splits


def make_study() -> Study:
    subjects = ("mouse-a", "mouse-b", "mouse-c", "mouse-d")
    labs = {"mouse-a": "lab-1", "mouse-b": "lab-1", "mouse-c": "lab-2", "mouse-d": "lab-3"}
    return Study.factorial(
        trials=3,
        subjects=subjects,
        session_label=lambda subject, order: f"{subject}-day-1",
        columns={"lab": [labs[subject] for subject in subjects for _trial in range(3)]},
    )


def main() -> None:
    study = make_study()
    print("Leave-subject-out folds")
    for split in leave_one_subject_out_splits(study):
        print(
            f"held out={split.held_out_group}, train={split.train_subjects}, "
            f"test={split.test_subjects}"
        )

    print("\nLeave-lab-out folds")
    for split in leave_one_lab_out_splits(study):
        print(
            f"held out={split.held_out_group}, train={split.train_subjects}, "
            f"test={split.test_subjects}"
        )


if __name__ == "__main__":
    main()
