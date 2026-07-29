"""Apply an auditable bout policy before photometry encoding."""

from behavio.ethograms import BehaviorAnnotations, BehaviorInterval
from behavio.interval_policy import (
    ContextualizeIntervals,
    FilterIntervals,
    IntervalPolicy,
    MergeIntervals,
    ResolveIntervalOverlaps,
    apply_interval_policy,
)


def run() -> dict[str, object]:
    bouts = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"cue": (0.5,)},
        intervals=(
            BehaviorInterval("groom", 1.0, 1.2, 0.91),
            BehaviorInterval("groom", 1.3, 3.0, 0.88),
            BehaviorInterval("locomotion", 4.0, 8.0, 0.96),
        ),
        source="keypoint-moseq",
        clock_id="acquisition",
    )
    task_epochs = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={},
        intervals=(
            BehaviorInterval("baseline", 0.0, 4.0),
            BehaviorInterval("test", 4.0, 10.0),
        ),
        source="experiment-log",
        clock_id="acquisition",
    )
    policy = IntervalPolicy(
        (
            MergeIntervals("merge-groom", maximum_gap_s=0.15, labels=("groom",)),
            FilterIntervals("minimum-bout", minimum_duration_s=0.5),
            ContextualizeIntervals(
                "task-context",
                context_source="task-epochs",
                multiple_matches="first",
            ),
            ResolveIntervalOverlaps("exclusive-labels", scope="all"),
        )
    )
    result = apply_interval_policy(
        bouts,
        policy,
        context_sources={"task-epochs": task_epochs},
    )
    return {
        "result": result,
        "encoding_inputs": result.annotations.interval_encoding_inputs(),
    }


if __name__ == "__main__":
    artifacts = run()
    result = artifacts["result"]
    print(result.annotations.intervals)
    print(result.evidence_fingerprint)
