"""Compose external pose and state outputs onto one clock and one Study.

Every column of the returned ``Study`` is reduced from the pose, ethograms and
synchronised covariate built above it. Nothing is asserted by hand: the trial
timing is declared from the BORIS cue events, moved onto the acquisition clock
through the fitted synchronisation, and each trial column is a windowed
reduction that carries its own coverage and status.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from behavio import Study
from behavio.ethograms import annotations_from_boris, annotations_from_moseq
from behavio.pose import pose_from_deeplabcut, pose_from_sleap
from behavio.sync import (
    ClockPulseMatches,
    ClockSynchronizationSpec,
    fit_clock_synchronization,
)
from behavio.trialization import (
    EventCount,
    FractionOfTimeInState,
    MaximumValue,
    TrialTiming,
    TrialWindow,
    attach_trial_columns,
    reduce_annotations_to_trials,
    reduce_covariate_to_trials,
)


class DLCResult:
    """Minimal dataframe-like stand-in for a real pandas DeepLabCut result."""

    def __init__(self, columns: dict[tuple[str, ...], list[float]]) -> None:
        self._columns = columns
        self.columns = tuple(columns)

    def __getitem__(self, name: tuple[str, ...]) -> list[float]:
        return self._columns[name]


def run() -> dict[str, Any]:
    fps = 10.0
    dlc = DLCResult(
        {
            ("demo-network", "nose", "x"): [0, 1, 2, 3, 4, 5],
            ("demo-network", "nose", "y"): [0, 0, 0, 0, 0, 0],
            ("demo-network", "nose", "likelihood"): [0.99] * 6,
        }
    )
    dlc_pose = pose_from_deeplabcut(
        dlc,
        subject="mouse-1",
        session="day-1",
        keypoint="nose",
        scorer="demo-network",
        fps=fps,
        clock_id="video",
    )

    sleap_tracks = np.zeros((6, 1, 1, 2), dtype=float)
    sleap_tracks[:, 0, 0, 0] = np.arange(6)
    sleap_pose = pose_from_sleap(
        sleap_tracks,
        subject="mouse-1",
        session="day-1",
        node_names=["nose"],
        node="nose",
        dims=("frame", "track", "node", "xy"),
        fps=fps,
        clock_id="video",
    )

    moseq = annotations_from_moseq(
        [0, 0, 1, 1, 1, 0],
        subject="mouse-1",
        session="day-1",
        fps=fps,
        labels={0: "pause", 1: "approach"},
        clock_id="video",
    )
    boris = annotations_from_boris(
        {
            "Behavior": ["cue", "cue", "investigate"],
            "Type": ["POINT", "POINT", "STATE"],
            "Start": [0.1, 0.3, 0.2],
            "Stop": [np.nan, np.nan, 0.5],
        },
        subject="mouse-1",
        session="day-1",
        behavior_column="Behavior",
        type_column="Type",
        start_column="Start",
        stop_column="Stop",
        clock_id="video",
    )

    video_pulses = np.asarray([0.0, 0.3, 0.6])
    acquisition_pulses = 0.02 + 1.001 * video_pulses
    clock_synchronization = fit_clock_synchronization(
        ClockPulseMatches.from_arrays(
            source_clock_id="video",
            target_clock_id="acquisition",
            source_time_s=video_pulses,
            target_time_s=acquisition_pulses,
        ),
        ClockSynchronizationSpec(
            maximum_absolute_residual_s=1e-6,
            maximum_drift_ppm=2_000,
            minimum_source_span_s=0.5,
        ),
    )
    dlc_pose = clock_synchronization.synchronize_pose(dlc_pose)
    sleap_pose = clock_synchronization.synchronize_pose(sleap_pose)
    moseq = clock_synchronization.synchronize_annotations(moseq)
    boris = clock_synchronization.synchronize_annotations(boris)

    acquisition_time = 0.02 + np.arange(6, dtype=float) / fps
    speed = dlc_pose.speed(minimum_confidence=0.9)
    aligned_speed = speed.aligned_to(
        acquisition_time,
        target_clock_id="acquisition",
        max_gap_s=0.11,
    )
    # The three mappings a consuming encoding model needs, on one clock.
    encoding_inputs = boris.interval_encoding_inputs(edge="onset")
    events = {**moseq.event_times(), **boris.event_times()}

    # Trial timing is declared, not guessed: the BORIS cue events delimit trials,
    # and they are already on the acquisition clock because the annotations were
    # synchronised above. The reductions below refuse any other clock.
    timing = TrialTiming.from_arrays(
        subject=dlc_pose.subject,
        session=dlc_pose.session,
        onset_s=boris.point_events["cue"],
        clock_id="acquisition",
        source="boris:cue",
        clock_synchronization_ids=boris.clock_synchronization_ids,
    )
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=0.2)
    peak_speed = reduce_covariate_to_trials(
        aligned_speed,
        timing=timing,
        window=window,
        reducer=MaximumValue(),
        max_gap_s=0.11,
        minimum_coverage=0.5,
        name="peak_speed",
    )
    approach_time = reduce_annotations_to_trials(
        moseq,
        timing=timing,
        window=window,
        reducer=FractionOfTimeInState("approach"),
        observed_span_s=(float(acquisition_time[0]), float(acquisition_time[-1])),
        minimum_coverage=0.5,
        name="approach_fraction",
    )
    investigate_bouts = reduce_annotations_to_trials(
        boris,
        timing=timing,
        window=window,
        reducer=EventCount("investigate", include_points=False),
        observed_span_s=(float(acquisition_time[0]), float(acquisition_time[-1])),
        minimum_coverage=0.5,
        name="investigate_onsets",
    )

    # One row per declared trial, with longitudinal coordinates the trial columns
    # are then joined onto by subject, session and trial.
    study = attach_trial_columns(
        Study.from_columns(
            {
                "subject": [timing.subject] * timing.n_trials,
                "session": [timing.session] * timing.n_trials,
                "trial": list(timing.trial),
                "session_order": [0] * timing.n_trials,
            }
        ),
        [peak_speed, approach_time, investigate_bouts],
    )
    return {
        "deeplabcut_pose": dlc_pose,
        "sleap_pose": sleap_pose,
        "moseq_annotations": moseq,
        "boris_annotations": boris,
        "clock_synchronization": clock_synchronization,
        "aligned_speed": aligned_speed,
        "events": events,
        "encoding_inputs": encoding_inputs,
        "trial_timing": timing,
        "trial_reductions": (peak_speed, approach_time, investigate_bouts),
        "study": study,
    }


if __name__ == "__main__":
    result = run()
    print("Event predictors:", sorted(result["events"]))
    print("Aligned speed valid samples:", int(result["aligned_speed"].valid.sum()))
    print("Study columns:", result["study"].columns)
    print("Peak speed per trial:", result["study"]["peak_speed"].tolist())
    print("Approach fraction per trial:", result["study"]["approach_fraction"].tolist())
    print("Trial coverage status:", result["study"]["peak_speed_status"].tolist())
