"""Trial-outcome-blind selection for the IBL 2021 public learning benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

RELEASE_TAG = "2021_Q1_IBL_et_al_Behaviour"
TRAINING_PROTOCOL = "trainingChoiceWorld"
BIASED_PROTOCOL = "biasedChoiceWorld"
SESSIONS_PER_PHASE = 3


def select_learning_panel(
    session_records: Iterable[Mapping[str, Any]],
    trial_table_sessions: set[str],
    *,
    sessions_per_phase: int = SESSIONS_PER_PHASE,
) -> list[dict[str, Any]]:
    """Select one trial-outcome-blind, transition-anchored trajectory per lab.

    Subjects must have a first ``biasedChoiceWorld`` transition and enough preceding
    ``trainingChoiceWorld`` sessions for disjoint early and late-training windows. Within
    each lab, the subject with the most pre-transition training sessions is selected.
    Total eligible task-session coverage and then subject identifier break ties.
    """

    if sessions_per_phase <= 0:
        raise ValueError("sessions_per_phase must be positive")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in session_records:
        session = _identifier(record, "id")
        protocol = str(record.get("task_protocol") or "")
        if session not in trial_table_sessions:
            continue
        if TRAINING_PROTOCOL not in protocol and BIASED_PROTOCOL not in protocol:
            continue
        grouped[(str(record.get("lab") or ""), str(record.get("subject") or ""))].append(record)

    candidates_by_lab: dict[str, list[tuple[int, int, str, list[dict[str, Any]]]]] = defaultdict(
        list
    )
    minimum_training_sessions = 2 * sessions_per_phase
    for (lab, subject), records in grouped.items():
        if not lab or not subject:
            continue
        ordered = sorted(records, key=_chronology_key)
        transition_index = next(
            (
                index
                for index, record in enumerate(ordered)
                if BIASED_PROTOCOL in str(record.get("task_protocol") or "")
            ),
            None,
        )
        if transition_index is None:
            continue
        training = [
            record
            for record in ordered[:transition_index]
            if TRAINING_PROTOCOL in str(record.get("task_protocol") or "")
        ]
        if len(training) < minimum_training_sessions:
            continue

        early = training[:sessions_per_phase]
        late = training[-sessions_per_phase:]
        selected_ids = {_identifier(record, "id") for record in early}
        if selected_ids & {_identifier(record, "id") for record in late}:
            raise RuntimeError("phase windows must be disjoint")

        order_by_session = {
            _identifier(record, "id"): order for order, record in enumerate(ordered)
        }
        selected: list[dict[str, Any]] = []
        for phase, phase_records in (("early", early), ("late_training", late)):
            for record in phase_records:
                session = _identifier(record, "id")
                selected.append(
                    {
                        "lab": lab,
                        "subject": subject,
                        "session": session,
                        "start_time": str(record.get("start_time") or ""),
                        "number": int(record.get("number") or 0),
                        "task_protocol": str(record.get("task_protocol") or ""),
                        "session_order": order_by_session[session],
                        "phase": phase,
                        "n_training_sessions_before_transition": len(training),
                        "transition_session": _identifier(ordered[transition_index], "id"),
                    }
                )
        candidates_by_lab[lab].append((len(training), len(ordered), subject, selected))

    panel: list[dict[str, Any]] = []
    for lab in sorted(candidates_by_lab):
        candidates = sorted(
            candidates_by_lab[lab],
            key=lambda candidate: (-candidate[0], -candidate[1], candidate[2]),
        )
        panel.extend(candidates[0][3])
    return panel


def manifest_digest(session_rows: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable digest of the ordered, pinned session records."""

    payload = json.dumps(
        list(session_rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _identifier(record: Mapping[str, Any], field: str) -> str:
    value = str(record.get(field) or "")
    return value.rstrip("/").rsplit("/", 1)[-1]


def _chronology_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("start_time") or ""),
        int(record.get("number") or 0),
        _identifier(record, "id"),
    )
