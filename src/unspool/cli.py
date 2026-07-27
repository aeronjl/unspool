"""Lean command line for protocol validation, execution, and evidence inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from unspool.compiler import compile_execution_plan, materialize_protocol
from unspool.evidence import (
    compare_evidence_bundles,
    read_evidence_bundle,
    replay_evidence_bundle,
)
from unspool.models import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BinaryQLearning,
    HierarchicalBernoulliHistoryGLM,
    HierarchicalSmoothBernoulliHistoryGLM,
    SmoothBernoulliHistoryGLM,
    WienerDriftDiffusion,
    model_capabilities,
)
from unspool.protocol import ProtocolState, StudyProtocol, protocol_from_json
from unspool.runner import run_nested_protocol, run_protocol
from unspool.study import Study
from unspool.validation import (
    cohort_forward_session_splits,
    forward_session_splits,
    historical_cohort_forecast_splits,
    leave_one_lab_out_session_forecast_splits,
    leave_one_lab_out_splits,
    leave_one_subject_out_splits,
    within_session_rolling_splits,
)

_MODELS: dict[str, type[Any]] = {
    "unspool.models.BernoulliHistoryGLM": BernoulliHistoryGLM,
    "unspool.models.SmoothBernoulliHistoryGLM": SmoothBernoulliHistoryGLM,
    "unspool.models.BernoulliGLMHMM": BernoulliGLMHMM,
    "unspool.models.BinaryQLearning": BinaryQLearning,
    "unspool.models.HierarchicalBernoulliHistoryGLM": HierarchicalBernoulliHistoryGLM,
    "unspool.models.HierarchicalSmoothBernoulliHistoryGLM": (HierarchicalSmoothBernoulliHistoryGLM),
    "unspool.models.WienerDriftDiffusion": WienerDriftDiffusion,
}
_SPLITTERS: dict[str, Callable[..., tuple[Any, ...]]] = {
    "unspool.validation.forward_session_splits": forward_session_splits,
    "unspool.validation.cohort_forward_session_splits": cohort_forward_session_splits,
    "unspool.validation.historical_cohort_forecast_splits": (historical_cohort_forecast_splits),
    "unspool.validation.leave_one_subject_out_splits": leave_one_subject_out_splits,
    "unspool.validation.leave_one_lab_out_splits": leave_one_lab_out_splits,
    "unspool.validation.leave_one_lab_out_session_forecast_splits": (
        leave_one_lab_out_session_forecast_splits
    ),
    "unspool.validation.within_session_rolling_splits": within_session_rolling_splits,
}


class CLIError(ValueError):
    """User-facing command failure without a Python traceback."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Unspool command line and return a process exit status."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (CLIError, ValueError, TypeError, OSError, KeyError) as error:
        print(f"unspool: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unspool",
        description="Validate and execute frozen longitudinal study protocols.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("protocol-validate", help="validate canonical protocol JSON")
    validate.add_argument("protocol", type=Path)
    validate.add_argument("--freeze-out", type=Path)
    validate.set_defaults(handler=_validate_protocol)

    execute = subparsers.add_parser("execute", help="execute built-in candidates under a protocol")
    execute.add_argument("protocol", type=Path)
    execute.add_argument("study", type=Path, help="JSON object containing a columns mapping")
    execute.add_argument("output", type=Path, help="new directory for the evaluation snapshot")
    execute.set_defaults(handler=_execute)

    inspect = subparsers.add_parser("inspect", help="verify and summarize an evidence bundle")
    inspect.add_argument("bundle", type=Path)
    inspect.set_defaults(handler=_inspect)

    compare = subparsers.add_parser("bundle-compare", help="compare two verified evidence bundles")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.set_defaults(handler=_compare)

    report = subparsers.add_parser("report", help="verify a bundle and emit its bounded report")
    report.add_argument("bundle", type=Path)
    report.add_argument("--output", type=Path)
    report.set_defaults(handler=_report)
    return parser


def _validate_protocol(arguments: argparse.Namespace) -> int:
    protocol = _read_protocol(arguments.protocol)
    output = None
    if arguments.freeze_out is not None:
        frozen = protocol.freeze() if protocol.state == ProtocolState.DRAFT else protocol
        if frozen.state != ProtocolState.FROZEN:
            raise CLIError("--freeze-out requires a draft or frozen pre-evidence protocol")
        _write_new(arguments.freeze_out, frozen.canonical_json() + "\n")
        output = str(arguments.freeze_out)
    _print_json(
        {
            "valid": True,
            "identifier": protocol.identifier,
            "schema_version": protocol.schema_version,
            "state": protocol.state.value,
            "fingerprint": protocol.fingerprint,
            "frozen_output": output,
        }
    )
    return 0


def _execute(arguments: argparse.Namespace) -> int:
    protocol = _read_protocol(arguments.protocol)
    if protocol.state == ProtocolState.DRAFT:
        protocol = protocol.freeze()
    if protocol.state != ProtocolState.FROZEN:
        raise CLIError("execute requires a draft or frozen pre-evidence protocol")
    if arguments.output.exists():
        raise CLIError(f"output path already exists: {arguments.output}")
    study = _read_study(arguments.study)
    models = _instantiate_models(protocol)
    materialized = materialize_protocol(protocol, study)
    splitter = _SPLITTERS.get(protocol.validation.splitter)
    if splitter is None:
        raise CLIError(
            f"splitter is not in the built-in execution registry: {protocol.validation.splitter!r}"
        )
    splitter_settings = _settings(protocol.validation.settings)
    splits = splitter(materialized.study, **splitter_settings)
    capabilities = {name: model_capabilities(model) for name, model in models.items()}
    inner_splitter = None
    if protocol.selection is not None:
        inner_function = _SPLITTERS.get(protocol.selection.inner_validation.splitter)
        if inner_function is None:
            raise CLIError(
                "inner splitter is not in the built-in execution registry: "
                f"{protocol.selection.inner_validation.splitter!r}"
            )
        inner_settings = _settings(protocol.selection.inner_validation.settings)

        def inner_splitter(training: Study, _fold: int) -> tuple[Any, ...]:
            return inner_function(training, **inner_settings)

    compiled = compile_execution_plan(
        materialized,
        splits,
        capabilities=capabilities,
        inner_splitter=inner_splitter,
    )
    if not compiled.plan.audit.passed:
        raise CLIError(
            "pre-fit audit failed: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in compiled.plan.audit.errors)
        )
    run = (
        run_nested_protocol(compiled, models)
        if protocol.selection is not None
        else run_protocol(compiled, models)
    )

    artifacts = {
        "protocol.json": run.protocol.to_dict(),
        "cohort.json": materialized.manifest.to_dict(),
        "plan.json": compiled.plan.to_dict(),
        "evaluation.json": run.report.to_dict(),
    }
    snapshot = {
        "schema_version": "unspool.evaluation-snapshot/1",
        "protocol_fingerprint": protocol.fingerprint,
        "state": run.protocol.state.value,
        "recovery_pending": bool(protocol.recovery),
        "artifacts": {
            name: hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()
            for name, value in artifacts.items()
        },
    }
    arguments.output.mkdir(parents=True)
    for name, value in artifacts.items():
        (arguments.output / name).write_text(_json_text(value) + "\n", encoding="utf-8")
    (arguments.output / "snapshot.json").write_text(_json_text(snapshot) + "\n", encoding="utf-8")
    decision = (
        {
            "decision": "nested-training-only-selection",
            "selection_counts": run.report.selection_counts,
        }
        if protocol.selection is not None
        else {
            "ranking_status": run.report.ranking.status.value,
            "winner": run.report.ranking.winner,
        }
    )
    _print_json(
        {
            "output": str(arguments.output),
            "protocol_fingerprint": protocol.fingerprint,
            "state": run.protocol.state.value,
            "recovery_pending": bool(protocol.recovery),
            **decision,
        }
    )
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    replay = replay_evidence_bundle(arguments.bundle)
    _print_json(asdict_without_none(replay))
    return 0


def _compare(arguments: argparse.Namespace) -> int:
    comparison = compare_evidence_bundles(arguments.left, arguments.right)
    _print_json(asdict_without_none(comparison))
    return 0


def _report(arguments: argparse.Namespace) -> int:
    bundle = read_evidence_bundle(arguments.bundle)
    replay_evidence_bundle(bundle)
    markdown = bundle.file("report/report.md").content.decode("utf-8")
    if arguments.output is None:
        print(markdown, end="")
    else:
        _write_new(arguments.output, markdown)
        _print_json({"bundle_id": bundle.bundle_id, "report": str(arguments.output)})
    return 0


def _read_protocol(path: Path) -> StudyProtocol:
    try:
        return protocol_from_json(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except Exception as error:
        raise CLIError(f"invalid protocol {path}: {error}") from error


def _read_study(path: Path) -> Study:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise CLIError(f"invalid study JSON: {error}") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"columns"}
        or not isinstance(value["columns"], dict)
    ):
        raise CLIError("study JSON must be an object containing only a columns mapping")
    return Study.from_columns(value["columns"])


def _instantiate_models(protocol: StudyProtocol) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for candidate in protocol.candidates:
        model_type = _MODELS.get(candidate.implementation)
        if model_type is None:
            raise CLIError(
                f"candidate implementation is not in the built-in registry: "
                f"{candidate.implementation!r}"
            )
        try:
            models[candidate.name] = model_type(**_settings(candidate.hyperparameters))
        except Exception as error:
            raise CLIError(f"cannot instantiate candidate {candidate.name!r}: {error}") from error
    return models


def _settings(settings: tuple[Any, ...]) -> dict[str, Any]:
    return {setting.name: setting.value for setting in settings}


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def _print_json(value: Any) -> None:
    print(_json_text(value))


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.value if isinstance(item, StrEnum) else item,
    )


def asdict_without_none(value: Any) -> dict[str, Any]:
    """Serialize one CLI result dataclass while retaining false and empty values."""

    return asdict(value)


def _reject_constant(value: str) -> None:
    raise CLIError(f"non-finite JSON constant is forbidden: {value}")


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
