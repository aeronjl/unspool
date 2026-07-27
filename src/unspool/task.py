"""Task-level observation contracts for behavioural modelling workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unspool.models.base import (
    BehaviourEstimator,
    FitResult,
    Prediction,
    PredictionMode,
    _protected_array,
    model_capabilities,
)
from unspool.response_times import ResponseTimeSpec
from unspool.study import REQUIRED_COLUMNS, Study


class TaskValidationError(ValueError):
    """Raised when trial observations do not satisfy a declared task contract."""


@dataclass(frozen=True, slots=True)
class ChoiceSpec:
    """Categorical choices, omissions, and optional trial-specific availability.

    ``options`` defines the stable action coordinate used across models and datasets.
    Omission values remain distinct from valid actions and are encoded as ``-1`` by
    :meth:`read`. Missing values can be treated as omissions only when explicitly enabled.
    """

    options: tuple[Any, ...]
    column: str = "choice"
    omission_values: tuple[Any, ...] = ()
    missing_is_omission: bool = False
    available_options_column: str | None = None

    def __post_init__(self) -> None:
        _column(self.column, "choice column")
        options = _labels(self.options, "choice options", minimum=2)
        omissions = _labels(self.omission_values, "choice omission values", minimum=0)
        overlap = {_label_key(value) for value in options} & {
            _label_key(value) for value in omissions
        }
        if overlap:
            raise TaskValidationError("choice options and omission values must not overlap")
        if not isinstance(self.missing_is_omission, bool):
            raise TaskValidationError("missing_is_omission must be boolean")
        if self.available_options_column is not None:
            _column(self.available_options_column, "available-options column")
            if self.available_options_column == self.column:
                raise TaskValidationError(
                    "available-options column must be distinct from the choice column"
                )
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "omission_values", omissions)

    def read(self, study: Study) -> ChoiceData:
        """Validate and encode the task's observed choices without dropping trials."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        if self.column not in study.columns:
            raise TaskValidationError(f"study is missing choice column {self.column!r}")

        option_index = {_label_key(value): index for index, value in enumerate(self.options)}
        omission_keys = {_label_key(value) for value in self.omission_values}
        codes = np.full(len(study), -1, dtype=np.int64)
        omitted = np.zeros(len(study), dtype=np.bool_)

        for row, raw in enumerate(study[self.column]):
            value = _scalar(raw)
            if _is_missing(value):
                if not self.missing_is_omission:
                    raise TaskValidationError(
                        f"choice is missing at row {row}; declare missing_is_omission=True "
                        "to retain it as an omission"
                    )
                omitted[row] = True
                continue
            key = _label_key(value)
            if key in omission_keys:
                omitted[row] = True
                continue
            if key not in option_index:
                raise TaskValidationError(
                    f"choice at row {row} is {value!r}, outside declared options and omissions"
                )
            codes[row] = option_index[key]

        available = self._read_available_options(study, codes, omitted)
        return ChoiceData(
            column=self.column,
            options=self.options,
            codes=codes,
            omitted=omitted,
            available=available,
        )

    def _read_available_options(
        self,
        study: Study,
        codes: NDArray[np.int64],
        omitted: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        available = np.ones((len(study), len(self.options)), dtype=np.bool_)
        if self.available_options_column is None:
            return available
        if self.available_options_column not in study.columns:
            raise TaskValidationError(
                f"study is missing available-options column {self.available_options_column!r}"
            )

        option_index = {_label_key(value): index for index, value in enumerate(self.options)}
        for row, raw in enumerate(study[self.available_options_column]):
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise TaskValidationError(
                    f"available options at row {row} must be a non-empty sequence"
                )
            row_options = tuple(_scalar(value) for value in raw)
            if not row_options:
                raise TaskValidationError(f"available options are empty at row {row}")
            keys = tuple(_label_key(value) for value in row_options)
            if len(set(keys)) != len(keys):
                raise TaskValidationError(f"available options contain duplicates at row {row}")
            unknown = [
                value
                for value, key in zip(row_options, keys, strict=True)
                if key not in option_index
            ]
            if unknown:
                raise TaskValidationError(
                    f"available options at row {row} contain undeclared values: {unknown!r}"
                )
            available[row] = False
            available[row, [option_index[key] for key in keys]] = True
            if not omitted[row] and not available[row, codes[row]]:
                raise TaskValidationError(
                    f"observed choice at row {row} was not available on that trial"
                )
        return available


@dataclass(frozen=True, slots=True)
class ChoiceData:
    """Validated categorical choices on a stable integer coordinate."""

    column: str
    options: tuple[Any, ...]
    codes: NDArray[np.int64]
    omitted: NDArray[np.bool_]
    available: NDArray[np.bool_]

    def __post_init__(self) -> None:
        codes = _protected_array(self.codes, dtype=np.int64)
        omitted = _protected_array(self.omitted, dtype=np.bool_)
        available = _protected_array(self.available, dtype=np.bool_)
        if codes.ndim != 1 or omitted.shape != codes.shape:
            raise TaskValidationError("choice codes and omission mask must be aligned vectors")
        if available.shape != (len(codes), len(self.options)):
            raise TaskValidationError("choice availability must have one column per option")
        if np.any((codes < -1) | (codes >= len(self.options))):
            raise TaskValidationError("choice codes fall outside the option coordinate")
        if np.any((codes == -1) != omitted):
            raise TaskValidationError("only omitted choices may use code -1")
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "omitted", omitted)
        object.__setattr__(self, "available", available)

    @property
    def counts(self) -> Mapping[Any, int]:
        """Observed, non-omission counts in declared option order."""

        return MappingProxyType(
            {option: int(np.sum(self.codes == index)) for index, option in enumerate(self.options)}
        )


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """One scalar reward observation with explicit support and missingness."""

    column: str = "reward"
    minimum: float | None = None
    maximum: float | None = None
    allow_missing: bool = False

    def __post_init__(self) -> None:
        _column(self.column, "reward column")
        if self.minimum is not None and not np.isfinite(self.minimum):
            raise TaskValidationError("reward minimum must be finite or null")
        if self.maximum is not None and not np.isfinite(self.maximum):
            raise TaskValidationError("reward maximum must be finite or null")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise TaskValidationError("reward minimum cannot exceed its maximum")
        if not isinstance(self.allow_missing, bool):
            raise TaskValidationError("allow_missing must be boolean")

    def read(self, study: Study) -> NDArray[np.float64]:
        """Validate scalar rewards and return a protected floating-point vector."""

        if self.column not in study.columns:
            raise TaskValidationError(f"study is missing reward column {self.column!r}")
        try:
            values = np.asarray(study[self.column], dtype=np.float64)
        except (TypeError, ValueError):
            raise TaskValidationError("rewards must be numeric values") from None
        missing = np.isnan(values)
        if np.any(missing) and not self.allow_missing:
            row = int(np.flatnonzero(missing)[0])
            raise TaskValidationError(f"reward is missing at row {row}")
        finite = values[~missing]
        if not np.all(np.isfinite(finite)):
            raise TaskValidationError("non-missing rewards must be finite")
        if self.minimum is not None and np.any(finite < self.minimum):
            raise TaskValidationError(f"rewards must be at least {self.minimum}")
        if self.maximum is not None and np.any(finite > self.maximum):
            raise TaskValidationError(f"rewards must be at most {self.maximum}")
        return _protected_array(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TaskValidation:
    """Compact denominator audit returned by :meth:`TaskSpec.validate`."""

    n_trials: int
    n_observed_choices: int
    n_omissions: int
    choice_counts: tuple[tuple[Any, int], ...]
    has_rewards: bool
    has_response_times: bool

    @property
    def counts(self) -> Mapping[Any, int]:
        return MappingProxyType(dict(self.choice_counts))


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Reusable task semantics independent of any particular model family."""

    choice: ChoiceSpec
    predictors: tuple[str, ...] = ()
    reward: RewardSpec | None = None
    response_time: ResponseTimeSpec | None = None
    block_column: str | None = None
    episode_column: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.choice, ChoiceSpec):
            raise TypeError("choice must be a ChoiceSpec")
        predictors = tuple(self.predictors)
        if len(set(predictors)) != len(predictors):
            raise TaskValidationError("predictor columns must be unique")
        for predictor in predictors:
            _column(predictor, "predictor column")
        for column, label in (
            (self.block_column, "block column"),
            (self.episode_column, "episode column"),
        ):
            if column is not None:
                _column(column, label)
        declared = [self.choice.column, *predictors]
        if self.reward is not None:
            declared.append(self.reward.column)
        if self.response_time is not None:
            declared.append(self.response_time.column)
        if self.block_column is not None:
            declared.append(self.block_column)
        if self.episode_column is not None:
            declared.append(self.episode_column)
        if self.choice.available_options_column is not None:
            declared.append(self.choice.available_options_column)
        if len(set(declared)) != len(declared):
            raise TaskValidationError("task roles must use distinct columns")
        object.__setattr__(self, "predictors", predictors)

    @property
    def outcome_columns(self) -> tuple[str, ...]:
        columns = [self.choice.column]
        if self.response_time is not None:
            columns.append(self.response_time.column)
        return tuple(columns)

    @property
    def declared_columns(self) -> tuple[str, ...]:
        """All source columns assigned a task role, in declaration order."""

        columns = [*self.outcome_columns, *self.predictors]
        if self.reward is not None:
            columns.append(self.reward.column)
        if self.block_column is not None:
            columns.append(self.block_column)
        if self.episode_column is not None:
            columns.append(self.episode_column)
        if self.choice.available_options_column is not None:
            columns.append(self.choice.available_options_column)
        return tuple(columns)

    def validate(self, study: Study) -> TaskValidation:
        """Validate task observations and return explicit analysis denominators."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        required = set(self.predictors)
        if self.block_column is not None:
            required.add(self.block_column)
        if self.episode_column is not None:
            required.add(self.episode_column)
        missing = required - set(study.columns)
        if missing:
            raise TaskValidationError(f"study is missing task columns: {sorted(missing)}")

        choices = self.choice.read(study)
        if self.reward is not None:
            self.reward.read(study)
        if self.response_time is not None:
            try:
                self.response_time.read(study)
            except ValueError as error:
                raise TaskValidationError(str(error)) from error
        return TaskValidation(
            n_trials=len(study),
            n_observed_choices=int(np.sum(~choices.omitted)),
            n_omissions=int(np.sum(choices.omitted)),
            choice_counts=tuple(choices.counts.items()),
            has_rewards=self.reward is not None,
            has_response_times=self.response_time is not None,
        )

    def validate_model(self, model: BehaviourEstimator) -> None:
        """Require a model's scored event to be declared by this task."""

        capabilities = model_capabilities(model)
        undeclared = set(capabilities.scored_columns) - set(self.outcome_columns)
        if undeclared:
            raise TaskValidationError(
                f"model scores observations not declared by the task: {sorted(undeclared)}"
            )
        required = getattr(model, "required_task_columns", ())
        if isinstance(required, str):
            raise TaskValidationError("model required_task_columns must be a tuple")
        required = tuple(required)
        if len(set(required)) != len(required) or any(
            not isinstance(column, str) or not column for column in required
        ):
            raise TaskValidationError(
                "model required_task_columns must contain unique non-empty strings"
            )
        missing_roles = set(required) - set(self.declared_columns)
        if missing_roles:
            raise TaskValidationError(
                f"model uses columns without a declared task role: {sorted(missing_roles)}"
            )


@dataclass(frozen=True, slots=True)
class FittedModel:
    """A fitted estimator bound to the task contract that validated its data."""

    model: BehaviourEstimator
    task: TaskSpec
    result: FitResult
    validation: TaskValidation

    def predict(
        self,
        study: Study,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        self.task.validate(study)
        return self.model.predict(study, self.result, mode=mode)

    def pointwise_log_prob(
        self,
        study: Study,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        self.task.validate(study)
        values = np.asarray(
            self.model.pointwise_log_prob(study, self.result, mode=mode),
            dtype=np.float64,
        )
        if values.shape != (len(study),) or not np.all(np.isfinite(values)):
            raise ValueError("pointwise_log_prob must return one finite value per trial")
        return _protected_array(values, dtype=np.float64)

    def audit(self):
        """Return the common numerical audit while retaining the raw fit result."""

        return self.result.audit()


def fit_model(model: BehaviourEstimator, study: Study, *, task: TaskSpec) -> FittedModel:
    """Validate a task, fit one estimator, and bind their inspectable results."""

    task.validate_model(model)
    validation = task.validate(study)
    result = model.fit(study)
    if not isinstance(result, FitResult):
        raise TypeError("model.fit must return a FitResult")
    if result.model_name != model.model_name or result.model_signature != model.signature:
        raise ValueError("fit result does not match the fitted estimator")
    if result.n_observations != len(study):
        raise ValueError("fit result n_observations must equal the study length")
    return FittedModel(model=model, task=task, result=result, validation=validation)


def _column(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise TaskValidationError(f"{label} must be a non-empty string")
    if value in REQUIRED_COLUMNS:
        raise TaskValidationError(f"{label} cannot replace a required Study column")


def _labels(values: tuple[Any, ...], label: str, *, minimum: int) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        raise TaskValidationError(f"{label} must be a tuple")
    if len(values) < minimum:
        raise TaskValidationError(f"{label} must contain at least {minimum} values")
    normalized = tuple(_scalar(value) for value in values)
    for value in normalized:
        if _is_missing(value):
            raise TaskValidationError(f"{label} cannot contain missing values")
        try:
            hash(_label_key(value))
        except TypeError:
            raise TaskValidationError(f"{label} must contain scalar hashable values") from None
    if len({_label_key(value) for value in normalized}) != len(normalized):
        raise TaskValidationError(f"{label} must be unique")
    return normalized


def _label_key(value: Any) -> tuple[Any, Any]:
    if isinstance(value, bool):
        return bool, value
    if isinstance(value, (int, float)):
        return "number", float(value)
    return type(value), value


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False
