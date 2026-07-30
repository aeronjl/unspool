"""``smooth(model)``: let every parameter of a model follow a path in clock time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.compose.trajectory import CoefficientTrajectory
from behavio.contracts.compose import (
    PenalisedDesign,
    PenalisedLinearEstimator,
)
from behavio.contracts.estimator import (
    FitResult,
    ModelDataError,
    ModelPrediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models._kernels.basis import (
    format_time,
    linear_time_basis,
    roughness_matrix,
    validated_knots,
)
from behavio.models._kernels.introspection import Describable
from behavio.study import Study

__all__ = ["SmoothModel", "smooth"]


def smooth(
    model: PenalisedLinearEstimator,
    *,
    over: str = "session_order",
    knots: Sequence[float] = (0.0, 1.0),
    smoothness: float = 1.0,
    group_smoothness: float | None = None,
    shared_trajectory: bool = False,
) -> SmoothModel:
    """Return ``model`` with every parameter replaced by a smooth path in ``over``.

    Each parameter becomes one value per knot, linearly interpolated between knots, with a
    spacing-scaled first-difference penalty -- a Gaussian random walk observed at the knots
    -- supplying the smoothness prior. The knots and their clock are part of the
    specification and are fixed before fitting, so held-out outcomes can never choose the
    temporal basis.

    ``group_smoothness`` is not used by the smooth model itself. It is the roughness a
    *group's deviation path* is given when :func:`behavio.compose.hierarchical` wraps this
    model, and it defaults to ``smoothness`` because a subject's deviation from a smooth
    population path is a path too. Setting it lower lets subjects wander more freely than
    the population does.
    """

    if hasattr(model, "varying_effects"):
        raise TypeError(
            "hierarchy is the outer combinator: write hierarchical(smooth(model)) rather "
            "than smooth(hierarchical(model)). A hierarchical estimator reports the "
            "population coordinate while fitting a joint one whose width depends on how "
            "many groups the study has, so it cannot be expanded again from outside"
        )
    if not isinstance(model, PenalisedLinearEstimator):
        raise TypeError(
            "smooth() needs a model satisfying behavio.contracts.compose."
            "PenalisedLinearEstimator; "
            f"{type(model).__name__} does not"
        )
    penalty = model.penalty_matrix()
    if penalty.shape != (len(model.parameter_names),) * 2:
        raise TypeError(
            f"{type(model).__name__} declares {len(model.parameter_names)} parameters but a "
            f"{penalty.shape} penalty, so its estimated coordinate is not the one it reports"
        )
    return SmoothModel(
        model=model,
        clock=over,
        knots=tuple(knots),
        smoothness=smoothness,
        group_smoothness=smoothness if group_smoothness is None else group_smoothness,
        shared_trajectory=shared_trajectory,
    )


@dataclass(frozen=True, slots=True)
class SmoothModel(Describable):
    """A model whose parameters are values at fixed knots of one clock column.

    Parameter naming is stable and mechanical: a parameter ``p`` of the wrapped model
    becomes ``p[clock=knot]`` for each knot, in coefficient-major, knot-minor order. So
    ``BernoulliHistoryGLM(covariates=("stimulus",))`` smoothed over ``session_order`` with
    knots ``(0, 4)`` has parameters ``intercept[session_order=0]``,
    ``intercept[session_order=4]``, ``stimulus[session_order=0]``,
    ``stimulus[session_order=4]``, ``choice_lag_1[session_order=0]``, ...
    """

    model: PenalisedLinearEstimator
    clock: str
    knots: tuple[float, ...]
    smoothness: float
    group_smoothness: float
    shared_trajectory: bool

    def __post_init__(self) -> None:
        knots = validated_knots(self.knots)
        if not isinstance(self.clock, str) or not self.clock:
            raise ValueError("over must be a non-empty Study column name")
        if self.clock in self.model.scored_columns:
            raise ValueError("the clock and the scored column must be distinct")
        for value, label in (
            (self.smoothness, "smoothness"),
            (self.group_smoothness, "group_smoothness"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if not isinstance(self.shared_trajectory, bool):
            raise ValueError("shared_trajectory must be boolean")
        object.__setattr__(self, "knots", knots)

    # -- identity ---------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"smooth-{self.model.model_name}"

    @property
    def signature(self) -> str:
        knots = ",".join(format_time(knot) for knot in self.knots)
        return (
            f"smooth[time={self.clock};knots={knots};smoothness={self.smoothness};"
            f"group_smoothness={self.group_smoothness};"
            f"shared_trajectory={self.shared_trajectory}]({self.model.signature})"
        )

    @property
    def time(self) -> str:
        """The clock column, under the name model introspection looks for."""

        return self.clock

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """The wrapped model's parameters, each of which now follows a path."""

        return tuple(self.model.parameter_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{coefficient}[{self.clock}={format_time(knot)}]"
            for coefficient in self.coefficient_names
            for knot in self.knots
        )

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return tuple(self.model.scored_columns)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return tuple(self.model.required_task_columns)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return tuple(self.model.supported_prediction_modes)

    @property
    def design_spec(self) -> Any:
        """The wrapped model's design, which the temporal basis multiplies."""

        return getattr(self.model, "design_spec", None)

    @property
    def declared_priors(self) -> tuple[str, ...]:
        return (
            f"random walk over {self.clock} knots {self.knots}: "
            f"first-difference penalty scaled by smoothness={self.smoothness}",
            *getattr(self.model, "declared_priors", ()),
        )

    @property
    def likelihood(self) -> Any:
        return self.model.likelihood

    @property
    def boundary_threshold(self) -> float:
        return float(self.model.boundary_threshold)

    # -- the penalised problem ---------------------------------------------------------

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped model's scored observation."""

        return self.model.outcomes(study)

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped design multiplied row-wise by the temporal basis."""

        features = self.model.design_matrix(study)
        basis = self.time_basis(study)
        return np.einsum("ij,ik->ijk", features, basis).reshape(len(study), -1)

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the wrapped penalty lifted onto knots plus the roughness penalty."""

        n_knots = len(self.knots)
        inner = self.model.penalty_matrix()
        lifted = np.kron(inner, np.eye(n_knots))
        roughness = self.smoothness * np.kron(np.eye(inner.shape[0]), roughness_matrix(self.knots))
        return lifted + roughness

    def fit_penalised(
        self,
        design: PenalisedDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a penalised problem with the wrapped model's own optimizer settings."""

        return self.model.fit_penalised(
            design, model_name=model_name, model_signature=model_signature
        )

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the prior on one group's *deviation path*, not its deviation values.

        This is the member that makes a hierarchical smooth model a composition rather than
        a sibling. A group's deviation from a smooth population path is itself a path, so it
        inherits the roughness penalty; drop that and every subject's deviation is free to
        jump between adjacent knots, which is a different and much weaker model.
        """

        coefficients = self._coefficient_blocks(columns)
        ridge = np.diag(1.0 / np.asarray(scales, dtype=np.float64) ** 2)
        roughness = self.group_smoothness * np.kron(
            np.eye(len(coefficients)), roughness_matrix(self.knots)
        )
        return ridge + roughness

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw one deviation path per group and varying coefficient."""

        coefficients = self._coefficient_blocks(columns)
        n_knots = len(self.knots)
        scale_array = np.asarray(scales, dtype=np.float64)
        roughness = self.group_smoothness * roughness_matrix(self.knots)
        covariances = [
            np.linalg.pinv(
                np.diag(1.0 / scale_array[block * n_knots : (block + 1) * n_knots] ** 2)
                + roughness,
                hermitian=True,
            )
            for block in range(len(coefficients))
        ]
        deviations = np.empty((groups, len(coefficients) * n_knots), dtype=np.float64)
        for group in range(groups):
            for block in range(len(coefficients)):
                deviations[group, block * n_knots : (block + 1) * n_knots] = (
                    generator.multivariate_normal(np.zeros(n_knots), covariances[block])
                )
        return deviations

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Collapse per-row knot values onto the wrapped coordinate and delegate."""

        values = np.asarray(coefficients, dtype=np.float64)
        n_coefficients = len(self.coefficient_names)
        if values.shape != (len(design), n_coefficients * len(self.knots)):
            raise ValueError("simulate_rows needs one knot value per parameter per study row")
        basis = self.time_basis(design)
        rows = np.einsum(
            "ijk,ik->ij", values.reshape(len(design), n_coefficients, len(self.knots)), basis
        )
        return self.model.simulate_rows(design, rows, seed=seed)

    # -- the estimator contract --------------------------------------------------------

    def fit(self, study: Study) -> FitResult:
        """Fit smooth parameter paths with a time-scaled random-walk penalty."""

        self._validate_study_scope(study)
        return self.fit_penalised(
            PenalisedDesign(
                parameter_names=self.parameter_names,
                design_matrix=self.design_matrix(study),
                outcomes=self.outcomes(study),
                penalty_matrix=self.penalty_matrix(),
                likelihood=self.likelihood,
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        """Return predictions under the fitted paths evaluated at each row's clock value."""

        self._validate_study_scope(study)
        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        linear_predictor = self.design_matrix(study) @ fit.estimates
        return self.likelihood.prediction(linear_predictor, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observation under the fitted paths."""

        outcomes = self.outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        return self.likelihood.pointwise_log_prob(prediction.linear_predictor, outcomes)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations under smooth parameter paths."""

        self._validate_study_scope(design)
        knot_values = self.parameter_vector(parameters).reshape(
            len(self.coefficient_names), len(self.knots)
        )
        coefficients = self.time_basis(design) @ knot_values.T
        return self.model.simulate_rows(design, coefficients, seed=seed)

    # -- reading paths back ------------------------------------------------------------

    def parameters_from_paths(self, paths: Mapping[str, Sequence[float]]) -> Mapping[str, float]:
        """Pack named parameter paths into this model's simulation coordinates."""

        expected = set(self.coefficient_names)
        observed = set(paths)
        if observed != expected:
            raise ValueError(
                "paths must match the model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values: dict[str, float] = {}
        offset = 0
        for coefficient in self.coefficient_names:
            path = np.asarray(paths[coefficient], dtype=np.float64)
            if path.shape != (len(self.knots),) or not np.all(np.isfinite(path)):
                raise ValueError(
                    f"path {coefficient!r} must contain one finite value per temporal knot"
                )
            for value in path:
                values[self.parameter_names[offset]] = float(value)
                offset += 1
        return MappingProxyType(values)

    def coefficient_trajectory(
        self, fit: FitResult, *, times: Sequence[float] | None = None
    ) -> CoefficientTrajectory:
        """Evaluate fitted parameter paths at requested clock values."""

        self._validate_fit(fit)
        return self.trajectory_from_knots(fit.estimates, times=times)

    def trajectory_from_knots(
        self,
        knot_values: NDArray[np.float64],
        *,
        times: Sequence[float] | None = None,
    ) -> CoefficientTrajectory:
        """Evaluate any flat knot vector on this model's basis."""

        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = linear_time_basis(time_array, self.knots)
        values = np.asarray(knot_values, dtype=np.float64).reshape(
            len(self.coefficient_names), len(self.knots)
        )
        return CoefficientTrajectory(
            clock=self.clock,
            times=time_array,
            coefficient_names=self.coefficient_names,
            values=basis @ values.T,
        )

    def time_basis(self, study: Study) -> NDArray[np.float64]:
        """Return the piecewise-linear interpolation weights for each row's clock value."""

        if self.clock not in study.columns:
            raise ModelDataError(f"study is missing temporal column {self.clock!r}")
        try:
            times = np.asarray(study[self.clock], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"temporal column {self.clock!r} must be numeric") from None
        if not np.all(np.isfinite(times)):
            raise ModelDataError(f"temporal column {self.clock!r} must be finite")
        try:
            return linear_time_basis(times, self.knots)
        except ValueError as error:
            raise ModelDataError(str(error)) from None

    def parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Validate a named parameter mapping and return it in model order."""

        expected = set(self.parameter_names)
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values = np.asarray([parameters[name] for name in self.parameter_names], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        return values

    # -- internals ---------------------------------------------------------------------

    def _coefficient_blocks(self, columns: NDArray[np.intp]) -> tuple[int, ...]:
        """Return the wrapped-parameter positions ``columns`` selects, whole paths only."""

        n_knots = len(self.knots)
        selected = np.asarray(columns, dtype=np.intp)
        blocks = tuple(dict.fromkeys(int(column) // n_knots for column in selected))
        expected = [block * n_knots + knot for block in blocks for knot in range(n_knots)]
        if list(selected) != expected:
            raise ValueError(
                "a smooth parameter varies by group as a whole path: name every knot of a "
                "coefficient, or name the coefficient before smoothing it"
            )
        return blocks

    def _validate_study_scope(self, study: Study) -> None:
        if len(study.subjects) > 1 and not self.shared_trajectory:
            raise ModelDataError(
                "smooth parameter paths are subject-specific by default; fit one subject "
                "at a time, wrap the model in hierarchical() so that between-subject "
                "variation is modelled, or set shared_trajectory=True to align subjects "
                "explicitly"
            )

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{self.model_name} does not support {prediction_mode.value!r} prediction"
            )
        return prediction_mode
