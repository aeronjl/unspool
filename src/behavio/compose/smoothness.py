"""``smooth(model)``: let declared parameters of a model follow a path in clock time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.compose.trajectory import CoefficientTrajectory
from behavio.contracts.bounded import (
    RowCoefficientDesign,
    RowObjective,
    require_composable,
    uses_row_coefficients,
    validated_row_coefficients,
)
from behavio.contracts.compose import (
    PenalisedDesign,
    PenalisedLinearEstimator,
    linear_predictor,
    validate_predictor_shape,
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
from behavio.models._kernels.introspection import Describable, ModelFinding
from behavio.trials import Study

__all__ = ["SmoothModel", "smooth"]


def smooth(
    model: PenalisedLinearEstimator,
    *,
    over: str = "session_order",
    knots: Sequence[float] = (0.0, 1.0),
    smoothness: float = 1.0,
    parameters: Sequence[str] | None = None,
    group_smoothness: float | None = None,
    shared_trajectory: bool = False,
) -> SmoothModel:
    """Return ``model`` with ``parameters`` replaced by smooth paths in ``over``.

    Each declared parameter becomes one value per knot, linearly interpolated between
    knots, with a spacing-scaled first-difference penalty -- a Gaussian random walk observed
    at the knots -- supplying the smoothness prior. The knots and their clock are part of the
    specification and are fixed before fitting, so held-out outcomes can never choose the
    temporal basis.

    ``parameters=None`` smooths every parameter, which is what the deleted
    ``SmoothBernoulliHistoryGLM`` did and all it could do. Naming a subset leaves the rest
    **stationary**: one coordinate rather than one per knot, and no roughness penalty. That
    is not a convenience -- a Wiener non-decision time that drifts between knots is a
    different and much weaker model than one that does not, and the hand-written smooth
    drift-diffusion family held it fixed for that reason.

    ``group_smoothness`` is not used by the smooth model itself. It is the roughness a
    *group's deviation path* is given when :func:`behavio.compose.hierarchical` wraps this
    model, and it defaults to ``smoothness`` because a subject's deviation from a smooth
    population path is a path too. Setting it lower lets subjects wander more freely than
    the population does.

    A model whose likelihood is not a penalised linear one is smoothed through
    :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` instead, and the *clock*
    then carries a restriction the linear families do not have: it must be constant within
    each block the model's likelihood recurses over. Smoothing a Q-learning agent over
    ``session_order`` is a per-session learning rate or policy; smoothing it over a
    within-session trial counter is refused, because a value trace written by a learning
    rate that changed part-way through cannot say which of its values produced which part
    of the trace.
    """

    if hasattr(model, "varying_effects"):
        raise TypeError(
            "hierarchy is the outer combinator: write hierarchical(smooth(model)) rather "
            "than smooth(hierarchical(model)). A hierarchical estimator reports the "
            "population coordinate while fitting a joint one whose width depends on how "
            "many groups the study has, so it cannot be expanded again from outside"
        )
    require_composable(model, combinator="smooth")
    available = tuple(model.parameter_names)
    if parameters is None:
        varying = available
    else:
        if isinstance(parameters, str):
            raise ValueError("parameters must be a sequence of parameter names")
        declared = tuple(parameters)
        unknown = [name for name in declared if name not in available]
        if not declared or len(set(declared)) != len(declared):
            raise ValueError("smoothed parameters must be non-empty and unique")
        if unknown:
            raise ValueError(
                f"smoothed parameters are not parameters of this model: {sorted(unknown)}; "
                f"available: {list(available)}"
            )
        varying = tuple(name for name in available if name in set(declared))
    return SmoothModel(
        model=model,
        clock=over,
        knots=tuple(knots),
        smoothness=smoothness,
        varying=varying,
        group_smoothness=smoothness if group_smoothness is None else group_smoothness,
        shared_trajectory=shared_trajectory,
    )


@dataclass(frozen=True, slots=True)
class SmoothModel(Describable):
    """A model whose declared parameters are values at fixed knots of one clock column.

    Parameter naming is stable and mechanical: a smoothed parameter ``p`` of the wrapped
    model becomes ``p[clock=knot]`` for each knot, in coefficient-major, knot-minor order,
    and a parameter that was not smoothed keeps its own name and its single coordinate. So
    ``BernoulliHistoryGLM(predictors=("stimulus",))`` smoothed over ``session_order`` with
    knots ``(0, 4)`` has parameters ``intercept[session_order=0]``,
    ``intercept[session_order=4]``, ``stimulus[session_order=0]``,
    ``stimulus[session_order=4]``, ``choice_lag_1[session_order=0]``, ...; smoothing only
    ``("stimulus",)`` gives ``intercept``, ``stimulus[session_order=0]``,
    ``stimulus[session_order=4]``, ``choice_lag_1``.
    """

    model: PenalisedLinearEstimator
    clock: str
    knots: tuple[float, ...]
    smoothness: float
    varying: tuple[str, ...]
    group_smoothness: float
    shared_trajectory: bool

    def __post_init__(self) -> None:
        knots = validated_knots(self.knots)
        varying = tuple(self.varying)
        available = tuple(self.model.parameter_names)
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
        if not varying or len(set(varying)) != len(varying):
            raise ValueError("smoothed parameters must be non-empty and unique")
        if set(varying) - set(available):
            raise ValueError("smoothed parameters must be parameters of the wrapped model")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "varying", tuple(name for name in available if name in varying))

    # -- identity ---------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"smooth-{self.model.model_name}"

    @property
    def signature(self) -> str:
        knots = ",".join(format_time(knot) for knot in self.knots)
        selection = "" if self.smooths_every_parameter else f"varying={','.join(self.varying)};"
        return (
            f"smooth[time={self.clock};knots={knots};{selection}"
            f"smoothness={self.smoothness};"
            f"group_smoothness={self.group_smoothness};"
            f"shared_trajectory={self.shared_trajectory}]({self.model.signature})"
        )

    @property
    def smooths_every_parameter(self) -> bool:
        """Whether every wrapped parameter follows a path, which is the default."""

        return self.varying == tuple(self.model.parameter_names)

    @property
    def time(self) -> str:
        """The clock column, under the name model introspection looks for."""

        return self.clock

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """The wrapped model's parameters, some of which now follow a path."""

        return tuple(self.model.parameter_names)

    @property
    def varying_coefficients(self) -> tuple[str, ...]:
        """The wrapped parameters that follow a path, in model order."""

        return tuple(self.varying)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for coefficient, _start, width in self.layout:
            if width == 1:
                names.append(coefficient)
            else:
                names.extend(
                    f"{coefficient}[{self.clock}={format_time(knot)}]" for knot in self.knots
                )
        return tuple(names)

    @property
    def layout(self) -> tuple[tuple[str, int, int], ...]:
        """``(coefficient, first column, width)`` per wrapped parameter, in model order."""

        varying = set(self.varying)
        blocks: list[tuple[str, int, int]] = []
        offset = 0
        for coefficient in self.coefficient_names:
            width = len(self.knots) if coefficient in varying else 1
            blocks.append((coefficient, offset, width))
            offset += width
        return tuple(blocks)

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
        paths = "every parameter" if self.smooths_every_parameter else ", ".join(self.varying)
        return (
            f"random walk over {self.clock} knots {self.knots} for {paths}: "
            f"first-difference penalty scaled by smoothness={self.smoothness}",
            *getattr(self.model, "declared_priors", ()),
        )

    @property
    def likelihood(self) -> Any:
        return self.model.likelihood

    @property
    def penalised_linear_refusal(self) -> str:
        """The wrapped model's refusal, so ``mix()`` still reports it by name.

        A smooth model has every structural member of
        :class:`~behavio.contracts.compose.PenalisedLinearEstimator` whatever it wraps, so
        the structural test alone would say yes for a smooth Q-learning agent and the
        failure would surface inside an optimizer. Forwarding the sentence is the same
        answer :class:`~behavio.models.BernoulliGLMHMM` gives, one combinator further out.
        """

        return str(getattr(self.model, "penalised_linear_refusal", "") or "")

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        """The wrapped model's cells: a path in clock time is not a new cell."""

        return tuple(self.model.predictor_cells)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """The wrapped model's channels: smoothing a parameter cannot change what is seen."""

        return tuple(self.model.outcome_channels)

    @property
    def categories(self) -> tuple[Any, ...]:
        """The wrapped model's outcome coordinate, when it scores a categorical outcome.

        Absent, and raising :class:`AttributeError` rather than returning ``None``, when
        the wrapped model has none: that is what keeps ``smooth(glm)`` from structurally
        satisfying :class:`~behavio.contracts.estimator.CategoricalBehaviourEstimator`.
        """

        return tuple(self.model.categories)

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the wrapped model's observed category codes."""

        return self.model.outcome_codes(study)

    # -- the penalised problem ---------------------------------------------------------

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped model's scored observation."""

        return self.model.outcomes(study)

    def predictor_offsets(self, study: Study) -> NDArray[np.float64] | None:
        """Return the wrapped model's offsets unchanged.

        An offset is a term on the linear predictor that no parameter multiplies, so
        letting a parameter follow a path in clock time cannot touch it: an option that
        was not offered on a trial is not offered on that trial at any point of the path.
        """

        return self.model.predictor_offsets(study)

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped design multiplied row-wise by the temporal basis.

        The all-varying case is written as the single ``einsum`` it has always been, because
        a fit published before parameters could be smoothed selectively must still be
        reproducible to the last bit; the selective case assembles the same product one
        coefficient block at a time.
        """

        features = validate_predictor_shape(self.model, self.model.design_matrix(study))
        basis = self.time_basis(study)
        if self.smooths_every_parameter:
            if features.ndim == 2:
                return np.einsum("ij,ik->ijk", features, basis).reshape(len(study), -1)
            return np.einsum("icj,ik->icjk", features, basis, optimize=True).reshape(
                len(study), features.shape[1], -1
            )
        columns: list[NDArray[np.float64]] = []
        for index, (_coefficient, _start, width) in enumerate(self.layout):
            feature = features[..., index]
            if width == 1:
                columns.append(feature[..., None])
            else:
                columns.append(
                    feature[..., None]
                    * basis.reshape(len(study), *([1] * (feature.ndim - 1)), width)
                )
        return np.concatenate(columns, axis=-1)

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the wrapped penalty lifted onto knots plus the roughness penalty."""

        n_knots = len(self.knots)
        inner = self.model.penalty_matrix()
        if self.smooths_every_parameter:
            lifted = np.kron(inner, np.eye(n_knots))
            roughness = self.smoothness * np.kron(
                np.eye(inner.shape[0]), roughness_matrix(self.knots)
            )
            return lifted + roughness
        size = len(self.parameter_names)
        penalty = np.zeros((size, size), dtype=np.float64)
        blocks = self.layout
        for row, (_left, left_start, left_width) in enumerate(blocks):
            for column, (_right, right_start, right_width) in enumerate(blocks):
                value = float(inner[row, column])
                if not value:
                    continue
                if left_width != right_width:
                    raise ValueError(
                        "the wrapped penalty couples a smoothed parameter to a stationary "
                        "one, and there is no knot grid the coupling can be lifted onto"
                    )
                block = value * np.eye(left_width)
                penalty[
                    left_start : left_start + left_width,
                    right_start : right_start + right_width,
                ] += block
        roughness = self.smoothness * roughness_matrix(self.knots)
        for _coefficient, start, width in blocks:
            if width > 1:
                penalty[start : start + width, start : start + width] += roughness
        return penalty

    def coordinate_box(self, study: Study) -> NDArray[np.float64] | None:
        """Return the wrapped box repeated over each parameter's knots.

        Every point of a path is a value of the parameter, so it is admissible exactly
        where the parameter is. A box does not become a different box for being sampled.
        """

        box = self.model.coordinate_box(study)
        if box is None:
            return None
        return np.vstack(
            [
                np.tile(box[index], (width, 1))
                for index, (_name, _start, width) in enumerate(self.layout)
            ]
        )

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the wrapped starts as flat paths, one knot value per wrapped value."""

        return tuple(self.expand_coefficients(point) for point in self.model.initial_points(study))

    def expand_coefficients(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one wrapped-coordinate vector as a constant path in this coordinate."""

        flat = np.asarray(values, dtype=np.float64)
        if flat.shape != (len(self.coefficient_names),):
            raise ValueError("expanding needs one value per wrapped parameter")
        return np.concatenate(
            [
                np.full(width, flat[index])
                for index, (_name, _start, width) in enumerate(self.layout)
            ]
        )

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return every knot of a coefficient, so a path varies by group as a whole path."""

        for index, (coefficient, start, width) in enumerate(self.layout):
            if coefficient == name:
                del index
                return self.parameter_names[start : start + width]
        return (name,)

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

    # -- the same problem, for a model composed through rows rather than a design ---------

    def row_objective(self, study: Study) -> _SmoothRowObjective:
        """Return the wrapped objective read through this model's temporal basis.

        The only new arithmetic smoothness contributes on this route is one row-wise linear
        map and its transpose, which is the same statement the design-matrix route makes by
        multiplying the design by the basis. Nothing here knows what the wrapped likelihood
        is, and the clock check is here rather than in the wrapped model because the clock
        is this combinator's argument.
        """

        inner = self.model.row_objective(study)
        basis = self.time_basis(study)
        self._require_clock_constant_within_blocks(basis, inner.row_blocks)
        return _SmoothRowObjective(model=self, inner=inner, basis=basis)

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a row-coefficient problem with the wrapped model's own optimizer settings."""

        return self.model.fit_rows(design, model_name=model_name, model_signature=model_signature)

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> ModelPrediction:
        """Collapse per-row knot values onto the wrapped coordinate and delegate."""

        return self.model.predict_rows(study, self.collapse_rows(study, coefficients), mode=mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Collapse per-row knot values onto the wrapped coordinate and delegate."""

        return self.model.pointwise_log_prob_rows(study, self.collapse_rows(study, coefficients))

    def collapse_rows(self, study: Study, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate one knot vector per row at that row's clock value.

        This is the map ``simulate_rows`` has always applied before delegating; naming it
        is what lets the likelihood, the prediction and the simulator all reach the wrapped
        model through the same arithmetic instead of three copies of it.
        """

        values = validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )
        return _collapse_paths(values, self.time_basis(study), self.layout)

    def _require_clock_constant_within_blocks(
        self, basis: NDArray[np.float64], blocks: NDArray[np.intp]
    ) -> None:
        n_blocks = int(blocks.max()) + 1 if len(blocks) else 0
        representative = np.zeros((n_blocks, basis.shape[1]), dtype=np.float64)
        representative[blocks] = basis
        if not np.array_equal(basis, representative[blocks]):
            raise ModelDataError(
                f"{self.model.model_name} scores its trials through a recursion, so a path "
                f"over {self.clock!r} is only defined if the clock is constant within each "
                "block that recursion runs over; smooth over a session-level clock instead"
            )

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the prior on one group's *deviation path*, not its deviation values.

        This is the member that makes a hierarchical smooth model a composition rather than
        a sibling. A group's deviation from a smooth population path is itself a path, so it
        inherits the roughness penalty; drop that and every subject's deviation is free to
        jump between adjacent knots, which is a different and much weaker model. A
        stationary parameter's deviation is a number and gets the ordinary ridge.
        """

        widths = self._selected_widths(columns)
        ridge = np.diag(1.0 / np.asarray(scales, dtype=np.float64) ** 2)
        roughness = self.group_smoothness * roughness_matrix(self.knots)
        penalty = np.array(ridge, dtype=np.float64)
        offset = 0
        for width in widths:
            if width > 1:
                penalty[offset : offset + width, offset : offset + width] += roughness
            offset += width
        return penalty

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw one deviation path per group and varying coefficient."""

        widths = self._selected_widths(columns)
        scale_array = np.asarray(scales, dtype=np.float64)
        roughness = self.group_smoothness * roughness_matrix(self.knots)
        covariances = []
        offset = 0
        for width in widths:
            block = np.diag(1.0 / scale_array[offset : offset + width] ** 2)
            if width > 1:
                block = block + roughness
            covariances.append(np.linalg.pinv(block, hermitian=True))
            offset += width
        deviations = np.empty((groups, int(sum(widths))), dtype=np.float64)
        for group in range(groups):
            offset = 0
            for width, covariance in zip(widths, covariances, strict=True):
                deviations[group, offset : offset + width] = generator.multivariate_normal(
                    np.zeros(width), covariance
                )
                offset += width
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
        if values.shape != (len(design), len(self.parameter_names)):
            raise ValueError("simulate_rows needs one knot value per parameter per study row")
        rows = _collapse_paths(values, self.time_basis(design), self.layout)
        return self.model.simulate_rows(design, rows, seed=seed)

    # -- the estimator contract --------------------------------------------------------

    def fit(self, study: Study) -> FitResult:
        """Fit smooth parameter paths with a time-scaled random-walk penalty."""

        self._validate_study_scope(study)
        if uses_row_coefficients(self.model):
            objective = self.row_objective(study)
            return self.fit_rows(
                RowCoefficientDesign(
                    parameter_names=self.parameter_names,
                    objective=objective,
                    expand=lambda joint: np.tile(joint, (objective.n_rows, 1)),
                    contract=lambda gradient: np.asarray(gradient, dtype=np.float64).sum(axis=0),
                    penalty_matrix=self.penalty_matrix(),
                    box=self.coordinate_box(study),
                    initial_points=self.initial_points(study),
                ),
                model_name=self.model_name,
                model_signature=self.signature,
            )
        return self.fit_penalised(
            PenalisedDesign(
                parameter_names=self.parameter_names,
                design_matrix=self.design_matrix(study),
                outcomes=self.outcomes(study),
                penalty_matrix=self.penalty_matrix(),
                likelihood=self.likelihood,
                offsets=self.predictor_offsets(study),
                box=self.coordinate_box(study),
                initial_points=self.initial_points(study),
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
        if uses_row_coefficients(self.model):
            return self.predict_rows(study, self._fitted_rows(study, fit), mode=prediction_mode)
        return self.likelihood.prediction(self.row_predictor(study, fit), mode=prediction_mode)

    def _fitted_rows(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Repeat one fitted coordinate over the study's rows, ready to be collapsed."""

        return np.tile(np.asarray(fit.estimates, dtype=np.float64), (len(study), 1))

    def row_predictor(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Return the linear predictor of each row under a fitted set of paths."""

        self._validate_fit(fit)
        return linear_predictor(
            self.design_matrix(study), fit.estimates, self.predictor_offsets(study)
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observation under the fitted paths.

        The likelihood is scored on the model's own linear predictor rather than on the one
        a :class:`~behavio.contracts.estimator.Prediction` carries. Those are the same array
        for a family whose prediction *is* its predictor, and they are not for a
        drift-diffusion family, whose four predictor cells produce one choice probability.
        """

        self._validate_study_scope(study)
        self._prediction_mode(mode)
        self._validate_fit(fit)
        if uses_row_coefficients(self.model):
            return self.pointwise_log_prob_rows(study, self._fitted_rows(study, fit))
        return self.likelihood.pointwise_log_prob(
            self.row_predictor(study, fit), self.outcomes(study)
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations under smooth parameter paths."""

        self._validate_study_scope(design)
        knot_values = self.knot_grid(self.parameter_vector(parameters))
        coefficients = self.time_basis(design) @ knot_values.T
        return self.model.simulate_rows(design, coefficients, seed=seed)

    # -- reading paths back ------------------------------------------------------------

    def parameters_from_paths(
        self, paths: Mapping[str, float | Sequence[float]]
    ) -> Mapping[str, float]:
        """Pack named parameter paths and stationary values into simulation coordinates."""

        expected = set(self.coefficient_names)
        observed = set(paths)
        if observed != expected:
            raise ValueError(
                "paths must match the model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values: dict[str, float] = {}
        offset = 0
        for coefficient, _start, width in self.layout:
            try:
                path = np.atleast_1d(np.asarray(paths[coefficient], dtype=np.float64))
            except (TypeError, ValueError):
                raise ValueError(
                    f"path {coefficient!r} must contain one finite value per temporal knot"
                ) from None
            if path.shape != (width,) or not np.all(np.isfinite(path)):
                requirement = (
                    "one finite value" if width == 1 else "one finite value per temporal knot"
                )
                raise ValueError(f"path {coefficient!r} must contain {requirement}")
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
        """Evaluate any flat coordinate vector on this model's basis."""

        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = linear_time_basis(time_array, self.knots)
        return CoefficientTrajectory(
            clock=self.clock,
            times=time_array,
            coefficient_names=self.coefficient_names,
            values=basis @ self.knot_grid(knot_values).T,
        )

    def knot_grid(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a flat coordinate vector as ``(coefficients, knots)``, stationary repeated."""

        flat = np.asarray(values, dtype=np.float64)
        if flat.shape != (len(self.parameter_names),):
            raise ValueError("a knot grid needs one value per parameter of this model")
        grid = np.empty((len(self.coefficient_names), len(self.knots)), dtype=np.float64)
        for index, (_coefficient, start, width) in enumerate(self.layout):
            grid[index] = flat[start] if width == 1 else flat[start : start + width]
        return grid

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

    def _selected_widths(self, columns: NDArray[np.intp]) -> tuple[int, ...]:
        """Return the widths of the coefficient blocks ``columns`` selects, whole blocks only."""

        selected = [int(column) for column in np.asarray(columns, dtype=np.intp)]
        widths: list[int] = []
        expected: list[int] = []
        for _coefficient, start, width in self.layout:
            if start in selected:
                widths.append(width)
                expected.extend(range(start, start + width))
        if selected != expected:
            raise ValueError(
                "a smooth parameter varies by group as a whole path: name every knot of a "
                "coefficient, or name the coefficient before smoothing it"
            )
        return tuple(widths)

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

    # -- what is wrong with fitting this here --------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Forward whatever the wrapped model has to say about this study.

        Smoothing adds no finding of its own -- an unsupported knot is already reported
        generically, from ``knots`` and ``time`` -- but it must not swallow the wrapped
        model's, which is what a combinator with no ``additional_findings`` at all does.
        """

        declared = getattr(self.model, "additional_findings", None)
        return () if declared is None else tuple(declared(study))

    def group_deviation_findings(
        self, study: Study, *, grouping: str, parameters: Sequence[str]
    ) -> tuple[ModelFinding, ...]:
        """Translate knot names back to coefficients and ask the wrapped model.

        ``hierarchical(smooth(psychometric))`` names its varying parameters as knots of a
        path -- ``lapse_logit[session_order=0]`` -- and the model that knows a lapse rate is
        bounded knows it under the name ``lapse_logit``. A path varies by group as a whole
        path, so a coefficient is named here as soon as any of its knots is.
        """

        declared = getattr(self.model, "group_deviation_findings", None)
        if declared is None:
            return ()
        named = set(parameters)
        coefficients = [
            coefficient
            for coefficient, start, width in self.layout
            if named & set(self.parameter_names[start : start + width])
        ]
        return tuple(declared(study, grouping=grouping, parameters=tuple(coefficients)))


def _collapse_paths(
    values: NDArray[np.float64],
    basis: NDArray[np.float64],
    layout: tuple[tuple[str, int, int], ...],
) -> NDArray[np.float64]:
    """Evaluate one knot vector per row at that row's clock value, coefficient by coefficient."""

    rows = np.empty((len(values), len(layout)), dtype=np.float64)
    for index, (_coefficient, start, width) in enumerate(layout):
        block = values[:, start : start + width]
        rows[:, index] = block[:, 0] if width == 1 else np.einsum("ik,ik->i", block, basis)
    return rows


@dataclass(frozen=True, slots=True)
class _SmoothRowObjective:
    """The wrapped objective composed with this model's temporal basis.

    A path in clock time is a *linear* map from knots to the value in force on a row, so its
    contribution to the chain rule is that map's transpose and nothing else: the wrapped
    model never learns that its coefficients came from a path, and this class never learns
    what the wrapped likelihood is.
    """

    model: SmoothModel
    inner: RowObjective
    basis: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        """The number of scored rows."""

        return int(self.inner.n_rows)

    @property
    def n_parameters(self) -> int:
        """The width of one row's knot coordinate."""

        return len(self.model.parameter_names)

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """The wrapped model's blocks: a path cannot subdivide a recursion."""

        return self.inner.row_blocks

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the wrapped negative log likelihood and its gradient in the knot rows."""

        values = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        value, inner_gradient = self.inner.value_and_gradient(
            _collapse_paths(values, self.basis, self.model.layout)
        )
        gradient = np.zeros_like(values)
        for index, (_coefficient, start, width) in enumerate(self.model.layout):
            if width == 1:
                gradient[:, start] = inner_gradient[:, index]
            else:
                gradient[:, start : start + width] = inner_gradient[:, index, None] * self.basis
        return float(value), gradient
