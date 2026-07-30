"""Reference generalized linear models for binary behavioural outcomes.

:class:`BernoulliHistoryGLM` is the only model here. Smoothness and hierarchy used to be
two more dataclasses in this file and one more in each of two sibling modules; they are now
:func:`behavio.compose.smooth` and :func:`behavio.compose.hierarchical`, applied to this
one. What that cost this module is the block of contract members at the end of the class --
:meth:`~BernoulliHistoryGLM.design_matrix`, :meth:`~BernoulliHistoryGLM.penalty_matrix`,
:meth:`~BernoulliHistoryGLM.fit_penalised`, :meth:`~BernoulliHistoryGLM.simulate_rows` and
their two group-prior neighbours -- which is what
:class:`behavio.contracts.compose.PenalisedLinearEstimator` asks of any model that wants to
be composable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio.contracts.compose import (
    PenalisedDesign,
    ridge_group_draw,
    ridge_group_penalty,
)
from behavio.design.matrix import DesignSpec
from behavio.models._kernels.bernoulli import (
    BERNOULLI,
    BernoulliLikelihood,
    ordered_session_indices,
)
from behavio.models._kernels.design import (
    build_matrix,
    extend,
    outcome_history_term,
    resolve_design,
    validate_design_choice,
)
from behavio.models._kernels.introspection import Describable
from behavio.models._kernels.penalised import fit_penalised_linear
from behavio.models.base import (
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.trials import REQUIRED_COLUMNS, Study


@dataclass(frozen=True, slots=True)
class BernoulliHistoryGLM(Describable):
    """A static Bernoulli GLM with exogenous predictors and choice history.

    Previous choices are constructed within subject/session boundaries and effect-coded as
    -1 and +1. Missing history at the beginning of each session is encoded as zero. During
    simulation, history is updated recursively from generated choices; during prediction,
    observed past choices provide one-step-ahead filtered history.

    The exogenous half of the linear predictor is a :class:`~behavio.design.DesignSpec`.
    ``predictors=("a", "b")`` is shorthand for one identity numeric term per name plus an
    intercept, and is exactly equal to writing that design out; pass ``design=`` instead to
    say anything the shorthand cannot -- an interaction, a fixed-level contrast, a weighted
    history kernel. The two are alternatives, not layers, and passing both is an error.

    ``choice_lags`` stays a model-level declaration under either spelling, because lagged
    outcomes are generated recursively during simulation and so cannot be an ordinary
    exogenous column. It is appended to whichever design supplied the exogenous terms.
    """

    predictors: tuple[str, ...] = ()
    outcome: str = "choice"
    choice_lags: int = 1
    l2: float = 0.0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0
    design: DesignSpec | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        predictors = tuple(self.predictors)
        validate_design_choice(self.design, predictors)
        if len(set(predictors)) != len(predictors):
            raise ValueError("predictors must be unique")
        if any(not isinstance(name, str) or not name for name in predictors):
            raise ValueError("predictor names must be non-empty strings")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS:
            raise ValueError("outcome cannot replace a required Study column")
        if self.outcome in predictors:
            raise ValueError("the outcome cannot also be a predictor")
        if isinstance(self.choice_lags, bool) or not isinstance(self.choice_lags, int):
            raise ValueError("choice_lags must be a non-negative integer")
        if self.choice_lags < 0:
            raise ValueError("choice_lags must be a non-negative integer")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            not np.isfinite(self.coefficient_warning_threshold)
            or self.coefficient_warning_threshold <= 0
        ):
            raise ValueError("coefficient_warning_threshold must be finite and positive")
        reserved = {"intercept", *(f"choice_lag_{lag}" for lag in range(1, self.choice_lags + 1))}
        conflict = reserved.intersection(predictors)
        if conflict:
            raise ValueError(f"predictor names conflict with model parameters: {sorted(conflict)}")
        object.__setattr__(self, "predictors", predictors)

    @property
    def model_name(self) -> str:
        return "bernoulli-history-glm"

    @property
    def signature(self) -> str:
        predictors = ",".join(self.predictors)
        return (
            f"{self.model_name}[outcome={self.outcome};predictors={predictors};"
            f"choice_lags={self.choice_lags};l2={self.l2}{self._design_signature}]"
        )

    @property
    def _design_signature(self) -> str:
        """The design's contribution to the signature, empty for a ``predictors`` model.

        A signature is a scientific fingerprint, so a design has to change it. It must
        equally not change for a model constructed the old way, because those signatures
        are already written into fitted artefacts and committed benchmark results -- hence
        an empty contribution rather than the equivalent design's own signature.
        """

        return "" if self.design is None else f";design={self.design.signature}"

    @property
    def exogenous_design(self) -> DesignSpec:
        """The declared design, or the one a ``predictors`` tuple denotes."""

        return resolve_design(self.design, self.predictors)

    @property
    def design_spec(self) -> DesignSpec:
        """The complete design this model fits, exogenous terms plus lagged outcomes."""

        if not self.choice_lags:
            return self.exogenous_design
        return extend(
            self.exogenous_design,
            outcome_history_term(self.outcome, self.choice_lags),
        )

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return self.design_spec.feature_names

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.coefficient_names

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """Design columns that must be declared as task predictors."""

        return tuple(
            column
            for column in self.exogenous_design.required_columns
            if column != self.outcome and column not in REQUIRED_COLUMNS
        )

    @property
    def declared_priors(self) -> tuple[str, ...]:
        """Human-readable statements of the penalties in force, for :meth:`describe`."""

        if not self.l2:
            return ()
        return (
            f"ridge on every non-intercept coefficient: Normal(0, {1.0 / self.l2**0.5:.4g}) "
            f"(l2={self.l2})",
        )

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def likelihood(self) -> BernoulliLikelihood:
        """The observation model this GLM's linear predictor feeds."""

        return BERNOULLI

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        """A binary choice is one number per row, so this family declares no cells."""

        return ()

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """A binary choice is one number per row, so this family declares no channels."""

        return ()

    def predictor_offsets(self, study: Study) -> None:
        """Return ``None``: nothing is added to this model's linear predictor."""

        return None

    def coordinate_box(self, study: Study) -> None:
        """Return ``None``: a log-odds coefficient is admissible anywhere on the line."""

        return None

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the origin, which is where a penalised logistic fit has always started."""

        return (np.zeros(len(self.parameter_names), dtype=np.float64),)

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return ``(name,)``: this model's parameters are numbers, not structured objects."""

        return (name,)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices in chronological order while preserving source row order."""

        coefficients = self._parameter_vector(parameters)
        rows = np.broadcast_to(coefficients, (len(design), len(coefficients)))
        return self.simulate_rows(design, rows, seed=seed)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices given one coefficient vector per row.

        This is the model's only simulator. Constant coefficients are the ordinary case,
        coefficients that differ by group are what :func:`behavio.compose.hierarchical`
        hands down, and coefficients that differ by clock value are what
        :func:`behavio.compose.smooth` hands down; the recursion over generated history is
        identical in all three and used to be written out three times.
        """

        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(design), len(self.coefficient_names)):
            raise ValueError("simulate_rows needs one coefficient per parameter per study row")
        predictors = self._predictor_matrix(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)

        exogenous = self.exogenous_design
        history_start = len(exogenous.feature_names)
        for session_indices in ordered_session_indices(design):
            generated_history: list[float] = []
            for index in session_indices:
                row = coefficients[index]
                if exogenous.intercept:
                    linear_predictor = row[0]
                    if predictors.shape[1]:
                        linear_predictor += float(predictors[index] @ row[1:history_start])
                else:
                    linear_predictor = float(predictors[index] @ row[:history_start])
                for lag in range(1, self.choice_lags + 1):
                    history_value = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                    linear_predictor += row[history_start + lag - 1] * history_value
                choice = int(generator.binomial(1, expit(linear_predictor)))
                choices[index] = choice
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        """Fit the penalized Bernoulli likelihood with deterministic L-BFGS-B."""

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

    def fit_penalised(
        self,
        design: PenalisedDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve any penalized problem this model's linear predictor feeds.

        The likelihood comes from the design rather than from this class, because a
        combinator may have replaced it: ``mix(glm, component)`` hands down a two-process
        density over a widened predictor, and the arithmetic that solves it is still this
        model's own. ``fit_bernoulli`` remains the entry point for the families that fit a
        design matrix against a binary outcome directly and never see a combinator.
        """

        return fit_penalised_linear(
            model_name=model_name,
            model_signature=model_signature,
            parameter_names=design.parameter_names,
            design_matrix=design.design_matrix,
            outcomes=design.outcomes,
            penalty_matrix=design.penalty_matrix,
            likelihood=design.likelihood,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
            offsets=design.offsets,
            box=design.box,
            initial_points=design.initial_points,
            derived_estimates=design.derived_estimates,
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the ridge on every non-intercept coefficient."""

        penalty = np.zeros(len(self.parameter_names), dtype=np.float64)
        penalty[1:] = self.l2
        return np.diag(penalty)

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the isotropic Gaussian prior on one group's coefficient deviations."""

        return ridge_group_penalty(scales)

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw independent Gaussian coefficient deviations for each group."""

        return ridge_group_draw(scales, groups=groups, generator=generator)

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return one-step-ahead probabilities using observed past choices only."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        linear_predictor = self.design_matrix(study) @ fit.estimates
        return Prediction(
            probability=expit(linear_predictor),
            linear_predictor=linear_predictor,
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observed choice without conditioning on future choices."""

        outcomes = self.outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        return self.likelihood.pointwise_log_prob(prediction.linear_predictor, outcomes)

    def _parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
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

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the validated binary choices this model scores."""

        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"outcome column {self.outcome!r} must be numeric") from None
        if not np.all(np.isfinite(outcomes)) or not np.all((outcomes == 0) | (outcomes == 1)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return outcomes

    def _predictor_matrix(self, study: Study) -> NDArray[np.float64]:
        """The exogenous block alone, without the intercept, as simulation consumes it."""

        design = self.exogenous_design
        if not design.terms:
            return np.empty((len(study), 0), dtype=np.float64)
        missing = [name for name in design.required_columns if name not in study.columns]
        if missing:
            raise ModelDataError(f"study is missing predictor columns: {missing}")
        return build_matrix(DesignSpec(terms=design.terms, intercept=False), study).values

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the filtered feature matrix, one column per coefficient.

        Lagged outcomes are read from the study, so a model with ``choice_lags`` needs the
        observed choices present even when only predicting: that is what makes the
        prediction one-step-ahead filtered rather than smoothed.
        """

        if self.choice_lags:
            self.outcomes(study)
        missing = [
            name for name in self.exogenous_design.required_columns if name not in study.columns
        ]
        if missing:
            raise ModelDataError(f"study is missing predictor columns: {missing}")
        return build_matrix(self.design_spec, study).values

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode
