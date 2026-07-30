"""Reference multinomial and omission-aware choice likelihoods.

:class:`MultinomialLogit` is a penalised linear model whose linear predictor is one number
per *category* rather than one number per row, which is the whole of what stopped it being
composable. Declaring :attr:`~MultinomialLogit.predictor_cells` and building a
``(rows, categories, parameters)`` design is all it takes for
:func:`behavio.compose.smooth` and :func:`behavio.compose.hierarchical` to apply, so a
drifting multinomial, a hierarchical multinomial and a hierarchical drifting multinomial
are expressions rather than three more classes in this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from behavio._internal.arrays import protected_array
from behavio.contracts.compose import (
    PenalisedDesign,
    linear_predictor,
    ridge_group_draw,
    ridge_group_penalty,
)
from behavio.design.matrix import (
    DesignSpec,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
)
from behavio.models._kernels.introspection import Describable
from behavio.models._kernels.multinomial import MultinomialLikelihood
from behavio.models._kernels.penalised import fit_penalised_linear
from behavio.models.base import (
    CategoricalPrediction,
    FitResult,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.task.spec import ChoiceData, ChoiceSpec, TaskValidationError
from behavio.trials import REQUIRED_COLUMNS, Study


@dataclass(frozen=True, slots=True)
class MultinomialLogit(Describable):
    """Treatment-coded softmax regression on a fixed task and design coordinate.

    ``include_omission=True`` pools every omission representation declared by
    :class:`~behavio.task.ChoiceSpec` into one additional modeled category. Trial-specific
    unavailable actions receive exactly zero probability; the omission category remains
    available because it represents failure to emit any valid action.

    Composability
    -------------
    This model satisfies :class:`behavio.contracts.compose.PenalisedLinearEstimator`, so
    ``smooth(model)``, ``hierarchical(model)`` and ``hierarchical(smooth(model))`` are all
    available and none of them is a class. The parameter coordinate is unchanged by that:
    a coefficient is still ``category['right']::stimulus``, a smoothed one is
    ``category['right']::stimulus[session_order=0]``, and a hierarchical fit reports the
    population coordinate with the group deviations read off
    :class:`~behavio.compose.HierarchicalFitResult` by group label. Per-category structure
    was always in the *name*; only the *predictor* had to be widened.

    Availability under composition is a modelling statement and not a broadcasting one. An
    unavailable category reaches the likelihood as ``-inf`` in its cell, which contributes
    zero probability, zero gradient and zero curvature, so a trial on which a category was
    not offered carries no information about that category's coefficients at any level. A
    group that was never offered a category therefore gets a deviation of exactly zero for
    it, with the prior standard deviation as its standard error -- the honest answer, and
    not one anything here had to be written to produce.
    """

    choice: ChoiceSpec
    design: DesignSpec = field(default_factory=DesignSpec)
    reference: Any | None = None
    include_omission: bool = False
    omission_label: Any | None = None
    l2: float = 0.0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.choice, ChoiceSpec):
            raise TypeError("choice must be a ChoiceSpec")
        if not isinstance(self.design, DesignSpec):
            raise TypeError("design must be a DesignSpec")
        if not isinstance(self.include_omission, bool):
            raise ValueError("include_omission must be boolean")
        if self.include_omission:
            if not self.choice.omission_values:
                raise ValueError("omission-aware likelihoods require ChoiceSpec.omission_values")
            label = (
                self.choice.omission_values[0]
                if self.omission_label is None
                else _scalar(self.omission_label)
            )
            if _label_key(label) not in {
                _label_key(value) for value in self.choice.omission_values
            }:
                raise ValueError("omission_label must be one of ChoiceSpec.omission_values")
            object.__setattr__(self, "omission_label", label)
        elif self.omission_label is not None:
            raise ValueError("omission_label requires include_omission=True")
        reference = self.choice.options[0] if self.reference is None else _scalar(self.reference)
        if _label_key(reference) not in {_label_key(value) for value in self.categories}:
            raise ValueError("reference must be one of the modeled categories")
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
        object.__setattr__(self, "reference", reference)
        for category in self.categories:
            _portable_category(category)

    @property
    def model_name(self) -> str:
        return "omission-aware-multinomial-logit" if self.include_omission else "multinomial-logit"

    @property
    def categories(self) -> tuple[Any, ...]:
        if self.include_omission:
            return (*self.choice.options, self.omission_label)
        return self.choice.options

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[choice={self.choice.column!r};categories={self.categories!r};"
            f"reference={self.reference!r};design={self.design.signature};l2={self.l2}]"
        )

    @property
    def design_spec(self) -> DesignSpec:
        """The design this model fits, under the name every family now uses for it."""

        return self.design

    @property
    def declared_priors(self) -> tuple[str, ...]:
        if not self.l2:
            return ()
        return (
            f"ridge on every non-intercept coefficient: Normal(0, "
            f"{1.0 / self.l2**0.5:.4g}) (l2={self.l2})",
        )

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.choice.column,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        columns = self.design.required_columns
        if self.choice.available_options_column is not None:
            columns = (*columns, self.choice.available_options_column)
        return tuple(
            column
            for column in dict.fromkeys(columns)
            if column != self.choice.column and column not in REQUIRED_COLUMNS
        )

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def reference_index(self) -> int:
        key = _label_key(self.reference)
        return next(
            index for index, value in enumerate(self.categories) if _label_key(value) == key
        )

    @property
    def estimated_category_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index in range(len(self.categories)) if index != self.reference_index
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names(self.design.feature_names)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """The estimated coordinate, under the name every composable family uses for it."""

        return self.parameter_names

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        """One cell per modeled category, reference category included.

        The reference category is a cell with no parameters rather than an absent one: its
        logit is fixed at zero, and a fixed zero is still a number a row's softmax has to
        normalise over. Naming it here keeps the cell coordinate the same length as
        :attr:`categories`, which is what makes a prediction readable.
        """

        return tuple(f"category[{category!r}]" for category in self.categories)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """A category code is one number per row, so this family declares no channels.

        Cells and channels are independent widenings, and this family is the proof: its
        predictor is one number per category while its observation stays a single code.
        """

        return ()

    @property
    def likelihood(self) -> MultinomialLikelihood:
        """The observation model this estimator's per-category logits feed."""

        return MultinomialLikelihood(self.categories)

    def coordinate_box(self, study: Study) -> None:
        """Return ``None``: a multinomial logit coefficient is admissible anywhere."""

        return None

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the origin, which is where a penalised softmax fit has always started."""

        return (np.zeros(len(self.parameter_names), dtype=np.float64),)

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return ``(name,)``: this model's parameters are numbers, not structured objects."""

        return (name,)

    def category_parameter_names(self, category: Any) -> tuple[str, ...]:
        """Return the parameter names belonging to one modeled category.

        ``hierarchical(model, parameters=model.category_parameter_names("up"))`` is how a
        single category is let vary by group without anyone writing ``repr`` quoting into a
        string literal. The reference category has no parameters and returns ``()``.
        """

        key = _label_key(_scalar(category))
        if key not in {_label_key(value) for value in self.categories}:
            raise ValueError(f"{category!r} is not a modeled category of this model")
        prefix = f"category[{_scalar(category)!r}]::"
        return tuple(name for name in self.parameter_names if name.startswith(prefix))

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        data = self._choice_data(study)
        codes = np.asarray(data.codes, dtype=np.int64).copy()
        if np.any(data.omitted):
            if not self.include_omission:
                raise ModelDataError(
                    "observed omissions require include_omission=True or explicit exclusion"
                )
            codes[data.omitted] = len(self.categories) - 1
        return protected_array(codes, dtype=np.int64)

    # -- the penalised problem ---------------------------------------------------------

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the observed category of each row, as the contract's float vector.

        A code is an index, not a quantity, and it is carried as a float only because one
        outcome vector serves every family. :class:`~behavio.models._kernels.multinomial.\
MultinomialLikelihood` checks that what it is handed is integral before it indexes with it.
        """

        return np.asarray(self.outcome_codes(study), dtype=np.float64)

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the ``(rows, categories, parameters)`` design of this treatment coding.

        Each estimated category owns a contiguous block of the coordinate and sees the
        design only in its own cell; the reference category's cell is all zeros, which is
        the treatment coding written as a tensor rather than as a reshape. Building it
        densely costs a factor of ``categories`` in memory over the ``(rows, features)``
        matrix and a private reshape, and buys the property that every combinator, every
        expansion and every solver in the package is the one the binary families use.
        """

        features = self.design.build(study).values
        n_features = features.shape[1]
        design = np.zeros(
            (len(study), len(self.categories), len(self.estimated_category_indices) * n_features),
            dtype=np.float64,
        )
        for block, category in enumerate(self.estimated_category_indices):
            design[:, category, block * n_features : (block + 1) * n_features] = features
        return design

    def predictor_offsets(self, study: Study) -> NDArray[np.float64] | None:
        """Return ``-inf`` in the cell of every category a row did not offer.

        This is the only route by which per-trial availability reaches the likelihood, and
        it is deliberately not a mask the likelihood is given: an offset is carried through
        every combinator untouched, so a group deviation or a smooth path on a category
        that was unavailable simply receives no data on that trial rather than receiving
        broadcast nonsense.
        """

        available = self._availability(study)
        if bool(np.all(available)):
            return None
        offsets = np.zeros(available.shape, dtype=np.float64)
        offsets[~available] = -np.inf
        return offsets

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the ridge on every non-intercept coefficient of every category."""

        feature_penalty = np.asarray(
            [self.l2 * (name != "intercept") for name in self.design.feature_names],
            dtype=np.float64,
        )
        return np.diag(np.tile(feature_penalty, len(self.estimated_category_indices)))

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the isotropic Gaussian prior on one group's coefficient deviations.

        A multinomial coefficient is an exchangeable number and not a sample of a
        structured object -- the per-category structure lives in which coefficient it is,
        not inside it -- so the deviation prior is the ordinary ridge.
        """

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

    def fit(self, study: Study) -> FitResult:
        """Fit the penalized softmax likelihood with deterministic L-BFGS-B."""

        return self.fit_penalised(
            PenalisedDesign(
                parameter_names=self.parameter_names,
                design_matrix=self.design_matrix(study),
                outcomes=self.outcomes(study),
                penalty_matrix=self.penalty_matrix(),
                likelihood=self.likelihood,
                offsets=self.predictor_offsets(study),
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
        """Solve any penalised softmax problem on this model's optimizer settings."""

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
            optimizer="L-BFGS-B (analytic softmax gradient)",
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> CategoricalPrediction:
        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        logits = linear_predictor(
            self.design_matrix(study), fit.estimates, self.predictor_offsets(study)
        )
        return self.likelihood.prediction(logits, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        prediction = self.predict(study, fit, mode=mode)
        return self.likelihood.pointwise_log_prob(prediction.linear_predictor, self.outcomes(study))

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        names = self.parameter_names
        if set(parameters) != set(names):
            raise ValueError("parameters must match the model and built design exactly")
        try:
            estimates = np.asarray([parameters[name] for name in names], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(estimates)):
            raise ValueError("parameters must contain finite numeric values")
        rows = np.broadcast_to(estimates, (len(design), len(estimates)))
        return self.simulate_rows(design, rows, seed=seed)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices given one coefficient vector per row.

        Constant coefficients are the ordinary case, coefficients that differ by group are
        what :func:`behavio.compose.hierarchical` hands down, and coefficients that differ
        by clock value are what :func:`behavio.compose.smooth` hands down. Availability is
        read from the design study, never from an outcome column, so an unavailable action
        is impossible during simulation rather than merely unlikely.
        """

        if any(_uses_outcome_history(term, self.choice.column) for term in self.design.terms):
            raise ModelDataError(
                "multinomial simulation currently requires outcome-independent design terms"
            )
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(design), len(self.parameter_names)):
            raise ValueError("simulate_rows needs one coefficient per parameter per study row")
        logits = np.einsum(
            "rcp,rp->rc", self.design_matrix(design), values, optimize=True
        ) + np.where(self._availability(design), 0.0, -np.inf)
        probability = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        codes = np.asarray(
            [generator.choice(len(self.categories), p=row) for row in probability],
            dtype=np.int64,
        )
        choices = np.asarray([self.categories[index] for index in codes], dtype=object)
        columns = {name: design[name] for name in design.columns}
        columns[self.choice.column] = choices
        return Study(columns)

    def _choice_data(self, study: Study) -> ChoiceData:
        try:
            return self.choice.read(study)
        except TaskValidationError as error:
            raise ModelDataError(str(error)) from error

    def _availability(self, study: Study) -> NDArray[np.bool_]:
        try:
            actions = self.choice.availability(study)
        except TaskValidationError as error:
            raise ModelDataError(str(error)) from error
        if self.include_omission:
            return np.column_stack((actions, np.ones(len(study), dtype=np.bool_)))
        return actions

    def _parameter_names(self, feature_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"category[{self.categories[category]!r}]::{feature}"
            for category in self.estimated_category_indices
            for feature in feature_names
        )

    def _validate_fit(self, fit: FitResult) -> None:
        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self.parameter_names
        ):
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode is not PredictionMode.FILTERED:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


def _uses_outcome_history(term: Any, outcome: str) -> bool:
    if isinstance(term, (HistoryTerm, HistoryKernelTerm)):
        return term.column == outcome
    if isinstance(term, InteractionTerm):
        return _uses_outcome_history(term.left, outcome) or _uses_outcome_history(
            term.right, outcome
        )
    return False


def _label_key(value: Any) -> tuple[Any, Any]:
    if isinstance(value, bool):
        return bool, value
    if isinstance(value, (int, float)):
        return "number", float(value)
    return type(value), value


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _portable_category(value: Any) -> None:
    scalar = _scalar(value)
    if scalar is None or isinstance(scalar, (str, bool, int)):
        return
    if isinstance(scalar, float) and np.isfinite(scalar):
        return
    raise ValueError(f"modeled categories must be finite JSON scalars: {scalar!r}")
