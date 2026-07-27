"""Backend-neutral labelled posterior results and optional ArviZ interchange."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

SAMPLE_DIMS: Final = ("chain", "draw")
POSTERIOR_GROUPS: Final = (
    "posterior",
    "unconstrained_posterior",
    "sample_stats",
    "log_likelihood",
    "posterior_predictive",
    "observed_data",
    "constant_data",
    "predictions",
    "predictions_constant_data",
)
_SAMPLED_GROUPS: Final = frozenset(
    {
        "posterior",
        "unconstrained_posterior",
        "sample_stats",
        "log_likelihood",
        "posterior_predictive",
        "predictions",
    }
)
_UNSAMPLED_GROUPS: Final = frozenset(
    {"observed_data", "constant_data", "predictions_constant_data"}
)


class PosteriorError(ValueError):
    """Raised when labelled posterior evidence violates the common contract."""


class ArviZUnavailableError(ImportError):
    """Raised when optional ArviZ interoperability is requested but unavailable."""


@dataclass(frozen=True, slots=True)
class PosteriorVariable:
    """One immutable labelled array within a posterior-result group.

    ``dims`` names every axis, including ``chain`` and ``draw`` for sampled values.
    Every dimension must have an explicit one-dimensional coordinate. This is a small
    NumPy-native interchange object, not a replacement for xarray.
    """

    name: str
    values: NDArray[Any] | Sequence[Any]
    dims: tuple[str, ...]
    coords: Mapping[str, Sequence[Any] | NDArray[Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PosteriorError("variable name must be a non-empty string")
        dims = tuple(self.dims)
        if any(not isinstance(dim, str) or not dim for dim in dims):
            raise PosteriorError("variable dimensions must be non-empty strings")
        if len(set(dims)) != len(dims):
            raise PosteriorError(f"variable {self.name!r} has duplicate dimensions")
        if self.name in dims:
            raise PosteriorError("variable names must not also be dimension names")
        values = np.array(self.values, copy=True)
        if values.ndim != len(dims):
            raise PosteriorError(
                f"variable {self.name!r} has {values.ndim} axes but {len(dims)} dimensions"
            )
        if not isinstance(self.coords, Mapping):
            raise PosteriorError("variable coords must be a mapping")
        if set(self.coords) != set(dims):
            raise PosteriorError(f"variable {self.name!r} coords must name exactly its dimensions")
        protected_coords: dict[str, NDArray[Any]] = {}
        for axis, dim in enumerate(dims):
            coordinate = np.array(self.coords[dim], copy=True)
            if coordinate.ndim != 1 or len(coordinate) != values.shape[axis]:
                raise PosteriorError(
                    f"coordinate {dim!r} must have one label per {self.name!r} value"
                )
            coordinate.setflags(write=False)
            protected_coords[dim] = coordinate
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", MappingProxyType(protected_coords))

    @property
    def intrinsic_dims(self) -> tuple[str, ...]:
        """Dimensions other than the canonical sampling dimensions."""

        return tuple(dim for dim in self.dims if dim not in SAMPLE_DIMS)


@dataclass(frozen=True, slots=True)
class PosteriorGroup:
    """A named conceptual group of labelled posterior-result variables."""

    name: str
    variables: tuple[PosteriorVariable, ...]
    attrs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name not in POSTERIOR_GROUPS:
            raise PosteriorError(
                f"unsupported posterior group {self.name!r}; expected one of {POSTERIOR_GROUPS}"
            )
        variables = tuple(self.variables)
        if not variables or any(not isinstance(item, PosteriorVariable) for item in variables):
            raise PosteriorError(f"group {self.name!r} must contain PosteriorVariable records")
        names = tuple(item.name for item in variables)
        if len(set(names)) != len(names):
            raise PosteriorError(f"group {self.name!r} has duplicate variable names")
        attrs = {} if self.attrs is None else dict(self.attrs)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "attrs", _freeze_metadata(attrs))

    def __getitem__(self, name: str) -> PosteriorVariable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Stable variable order within the group."""

        return tuple(variable.name for variable in self.variables)


@dataclass(frozen=True, slots=True)
class PosteriorResult:
    """A backend-neutral Bayesian fit result in natural parameter coordinates.

    The object retains labelled samples and the observations they condition on while
    keeping xarray and ArviZ optional. Posterior variables named by ``parameter_names``
    must be in the model's constrained, natural parameter space.
    """

    model_name: str
    model_signature: str
    inference_library: str
    inference_library_version: str
    parameter_names: tuple[str, ...]
    groups: tuple[PosteriorGroup, ...]
    parameter_space_fingerprint: str | None = None
    attrs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.model_name, "model_name"),
            (self.model_signature, "model_signature"),
            (self.inference_library, "inference_library"),
            (self.inference_library_version, "inference_library_version"),
        ):
            if not isinstance(value, str) or not value:
                raise PosteriorError(f"{label} must be a non-empty string")
        parameter_names = tuple(self.parameter_names)
        if (
            not parameter_names
            or len(set(parameter_names)) != len(parameter_names)
            or any(not isinstance(name, str) or not name for name in parameter_names)
        ):
            raise PosteriorError("parameter_names must be non-empty and unique")
        if self.parameter_space_fingerprint is not None and (
            not isinstance(self.parameter_space_fingerprint, str)
            or not self.parameter_space_fingerprint
        ):
            raise PosteriorError("parameter_space_fingerprint must be null or non-empty")
        groups = tuple(self.groups)
        if not groups or any(not isinstance(group, PosteriorGroup) for group in groups):
            raise PosteriorError("groups must contain PosteriorGroup records")
        group_names = tuple(group.name for group in groups)
        if len(set(group_names)) != len(group_names):
            raise PosteriorError("posterior result group names must be unique")
        if "posterior" not in group_names:
            raise PosteriorError("posterior result must contain a posterior group")
        posterior = next(group for group in groups if group.name == "posterior")
        missing = set(parameter_names) - set(posterior.variable_names)
        if missing:
            raise PosteriorError(f"posterior is missing declared parameters {sorted(missing)}")
        _validate_group_dimensions(groups)
        _validate_shared_coordinates(groups)
        _validate_group_relations(groups)
        attrs = {} if self.attrs is None else dict(self.attrs)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "attrs", _freeze_metadata(attrs))

    def __getitem__(self, name: str) -> PosteriorGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(name)

    @property
    def group_names(self) -> tuple[str, ...]:
        """Stable group order within the result."""

        return tuple(group.name for group in self.groups)

    @property
    def n_chains(self) -> int:
        """Number of posterior chains."""

        return len(self["posterior"].variables[0].coords["chain"])

    @property
    def n_draws(self) -> int:
        """Number of retained draws in each posterior chain."""

        return len(self["posterior"].variables[0].coords["draw"])

    def to_arviz(self) -> Any:
        """Export to the installed ArviZ labelled inference container.

        ArviZ 0.22-0.23 return ``InferenceData``; ArviZ 1.2 and later return an
        ``xarray.DataTree``. Both representations follow the same group contract.
        """

        arviz = _import_arviz()
        data = {
            group.name: {variable.name: variable.values for variable in group.variables}
            for group in self.groups
        }
        fitting_groups = tuple(group for group in self.groups if "predictions" not in group.name)
        prediction_groups = tuple(group for group in self.groups if "predictions" in group.name)
        coords = _collect_coordinates(fitting_groups)
        pred_coords = _collect_coordinates(prediction_groups)
        dims: dict[str, list[str]] = {}
        pred_dims: dict[str, list[str]] = {}
        for group in self.groups:
            target = pred_dims if "predictions" in group.name else dims
            for variable in group.variables:
                intrinsic = list(variable.intrinsic_dims)
                previous = target.get(variable.name)
                if previous is not None and previous != intrinsic:
                    raise PosteriorError(
                        f"variable {variable.name!r} uses conflicting intrinsic dimensions"
                    )
                target[variable.name] = intrinsic
        root_attrs = {
            **_thaw_metadata(self.attrs),
            "name": self.model_name,
            "model_signature": self.model_signature,
            "creation_library": "Unspool",
            "creation_library_version": _package_version(),
            "creation_library_language": "Python",
            "inference_library": self.inference_library,
            "inference_library_version": self.inference_library_version,
        }
        if self.parameter_space_fingerprint is not None:
            root_attrs["parameter_space_fingerprint"] = self.parameter_space_fingerprint
        group_attrs = {
            group.name: {
                **_thaw_metadata(group.attrs),
                **({"sample_dims": list(SAMPLE_DIMS)} if group.name in _SAMPLED_GROUPS else {}),
                **(
                    {"sampled_variables": list(self.parameter_names)}
                    if group.name == "posterior"
                    else {}
                ),
            }
            for group in self.groups
        }

        if _arviz_uses_nested_dict(arviz):
            attrs = {"/": root_attrs, **group_attrs}
            return arviz.from_dict(
                data,
                name=self.model_name,
                sample_dims=list(SAMPLE_DIMS),
                coords=coords,
                dims=dims,
                pred_dims=pred_dims,
                pred_coords=pred_coords,
                attrs=attrs,
            )
        return _to_legacy_arviz(arviz, self.groups, root_attrs, group_attrs)


def posterior_result_from_arviz(
    data: Any,
    *,
    model_name: str,
    model_signature: str,
    inference_library: str,
    inference_library_version: str,
    parameter_names: Sequence[str] | None = None,
    parameter_space_fingerprint: str | None = None,
) -> PosteriorResult:
    """Import standard groups from an ArviZ ``InferenceData`` or ``DataTree``.

    Explicit model and backend identity is required so importing a convenient analysis
    object cannot silently redefine scientific provenance.
    """

    _import_arviz()
    available = _arviz_group_names(data)
    groups: list[PosteriorGroup] = []
    for group_name in POSTERIOR_GROUPS:
        if group_name not in available:
            continue
        dataset = _arviz_dataset(data, group_name)
        variables = []
        for variable_name in dataset.data_vars:
            array = dataset[variable_name]
            coords = {dim: np.asarray(array.coords[dim].values) for dim in array.dims}
            variables.append(
                PosteriorVariable(
                    name=str(variable_name),
                    values=np.asarray(array.values),
                    dims=tuple(str(dim) for dim in array.dims),
                    coords=coords,
                )
            )
        if variables:
            groups.append(
                PosteriorGroup(
                    name=group_name,
                    variables=tuple(variables),
                    attrs=_plain_metadata(dataset.attrs),
                )
            )
    if not groups or groups[0].name != "posterior":
        raise PosteriorError("ArviZ data must contain a non-empty posterior group")
    posterior_attrs = groups[0].attrs
    if parameter_names is None:
        declared = posterior_attrs.get("sampled_variables")
        parameter_names = (
            tuple(str(name) for name in declared)
            if isinstance(declared, (list, tuple)) and declared
            else groups[0].variable_names
        )
    return PosteriorResult(
        model_name=model_name,
        model_signature=model_signature,
        inference_library=inference_library,
        inference_library_version=inference_library_version,
        parameter_names=tuple(parameter_names),
        groups=tuple(groups),
        parameter_space_fingerprint=parameter_space_fingerprint,
    )


def _validate_group_dimensions(groups: tuple[PosteriorGroup, ...]) -> None:
    for group in groups:
        for variable in group.variables:
            has_chain = "chain" in variable.dims
            has_draw = "draw" in variable.dims
            if has_chain != has_draw:
                raise PosteriorError(
                    f"{group.name}.{variable.name} must contain both chain and draw or neither"
                )
            if group.name in _UNSAMPLED_GROUPS and (has_chain or has_draw):
                raise PosteriorError(f"{group.name} variables must not contain sample dimensions")
            if group.name in _SAMPLED_GROUPS and variable.dims[:2] != SAMPLE_DIMS:
                raise PosteriorError(
                    f"{group.name}.{variable.name} must lead with chain and draw dimensions"
                )


def _validate_shared_coordinates(groups: tuple[PosteriorGroup, ...]) -> None:
    _collect_coordinates(tuple(group for group in groups if "predictions" not in group.name))
    _collect_coordinates(tuple(group for group in groups if "predictions" in group.name))
    posterior = next(group for group in groups if group.name == "posterior")
    reference = posterior.variables[0]
    for variable in posterior.variables[1:]:
        for dim in SAMPLE_DIMS:
            if not np.array_equal(variable.coords[dim], reference.coords[dim]):
                raise PosteriorError(
                    "all posterior variables must share chain and draw coordinates"
                )
    for group in groups:
        if group.name not in _SAMPLED_GROUPS:
            continue
        for variable in group.variables:
            if "chain" not in variable.dims:
                continue
            for dim in SAMPLE_DIMS:
                if not np.array_equal(variable.coords[dim], reference.coords[dim]):
                    raise PosteriorError(
                        f"{group.name}.{variable.name} does not match posterior samples"
                    )


def _validate_group_relations(groups: tuple[PosteriorGroup, ...]) -> None:
    by_name = {group.name: group for group in groups}
    observed = by_name.get("observed_data")
    if observed is not None:
        observed_by_name = {variable.name: variable for variable in observed.variables}
        predictive = by_name.get("posterior_predictive")
        if predictive is not None:
            for variable in predictive.variables:
                counterpart = observed_by_name.get(variable.name)
                if counterpart is not None and variable.intrinsic_dims != counterpart.dims:
                    raise PosteriorError(
                        f"posterior_predictive.{variable.name} dimensions do not match "
                        "observed_data"
                    )
        likelihood = by_name.get("log_likelihood")
        if likelihood is not None:
            for variable in likelihood.variables:
                counterpart = observed_by_name.get(variable.name)
                if counterpart is None:
                    continue
                for dim in set(variable.intrinsic_dims) & set(counterpart.dims):
                    if not np.array_equal(variable.coords[dim], counterpart.coords[dim]):
                        raise PosteriorError(
                            f"log_likelihood.{variable.name} coordinates do not match observed_data"
                        )


def _collect_coordinates(groups: tuple[PosteriorGroup, ...]) -> dict[str, NDArray[Any]]:
    coordinates: dict[str, NDArray[Any]] = {}
    for group in groups:
        for variable in group.variables:
            for dim, coordinate in variable.coords.items():
                previous = coordinates.get(dim)
                if previous is not None and not np.array_equal(previous, coordinate):
                    raise PosteriorError(
                        f"dimension {dim!r} has conflicting coordinates across posterior groups"
                    )
                coordinates[dim] = coordinate
    return coordinates


def _import_arviz() -> Any:
    try:
        return importlib.import_module("arviz")
    except ImportError as error:
        raise ArviZUnavailableError(
            "ArviZ interoperability requires `pip install 'unspool[probabilistic]'`"
        ) from error


def _arviz_uses_nested_dict(arviz: Any) -> bool:
    import inspect

    parameters = inspect.signature(arviz.from_dict).parameters
    return "data" in parameters and "posterior" not in parameters


def _to_legacy_arviz(
    arviz: Any,
    groups: tuple[PosteriorGroup, ...],
    root_attrs: Mapping[str, Any],
    group_attrs: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Build old InferenceData without its cross-group prediction-dimension collision."""

    xarray = importlib.import_module("xarray")
    datasets = {}
    for group in groups:
        coords: dict[str, NDArray[Any]] = {}
        data_vars = {}
        for variable in group.variables:
            data_vars[variable.name] = (variable.dims, variable.values)
            coords.update(variable.coords)
        datasets[group.name] = xarray.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=dict(group_attrs[group.name]),
        )
    return arviz.InferenceData(attrs=dict(root_attrs), **datasets)


def _arviz_group_names(data: Any) -> set[str]:
    children = getattr(data, "children", None)
    if children is not None:
        return {str(name) for name in children}
    groups = getattr(data, "groups", None)
    if callable(groups):
        return {str(name) for name in groups()}
    raise PosteriorError("data is not an ArviZ InferenceData or xarray DataTree")


def _arviz_dataset(data: Any, group_name: str) -> Any:
    children = getattr(data, "children", None)
    if children is not None:
        return data[group_name].to_dataset()
    return getattr(data, group_name)


def _plain_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            plain[str(key)] = value.tolist()
        elif isinstance(value, np.generic):
            plain[str(key)] = value.item()
        else:
            plain[str(key)] = value
    return plain


def _freeze_metadata(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise PosteriorError("metadata keys must be non-empty strings")
        if isinstance(value, Mapping):
            frozen[key] = _freeze_metadata(value)
        elif isinstance(value, (list, tuple)):
            frozen[key] = tuple(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            frozen[key] = value
        else:
            raise PosteriorError(f"metadata value for {key!r} is not JSON-like")
    return MappingProxyType(frozen)


def _thaw_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    thawed: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            thawed[key] = _thaw_metadata(value)
        elif isinstance(value, tuple):
            thawed[key] = list(value)
        else:
            thawed[key] = value
    return thawed


def _package_version() -> str:
    try:
        return importlib.metadata.version("unspool")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
