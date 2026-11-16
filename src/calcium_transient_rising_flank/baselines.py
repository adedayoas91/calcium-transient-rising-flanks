"""Optional adapters for external deconvolution and PAG baselines.

Project-standard trace matrices are always shaped ``(n_rois, n_timepoints)``.
Adapters handle any library-specific reordering internally and preserve the raw
external outputs in their result objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib
from typing import Any

import numpy as np

from .estimators import benjamini_hochberg
from .preprocessing import validate_traces
from .sensitivity import PartialAncestralGraph


def _optional_dependency_error(
    *, extra: str, package: str, feature: str, cause: ImportError
) -> ImportError:
    error = ImportError(
        f"{feature} requires the optional '{extra}' extra "
        f"(install '{package}' or "
        f"'calcium-transient-rising-flank[{extra}]')."
    )
    error.__cause__ = cause
    return error


def _load_oasis_deconvolve() -> Callable[..., Any]:
    last_error: ImportError | None = None
    for module_name, attr_name in (
        ("oasis", "deconvolve"),
        ("oasis.functions", "deconvolve"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue
        backend = getattr(module, attr_name, None)
        if callable(backend):
            return backend
    if last_error is None:
        last_error = ImportError("could not locate deconvolve() in oasis modules")
    raise _optional_dependency_error(
        extra="deconvolution",
        package="oasis-deconv>=0.3.2",
        feature="OASIS deconvolution",
        cause=last_error,
    )


def _load_tigramite_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        dataframe_module = importlib.import_module("tigramite.data_processing")
        lpcmci_module = importlib.import_module("tigramite.lpcmci")
        parcorr_module = importlib.import_module(
            "tigramite.independence_tests.parcorr"
        )
    except ImportError as exc:
        missing_package = getattr(exc, "name", None)
        missing_tigramite = missing_package == "tigramite" or (
            missing_package is not None
            and missing_package.startswith("tigramite.")
        )
        if missing_package and not missing_tigramite:
            error = ImportError(
                "LPCMCI PAG discovery could not import Tigramite because its "
                f"runtime dependency '{missing_package}' is unavailable. "
                "Reinstall 'calcium-transient-rising-flank[pag]' so all PAG "
                "dependencies are present."
            )
            error.__cause__ = exc
            raise error from exc
        raise _optional_dependency_error(
            extra="pag",
            package="tigramite>=5.2.10.1",
            feature="LPCMCI PAG discovery",
            cause=exc,
        ) from exc
    return (
        dataframe_module.DataFrame,
        lpcmci_module.LPCMCI,
        parcorr_module.ParCorr,
    )


def _load_tigramite_pcmci_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        dataframe_module = importlib.import_module("tigramite.data_processing")
        pcmci_module = importlib.import_module("tigramite.pcmci")
        parcorr_module = importlib.import_module(
            "tigramite.independence_tests.parcorr"
        )
    except ImportError as exc:
        raise _optional_dependency_error(
            extra="pag",
            package="tigramite>=5.2.10.1",
            feature="PCMCI+ causal discovery",
            cause=exc,
        ) from exc
    return (
        dataframe_module.DataFrame,
        pcmci_module.PCMCI,
        parcorr_module.ParCorr,
    )


def _load_f_distribution() -> Any:
    try:
        return importlib.import_module("scipy.stats").f
    except ImportError as exc:
        raise _optional_dependency_error(
            extra="pag",
            package="scipy",
            feature="VAR Granger F-tests",
            cause=exc,
        ) from exc


def _as_float_scalar(value: Any, *, name: str) -> float:
    scalar = np.asarray(value, dtype=float)
    if scalar.size != 1:
        raise ValueError(f"{name} must be scalar per ROI")
    return float(scalar.reshape(()))


def _normalise_trace_output(
    value: Any, *, expected_length: int, name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.size != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} samples per ROI"
        )
    return array.reshape(expected_length).copy()


def _normalise_ar_params(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return (float(array.reshape(())),)
    if array.ndim != 1:
        raise ValueError("AR parameters must be scalar or one-dimensional")
    return tuple(float(x) for x in array.tolist())


def _coerce_oasis_result(
    result: Any, *, expected_length: int
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, ...], float]:
    if all(hasattr(result, name) for name in ("c", "s", "b", "g", "lam")):
        denoised = result.c
        spikes = result.s
        baseline = result.b
        ar_params = result.g
        penalty_lambda = result.lam
    else:
        unpacked = tuple(result)
        if len(unpacked) != 5:
            raise ValueError("OASIS backend must return five values: c, s, b, g, lam")
        denoised, spikes, baseline, ar_params, penalty_lambda = unpacked
    return (
        _normalise_trace_output(
            denoised, expected_length=expected_length, name="denoised trace"
        ),
        _normalise_trace_output(
            spikes, expected_length=expected_length, name="spike trace"
        ),
        _as_float_scalar(baseline, name="baseline"),
        _normalise_ar_params(ar_params),
        _as_float_scalar(penalty_lambda, name="penalty lambda"),
    )


def project_lossy_lagged_pag_skeleton(graph: np.ndarray) -> np.ndarray:
    """Return a lossy lagged-only skeleton from a raw LPCMCI PAG tensor."""

    values = np.asarray(graph)
    if values.ndim != 3:
        raise ValueError("graph must have shape (n_rois, n_rois, tau_max + 1)")
    if values.shape[2] <= 1:
        return np.zeros(values.shape[:2], dtype=bool)
    skeleton = np.any(values[:, :, 1:] != "", axis=2)
    np.fill_diagonal(skeleton, False)
    return skeleton


def project_lagged_directed_graph(graph: np.ndarray) -> np.ndarray:
    """Project lagged Tigramite ``-->`` marks to ``[source, target]`` edges."""

    values = np.asarray(graph)
    if values.ndim != 3:
        raise ValueError("graph must have shape (n_rois, n_rois, tau_max + 1)")
    if values.shape[2] <= 1:
        return np.zeros(values.shape[:2], dtype=bool)
    adjacency = np.any(values[:, :, 1:] == "-->", axis=2)
    np.fill_diagonal(adjacency, False)
    return adjacency


def project_significant_lagged_directed_graph(
    graph: np.ndarray,
    p_matrix: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Project lagged directed links whose adjusted p-value meets ``alpha``."""

    values = np.asarray(graph)
    p_values = np.asarray(p_matrix, dtype=float)
    if values.ndim != 3 or p_values.shape != values.shape:
        raise ValueError("graph and p_matrix must be matching three-dimensional arrays")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if values.shape[2] <= 1:
        return np.zeros(values.shape[:2], dtype=bool)
    adjacency = np.any(
        (values[:, :, 1:] == "-->") & (p_values[:, :, 1:] <= alpha),
        axis=2,
    )
    np.fill_diagonal(adjacency, False)
    return adjacency


@dataclass(frozen=True)
class OASISResult:
    """Per-ROI OASIS deconvolution output on the original ROI x time axis."""

    denoised: np.ndarray
    spikes: np.ndarray
    baseline: np.ndarray
    ar_params: tuple[tuple[float, ...], ...]
    penalty_lambda: np.ndarray
    method: str = "oasis"


@dataclass(frozen=True)
class OASISDeconvolver:
    """Thin adapter around ``oasis.deconvolve`` with lazy optional imports."""

    backend: Callable[..., Any] | None = None
    backend_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def fit(self, traces: np.ndarray) -> OASISResult:
        values = validate_traces(traces)
        backend = self.backend or _load_oasis_deconvolve()
        kwargs = dict(self.backend_kwargs)

        denoised = np.zeros_like(values)
        spikes = np.zeros_like(values)
        baseline = np.zeros(values.shape[0], dtype=float)
        penalty_lambda = np.zeros(values.shape[0], dtype=float)
        ar_params: list[tuple[float, ...]] = []

        for roi, row in enumerate(values):
            c, s, b, g, lam = _coerce_oasis_result(
                backend(row.copy(), **kwargs),
                expected_length=values.shape[1],
            )
            denoised[roi] = c
            spikes[roi] = s
            baseline[roi] = b
            penalty_lambda[roi] = lam
            ar_params.append(g)

        return OASISResult(
            denoised=denoised,
            spikes=spikes,
            baseline=baseline,
            ar_params=tuple(ar_params),
            penalty_lambda=penalty_lambda,
        )


@dataclass(frozen=True, kw_only=True)
class LPCMCIResult(PartialAncestralGraph):
    """Raw LPCMCI PAG output plus a named lossy lagged skeleton projection."""

    graph: np.ndarray
    p_matrix: np.ndarray
    val_matrix: np.ndarray
    tau_max: int
    cond_ind_test: str

    @property
    def raw_pag(self) -> np.ndarray:
        return self.graph

    def lossy_lagged_skeleton(self) -> np.ndarray:
        return project_lossy_lagged_pag_skeleton(self.graph)


@dataclass(frozen=True)
class LPCMCIAdapter:
    """PAGEstimator-compatible LPCMCI adapter with injectable test backends."""

    tau_max: int
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = (
        "raw LPCMCI PAG preserved in graph/raw_pag",
        "endpoint_marks aliases the full PAG tensor to avoid flattening",
    )
    verbosity: int = 0
    dataframe_factory: Callable[[np.ndarray], Any] | None = None
    cond_ind_test_factory: Callable[[], Any] | None = None
    estimator_factory: Callable[[Any, Any, int], Any] | None = None

    def _build_backend(self, traces_t_by_n: np.ndarray) -> tuple[Any, Any, Any]:
        if self.dataframe_factory is None:
            dataframe_class, estimator_class, parcorr_class = _load_tigramite_classes()
            dataframe = dataframe_class(traces_t_by_n)
        else:
            dataframe = self.dataframe_factory(traces_t_by_n)
            estimator_class = None
            parcorr_class = None

        if self.cond_ind_test_factory is not None:
            cond_ind_test = self.cond_ind_test_factory()
        else:
            if parcorr_class is None:
                _, _, parcorr_class = _load_tigramite_classes()
            cond_ind_test = parcorr_class()

        if self.estimator_factory is not None:
            estimator = self.estimator_factory(
                dataframe, cond_ind_test, self.verbosity
            )
        else:
            if estimator_class is None:
                _, estimator_class, _ = _load_tigramite_classes()
            estimator = estimator_class(
                dataframe=dataframe,
                cond_ind_test=cond_ind_test,
                verbosity=self.verbosity,
            )
        return dataframe, cond_ind_test, estimator

    def fit(self, traces: np.ndarray) -> LPCMCIResult:
        if self.tau_max < 0:
            raise ValueError("tau_max must be non-negative")
        if "tau_max" in self.run_kwargs:
            raise ValueError("run_kwargs must not override tau_max")

        values = validate_traces(traces)
        traces_t_by_n = values.T.copy()
        _, cond_ind_test, estimator = self._build_backend(traces_t_by_n)

        result = estimator.run_lpcmci(
            tau_max=self.tau_max,
            **dict(self.run_kwargs),
        )

        graph = np.asarray(result["graph"]).copy()
        p_matrix = np.asarray(result["p_matrix"], dtype=float).copy()
        val_matrix = np.asarray(result["val_matrix"], dtype=float).copy()
        expected_shape = (values.shape[0], values.shape[0], self.tau_max + 1)

        if graph.shape != expected_shape:
            raise ValueError(
                "LPCMCI graph must have shape "
                f"{expected_shape}, got {graph.shape}"
            )
        if p_matrix.shape != expected_shape or val_matrix.shape != expected_shape:
            raise ValueError(
                "LPCMCI p/val matrices must match the raw PAG tensor shape"
            )

        return LPCMCIResult(
            endpoint_marks=graph.copy(),
            method="lpcmci",
            assumptions=self.assumptions,
            graph=graph,
            p_matrix=p_matrix,
            val_matrix=val_matrix,
            tau_max=self.tau_max,
            cond_ind_test=type(cond_ind_test).__name__,
        )


@dataclass(frozen=True)
class PCMCIPlusResult:
    """Raw PCMCI+ graph and test tensors with a lagged directed projection."""

    graph: np.ndarray
    p_matrix: np.ndarray
    val_matrix: np.ndarray
    tau_max: int
    cond_ind_test: str
    assumptions: tuple[str, ...]
    method: str = "pcmciplus"

    def lagged_adjacency(self) -> np.ndarray:
        return project_lagged_directed_graph(self.graph)


@dataclass(frozen=True)
class PCMCIPlusAdapter:
    """Fast Tigramite PCMCI+ adapter for causally sufficient time series."""

    tau_max: int
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = (
        "causal sufficiency (no unobserved common causes)",
        "stationarity over the fitted interval",
        "ParCorr tests linear conditional dependence",
        "the native graph is selected by pc_alpha before optional p-value FDR",
        "lagged projection excludes contemporaneous endpoint marks",
    )
    verbosity: int = 0
    dataframe_factory: Callable[[np.ndarray], Any] | None = None
    cond_ind_test_factory: Callable[[], Any] | None = None
    estimator_factory: Callable[[Any, Any, int], Any] | None = None

    def _build_backend(self, traces_t_by_n: np.ndarray) -> tuple[Any, Any, Any]:
        if self.dataframe_factory is None:
            dataframe_class, estimator_class, parcorr_class = (
                _load_tigramite_pcmci_classes()
            )
            dataframe = dataframe_class(traces_t_by_n)
        else:
            dataframe = self.dataframe_factory(traces_t_by_n)
            estimator_class = None
            parcorr_class = None

        if self.cond_ind_test_factory is not None:
            cond_ind_test = self.cond_ind_test_factory()
        else:
            if parcorr_class is None:
                _, _, parcorr_class = _load_tigramite_pcmci_classes()
            cond_ind_test = parcorr_class()

        if self.estimator_factory is not None:
            estimator = self.estimator_factory(
                dataframe, cond_ind_test, self.verbosity
            )
        else:
            if estimator_class is None:
                _, estimator_class, _ = _load_tigramite_pcmci_classes()
            estimator = estimator_class(
                dataframe=dataframe,
                cond_ind_test=cond_ind_test,
                verbosity=self.verbosity,
            )
        return dataframe, cond_ind_test, estimator

    def fit(self, traces: np.ndarray) -> PCMCIPlusResult:
        if self.tau_max < 1:
            raise ValueError("tau_max must be positive")
        if "tau_max" in self.run_kwargs:
            raise ValueError("run_kwargs must not override tau_max")

        values = validate_traces(traces)
        _, cond_ind_test, estimator = self._build_backend(values.T.copy())
        kwargs = {"tau_min": 0, "pc_alpha": 0.05, **dict(self.run_kwargs)}
        result = estimator.run_pcmciplus(tau_max=self.tau_max, **kwargs)

        graph = np.asarray(result["graph"]).copy()
        p_matrix = np.asarray(result["p_matrix"], dtype=float).copy()
        val_matrix = np.asarray(result["val_matrix"], dtype=float).copy()
        expected_shape = (values.shape[0], values.shape[0], self.tau_max + 1)
        if graph.shape != expected_shape:
            raise ValueError(
                f"PCMCI+ graph must have shape {expected_shape}, got {graph.shape}"
            )
        if p_matrix.shape != expected_shape or val_matrix.shape != expected_shape:
            raise ValueError("PCMCI+ p/val matrices must match the graph tensor")
        return PCMCIPlusResult(
            graph=graph,
            p_matrix=p_matrix,
            val_matrix=val_matrix,
            tau_max=self.tau_max,
            cond_ind_test=type(cond_ind_test).__name__,
            assumptions=self.assumptions,
        )


@dataclass(frozen=True)
class VARGrangerResult:
    """Conditional multivariate Granger results from nested VAR regressions."""

    adjacency: np.ndarray
    p_values: np.ndarray
    scores: np.ndarray
    best_lags: np.ndarray
    coefficients: np.ndarray
    companion_spectral_radius: float
    stable: bool
    max_lag: int
    alpha: float
    fdr: bool
    assumptions: tuple[str, ...]
    method: str = "var-granger"


@dataclass(frozen=True)
class VARGrangerAdapter:
    """Conditional linear VAR Granger learner with nested-model F-tests."""

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    assumptions: tuple[str, ...] = (
        "linear stationary VAR dynamics",
        "no omitted common causes sufficient to induce spurious predictability",
        "independent homoscedastic approximately Gaussian regression innovations",
    )
    f_distribution: Any | None = None

    def fit(self, traces: np.ndarray) -> VARGrangerResult:
        values = validate_traces(traces)
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        n_nodes, n_steps = values.shape
        n_observations = n_steps - self.max_lag
        n_predictors = n_nodes * self.max_lag
        if n_observations <= n_predictors + 1:
            raise ValueError(
                "not enough timepoints for the requested conditional VAR order"
            )

        lagged = np.column_stack(
            [
                values[:, self.max_lag - lag : n_steps - lag].T
                for lag in range(1, self.max_lag + 1)
            ]
        )
        design = np.column_stack([np.ones(n_observations), lagged])
        outcomes = values[:, self.max_lag :].T
        unrestricted_coef, *_ = np.linalg.lstsq(design, outcomes, rcond=None)
        unrestricted_residual = outcomes - design @ unrestricted_coef
        unrestricted_rss = np.sum(unrestricted_residual**2, axis=0)
        unrestricted_rank = int(np.linalg.matrix_rank(design))
        denominator_df = n_observations - unrestricted_rank
        if denominator_df <= 0:
            raise ValueError("unrestricted VAR has no residual degrees of freedom")
        lag_matrices = [
            unrestricted_coef[
                1 + lag * n_nodes : 1 + (lag + 1) * n_nodes
            ].T
            for lag in range(self.max_lag)
        ]
        companion = np.zeros(
            (n_nodes * self.max_lag, n_nodes * self.max_lag), dtype=float
        )
        companion[:n_nodes] = np.hstack(lag_matrices)
        if self.max_lag > 1:
            companion[n_nodes:, :-n_nodes] = np.eye(
                n_nodes * (self.max_lag - 1)
            )
        companion_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(companion)))
        )

        scores = np.zeros((n_nodes, n_nodes), dtype=float)
        p_values = np.ones((n_nodes, n_nodes), dtype=float)
        best_lags = np.zeros((n_nodes, n_nodes), dtype=int)
        coefficients = np.zeros((n_nodes, n_nodes, self.max_lag), dtype=float)
        f_dist = self.f_distribution or _load_f_distribution()
        eps = np.finfo(float).eps

        for source in range(n_nodes):
            source_columns = np.asarray(
                [1 + lag * n_nodes + source for lag in range(self.max_lag)],
                dtype=int,
            )
            restricted = np.delete(design, source_columns, axis=1)
            restricted_coef, *_ = np.linalg.lstsq(
                restricted, outcomes, rcond=None
            )
            restricted_residual = outcomes - restricted @ restricted_coef
            restricted_rss = np.sum(restricted_residual**2, axis=0)
            restrictions = max(
                1,
                unrestricted_rank - int(np.linalg.matrix_rank(restricted)),
            )
            source_coefficients = unrestricted_coef[source_columns, :]
            coefficients[source] = source_coefficients.T
            best_lags[source] = np.argmax(
                np.abs(source_coefficients), axis=0
            ) + 1
            for target in range(n_nodes):
                if source == target:
                    continue
                rss_u = float(np.maximum(float(unrestricted_rss[target]), eps))
                rss_r = float(np.maximum(float(restricted_rss[target]), rss_u))
                scores[source, target] = max(0.0, float(np.log(rss_r / rss_u)))
                statistic = max(
                    0.0,
                    ((rss_r - rss_u) / restrictions)
                    / (rss_u / denominator_df),
                )
                p_values[source, target] = float(
                    f_dist.sf(statistic, restrictions, denominator_df)
                )

        adjacency = (
            benjamini_hochberg(p_values, self.alpha)
            if self.fdr
            else p_values <= self.alpha
        )
        np.fill_diagonal(adjacency, False)
        np.fill_diagonal(p_values, 1.0)
        np.fill_diagonal(scores, 0.0)
        np.fill_diagonal(best_lags, 0)
        return VARGrangerResult(
            adjacency=np.asarray(adjacency, dtype=bool),
            p_values=p_values,
            scores=scores,
            best_lags=best_lags,
            coefficients=coefficients,
            companion_spectral_radius=companion_spectral_radius,
            stable=companion_spectral_radius < 1.0,
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            assumptions=self.assumptions,
        )


__all__ = [
    "LPCMCIAdapter",
    "LPCMCIResult",
    "OASISDeconvolver",
    "OASISResult",
    "PCMCIPlusAdapter",
    "PCMCIPlusResult",
    "VARGrangerAdapter",
    "VARGrangerResult",
    "project_lagged_directed_graph",
    "project_significant_lagged_directed_graph",
    "project_lossy_lagged_pag_skeleton",
]
