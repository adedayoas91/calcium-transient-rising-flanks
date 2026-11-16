"""Orchestration for paired rise/fall c-GC deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .estimators import (
    CausalisedGC,
    GraphResult,
    RiseFlankCandidateResult,
    selected_frame_indices,
)
from .metrics import BilateralMetric, delta_w_ic, w_ic, w_rc
from .preprocessing import ScenarioData, build_scenarios
from .representations import RepresentationBundle, build_representations


@dataclass(frozen=True)
class AnalysisConfig:
    """Prespecified settings for a complete trace-analysis deployment."""

    max_lag: int = 1
    tau: int | None = None
    n_pasts: int | None = None
    n_surrogates: int = 0
    alpha: float = 0.05
    random_state: int | None = None
    tolerance: float | np.ndarray = 0.0
    gamma: float | np.ndarray | None = None
    bad_neurons: tuple[int, ...] = ()
    artifact_frames: tuple[int, ...] = ()
    smoothing_window: int = 3
    metric_binary: bool = False
    fdr: bool = True
    score_threshold: float = 0.0
    event_mode: str = "compressed"
    method: str = "cgc"
    beta: float | None = None
    segment_ids_by_representation: dict[str, np.ndarray] | None = None
    min_rise_run_samples: int = 1
    rise_candidate_filter: bool = False
    rise_match_min_lag: int = 1
    rise_match_max_lag: int | None = None
    rise_match_min_overlap_samples: int | None = None
    rise_match_min_overlap_fraction: float = 0.5
    rise_run_context_samples: int | None = None


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioData
    representations: RepresentationBundle
    cgc: dict[str, GraphResult]
    w_ic: dict[str, BilateralMetric | None]
    w_rc: dict[str, BilateralMetric | None] = field(default_factory=dict)
    delta_w_ic: float | None = None
    rise_flank_candidates: RiseFlankCandidateResult | None = None


@dataclass(frozen=True)
class PipelineResult:
    scenarios: dict[str, ScenarioResult]
    config: AnalysisConfig


def _metric_matrix(result: GraphResult, binary: bool) -> np.ndarray:
    return result.adjacency if binary else result.retained_scores


def _scenario_parameter(
    parameter: float | np.ndarray | None,
    node_indices: np.ndarray,
    n_original: int,
) -> float | np.ndarray | None:
    if parameter is None:
        return None
    values = np.asarray(parameter)
    if values.ndim == 0:
        return float(values)
    if values.shape != (n_original,):
        raise ValueError("ROI-specific parameters must match the original ROI count")
    return values[node_indices]


def _safe_w_ic(
    matrix: np.ndarray, sides: np.ndarray, binary: bool
) -> BilateralMetric | None:
    try:
        return w_ic(matrix, sides, binary)
    except ValueError:
        return None


def _safe_w_rc(
    matrix: np.ndarray, sides: np.ndarray, positions: np.ndarray, binary: bool
) -> BilateralMetric | None:
    try:
        return w_rc(matrix, sides, positions, binary)
    except ValueError:
        return None


def run_pipeline(
    traces: np.ndarray,
    sides: Iterable[str],
    positions: Iterable[float] | None = None,
    config: AnalysisConfig | None = None,
) -> PipelineResult:
    """Run all representations and A-D preprocessing scenarios.

    Side and position labels refer to the original ROI ordering and are
    subsetted after declared poor-quality-neuron removal.
    """

    settings = AnalysisConfig() if config is None else config
    if (
        settings.segment_ids_by_representation is not None
        and settings.event_mode != "physical"
    ):
        raise NotImplementedError(
            "segment-aware GC is only available with event_mode='physical'"
        )
    side_labels = np.asarray(list(sides))
    location = None if positions is None else np.asarray(list(positions), dtype=float)
    scenarios = build_scenarios(
        traces,
        settings.bad_neurons,
        settings.artifact_frames,
        settings.smoothing_window,
    )
    if side_labels.shape != (scenarios["A"].traces.shape[0],):
        raise ValueError("sides must contain one label per original ROI")
    if location is not None and location.shape != side_labels.shape:
        raise ValueError("positions must contain one value per original ROI")
    scenario_results: dict[str, ScenarioResult] = {}
    for name, scenario in scenarios.items():
        tolerance = _scenario_parameter(
            settings.tolerance, scenario.node_indices, side_labels.size
        )
        gamma = _scenario_parameter(
            settings.gamma, scenario.node_indices, side_labels.size
        )
        representations = build_representations(
            scenario.traces, tolerance if tolerance is not None else 0.0, gamma
        )
        cgc_estimator = CausalisedGC(
            settings.max_lag,
            settings.n_surrogates,
            settings.alpha,
            settings.random_state,
            settings.fdr,
            settings.score_threshold,
            event_mode=settings.event_mode,
            method=settings.method,
            beta=settings.beta,
            tau=settings.tau,
            n_pasts=settings.n_pasts,
            min_rise_run_samples=settings.min_rise_run_samples,
            rise_candidate_filter=settings.rise_candidate_filter,
            rise_match_min_lag=settings.rise_match_min_lag,
            rise_match_max_lag=settings.rise_match_max_lag,
            rise_match_min_overlap_samples=settings.rise_match_min_overlap_samples,
            rise_match_min_overlap_fraction=(
                settings.rise_match_min_overlap_fraction
            ),
            rise_run_context_samples=settings.rise_run_context_samples,
        )
        represented = representations.as_dict()
        cgc: dict[str, GraphResult] = {}
        rise_flank_candidates: RiseFlankCandidateResult | None = None
        for label, values in represented.items():
            segment_ids = None
            if settings.segment_ids_by_representation is not None:
                segment_ids = settings.segment_ids_by_representation.get(label)
                if segment_ids is not None:
                    segment_ids = np.asarray(segment_ids, dtype=int)
                    if segment_ids.shape != (scenario.traces.shape[1],):
                        raise ValueError(
                            "segment IDs must contain one value per timepoint"
                        )
            if label == "rise":
                if settings.min_rise_run_samples > 1 or settings.rise_candidate_filter:
                    rise_flank_candidates = cgc_estimator.rise_flank_candidates(values)
                    cgc[label] = cgc_estimator.fit(
                        scenario.traces,
                        segment_ids=segment_ids,
                        event_indices=rise_flank_candidates.event_indices,
                        candidate_adjacency=(
                            rise_flank_candidates.candidate_adjacency
                            if settings.rise_candidate_filter
                            else None
                        ),
                    )
                else:
                    cgc[label] = cgc_estimator.fit(
                        scenario.traces,
                        segment_ids=segment_ids,
                        event_indices=selected_frame_indices(values),
                    )
            elif label in {"fall", "fall_residual"}:
                cgc[label] = cgc_estimator.fit(
                    scenario.traces,
                    segment_ids=segment_ids,
                    event_indices=selected_frame_indices(values),
                )
            else:
                cgc[label] = cgc_estimator.fit(values, segment_ids=segment_ids)
        selected_sides = side_labels[scenario.node_indices]
        lateral: dict[str, BilateralMetric | None] = {}
        directional: dict[str, BilateralMetric | None] = {}
        for label, graph in cgc.items():
            lateral[label] = _safe_w_ic(
                _metric_matrix(graph, settings.metric_binary),
                selected_sides,
                settings.metric_binary,
            )
            if location is not None:
                directional[label] = _safe_w_rc(
                    _metric_matrix(graph, settings.metric_binary),
                    selected_sides,
                    location[scenario.node_indices],
                    settings.metric_binary,
                )
        paired_delta: float | None
        try:
            paired_delta = delta_w_ic(
                _metric_matrix(cgc["rise"], settings.metric_binary),
                _metric_matrix(cgc["fall"], settings.metric_binary),
                selected_sides,
                settings.metric_binary,
            )
        except ValueError:
            paired_delta = None
        scenario_results[name] = ScenarioResult(
            scenario=scenario,
            representations=representations,
            cgc=cgc,
            w_ic=lateral,
            w_rc=directional,
            delta_w_ic=paired_delta,
            rise_flank_candidates=rise_flank_candidates,
        )
    return PipelineResult(scenario_results, settings)
