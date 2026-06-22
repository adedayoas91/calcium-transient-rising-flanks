"""Event-aware directed-structure analysis for calcium transients."""

from .diagnostics import (
    ResidualDiagnostics,
    TransientSummary,
    characterize_transients,
    residual_diagnostics,
)
from .comparison import (
    add_paired_deltas,
    graph_summary,
    positions_from_mid,
    sides_from_mid,
    summarize_adjacency_cache,
    write_summary_csv,
)
from .estimators import CausalGranger, GraphResult, selected_frame_indices
from .metrics import delta_w_ic, edge_recovery, graph_stability, w_ic, w_rc
from .pipeline import AnalysisConfig, PipelineResult, run_pipeline
from .plotting import (
    plot_directed_graph,
    plot_matrix,
    plot_topographic_graph,
    plot_topographic_pair,
)
from .preprocessing import ScenarioData, build_scenarios
from .representations import RepresentationBundle, build_representations
from .robustness import (
    SensitivityRun,
    SyntheticCondition,
    SyntheticGridRun,
    downsample_dataset,
    run_analysis_sensitivity,
    run_synthetic_grid,
    split_calibration_evaluation,
)
from .sensitivity import PartialAncestralGraph, run_latent_confounding_sensitivity
from .validation import (
    EventNullControlResult,
    StabilityResult,
    bootstrap_event_indices,
    cross_recording_surrogate,
    jitter_event_indices,
    permute_phase_event_indices,
    reverse_event_indices,
    run_event_bootstrap_stability,
    run_event_null_controls,
    run_null_controls,
    simulate_calcium_dataset,
    validate_representations,
)

__all__ = [
    "AnalysisConfig",
    "CausalGranger",
    "EventNullControlResult",
    "GraphResult",
    "PartialAncestralGraph",
    "PipelineResult",
    "ResidualDiagnostics",
    "RepresentationBundle",
    "ScenarioData",
    "SensitivityRun",
    "StabilityResult",
    "SyntheticCondition",
    "SyntheticGridRun",
    "TransientSummary",
    "add_paired_deltas",
    "build_representations",
    "build_scenarios",
    "bootstrap_event_indices",
    "characterize_transients",
    "cross_recording_surrogate",
    "delta_w_ic",
    "downsample_dataset",
    "edge_recovery",
    "graph_stability",
    "graph_summary",
    "jitter_event_indices",
    "permute_phase_event_indices",
    "positions_from_mid",
    "plot_directed_graph",
    "plot_matrix",
    "plot_topographic_graph",
    "plot_topographic_pair",
    "residual_diagnostics",
    "reverse_event_indices",
    "run_analysis_sensitivity",
    "run_event_bootstrap_stability",
    "run_event_null_controls",
    "run_latent_confounding_sensitivity",
    "run_pipeline",
    "run_null_controls",
    "run_synthetic_grid",
    "selected_frame_indices",
    "simulate_calcium_dataset",
    "sides_from_mid",
    "split_calibration_evaluation",
    "summarize_adjacency_cache",
    "validate_representations",
    "w_ic",
    "w_rc",
    "write_summary_csv",
]
