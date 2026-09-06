"""Run the full c-GC diagnostic suite for LPCMCI and OASIS baselines.

The static baseline runner owns the large matched input grid.  This companion
stage consumes that completed grid and adds the analyses performed in the
``c-GC.ipynb`` and ``c-GC-star.ipynb`` notebooks: representative transient and
recovery summaries, locked rise/fall comparisons, cyclic-shift and reverse-time
nulls, the falling-flank comparator, bilateral-consistency contrasts, and the
prespecified noise and frame-rate sweeps.

LPCMCI results retain their raw PAG and use an explicitly lossy undirected
lagged-skeleton projection for recovery metrics.  OASIS is not treated as a
causal learner; graph metrics are produced by the same downstream c-GC and
c-GC* estimators used by the OASIS baseline grid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calcium_transient_rising_flank import (
    LPCMCIAdapter,
    OASISDeconvolver,
    build_representations,
    characterize_transients,
    edge_recovery,
    generate_static_validation_network,
    graph_summary,
    plot_directed_graph,
    plot_matrix,
    project_lossy_lagged_pag_skeleton,
    simulate_static_calcium_like,
    static_calcium_kernel,
    static_gamma_from_tau,
)
from calcium_transient_rising_flank.checkpointing import (
    JsonUnitCheckpointStore,
    atomic_write_json,
    format_progress,
)
from calcium_transient_rising_flank.validation import cyclic_shift_surrogate


def _load_grid_module() -> Any:
    path = Path(__file__).with_name("simulation_baselines.py")
    spec = importlib.util.spec_from_file_location("_simulation_baselines_grid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load matched-grid runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


grid = _load_grid_module()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("lpcmci", "oasis")
ANALYSIS_REVISION = "full-cgc-parity-v1"
CHAIN_EDGES = ((0, 1), (1, 2), (3, 4), (4, 5))
CHAIN_MIDDLE = 3
NOISE_LEVELS = (0.03, 0.06, 0.12, 0.24)
DOWNSAMPLE_FACTORS = (1, 2, 3)
SWEEP_SEED_COUNT = 3
NULL_MINIMUM_SHIFT = 3


@dataclass(frozen=True)
class DiagnosticUnit:
    """One independently checkpointed estimator input."""

    phase: str
    representation: str
    comparator: str | None = None
    run: int | None = None
    seed: int | None = None
    parameter: float | int | None = None

    @property
    def unit_id(self) -> str:
        parts = [self.phase, self.representation]
        if self.comparator is not None:
            parts.append(self.comparator)
        if self.run is not None:
            parts.append(f"run-{self.run}")
        if self.seed is not None:
            parts.append(f"seed-{self.seed}")
        if self.parameter is not None:
            parts.append(f"parameter-{self.parameter}")
        return "|".join(parts)

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "representation": self.representation,
            "comparator": self.comparator,
            "run": self.run,
            "seed": self.seed,
            "parameter": self.parameter,
            "unit_id": self.unit_id,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-null", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_baseline_summary(path: Path, baseline: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Run the {baseline} baseline grid first; missing {path}"
        ) from error
    if payload.get("status") != "complete":
        raise ValueError(f"Baseline summary is not complete: {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Baseline summary has no configuration: {path}")
    components = set(config.get("components", ()))
    if baseline not in components:
        raise ValueError(
            f"Baseline summary components {sorted(components)} do not contain {baseline}"
        )
    return payload


def _methods(baseline: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    if baseline == "lpcmci":
        return ("lpcmci",)
    methods = tuple(str(value) for value in config.get("cgc_methods", ()))
    if not methods:
        raise ValueError("OASIS baseline configuration has no downstream c-GC methods")
    return methods


def _primary_representations(baseline: str) -> tuple[str, ...]:
    # LPCMCI is evaluated on the rise representation used by the c-GC H3/H4
    # analysis. OASIS is a preprocessing baseline, so its corresponding primary
    # input is the inferred spike representation.
    return ("rise",) if baseline == "lpcmci" else ("oasis",)


def diagnostic_units(
    baseline: str,
    config: Mapping[str, Any],
    *,
    n_null: int,
) -> tuple[DiagnosticUnit, ...]:
    """Return the complete deterministic H2/H3/H4 unit plan."""

    if n_null < 1:
        raise ValueError("n_null must be positive")
    representations = tuple(str(value) for value in config["representations"])
    required = {"full", "deconvolved", "rise", "fall", "fall_residual"}
    if baseline == "oasis":
        required.add("oasis")
    missing = sorted(required - set(representations))
    if missing:
        raise ValueError(
            "Baseline grid is missing full-analysis representations: "
            + ", ".join(missing)
        )

    units: list[DiagnosticUnit] = [
        DiagnosticUnit(phase="representative", representation=representation)
        for representation in representations
    ]
    comparators = (
        "observed",
        *(f"cyclic_shift_{index}" for index in range(1, n_null + 1)),
        "reverse_time",
        "fall",
    )
    for representation in _primary_representations(baseline):
        units.extend(
            DiagnosticUnit(
                phase="null",
                representation=representation,
                comparator=comparator,
            )
            for comparator in comparators
        )

    n_runs = int(config["n_runs_outer"])
    n_seeds = int(config["n_seeds_per_run"])
    for run in range(n_runs):
        seed_offset = run * (n_seeds + 100)
        seeds = range(seed_offset + 1, seed_offset + min(SWEEP_SEED_COUNT, n_seeds) + 1)
        for seed in seeds:
            for representation in _primary_representations(baseline):
                units.extend(
                    DiagnosticUnit(
                        phase="noise",
                        representation=representation,
                        run=run,
                        seed=seed,
                        parameter=noise,
                    )
                    for noise in NOISE_LEVELS
                )
                units.extend(
                    DiagnosticUnit(
                        phase="frame_rate",
                        representation=representation,
                        run=run,
                        seed=seed,
                        parameter=factor,
                    )
                    for factor in DOWNSAMPLE_FACTORS
                )
    return tuple(units)


def _chain_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    truth = np.zeros((6, 6), dtype=float)
    for source, target in CHAIN_EDGES:
        truth[source, target] = 1.0
    data = simulate_static_calcium_like(
        truth,
        int(config["n_steps"]),
        rise_tau=grid.RISE_TAU,
        decay_tau=grid.DECAY_TAU,
        noise_std=grid.NOISE_STD,
        shared_noise_std=0.0,
        spontaneous_rate=grid.SPONTANEOUS_RATE,
        transmission_probability=grid.TRANSMISSION_PROBABILITY,
        amplitude_jitter=grid.AMP_JITTER,
        smooth_window=grid.SMOOTH_WINDOW,
        propagation_delay=grid.PROPAGATION_DELAY,
        random_state=grid.BASE_SEED,
    )
    return {
        "truth": truth,
        "traces": np.asarray(data.fluorescence, dtype=float),
        "events": np.asarray(data.events, dtype=float),
        "middle": CHAIN_MIDDLE,
        "downsample": 1,
    }


def _sweep_dataset(unit: DiagnosticUnit, config: Mapping[str, Any]) -> dict[str, Any]:
    if unit.run is None or unit.seed is None or unit.parameter is None:
        raise ValueError(f"Incomplete sweep unit: {unit}")
    network_seed = grid.BASE_SEED + unit.run * 10
    n_rois = int(np.random.RandomState(network_seed).uniform(6, 13))
    truth, _, _, middle = generate_static_validation_network(
        n_rois,
        seed=network_seed,
        ipsilateral_fraction=grid.IPSILATERAL_FRACTION,
    )
    noise_std = float(unit.parameter) if unit.phase == "noise" else grid.NOISE_STD
    factor = int(unit.parameter) if unit.phase == "frame_rate" else 1
    data = simulate_static_calcium_like(
        truth,
        int(config["n_steps"]),
        rise_tau=grid.RISE_TAU,
        decay_tau=grid.DECAY_TAU,
        noise_std=noise_std,
        shared_noise_std=0.0,
        spontaneous_rate=grid.SPONTANEOUS_RATE,
        transmission_probability=grid.TRANSMISSION_PROBABILITY,
        amplitude_jitter=grid.AMP_JITTER,
        smooth_window=grid.SMOOTH_WINDOW,
        propagation_delay=grid.PROPAGATION_DELAY,
        random_state=unit.seed,
    )
    traces = np.asarray(data.fluorescence, dtype=float)[:, ::factor]
    events = grid._align_events_to_observation_grid(data.events, factor)
    return {
        "truth": np.asarray(truth, dtype=float),
        "traces": traces,
        "events": events,
        "middle": int(middle),
        "downsample": factor,
    }


def _standard_representations(
    traces: np.ndarray,
    *,
    downsample: int,
) -> dict[str, np.ndarray]:
    bundle = build_representations(
        traces,
        tolerance=grid.REPRESENTATION_TOLERANCE,
        gamma=static_gamma_from_tau(grid.DECAY_TAU) ** downsample,
    )
    return {
        "full": bundle.full,
        "deconvolved": bundle.deconvolved,
        "rise": bundle.rise,
        "fall": bundle.fall,
        "fall_residual": bundle.fall_residual,
    }


def _oasis_representation(traces: np.ndarray) -> np.ndarray:
    return (
        OASISDeconvolver(backend_kwargs={"penalty": 1, "optimize_g": 0})
        .fit(traces)
        .spikes
    )


def _representation(
    name: str,
    traces: np.ndarray,
    *,
    downsample: int,
) -> np.ndarray:
    if name == "oasis":
        return _oasis_representation(traces)
    representations = _standard_representations(traces, downsample=downsample)
    try:
        return representations[name]
    except KeyError as error:
        raise ValueError(f"Unsupported diagnostic representation: {name}") from error


def _unit_input(
    unit: DiagnosticUnit,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    dataset = (
        _sweep_dataset(unit, config)
        if unit.phase in {"noise", "frame_rate"}
        else _chain_dataset(config)
    )
    truth = np.asarray(dataset["truth"], dtype=float)
    traces = np.asarray(dataset["traces"], dtype=float)
    events = np.asarray(dataset["events"], dtype=float)
    middle = int(dataset["middle"])
    downsample = int(dataset["downsample"])
    selected = _representation(unit.representation, traces, downsample=downsample)

    if unit.phase == "null":
        if unit.comparator is None:
            raise ValueError(f"Null unit has no comparator: {unit}")
        if unit.comparator.startswith("cyclic_shift_"):
            index = int(unit.comparator.rsplit("_", maxsplit=1)[1])
            selected = cyclic_shift_surrogate(
                selected,
                random_state=1000 + index - 1,
                minimum_shift=NULL_MINIMUM_SHIFT,
            )
        elif unit.comparator == "reverse_time":
            selected = selected[:, ::-1]
        elif unit.comparator == "fall":
            selected = _standard_representations(traces, downsample=downsample)["fall"]
        elif unit.comparator != "observed":
            raise ValueError(f"Unsupported null comparator: {unit.comparator}")
    return selected, truth, events, middle


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _artifact_path(output_dir: Path, unit_id: str, method: str) -> Path:
    digest = hashlib.sha256(f"{unit_id}|{method}".encode("utf-8")).hexdigest()
    return output_dir / "graph_artifacts" / f"{digest}.npz"


def _skeleton_recovery(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | None]:
    true_skeleton = np.asarray(truth, dtype=bool) | np.asarray(truth, dtype=bool).T
    predicted_skeleton = (
        np.asarray(predicted, dtype=bool) | np.asarray(predicted, dtype=bool).T
    )
    upper = np.triu(np.ones(true_skeleton.shape, dtype=bool), k=1)
    tp = int(np.count_nonzero(true_skeleton & predicted_skeleton & upper))
    fp = int(np.count_nonzero(~true_skeleton & predicted_skeleton & upper))
    fn = int(np.count_nonzero(true_skeleton & ~predicted_skeleton & upper))
    negatives = int(np.count_nonzero(~true_skeleton & upper))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "orientation_accuracy": None,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _fit_lpcmci(
    matrix: np.ndarray,
    truth: np.ndarray,
    middle: int,
    *,
    unit: DiagnosticUnit,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    result = LPCMCIAdapter(
        tau_max=int(config["lpcmci_tau_max"]),
        run_kwargs={"tau_min": 0, "pc_alpha": float(config["lpcmci_pc_alpha"])},
        verbosity=1,
    ).fit(matrix)
    skeleton = result.lossy_lagged_skeleton()
    symmetric_skeleton = (
        np.asarray(skeleton, dtype=bool) | np.asarray(skeleton, dtype=bool).T
    )
    artifact = _artifact_path(output_dir, unit.unit_id, "lpcmci")
    _save_npz(
        artifact,
        adjacency=symmetric_skeleton,
        graph=result.graph,
        p_matrix=result.p_matrix,
        val_matrix=result.val_matrix,
    )
    summary = graph_summary(symmetric_skeleton.astype(float), middle, binary=True)
    return {
        "row_kind": "graph",
        "baseline": "lpcmci",
        "method": "lpcmci",
        "metric_semantics": "lossy_undirected_lagged_pag_skeleton",
        "graph_artifact_path": str(artifact.relative_to(output_dir)),
        "retained_edges": int(np.count_nonzero(np.triu(symmetric_skeleton, k=1))),
        "comparison_w_ic": summary["w_ic"],
        "recovered_skeleton_w_ic": summary["w_ic"],
        "wic_semantics": "binary_lossy_lagged_pag_skeleton",
        **unit.metadata(),
        **_skeleton_recovery(truth, symmetric_skeleton),
    }


def _fit_oasis_downstream(
    matrix: np.ndarray,
    truth: np.ndarray,
    middle: int,
    *,
    unit: DiagnosticUnit,
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _methods("oasis", config):
        adjacency = grid._cgc_graph(
            matrix,
            method=method,
            n_surrogates=int(config["n_cgc_surrogates"]),
            seed=_stable_seed(unit.unit_id, method),
        )
        recovery = edge_recovery(truth, adjacency)
        denominator = recovery.precision + recovery.recall
        artifact = _artifact_path(output_dir, unit.unit_id, method)
        _save_npz(artifact, adjacency=np.asarray(adjacency, dtype=float))
        rows.append(
            {
                "row_kind": "graph",
                "baseline": "oasis",
                "method": method,
                "metric_semantics": "directed_edge_recovery",
                "graph_artifact_path": str(artifact.relative_to(output_dir)),
                "precision": recovery.precision,
                "recall": recovery.recall,
                "f1": (
                    2 * recovery.precision * recovery.recall / denominator
                    if denominator
                    else 0.0
                ),
                "false_positive_rate": recovery.false_positive_rate,
                "orientation_accuracy": recovery.orientation_accuracy,
                "true_positives": recovery.true_positives,
                "false_positives": recovery.false_positives,
                "false_negatives": recovery.false_negatives,
                "retained_edges": int(np.count_nonzero(adjacency)),
                "comparison_w_ic": graph_summary(adjacency, middle)["w_ic"],
                "recovered_w_ic": graph_summary(adjacency, middle)["w_ic"],
                "wic_semantics": "directed_adjacency",
                **unit.metadata(),
            }
        )
    return rows


def _oasis_event_rows(
    matrix: np.ndarray,
    truth_events: np.ndarray,
    *,
    unit: DiagnosticUnit,
) -> list[dict[str, Any]]:
    if unit.representation != "oasis":
        return []
    calibration_frames = max(20, matrix.shape[1] // 4)
    rows: list[dict[str, Any]] = []
    for threshold_rule, thresholds in (
        ("positive", np.zeros(matrix.shape[0], dtype=float)),
        ("held_out_mad3", grid._mad_thresholds(matrix, calibration_frames)),
    ):
        rows.append(
            {
                "row_kind": "event",
                "baseline": "oasis",
                "method": "oasis",
                "metric_semantics": "event_recovery",
                "threshold_rule": threshold_rule,
                "calibration_frames": calibration_frames,
                **unit.metadata(),
                **grid.event_recovery_metrics(
                    truth_events,
                    matrix,
                    thresholds=thresholds,
                    tolerance=2,
                ),
            }
        )
    return rows


def _run_unit(
    unit: DiagnosticUnit,
    *,
    baseline: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    matrix, truth, truth_events, middle = _unit_input(unit, config)
    if baseline == "lpcmci":
        return [
            _fit_lpcmci(
                matrix,
                truth,
                middle,
                unit=unit,
                config=config,
                output_dir=output_dir,
            )
        ]
    return [
        *_fit_oasis_downstream(
            matrix,
            truth,
            middle,
            unit=unit,
            config=config,
            output_dir=output_dir,
        ),
        *_oasis_event_rows(matrix, truth_events, unit=unit),
    ]


def _cached_rows_valid(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    graph_rows = [row for row in rows if row.get("row_kind") == "graph"]
    if not graph_rows:
        return False
    for row in graph_rows:
        relative = row.get("graph_artifact_path")
        if not relative or not (output_dir / str(relative)).is_file():
            return False
    return True


def _resolve_pag_path(raw_path: str, baseline_dir: Path) -> Path:
    declared = Path(raw_path)
    candidates = (
        declared,
        PROJECT_ROOT / declared,
        baseline_dir / "raw_pag" / declared.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve LPCMCI PAG artifact: {raw_path}")


def _lpcmci_grid_w_ic(row: Mapping[str, Any], baseline_dir: Path) -> float | None:
    path = _resolve_pag_path(str(row["raw_pag_path"]), baseline_dir)
    with np.load(path, allow_pickle=False) as payload:
        skeleton = project_lossy_lagged_pag_skeleton(payload["graph"])
    symmetric = np.asarray(skeleton, dtype=bool) | np.asarray(skeleton, dtype=bool).T
    return graph_summary(symmetric.astype(float), int(row["n_rois"] // 2), binary=True)[
        "w_ic"
    ]


def prepare_grid_analysis(
    graph_rows: pd.DataFrame,
    *,
    baseline: str,
    baseline_dir: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate and normalize the completed matched grid for common analysis."""

    frame = graph_rows.copy()
    expected_methods = set(_methods(baseline, config))
    expected_representations = set(str(value) for value in config["representations"])
    expected_rows = (
        int(config["n_runs_outer"])
        * len(config["conditions"])
        * int(config["n_seeds_per_run"])
        * len(expected_representations)
        * len(expected_methods)
    )
    key_columns = ["run", "condition", "seed", "method", "representation"]
    if len(frame) != expected_rows or frame.duplicated(key_columns).any():
        raise ValueError(
            f"Matched grid is incomplete or duplicated: expected {expected_rows} unique rows, "
            f"found {len(frame)} rows"
        )
    if set(frame["method"]) != expected_methods:
        raise ValueError(
            "Matched grid method set does not match its saved configuration"
        )
    if set(frame["representation"]) != expected_representations:
        raise ValueError(
            "Matched grid representation set does not match its saved configuration"
        )

    if baseline == "lpcmci":
        frame["precision"] = frame["skeleton_precision"]
        frame["recall"] = frame["skeleton_recall"]
        true_edges = (
            frame["skeleton_true_positives"] + frame["skeleton_false_negatives"]
        )
        negatives = frame["n_rois"] * (frame["n_rois"] - 1) / 2 - true_edges
        frame["false_positive_rate"] = np.where(
            negatives > 0,
            frame["skeleton_false_positives"] / negatives,
            0.0,
        )
        frame["orientation_accuracy"] = np.nan
        frame["recovered_skeleton_w_ic"] = [
            _lpcmci_grid_w_ic(row, baseline_dir)
            for row in frame.to_dict(orient="records")
        ]
        frame["comparison_w_ic"] = frame["recovered_skeleton_w_ic"]
        frame["wic_semantics"] = "binary_lossy_lagged_pag_skeleton"
        frame["metric_semantics"] = "lossy_undirected_lagged_pag_skeleton"
    else:
        frame["comparison_w_ic"] = frame["recovered_w_ic"]
        frame["wic_semantics"] = "directed_adjacency"
        frame["metric_semantics"] = "directed_edge_recovery"
    denominator = frame["precision"] + frame["recall"]
    frame["f1"] = np.where(
        denominator > 0,
        2 * frame["precision"] * frame["recall"] / denominator,
        0.0,
    )
    return frame


def paired_signflip_pvalue(values: Sequence[float]) -> tuple[float, float]:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(differences))
    if differences.size <= 20:
        assignments = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(differences.size, dtype=np.uint64)) & 1
        signs = np.where(bits == 0, -1.0, 1.0)
    else:
        signs = np.random.default_rng(grid.BASE_SEED).choice(
            (-1.0, 1.0), size=(20_000, differences.size)
        )
    null_means = (signs * differences).mean(axis=1)
    p_value = float(np.mean(np.abs(null_means) >= abs(observed) - 1e-12))
    return observed, p_value


def build_grid_summaries(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = frame[frame["split"] == "evaluation"].copy()
    metric_columns = ["precision", "recall", "false_positive_rate", "comparison_w_ic"]
    locked_summary = (
        locked.groupby(["method", "representation"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    locked_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in locked_summary.columns
    ]

    run_recall = (
        locked.groupby(["method", "condition", "run", "representation"])["recall"]
        .mean()
        .unstack("representation")
    )
    paired_rows: list[dict[str, Any]] = []
    method_semantics = (
        locked[["method", "metric_semantics"]]
        .drop_duplicates("method")
        .set_index("method")["metric_semantics"]
        .to_dict()
    )
    for (method, condition), group in run_recall.groupby(level=["method", "condition"]):
        differences = group["rise"] - group["fall"]
        observed, p_value = paired_signflip_pvalue(differences.to_numpy())
        paired_rows.append(
            {
                "method": method,
                "condition": condition,
                "metric": "recall",
                "metric_semantics": method_semantics[method],
                "mean_rise_minus_fall": observed,
                "n_runs": int(differences.notna().sum()),
                "p_value": p_value,
            }
        )
    paired = pd.DataFrame(paired_rows)

    wic_wide = frame.pivot(
        index=["method", "run", "condition", "seed"],
        columns="representation",
        values="comparison_w_ic",
    ).reset_index()
    truth_wic = frame[
        ["method", "run", "condition", "seed", "true_w_ic"]
    ].drop_duplicates(["method", "run", "condition", "seed"])
    wic_wide = wic_wide.merge(
        truth_wic,
        on=["method", "run", "condition", "seed"],
        validate="one_to_one",
    )
    wic_wide["delta_comparison_w_ic_rise_minus_fall"] = (
        wic_wide["rise"] - wic_wide["fall"]
    )
    skeleton_rows = frame["wic_semantics"].eq("binary_lossy_lagged_pag_skeleton").all()
    delta_column = (
        "delta_skeleton_w_ic_rise_minus_fall"
        if skeleton_rows
        else "delta_w_ic_rise_minus_fall"
    )
    wic_wide[delta_column] = wic_wide["delta_comparison_w_ic_rise_minus_fall"]
    wic_summary = (
        wic_wide.groupby(["method", "condition"])[
            "delta_comparison_w_ic_rise_minus_fall"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return (
        locked,
        locked_summary,
        paired,
        wic_wide.merge(
            wic_summary,
            on=["method", "condition"],
            suffixes=("", "_condition"),
        ),
    )


def _condition_recall_summary(locked: pd.DataFrame) -> pd.DataFrame:
    return (
        locked.groupby(["method", "condition", "representation"])["recall"]
        .mean()
        .unstack("representation")
        .reset_index()
    )


def _wic_condition_summary(wic_rows: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in (
            "rise",
            "fall",
            "delta_comparison_w_ic_rise_minus_fall",
            "delta_skeleton_w_ic_rise_minus_fall",
            "delta_w_ic_rise_minus_fall",
        )
        if column in wic_rows
    ]
    return wic_rows.groupby(["method", "condition"])[value_columns].mean().reset_index()


def _robustness_summary(sweep_rows: pd.DataFrame, phase: str) -> pd.DataFrame:
    phase_rows = sweep_rows[sweep_rows["phase"] == phase]
    return (
        phase_rows.groupby(["method", "representation", "run", "parameter"])[
            ["precision", "recall", "false_positive_rate"]
        ]
        .mean()
        .reset_index()
    )


def _h1_rows(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = _chain_dataset(config)
    traces = np.asarray(dataset["traces"], dtype=float)
    transient = characterize_transients(traces, tolerance=grid.REPRESENTATION_TOLERANCE)
    rows = pd.DataFrame(
        {
            "gamma_hat": transient.gamma,
            "event_density": transient.event_density,
            "rise_count": transient.rise_count,
            "median_rise_dur": transient.median_rise_duration,
            "SNR": transient.signal_to_noise,
        }
    )
    rows.index.name = "ROI"
    rows = rows.reset_index()
    return rows, dataset


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _plot_ground_truth_and_waveform(
    output_dir: Path,
    dataset: Mapping[str, Any],
) -> None:
    truth = np.asarray(dataset["truth"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_matrix(truth, CHAIN_MIDDLE, ax=axes[0])
    axes[0].set_title("Ground-truth adjacency")
    plot_directed_graph(truth, CHAIN_MIDDLE, ax=axes[1])
    axes[1].set_title("Ground-truth directed graph")
    figure.tight_layout()
    figure.savefig(output_dir / "ground_truth.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    kernel = static_calcium_kernel(grid.RISE_TAU, grid.DECAY_TAU)
    rise_length = int(np.argmax(kernel))
    isolated = np.zeros(90)
    isolated[10] = 1.0
    response = np.convolve(isolated, kernel)[:90]
    rising = np.flatnonzero(np.diff(response) > 0) + 1
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(kernel, color="crimson")
    axes[0].axvline(rise_length, linestyle=":", color="0.5")
    axes[0].set_title("Motoneuron-like calcium kernel")
    axes[0].set_xlabel("frame")
    axes[1].plot(response, color="0.3", label="transient")
    axes[1].plot(rising, response[rising], "r.", markersize=7, label="rise frames")
    axes[1].set_title(f"Single transient: rise spans ~{rise_length} frames")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "waveform.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    traces = np.asarray(dataset["traces"], dtype=float)
    representations = _standard_representations(traces, downsample=1)
    rise = representations["rise"]
    fall = representations["fall"]
    stop = min(300, traces.shape[1])
    times = np.arange(stop)
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    rise_frames = np.flatnonzero(rise[0] > 0)
    rise_frames = rise_frames[rise_frames < stop]
    axes[0].plot(times, traces[0, :stop], color="0.35", linewidth=1)
    axes[0].plot(
        rise_frames,
        traces[0, rise_frames],
        "r.",
        markersize=5,
        label="rising-flank frames",
    )
    axes[0].set_ylabel("ROI 0 dF/F")
    axes[0].legend(loc="upper right")
    axes[1].plot(times, rise[0, :stop], color="crimson", linewidth=1, label="rise")
    axes[1].plot(
        times,
        fall[0, :stop],
        color="steelblue",
        linewidth=1,
        alpha=0.7,
        label="fall",
    )
    axes[1].set_ylabel("increment")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="upper right")
    figure.suptitle("H1: kinetic asymmetry and rising-flank representation")
    figure.tight_layout()
    figure.savefig(output_dir / "h1_transient.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_recovery(
    output_dir: Path,
    representative: pd.DataFrame,
    locked: pd.DataFrame,
) -> None:
    metric = representative.pivot_table(
        index="representation",
        columns="method",
        values=["precision", "recall", "f1"],
        aggfunc="mean",
    )
    axes = metric.plot.bar(figsize=(9, 4), ylim=(0, 1.05))
    axes.set_ylabel("recovery metric")
    axes.set_title("H2: representative recovery by representation")
    axes.figure.tight_layout()
    axes.figure.savefig(
        output_dir / "h2_representations.png", dpi=150, bbox_inches="tight"
    )
    plt.close(axes.figure)

    per_run = (
        locked.groupby(["method", "condition", "run", "representation"])["recall"]
        .mean()
        .reset_index()
    )
    means = per_run.pivot_table(
        index="condition", columns=["method", "representation"], values="recall"
    )
    errors = per_run.pivot_table(
        index="condition",
        columns=["method", "representation"],
        values="recall",
        aggfunc="std",
    ).reindex(index=means.index, columns=means.columns)
    axes = means.plot.bar(
        yerr=errors.fillna(0.0), capsize=3, figsize=(13, 5), ylim=(0, 1.05)
    )
    axes.set_ylabel("locked-evaluation recall")
    axes.set_title("H2/H4: matched-condition recovery")
    axes.legend(fontsize=7, ncol=3)
    axes.figure.tight_layout()
    axes.figure.savefig(output_dir / "h4_conditions.png", dpi=150, bbox_inches="tight")
    plt.close(axes.figure)


def _plot_nulls(output_dir: Path, null_rows: pd.DataFrame) -> None:
    labels = null_rows["comparator"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for (method, representation), group in null_rows.groupby(
        ["method", "representation"]
    ):
        indexed = group.set_index("comparator").reindex(labels)
        label = f"{method}/{representation}"
        axes[0].plot(labels, indexed["recall"], marker="o", label=label)
        axes[1].plot(labels, indexed["retained_edges"], marker="o", label=label)
    axes[0].set_title("H3: null/comparator recovery")
    axes[0].set_ylabel("recall")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("H3: retained relations")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "h3_nulls.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_wic(output_dir: Path, wic_rows: pd.DataFrame) -> None:
    methods = wic_rows["method"].drop_duplicates().tolist()
    conditions = wic_rows["condition"].drop_duplicates().tolist()
    grouped = [
        wic_rows.loc[
            (wic_rows["method"] == method) & (wic_rows["condition"] == condition),
            "delta_comparison_w_ic_rise_minus_fall",
        ].dropna()
        for method in methods
        for condition in conditions
    ]
    labels = [
        f"{method}\n{condition}" for method in methods for condition in conditions
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].boxplot(grouped, tick_labels=labels)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("rise - fall W_IC")
    axes[0].set_title("Paired ipsilateral-consistency contrast")
    for method, group in wic_rows.groupby("method"):
        axes[1].scatter(group["fall"], group["rise"], s=18, label=method, alpha=0.7)
    axes[1].plot([0, 1.02], [0, 1.02], "k--", linewidth=1)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("fall W_IC")
    axes[1].set_ylabel("rise W_IC")
    axes[1].set_title("rise vs. fall W_IC")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "wic_delta.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_sweeps(output_dir: Path, sweep_rows: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, phase in zip(axes, ("noise", "frame_rate"), strict=True):
        per_run = _robustness_summary(sweep_rows, phase)
        for (method, representation), group in per_run.groupby(
            ["method", "representation"]
        ):
            for metric, marker in (("recall", "o"), ("precision", "s")):
                means = group.groupby("parameter")[metric].mean()
                stds = group.groupby("parameter")[metric].std().fillna(0.0)
                axis.errorbar(
                    means.index.astype(float),
                    means,
                    yerr=stds,
                    marker=marker,
                    capsize=3,
                    label=f"{method}/{representation}/{metric}",
                )
    axes[0].set_title("H4: robustness to noise")
    axes[0].set_xlabel("noise_std")
    axes[1].set_title("H4: robustness to subsampling")
    axes[1].set_xlabel("downsample factor")
    for axis in axes:
        axis.set_ylabel("recall")
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "h4_sweeps.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _write_progress(
    path: Path,
    *,
    baseline: str,
    status: str,
    completed_units: int,
    total_units: int,
    completed_fits: int,
    total_fits: int,
    active_unit: str | None,
) -> None:
    atomic_write_json(
        path,
        {
            "status": status,
            "baseline": baseline,
            "completed_unit_count": completed_units,
            "expected_unit_count": total_units,
            "completed_fit_count": completed_fits,
            "expected_fit_count": total_fits,
            "active_unit": active_unit,
        },
    )


def main() -> None:
    args = _parse_args()
    baseline_summary = _read_baseline_summary(
        args.baseline_dir / "summary.json", args.baseline
    )
    config = dict(baseline_summary["config"])
    units = diagnostic_units(args.baseline, config, n_null=args.n_null)
    methods = _methods(args.baseline, config)
    total_fits = len(units) * len(methods)
    diagnostic_config = {
        "analysis_revision": ANALYSIS_REVISION,
        "baseline": args.baseline,
        "baseline_config": config,
        "n_null": args.n_null,
        "noise_levels": list(NOISE_LEVELS),
        "downsample_factors": list(DOWNSAMPLE_FACTORS),
        "sweep_seed_count": SWEEP_SEED_COUNT,
        "primary_representations": list(_primary_representations(args.baseline)),
        "methods": list(methods),
    }
    store = JsonUnitCheckpointStore(
        output_dir=args.output_dir,
        namespace="full_analysis",
        config=diagnostic_config,
    )
    store.initialize(resume=args.resume)
    progress_path = args.output_dir / "progress.json"
    all_rows: list[dict[str, Any]] = []
    completed_fits = 0
    _write_progress(
        progress_path,
        baseline=args.baseline,
        status="running",
        completed_units=0,
        total_units=len(units),
        completed_fits=0,
        total_fits=total_fits,
        active_unit=None,
    )

    for index, unit in enumerate(units, start=1):
        _write_progress(
            progress_path,
            baseline=args.baseline,
            status="running",
            completed_units=index - 1,
            total_units=len(units),
            completed_fits=completed_fits,
            total_fits=total_fits,
            active_unit=unit.unit_id,
        )
        rows = store.load_rows(unit.unit_id) if args.resume else None
        if rows is None or not _cached_rows_valid(args.output_dir, rows):
            print(
                f"[{args.baseline}] starting diagnostic unit {index}/{len(units)}: "
                f"{unit.unit_id}",
                flush=True,
            )
            rows = _run_unit(
                unit,
                baseline=args.baseline,
                config=config,
                output_dir=args.output_dir,
            )
            store.save_rows(unit.unit_id, rows)
        else:
            print(
                f"[{args.baseline}] loaded diagnostic unit {index}/{len(units)}: "
                f"{unit.unit_id}",
                flush=True,
            )
        all_rows.extend(rows)
        completed_fits += len([row for row in rows if row.get("row_kind") == "graph"])
        print(
            format_progress(index, len(units), label="Full-analysis units"),
            flush=True,
        )

    if completed_fits != total_fits:
        raise RuntimeError(
            f"Diagnostic graph-fit count mismatch: expected {total_fits}, "
            f"found {completed_fits}"
        )

    graph_diagnostics = pd.DataFrame(
        row for row in all_rows if row.get("row_kind") == "graph"
    )
    event_diagnostics = pd.DataFrame(
        row for row in all_rows if row.get("row_kind") == "event"
    )
    expected_event_rows = (
        2 * sum(unit.representation == "oasis" for unit in units)
        if args.baseline == "oasis"
        else 0
    )
    if len(event_diagnostics) != expected_event_rows:
        raise RuntimeError(
            f"Diagnostic event-row count mismatch: expected {expected_event_rows}, "
            f"found {len(event_diagnostics)}"
        )
    graph_grid = pd.read_csv(args.baseline_dir / "graph_recovery_rows.csv")
    grid_analysis = prepare_grid_analysis(
        graph_grid,
        baseline=args.baseline,
        baseline_dir=args.baseline_dir,
        config=config,
    )
    locked, locked_summary, paired, wic_rows = build_grid_summaries(grid_analysis)
    h1_rows, representative_dataset = _h1_rows(config)
    representative = graph_diagnostics[
        graph_diagnostics["phase"] == "representative"
    ].copy()
    null_rows = graph_diagnostics[graph_diagnostics["phase"] == "null"].copy()
    sweep_rows = graph_diagnostics[
        graph_diagnostics["phase"].isin(("noise", "frame_rate"))
    ].copy()
    locked_recall = _condition_recall_summary(locked)
    wic_summary = _wic_condition_summary(wic_rows)
    robustness_noise = _robustness_summary(sweep_rows, "noise")
    robustness_framerate = _robustness_summary(sweep_rows, "frame_rate")

    outputs: dict[str, pd.DataFrame] = {
        "h1_transient_characterization.csv": h1_rows,
        "representative_recovery.csv": representative,
        "grid_analysis_rows.csv": grid_analysis,
        "locked_recovery_by_representation.csv": locked_summary,
        "locked_recall_by_condition.csv": locked_recall,
        "paired_rise_vs_fall_test.csv": paired,
        "wic_delta_rows.csv": wic_rows,
        "wic_delta_by_condition.csv": wic_summary,
        "null_comparator_rows.csv": null_rows,
        "robustness_rows.csv": sweep_rows,
        "robustness_noise.csv": robustness_noise,
        "robustness_framerate.csv": robustness_framerate,
    }
    if not event_diagnostics.empty:
        outputs["oasis_event_diagnostics.csv"] = event_diagnostics
    for filename, frame in outputs.items():
        _write_csv(args.output_dir / filename, frame)

    _plot_ground_truth_and_waveform(args.output_dir, representative_dataset)
    _plot_recovery(args.output_dir, representative, locked)
    _plot_nulls(args.output_dir, null_rows)
    _plot_wic(args.output_dir, wic_rows)
    _plot_sweeps(args.output_dir, sweep_rows)

    summary = {
        "status": "complete",
        "baseline": args.baseline,
        "analysis_revision": ANALYSIS_REVISION,
        "analysis_contract": (
            "Full H1-H4 parity with c-GC/c-GC*: representative characterization and "
            "recovery, matched locked grid, rise/fall paired tests, cyclic-shift and "
            "reverse-time nulls, falling comparator, W_IC contrast, and noise/frame sweeps."
        ),
        "metric_semantics": (
            "lossy undirected lagged PAG skeleton; raw PAG retained in graph artifacts"
            if args.baseline == "lpcmci"
            else "OASIS event/preprocessing baseline with downstream directed c-GC/c-GC*"
        ),
        "config": diagnostic_config,
        "expected_unit_count": len(units),
        "completed_unit_count": len(units),
        "expected_fit_count": total_fits,
        "completed_fit_count": completed_fits,
        "expected_event_diagnostic_rows": expected_event_rows,
        "n_grid_rows": len(grid_analysis),
        "n_h1_rows": len(h1_rows),
        "n_representative_rows": len(representative),
        "n_null_rows": len(null_rows),
        "n_robustness_rows": len(sweep_rows),
        "n_event_diagnostic_rows": len(event_diagnostics),
        "outputs": [
            *outputs,
            "ground_truth.png",
            "waveform.png",
            "h1_transient.png",
            "h2_representations.png",
            "h3_nulls.png",
            "h4_conditions.png",
            "wic_delta.png",
            "h4_sweeps.png",
            "graph_artifacts/*.npz",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    store.finish(completed_units=len(units), total_units=len(units))
    _write_progress(
        progress_path,
        baseline=args.baseline,
        status="complete",
        completed_units=len(units),
        total_units=len(units),
        completed_fits=completed_fits,
        total_fits=total_fits,
        active_unit=None,
    )
    print(
        format_progress(len(units), len(units), label="Full-analysis units")
        + f" | complete; summary={args.output_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
