"""Run matched rectification and cross-representation reviewer audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
for import_root in (PACKAGE_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from calcium_transient_rising_flank import (  # noqa: E402
    PCMCIPlusAdapter,
    VARGrangerAdapter,
    array_input_digest,
    benjamini_hochberg,
    build_representations,
    edge_recovery,
    graph_summary,
    signed_ar1_innovation,
    signed_difference,
    static_input_digest,
)
from calcium_transient_rising_flank.checkpointing import (  # noqa: E402
    JsonUnitCheckpointStore,
    atomic_write_json,
    format_progress,
)

from examples.run_fast_causal_baselines import (  # noqa: E402
    _fit,
    dynamic_extension_datasets,
    load_case_records,
)


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger")
DATASETS = ("simulations", "motorneurons")
SIGNED_REPRESENTATIONS = (
    "full",
    "deconvolved_rectified",
    "rise",
    "fall",
    "signed_difference",
    "signed_innovation",
)
SIGNED_CONDITIONS = (
    "all_excitatory_control",
    "mixed_excitation_inhibition",
    "post_inhibitory_rebound",
)
DEFAULT_CASE_DATA_DIR = Path("data/motoneurons")


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _masked_recall(mask: np.ndarray, adjacency: np.ndarray) -> float | None:
    truth = np.asarray(mask, dtype=bool)
    denominator = int(np.count_nonzero(truth))
    if denominator == 0:
        return None
    return float(
        np.count_nonzero(truth & np.asarray(adjacency, dtype=bool)) / denominator
    )


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def _stable_seed(base_seed: int, unit: str) -> int:
    digest = hashlib.sha256(unit.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big") % 1_000_000


def _records_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    """Fingerprint the selected empirical inputs for safe checkpoint reuse."""

    manifest = [
        {
            "case": record["fluo_type"],
            "recording": record["recording"],
            "input_digest": record["input_digest"],
        }
        for record in records
    ]
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def signed_effect_dataset(
    condition: str,
    *,
    seed: int,
    n_steps: int,
) -> dict[str, Any]:
    """Generate calcium traces with declared signed and delayed edge effects.

    The rebound condition is deliberately biphasic: the same directed links
    have a negative lag-one effect and a positive lag-two effect. This tests
    information loss from nonnegative innovation rectification without claiming
    that this linear stress model is a complete biophysical rebound model.
    """

    if condition not in SIGNED_CONDITIONS:
        raise ValueError(f"unsupported signed-effects condition: {condition}")
    n_nodes = 5
    gamma = 0.88
    lag1: np.ndarray = np.zeros((n_nodes, n_nodes), dtype=float)
    lag2: np.ndarray = np.zeros_like(lag1)
    lag1[0, 1] = 0.45
    lag1[1, 2] = 0.40
    lag1[2, 3] = 0.45
    lag1[3, 4] = 0.40
    if condition != "all_excitatory_control":
        lag1[1, 2] = -0.55
        lag1[3, 4] = -0.50
    if condition == "post_inhibitory_rebound":
        lag2[1, 2] = 0.65
        lag2[3, 4] = 0.60

    rng = np.random.default_rng(seed)
    innovations: np.ndarray = np.zeros((n_nodes, n_steps), dtype=float)
    calcium: np.ndarray = np.zeros_like(innovations)
    shocks: np.ndarray = rng.normal(scale=0.35, size=innovations.shape)
    for time in range(n_steps):
        current = shocks[:, time].copy()
        if time >= 1:
            current += innovations[:, time - 1] @ lag1
        if time >= 2:
            current += innovations[:, time - 2] @ lag2
        innovations[:, time] = current
        previous = 0.0 if time == 0 else gamma * calcium[:, time - 1]
        calcium[:, time] = previous + current
    traces = calcium + rng.normal(scale=0.02, size=calcium.shape)
    truth = (lag1 != 0.0) | (lag2 != 0.0)
    return {
        "condition": condition,
        "traces": traces,
        "truth": truth,
        "excitatory_mask": lag1 > 0.0,
        "inhibitory_mask": lag1 < 0.0,
        "rebound_mask": (lag1 < 0.0) & (lag2 > 0.0),
        "lag1_weights": lag1,
        "lag2_weights": lag2,
        "gamma": gamma,
        "max_lag": 2 if np.any(lag2) else 1,
        "trace_digest": static_input_digest(truth, traces),
    }


def _signed_representations(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    traces = np.asarray(dataset["traces"], dtype=float)
    gamma = float(dataset["gamma"])
    bundle = build_representations(traces, gamma=gamma)
    values = {
        "full": bundle.full,
        "deconvolved_rectified": bundle.deconvolved,
        "rise": bundle.rise,
        "fall": bundle.fall,
        "signed_difference": signed_difference(traces),
        "signed_innovation": signed_ar1_innovation(traces, gamma=gamma),
    }
    if not np.allclose(
        values["deconvolved_rectified"],
        np.maximum(values["signed_innovation"], 0.0),
    ):
        raise RuntimeError("rectified and signed AR(1) innovations are not paired")
    return values


def _stable_empirical_representations(traces: np.ndarray) -> dict[str, np.ndarray]:
    """Build bitwise-reproducible empirical inputs across independent methods."""

    values = np.asarray(traces, dtype=float)
    estimated = build_representations(values)
    gamma = np.round(estimated.gamma, decimals=12)
    bundle = build_representations(values, gamma=gamma)
    represented = bundle.as_dict()
    represented["signed_difference"] = signed_difference(values)
    represented["signed_innovation"] = signed_ar1_innovation(values, gamma=gamma)
    return represented


def _fit_graph(
    algorithm: str,
    values: np.ndarray,
    *,
    max_lag: int,
    alpha: float,
    n_surrogates: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    return _fit(
        algorithm,
        values,
        max_lag=max_lag,
        alpha=alpha,
        n_surrogates=n_surrogates,
        random_state=random_state,
        n_pasts=max_lag,
        tau=None,
    )


def _lag_specific_adjacency(
    algorithm: str,
    arrays: dict[str, np.ndarray],
    *,
    lag: int,
    alpha: float,
) -> np.ndarray:
    adjacency = np.asarray(arrays["adjacency"], dtype=bool)
    if algorithm == "pcmciplus":
        graph = np.asarray(arrays["graph"])
        p_values = np.asarray(arrays["p_matrix"], dtype=float)
        if lag >= graph.shape[2]:
            return np.zeros_like(adjacency)
        selected = (graph[:, :, lag] == "-->") & (p_values[:, :, lag] <= alpha)
    else:
        best_lags = np.asarray(arrays["best_lags"], dtype=int)
        selected = adjacency & (best_lags == lag)
    selected = np.asarray(selected, dtype=bool)
    np.fill_diagonal(selected, False)
    return selected


def _fit_cross_graph(
    algorithm: str,
    sources: np.ndarray,
    outcomes: np.ndarray,
    *,
    alpha: float,
    n_surrogates: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if sources.shape != outcomes.shape:
        raise ValueError(
            "cross-representation sources and outcomes must have equal shape"
        )
    n_nodes = sources.shape[0]
    if algorithm in {"cgc", "cgc-star"}:
        from calcium_transient_rising_flank import CausalisedGC

        result = CausalisedGC(
            max_lag=1,
            tau=1,
            n_pasts=1,
            n_surrogates=n_surrogates,
            alpha=alpha,
            beta=alpha,
            fdr=True,
            score_threshold=0.1,
            method=algorithm,
            event_mode="physical",
            simulation=False,
            random_state=random_state,
        ).fit(sources, outcomes=outcomes)
        adjacency = np.asarray(result.adjacency, dtype=bool).copy()
        np.fill_diagonal(adjacency, False)
        return (
            adjacency,
            {
                "cross_fit": "native_source_outcome_fit",
                "projection": "lag1_source_to_target",
            },
            {
                "adjacency": adjacency,
                "scores": result.scores,
                "p_values": result.p_values,
                "best_lags": result.best_lags,
            },
        )

    combined = np.concatenate((sources, outcomes), axis=0)
    eligible: np.ndarray = ~np.eye(n_nodes, dtype=bool)
    if algorithm == "pcmciplus":
        result = PCMCIPlusAdapter(
            tau_max=1,
            run_kwargs={
                "tau_min": 1,
                "pc_alpha": alpha,
                "fdr_method": "none",
            },
        ).fit(combined)
        cross_p_values = np.asarray(
            result.p_matrix[:n_nodes, n_nodes : 2 * n_nodes, 1], dtype=float
        )
        selected = benjamini_hochberg(cross_p_values, alpha, eligible_mask=eligible)
        cross_marks = result.graph[:n_nodes, n_nodes : 2 * n_nodes, 1] == "-->"
        adjacency = np.asarray(selected & cross_marks, dtype=bool)
        arrays = {
            "combined_graph": result.graph,
            "combined_p_matrix": result.p_matrix,
            "combined_val_matrix": result.val_matrix,
            "cross_p_values": cross_p_values,
        }
        method_metadata = {
            "conditional_independence_test": result.cond_ind_test,
            "multiple_testing": "BH over off-diagonal source-to-target hypotheses",
        }
    elif algorithm == "var-granger":
        result = VARGrangerAdapter(max_lag=1, alpha=alpha, fdr=False).fit(combined)
        cross_p_values = np.asarray(
            result.p_values[:n_nodes, n_nodes : 2 * n_nodes], dtype=float
        )
        adjacency = benjamini_hochberg(cross_p_values, alpha, eligible_mask=eligible)
        arrays = {
            "combined_adjacency_uncorrected": result.adjacency,
            "combined_p_values": result.p_values,
            "combined_scores": result.scores,
            "combined_best_lags": result.best_lags,
            "combined_coefficients": result.coefficients,
            "cross_p_values": cross_p_values,
        }
        method_metadata = {
            "conditional_independence_test": "nested joint-VAR F-test",
            "multiple_testing": "BH over off-diagonal source-to-target hypotheses",
        }
    else:
        raise ValueError(f"unsupported cross-representation algorithm: {algorithm}")
    np.fill_diagonal(adjacency, False)
    cross_arrays = {"adjacency": adjacency, **arrays}
    return (
        adjacency,
        {
            **method_metadata,
            "cross_fit": "joint_2N_system_cross_block_bh_family",
            "projection": "lag1_source_block_to_target_block",
        },
        cross_arrays,
    )


def _signed_rows(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = args.n_seeds * len(args.signed_conditions) * len(SIGNED_REPRESENTATIONS)
    completed = 0
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for condition in args.signed_conditions:
            dataset = signed_effect_dataset(condition, seed=seed, n_steps=args.n_steps)
            represented = _signed_representations(dataset)
            for representation in SIGNED_REPRESENTATIONS:
                unit = f"signed|{condition}|{seed}|{args.algorithm}|{representation}"
                cached = store.load_rows(unit) if args.resume else None
                if cached is not None:
                    rows.extend(cached)
                    completed += 1
                    continue
                values = np.asarray(represented[representation], dtype=float).copy()
                representation_digest = array_input_digest(values)
                adjacency, metadata, arrays = _fit_graph(
                    args.algorithm,
                    values.copy(),
                    max_lag=int(dataset["max_lag"]),
                    alpha=args.alpha,
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                )
                truth = np.asarray(dataset["truth"], dtype=bool)
                recovery = edge_recovery(truth, adjacency)
                lag1_adjacency = _lag_specific_adjacency(
                    args.algorithm, arrays, lag=1, alpha=args.alpha
                )
                lag2_adjacency = _lag_specific_adjacency(
                    args.algorithm, arrays, lag=2, alpha=args.alpha
                )
                artifact = (
                    args.output_dir / "artifacts" / f"{store.digest}__{completed}.npz"
                )
                _atomic_npz(
                    artifact,
                    **arrays,
                    truth=truth,
                    excitatory_mask=np.asarray(dataset["excitatory_mask"], dtype=bool),
                    inhibitory_mask=np.asarray(dataset["inhibitory_mask"], dtype=bool),
                    rebound_mask=np.asarray(dataset["rebound_mask"], dtype=bool),
                    lag1_weights=np.asarray(dataset["lag1_weights"], dtype=float),
                    lag2_weights=np.asarray(dataset["lag2_weights"], dtype=float),
                    input_values=values,
                )
                row = {
                    "dataset": "simulations",
                    "analysis_family": "signed_information",
                    "algorithm": args.algorithm,
                    "condition": condition,
                    "seed": seed,
                    "representation": representation,
                    "n_timepoints": values.shape[1],
                    "n_rois": values.shape[0],
                    "max_lag": int(dataset["max_lag"]),
                    "alpha": args.alpha,
                    "n_surrogates": (
                        args.n_surrogates
                        if args.algorithm in {"cgc", "cgc-star"}
                        else None
                    ),
                    "trace_digest": dataset["trace_digest"],
                    "representation_digest": representation_digest,
                    "negative_sample_fraction": float(np.mean(values < 0.0)),
                    "precision": recovery.precision,
                    "recall": recovery.recall,
                    "false_positive_rate": recovery.false_positive_rate,
                    "f1": _f1(recovery.precision, recovery.recall),
                    "excitatory_recall": _masked_recall(
                        dataset["excitatory_mask"], adjacency
                    ),
                    "inhibitory_recall": _masked_recall(
                        dataset["inhibitory_mask"], adjacency
                    ),
                    "inhibitory_lag1_recall": _masked_recall(
                        dataset["inhibitory_mask"], lag1_adjacency
                    ),
                    "rebound_recall": _masked_recall(
                        dataset["rebound_mask"], adjacency
                    ),
                    "rebound_lag2_recall": _masked_recall(
                        dataset["rebound_mask"], lag2_adjacency
                    ),
                    "retained_edges": int(np.count_nonzero(adjacency)),
                    "edge_sign_scored": False,
                    "sign_scope": (
                        "absolute dependence does not identify edge sign"
                        if args.algorithm in {"cgc", "cgc-star"}
                        else "native signed strengths retained; common analysis scores support and lag"
                    ),
                    "artifact_path": str(artifact),
                    **metadata,
                }
                store.save_rows(unit, [row])
                rows.append(row)
                completed += 1
                print(
                    format_progress(completed, total, label="Signed-information fits"),
                    flush=True,
                )
    return rows


def _cross_simulation_rows(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = args.n_seeds * 2
    completed = 0
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        available = {
            item["condition"]: item
            for item in dynamic_extension_datasets(seed, args.n_steps)
        }
        dataset = available["dynamic_a_noncausal_fall"]
        bundle = build_representations(
            np.asarray(dataset["traces"], dtype=float), gamma=float(dataset["gamma"])
        )
        represented = bundle.as_dict()
        trace_digest = dataset["input_digest"]
        truth = np.asarray(dataset["truth_targets"]["rise"], dtype=bool)
        for role in (
            "rise_graph_context",
            "fall_residual_to_rise_falsification",
        ):
            unit = f"cross|dynamic_a_noncausal_fall|{seed}|{args.algorithm}|{role}"
            cached = store.load_rows(unit) if args.resume else None
            if cached is not None:
                rows.extend(cached)
                completed += 1
                continue
            if role == "rise_graph_context":
                source_name = "rise"
                source_values = np.asarray(represented[source_name], dtype=float).copy()
                target_values = np.asarray(represented["rise"], dtype=float).copy()
                adjacency, metadata, arrays = _fit_graph(
                    args.algorithm,
                    source_values.copy(),
                    max_lag=1,
                    alpha=args.alpha,
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                )
                recovery = edge_recovery(truth, adjacency)
                precision: float | None = recovery.precision
                recall: float | None = recovery.recall
                f1: float | None = _f1(recovery.precision, recovery.recall)
                null_fpr: float | None = None
            else:
                source_name = "fall_residual"
                source_values = np.asarray(represented[source_name], dtype=float).copy()
                target_values = np.asarray(represented["rise"], dtype=float).copy()
                adjacency, metadata, arrays = _fit_cross_graph(
                    args.algorithm,
                    source_values.copy(),
                    target_values.copy(),
                    alpha=args.alpha,
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                )
                precision = recall = f1 = None
                eligible = adjacency.shape[0] * (adjacency.shape[0] - 1)
                null_fpr = float(np.count_nonzero(adjacency) / eligible)
            artifact = (
                args.output_dir
                / "artifacts"
                / f"{store.digest}__cross__{completed}.npz"
            )
            _atomic_npz(
                artifact,
                **{
                    **arrays,
                    "adjacency": adjacency,
                    "rise_truth": truth,
                    "source_values": source_values,
                    "target_values": target_values,
                },
            )
            row = {
                "dataset": "simulations",
                "analysis_family": "cross_representation",
                "algorithm": args.algorithm,
                "condition": "dynamic_a_noncausal_fall",
                "seed": seed,
                "test_role": role,
                "source_representation": source_name,
                "target_representation": "rise",
                "cross_phase_truth": "empty"
                if "falsification" in role
                else "not_applicable_contextual_rise_graph",
                "n_timepoints": represented["rise"].shape[1],
                "n_rois": represented["rise"].shape[0],
                "max_lag": 1,
                "alpha": args.alpha,
                "n_surrogates": (
                    args.n_surrogates if args.algorithm in {"cgc", "cgc-star"} else None
                ),
                "trace_digest": trace_digest,
                "source_digest": array_input_digest(source_values),
                "target_digest": array_input_digest(target_values),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "null_false_positive_rate": null_fpr,
                "retained_edges": int(np.count_nonzero(adjacency)),
                "edge_density": float(
                    np.count_nonzero(adjacency)
                    / (adjacency.shape[0] * (adjacency.shape[0] - 1))
                ),
                "artifact_path": str(artifact),
                **metadata,
            }
            store.save_rows(unit, [row])
            rows.append(row)
            completed += 1
            print(
                format_progress(completed, total, label="Cross-representation fits"),
                flush=True,
            )
    return rows


def _motor_rows(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    records = load_case_records(
        args.case_data_dir,
        cases=args.cases,
        recordings=args.recordings,
    )
    rows: list[dict[str, Any]] = []
    total = len(records) * (len(SIGNED_REPRESENTATIONS) + 2)
    completed = 0
    for record in records:
        represented = _stable_empirical_representations(record["traces"])
        represented["deconvolved_rectified"] = represented.pop("deconvolved")
        fitted: dict[str, np.ndarray] = {}
        for representation in SIGNED_REPRESENTATIONS:
            unit = (
                f"motor-signed|{record['fluo_type']}|{record['recording']}|"
                f"{args.algorithm}|{representation}"
            )
            cached = store.load_rows(unit) if args.resume else None
            if cached is not None:
                rows.extend(cached)
                with np.load(Path(str(cached[0]["artifact_path"]))) as saved:
                    fitted[representation] = np.asarray(saved["adjacency"], dtype=bool)
                completed += 1
                continue
            values = np.asarray(represented[representation], dtype=float).copy()
            representation_digest = array_input_digest(values)
            adjacency, metadata, arrays = _fit_graph(
                args.algorithm,
                values.copy(),
                max_lag=1,
                alpha=args.alpha,
                n_surrogates=args.n_surrogates,
                random_state=_stable_seed(args.seed, unit),
            )
            fitted[representation] = adjacency
            artifact = (
                args.output_dir
                / "artifacts"
                / f"{store.digest}__motor__{completed}.npz"
            )
            _atomic_npz(
                artifact,
                **{**arrays, "adjacency": adjacency, "input_values": values},
            )
            row = {
                "dataset": "motorneurons",
                "analysis_family": "signed_information",
                "algorithm": args.algorithm,
                "condition": "observed",
                "case": record["fluo_type"],
                "recording": record["recording"],
                "fish": record["fish"],
                "trial": record["trial"],
                "representation": representation,
                "n_rois": values.shape[0],
                "n_timepoints": values.shape[1],
                "max_lag": 1,
                "alpha": args.alpha,
                "n_surrogates": (
                    args.n_surrogates if args.algorithm in {"cgc", "cgc-star"} else None
                ),
                "trace_digest": record["input_digest"],
                "representation_digest": representation_digest,
                "negative_sample_fraction": float(np.mean(values < 0.0)),
                "retained_edges": int(np.count_nonzero(adjacency)),
                "artifact_path": str(artifact),
                **graph_summary(adjacency.astype(float), record["mid"], binary=True),
                **metadata,
            }
            store.save_rows(unit, [row])
            rows.append(row)
            completed += 1
            print(
                format_progress(completed, total, label="Motor signed-input fits"),
                flush=True,
            )

        for role in (
            "rise_graph_context",
            "fall_residual_to_rise_falsification",
        ):
            unit = (
                f"motor-cross|{record['fluo_type']}|{record['recording']}|"
                f"{args.algorithm}|{role}"
            )
            cached = store.load_rows(unit) if args.resume else None
            if cached is not None:
                rows.extend(cached)
                completed += 1
                continue
            if role == "rise_graph_context":
                source_name = "rise"
                source_values = np.asarray(represented[source_name], dtype=float).copy()
                target_values = np.asarray(represented["rise"], dtype=float).copy()
                adjacency, metadata, arrays = _fit_graph(
                    args.algorithm,
                    source_values.copy(),
                    max_lag=1,
                    alpha=args.alpha,
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                )
            else:
                source_name = "fall_residual"
                source_values = np.asarray(represented[source_name], dtype=float).copy()
                target_values = np.asarray(represented["rise"], dtype=float).copy()
                adjacency, metadata, arrays = _fit_cross_graph(
                    args.algorithm,
                    source_values.copy(),
                    target_values.copy(),
                    alpha=args.alpha,
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                )
            artifact = (
                args.output_dir
                / "artifacts"
                / f"{store.digest}__motor-cross__{completed}.npz"
            )
            _atomic_npz(
                artifact,
                **{
                    **arrays,
                    "adjacency": adjacency,
                    "source_values": source_values,
                    "target_values": target_values,
                },
            )
            row = {
                "dataset": "motorneurons",
                "analysis_family": "cross_representation",
                "algorithm": args.algorithm,
                "condition": "observed",
                "case": record["fluo_type"],
                "recording": record["recording"],
                "fish": record["fish"],
                "trial": record["trial"],
                "test_role": role,
                "source_representation": source_name,
                "target_representation": "rise",
                "n_rois": adjacency.shape[0],
                "n_timepoints": represented["rise"].shape[1],
                "max_lag": 1,
                "alpha": args.alpha,
                "n_surrogates": (
                    args.n_surrogates if args.algorithm in {"cgc", "cgc-star"} else None
                ),
                "trace_digest": record["input_digest"],
                "source_digest": array_input_digest(source_values),
                "target_digest": array_input_digest(target_values),
                "retained_edges": int(np.count_nonzero(adjacency)),
                "edge_density": float(
                    np.count_nonzero(adjacency)
                    / (adjacency.shape[0] * (adjacency.shape[0] - 1))
                ),
                "artifact_path": str(artifact),
                **metadata,
            }
            store.save_rows(unit, [row])
            rows.append(row)
            completed += 1
            print(
                format_progress(
                    completed, total, label="Motor cross-representation fits"
                ),
                flush=True,
            )

        signed_overlap = _jaccard(
            fitted["deconvolved_rectified"], fitted["signed_innovation"]
        )
        for row in rows:
            if (
                row.get("case") != record["fluo_type"]
                or row.get("recording") != record["recording"]
            ):
                continue
            if row.get("analysis_family") == "signed_information":
                row["signed_vs_rectified_jaccard"] = signed_overlap
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=91021)
    parser.add_argument("--signed-conditions", default=",".join(SIGNED_CONDITIONS))
    parser.add_argument("--case-data-dir", type=Path, default=DEFAULT_CASE_DATA_DIR)
    parser.add_argument("--cases", default="C,D")
    parser.add_argument("--recordings", default="F3T1,F3T2,F5T2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.signed_conditions = _parse_list(args.signed_conditions)
    unknown = sorted(set(args.signed_conditions) - set(SIGNED_CONDITIONS))
    if unknown:
        raise SystemExit(f"unsupported signed conditions: {', '.join(unknown)}")
    args.cases = tuple(item.upper() for item in _parse_list(args.cases))
    args.recordings = _parse_list(args.recordings)
    if args.n_seeds < 1 or args.n_steps < 300 or args.n_surrogates < 1:
        raise SystemExit("seeds/surrogates must be positive and n-steps at least 300")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("alpha must lie in (0, 1)")
    return args


def _checkpoint_config(
    args: argparse.Namespace,
    selected_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "n_steps": args.n_steps,
        "n_surrogates": (
            args.n_surrogates if args.algorithm in {"cgc", "cgc-star"} else None
        ),
        "alpha": args.alpha,
        "random_seed": args.seed,
        "signed_conditions": list(args.signed_conditions),
        "signed_representations": list(SIGNED_REPRESENTATIONS),
        "cross_condition": (
            "dynamic_a_noncausal_fall"
            if args.dataset == "simulations"
            else "observed_motorneuron_recording"
        ),
        "cross_estimand": "source fall residual at t-1 to target rise at t",
        "cases": list(args.cases) if args.dataset == "motorneurons" else [],
        "recordings": list(args.recordings) if args.dataset == "motorneurons" else [],
        "case_data_dir": (
            str(args.case_data_dir.resolve())
            if args.dataset == "motorneurons"
            else None
        ),
        "input_fingerprint": (
            _records_fingerprint(selected_records)
            if args.dataset == "motorneurons"
            else "deterministic_generated_inputs_v2"
        ),
        "runner_fingerprint": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "schema_version": 2,
    }


def main() -> None:
    args = parse_args()
    selected_records = (
        load_case_records(
            args.case_data_dir,
            cases=args.cases,
            recordings=args.recordings,
        )
        if args.dataset == "motorneurons"
        else []
    )
    config = _checkpoint_config(args, selected_records)
    store = JsonUnitCheckpointStore(
        args.output_dir,
        namespace=f"representation_bias_{args.dataset}_{args.algorithm}",
        config=config,
    )
    store.initialize(resume=args.resume)
    if args.dataset == "simulations":
        signed_rows = _signed_rows(args, store)
        cross_rows = _cross_simulation_rows(args, store)
    else:
        all_rows = _motor_rows(args, store)
        signed_rows = [
            row for row in all_rows if row["analysis_family"] == "signed_information"
        ]
        cross_rows = [
            row for row in all_rows if row["analysis_family"] == "cross_representation"
        ]
    _write_csv(args.output_dir / "signed_information_rows.csv", signed_rows)
    _write_csv(args.output_dir / "cross_representation_rows.csv", cross_rows)
    total = len(signed_rows) + len(cross_rows)
    store.finish(completed_units=total, total_units=total)
    summary = {
        "status": "complete",
        "config": config,
        "outputs": ["signed_information_rows.csv", "cross_representation_rows.csv"],
        "n_signed_rows": len(signed_rows),
        "n_cross_rows": len(cross_rows),
        "limitations": [
            "c-GC/c-GC* learn causal structure but their absolute dependence score does not label edge sign.",
            "Lag-specific recall measures support plus selected-lag localization, not formal edge-sign recovery.",
            "The rebound simulation is a linear biphasic stress test, not a complete biophysical rebound model.",
            "Motor-neuron outputs are descriptive sensitivity results because no signed or cross-phase graph truth is available.",
            "PCMCI+ and VAR cross-representation results use the source-to-target block of a joint 2N-variable system.",
            "Cross-method falsification results share inputs and the lag-one cross-block estimand but retain each algorithm's native conditioning system.",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
