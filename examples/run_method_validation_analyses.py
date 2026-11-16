"""Execute the prespecified falsification and sensitivity analyses."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    build_representations,
    edge_recovery,
    graph_summary,
    signed_ar1_innovation,
    signed_difference,
    static_input_digest,
)
from calcium_transient_rising_flank.checkpointing import atomic_write_json


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from examples.empirical_baselines import load_motoneuron_records  # noqa: E402
from examples.run_fast_causal_baselines import episodic_dataset  # noqa: E402
from examples.simulation_baselines import matched_static_dataset  # noqa: E402


COMPONENTS = ("cross", "threshold", "permutation", "depth", "signed")
METHODS = ("cgc", "cgc-star")
DEFAULT_DATA_FILE = Path("data/motoneurons/df_motorneurons_F3T1_F3T2_F5T2.pkl")


def _parse_list(value: str, cast=str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def _mean_pairwise_jaccard(graphs: Sequence[np.ndarray]) -> float:
    pairs = list(itertools.combinations(graphs, 2))
    return 1.0 if not pairs else float(np.mean([_jaccard(a, b) for a, b in pairs]))


def _bh_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values over off-diagonal directed hypotheses."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("p_values must be a square matrix")
    eligible: np.ndarray = ~np.eye(values.shape[0], dtype=bool)
    flattened = values[eligible]
    if np.any(~np.isfinite(flattened)) or np.any((flattened < 0) | (flattened > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    order = np.argsort(flattened)
    ordered = flattened[order]
    scaled = ordered * flattened.size / np.arange(1, flattened.size + 1)
    adjusted_ordered = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_flat = np.empty_like(flattened)
    adjusted_flat[order] = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.ones_like(values)
    adjusted[eligible] = adjusted_flat
    return adjusted


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _first_rank_bh_resolution(n_nodes: int, alpha: float = 0.05) -> dict[str, Any]:
    n_tests = n_nodes * (n_nodes - 1)
    return {
        "n_tests": n_tests,
        "alpha": alpha,
        "first_rank_bh_cutoff": alpha / n_tests,
        "minimum_surrogates_for_first_rank_bh": math.ceil(n_tests / alpha) - 1,
    }


def _fit_cgc(
    values: np.ndarray,
    *,
    method: str,
    seed: int,
    n_surrogates: int,
    n_pasts: int,
    outcomes: np.ndarray | None = None,
) -> Any:
    return CausalisedGC(
        max_lag=1,
        n_pasts=n_pasts,
        n_surrogates=n_surrogates,
        alpha=0.05,
        beta=0.05,
        fdr=True,
        score_threshold=0.1,
        method=method,
        event_mode="physical",
        random_state=seed,
    ).fit(values, outcomes=outcomes)


def _recovery_fields(truth: np.ndarray, adjacency: np.ndarray) -> dict[str, float]:
    recovery = edge_recovery(truth, adjacency)
    return {
        "precision": recovery.precision,
        "recall": recovery.recall,
        "false_positive_rate": recovery.false_positive_rate,
        "orientation_accuracy": recovery.orientation_accuracy,
        "f1": _f1(recovery.precision, recovery.recall),
    }


def cross_representation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.simulation_seeds):
        static = matched_static_dataset(
            run_index=0,
            condition="native",
            seed=seed,
            n_steps=args.simulation_steps,
            n_seeds=max(args.simulation_seeds, 2),
        )
        datasets = [
            {
                **static,
                "simulation_kind": "static",
                "condition": "native",
                "seed": seed,
            },
            {**episodic_dataset(seed, args.simulation_steps), "seed": seed},
        ]
        for dataset in datasets:
            bundle = build_representations(dataset["traces"])
            for method_index, method in enumerate(METHODS):
                for source_name, source in (
                    ("rise", bundle.rise),
                    ("fall_residual", bundle.fall_residual),
                ):
                    outcomes = bundle.rise if source_name == "fall_residual" else None
                    result = _fit_cgc(
                        source,
                        outcomes=outcomes,
                        method=method,
                        seed=seed * 100 + method_index,
                        n_surrogates=args.simulation_surrogates,
                        n_pasts=args.default_depth,
                    )
                    rows.append(
                        {
                            "simulation_kind": dataset["simulation_kind"],
                            "condition": dataset["condition"],
                            "seed": seed,
                            "method": method,
                            "source_representation": source_name,
                            "target_representation": "rise",
                            "cross_representation": outcomes is not None,
                            "test_role": (
                                "fall-residual-to-rise falsification"
                                if outcomes is not None
                                else "rise-to-rise positive control"
                            ),
                            "n_surrogates": args.simulation_surrogates,
                            "n_pasts": args.default_depth,
                            "retained_edges": int(np.count_nonzero(result.adjacency)),
                            "mean_score": float(np.mean(result.scores)),
                            "input_digest": dataset["input_digest"],
                            "truth_interpretation": (
                                "event-propagation graph reference; no declared "
                                "fall-to-rise cross-phase ground truth"
                            ),
                            **_recovery_fields(dataset["truth"], result.adjacency),
                        }
                    )
    return rows


def conditioning_depth_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.simulation_seeds):
        for condition in ("native", "shared_input"):
            dataset = matched_static_dataset(
                run_index=0,
                condition=condition,
                seed=seed,
                n_steps=args.simulation_steps,
                n_seeds=max(args.simulation_seeds, 2),
            )
            represented = build_representations(dataset["traces"]).as_dict()
            for depth in args.depths:
                for representation in ("rise", "fall", "fall_residual"):
                    for method_index, method in enumerate(METHODS):
                        result = _fit_cgc(
                            represented[representation],
                            method=method,
                            seed=seed * 1000 + depth * 10 + method_index,
                            n_surrogates=args.simulation_surrogates,
                            n_pasts=depth,
                        )
                        rows.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "method": method,
                                "representation": representation,
                                "n_pasts": depth,
                                "n_surrogates": args.simulation_surrogates,
                                "retained_edges": int(
                                    np.count_nonzero(result.adjacency)
                                ),
                                "input_digest": dataset["input_digest"],
                                **_recovery_fields(
                                    dataset["truth"], result.adjacency
                                ),
                            }
                        )
    return rows


def signed_representation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.simulation_seeds):
        static = matched_static_dataset(
            run_index=0,
            condition="native",
            seed=seed,
            n_steps=args.simulation_steps,
            n_seeds=max(args.simulation_seeds, 2),
        )
        static_truth = np.asarray(static["truth"], dtype=bool)
        datasets = [
            {
                **static,
                "condition": "static_calcium_all_excitatory",
                "excitatory_mask": static_truth,
                "inhibitory_mask": np.zeros_like(static_truth),
            },
            signed_event_calcium_dataset(seed, args.simulation_steps),
        ]
        for dataset in datasets:
            bundle = build_representations(dataset["traces"])
            represented = {
                "rise": bundle.rise,
                "fall": bundle.fall,
                "deconvolved_rectified": bundle.deconvolved,
                "signed_difference": signed_difference(dataset["traces"]),
                "signed_innovation": signed_ar1_innovation(
                    dataset["traces"], gamma=bundle.gamma
                ),
            }
            for representation, values in represented.items():
                for method_index, method in enumerate(METHODS):
                    result = _fit_cgc(
                        values,
                        method=method,
                        seed=seed * 100 + method_index,
                        n_surrogates=args.simulation_surrogates,
                        n_pasts=args.default_depth,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "condition": dataset["condition"],
                            "method": method,
                            "representation": representation,
                            "n_surrogates": args.simulation_surrogates,
                            "n_pasts": args.default_depth,
                            "negative_sample_fraction": float(np.mean(values < 0.0)),
                            "retained_edges": int(np.count_nonzero(result.adjacency)),
                            "excitatory_recall": _masked_recall(
                                dataset["excitatory_mask"], result.adjacency
                            ),
                            "inhibitory_recall": _masked_recall(
                                dataset["inhibitory_mask"], result.adjacency
                            ),
                            "score_semantics": "absolute conditional correlation",
                            "sign_identifiable": False,
                            "input_digest": dataset["input_digest"],
                            **_recovery_fields(dataset["truth"], result.adjacency),
                        }
                    )
    return rows


def _masked_recall(mask: np.ndarray, adjacency: np.ndarray) -> float:
    truth = np.asarray(mask, dtype=bool)
    count = np.count_nonzero(truth)
    return float("nan") if count == 0 else float(np.count_nonzero(adjacency & truth) / count)


def signed_event_calcium_dataset(seed: int, n_steps: int) -> dict[str, Any]:
    """Simulate calcium from a signed lag-one Bernoulli event network."""

    weights: np.ndarray = np.zeros((5, 5), dtype=float)
    weights[0, 1] = 2.0
    weights[1, 2] = -2.4
    weights[2, 3] = 1.8
    weights[3, 4] = -2.2
    weights[0, 4] = 1.5
    rng = np.random.default_rng(seed)
    events: np.ndarray = np.zeros((5, n_steps), dtype=float)
    events[:, 0] = rng.random(5) < 0.12
    for time in range(1, n_steps):
        logits = -2.0 + events[:, time - 1] @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        events[:, time] = rng.random(5) < probabilities
    calcium = np.zeros_like(events)
    calcium[:, 0] = events[:, 0]
    for time in range(1, n_steps):
        calcium[:, time] = 0.88 * calcium[:, time - 1] + events[:, time]
    traces = calcium + rng.normal(scale=0.04, size=calcium.shape)
    truth = weights != 0.0
    return {
        "condition": "signed_event_calcium",
        "truth": truth,
        "traces": traces,
        "excitatory_mask": weights > 0.0,
        "inhibitory_mask": weights < 0.0,
        "input_digest": static_input_digest(truth, traces),
    }


def _mad_scale(traces: np.ndarray) -> np.ndarray:
    differences = np.diff(np.asarray(traces, dtype=float), axis=1)
    center = np.median(differences, axis=1)
    return 1.4826 * np.median(
        np.abs(differences - center[:, None]), axis=1
    )


def empirical_threshold_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    record = load_motoneuron_records(
        args.data_file,
        fluo_types=(args.fluo_type,),
        recordings=(args.recording,),
    )[0]
    settings: list[tuple[str, str, np.ndarray]] = []
    for value in args.absolute_deltas:
        settings.append(
            ("absolute", f"delta={value:g}", np.full(record["traces"].shape[0], value))
        )
    scale = _mad_scale(record["traces"])
    for multiplier in args.mad_multipliers:
        settings.append(
            ("mad", f"mad={multiplier:g}", multiplier * scale)
        )

    fitted: list[tuple[dict[str, Any], np.ndarray]] = []
    for setting_index, (family, label, threshold) in enumerate(settings):
        bundle = build_representations(record["traces"], tolerance=threshold)
        for representation in ("rise", "fall"):
            values = bundle.as_dict()[representation]
            for method_index, method in enumerate(METHODS):
                result = _fit_cgc(
                    values,
                    method=method,
                    seed=args.permutation_seed_start + setting_index * 10 + method_index,
                    n_surrogates=args.threshold_surrogates,
                    n_pasts=args.default_depth,
                )
                row = {
                    "recording": record["recording"],
                    "fluo_type": record["fluo_type"],
                    "method": method,
                    "representation": representation,
                    "threshold_family": family,
                    "threshold_label": label,
                    "threshold_mean": float(np.mean(threshold)),
                    "threshold_max": float(np.max(threshold)),
                    "selected_sample_fraction": float(np.mean(values > 0.0)),
                    "n_surrogates": args.threshold_surrogates,
                    "n_pasts": args.default_depth,
                    "retained_edges": int(np.count_nonzero(result.adjacency)),
                    "input_digest": record["input_digest"],
                    **graph_summary(
                        result.adjacency.astype(float), record["mid"], binary=True
                    ),
                }
                fitted.append((row, result.adjacency))

    reference: dict[tuple[str, str], np.ndarray] = {}
    previous: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for row, adjacency in fitted:
        key = (str(row["method"]), str(row["representation"]))
        reference.setdefault(key, adjacency)
        row["jaccard_to_delta_zero"] = _jaccard(adjacency, reference[key])
        row["jaccard_to_previous_threshold"] = _jaccard(
            adjacency, previous.get(key, adjacency)
        )
        previous[key] = adjacency
        rows.append(row)
    return rows


def empirical_permutation_rows(
    args: argparse.Namespace,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record = load_motoneuron_records(
        args.data_file,
        fluo_types=(args.fluo_type,),
        recordings=(args.recording,),
    )[0]
    represented = build_representations(record["traces"]).as_dict()
    resolution = _first_rank_bh_resolution(record["traces"].shape[0])
    rows: list[dict[str, Any]] = []
    fitted: dict[
        tuple[str, str, int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for representation in ("rise", "fall"):
        for method in METHODS:
            for n_surrogates in args.permutation_counts:
                for seed_offset in range(args.permutation_seeds):
                    seed = args.permutation_seed_start + seed_offset
                    result = _fit_cgc(
                        represented[representation],
                        method=method,
                        seed=seed,
                        n_surrogates=n_surrogates,
                        n_pasts=args.default_depth,
                    )
                    key = (method, representation, n_surrogates, seed)
                    adjusted_p_values = _bh_adjusted_p_values(result.p_values)
                    fitted[key] = (
                        result.adjacency,
                        result.p_values,
                        adjusted_p_values,
                    )
                    artifact = artifact_dir / (
                        f"{method}__{representation}__B-{n_surrogates}__seed-{seed}.npz"
                    )
                    with artifact.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            adjacency=result.adjacency,
                            p_values=result.p_values,
                            adjusted_p_values=adjusted_p_values,
                            input_digest=np.asarray(record["input_digest"]),
                        )
                    rows.append(
                        {
                            "recording": record["recording"],
                            "fluo_type": record["fluo_type"],
                            "method": method,
                            "representation": representation,
                            "n_surrogates": n_surrogates,
                            "n_pasts": args.default_depth,
                            "seed": seed,
                            "retained_edges": int(
                                np.count_nonzero(result.adjacency)
                            ),
                            "minimum_p": float(
                                np.min(
                                    result.p_values[
                                        ~np.eye(result.p_values.shape[0], dtype=bool)
                                    ]
                                )
                            ),
                            "minimum_adjusted_p": float(
                                np.min(
                                    adjusted_p_values[
                                        ~np.eye(
                                            adjusted_p_values.shape[0], dtype=bool
                                        )
                                    ]
                                )
                            ),
                            "minimum_attainable_add_one_p": 1.0
                            / (n_surrogates + 1.0),
                            **resolution,
                            "artifact_path": str(artifact),
                            "input_digest": record["input_digest"],
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    largest = max(args.permutation_counts)
    for representation in ("rise", "fall"):
        for method in METHODS:
            for n_surrogates in args.permutation_counts:
                keys = [
                    (method, representation, n_surrogates, args.permutation_seed_start + i)
                    for i in range(args.permutation_seeds)
                ]
                graphs = [fitted[key][0] for key in keys]
                p_values = np.stack([fitted[key][1] for key in keys])
                adjusted_p_values = np.stack([fitted[key][2] for key in keys])
                off_diagonal: np.ndarray = ~np.eye(
                    p_values.shape[1], dtype=bool
                )
                reference_jaccard = [
                    _jaccard(
                        fitted[key][0],
                        fitted[(method, representation, largest, key[3])][0],
                    )
                    for key in keys
                ]
                summary_rows.append(
                    {
                        "recording": record["recording"],
                        "fluo_type": record["fluo_type"],
                        "input_digest": record["input_digest"],
                        "method": method,
                        "representation": representation,
                        "n_surrogates": n_surrogates,
                        "n_pasts": args.default_depth,
                        "n_seeds": args.permutation_seeds,
                        **resolution,
                        "within_B_mean_graph_jaccard": _mean_pairwise_jaccard(graphs),
                        "edge_count_mean": float(
                            np.mean([np.count_nonzero(graph) for graph in graphs])
                        ),
                        "edge_count_std": float(
                            np.std([np.count_nonzero(graph) for graph in graphs])
                        ),
                        "mean_offdiagonal_p_value_sd": float(
                            np.mean(np.std(p_values[:, off_diagonal], axis=0))
                        ),
                        "mean_offdiagonal_adjusted_p_value_sd": float(
                            np.mean(
                                np.std(
                                    adjusted_p_values[:, off_diagonal], axis=0
                                )
                            )
                        ),
                        "max_offdiagonal_adjusted_p_value_sd": float(
                            np.max(
                                np.std(
                                    adjusted_p_values[:, off_diagonal], axis=0
                                )
                            )
                        ),
                        "mean_jaccard_to_largest_B_same_seed": float(
                            np.mean(reference_jaccard)
                        ),
                        "interpretation": (
                            "empirical seed/B stabilization estimate; distinct from "
                            "the analytical minimum attainable p-value"
                        ),
                    }
                )
    return rows, summary_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", default=",".join(COMPONENTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/method_validation_analyses"),
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--recording", default="F3T1")
    parser.add_argument("--fluo-type", default="dff")
    parser.add_argument("--simulation-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--simulation-steps", type=int, default=1500)
    parser.add_argument("--simulation-surrogates", type=int, default=0)
    parser.add_argument("--default-depth", type=int, default=2)
    parser.add_argument("--depths", default="1,2,3,5")
    parser.add_argument("--absolute-deltas", default="0,0.01,0.02,0.05")
    parser.add_argument("--mad-multipliers", default="0.5,1,2,3")
    parser.add_argument("--threshold-surrogates", type=int, default=5000)
    parser.add_argument("--permutation-counts", default="500,1000,2500,5000")
    parser.add_argument("--permutation-seeds", type=int, default=3)
    parser.add_argument("--permutation-seed-start", type=int, default=101)
    args = parser.parse_args()
    args.components = _parse_list(args.components)
    invalid = sorted(set(args.components) - set(COMPONENTS))
    if invalid:
        raise SystemExit(f"unsupported components: {', '.join(invalid)}")
    args.depths = _parse_list(args.depths, int)
    args.absolute_deltas = _parse_list(args.absolute_deltas, float)
    args.mad_multipliers = _parse_list(args.mad_multipliers, float)
    args.permutation_counts = _parse_list(args.permutation_counts, int)
    if min(args.depths) < 1 or min(args.permutation_counts) < 1:
        raise SystemExit("depths and permutation counts must be positive")
    if 0.0 not in args.absolute_deltas:
        raise SystemExit("--absolute-deltas must include 0 for the reference graph")
    if min(args.absolute_deltas) < 0.0 or min(args.mad_multipliers) < 0.0:
        raise SystemExit("thresholds and MAD multipliers cannot be negative")
    if (
        args.simulation_seeds < 1
        or args.simulation_steps < 100
        or args.simulation_surrogates < 0
        or args.default_depth < 1
        or args.threshold_surrogates < 1
        or args.permutation_seeds < 1
    ):
        raise SystemExit("invalid simulation, conditioning, or surrogate count")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, int] = {}
    permutation_resolution: dict[str, Any] | None = None
    if "cross" in args.components:
        rows = cross_representation_rows(args)
        _write_csv(args.output_dir / "cross_representation_rows.csv", rows)
        outputs["cross_representation_rows"] = len(rows)
    if "depth" in args.components:
        rows = conditioning_depth_rows(args)
        _write_csv(args.output_dir / "conditioning_depth_rows.csv", rows)
        outputs["conditioning_depth_rows"] = len(rows)
    if "signed" in args.components:
        rows = signed_representation_rows(args)
        _write_csv(args.output_dir / "signed_representation_rows.csv", rows)
        outputs["signed_representation_rows"] = len(rows)
    if "threshold" in args.components:
        rows = empirical_threshold_rows(args)
        _write_csv(args.output_dir / "empirical_threshold_rows.csv", rows)
        outputs["empirical_threshold_rows"] = len(rows)
    if "permutation" in args.components:
        rows, summary_rows = empirical_permutation_rows(
            args, args.output_dir / "permutation_artifacts"
        )
        _write_csv(args.output_dir / "empirical_permutation_rows.csv", rows)
        _write_csv(
            args.output_dir / "empirical_permutation_stability.csv", summary_rows
        )
        outputs["empirical_permutation_rows"] = len(rows)
        outputs["empirical_permutation_stability"] = len(summary_rows)
        if rows:
            permutation_resolution = {
                key: rows[0][key]
                for key in (
                    "n_tests",
                    "alpha",
                    "first_rank_bh_cutoff",
                    "minimum_surrogates_for_first_rank_bh",
                )
            }
    summary = {
        "status": "complete",
        "components": list(args.components),
        "config": {
            "recording": args.recording,
            "fluo_type": args.fluo_type,
            "simulation_seeds": args.simulation_seeds,
            "simulation_steps": args.simulation_steps,
            "simulation_surrogates": args.simulation_surrogates,
            "default_depth": args.default_depth,
            "depths": list(args.depths),
            "absolute_deltas": list(args.absolute_deltas),
            "mad_multipliers": list(args.mad_multipliers),
            "threshold_surrogates": args.threshold_surrogates,
            "permutation_counts": list(args.permutation_counts),
            "permutation_seeds": args.permutation_seeds,
            "permutation_seed_start": args.permutation_seed_start,
        },
        "outputs": outputs,
        "limitations": [
            "The episodic generator declares no fall-to-next-rise causal graph; "
            "fall-residual-to-rise is therefore a falsification/negative-control test.",
            "Signed inputs retain negative samples, but c-GC/c-GC* still test "
            "absolute dependence and cannot label inhibitory edges.",
            "The signed-event benchmark models lagged excitation and event-rate "
            "suppression, not delayed post-inhibitory rebound.",
            "The shared-input condition is common observation noise rather than a "
            "fully parameterized latent biological driver.",
        ],
        "permutation_correction": (
            "Add-one Monte Carlo correction is applied before BH adjustment; "
            "reported stability includes raw and BH-adjusted p-values."
        ),
        "permutation_resolution": permutation_resolution,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
