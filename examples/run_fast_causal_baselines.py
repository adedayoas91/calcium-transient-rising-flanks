"""Run PCMCI+ or conditional VAR-Granger on matched project datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    DynamicSimulationConfig,
    PCMCIPlusAdapter,
    VARGrangerAdapter,
    build_representations,
    edge_recovery,
    graph_summary,
    project_significant_lagged_directed_graph,
    signed_ar1_innovation,
    signed_difference,
    simulate_calcium_dataset,
    static_input_digest,
)
from calcium_transient_rising_flank.checkpointing import (
    JsonUnitCheckpointStore,
    atomic_write_json,
    format_progress,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from examples.empirical_baselines import load_motoneuron_records  # noqa: E402
from examples.simulation_baselines import matched_static_dataset  # noqa: E402


ALGORITHMS = ("pcmciplus", "var-granger")
DATASETS = ("simulations", "motorneurons")
SIMULATION_KINDS = ("static", "episodic")
REPRESENTATIONS = (
    "full",
    "deconvolved",
    "rise",
    "fall",
    "fall_residual",
    "signed_difference",
    "signed_innovation",
)
DEFAULT_DATA_FILE = Path("data/motoneurons/df_motorneurons_F3T1_F3T2_F5T2.pkl")


def _parse_choices(value: str, choices: Sequence[str], name: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(choices))
    if not selected or invalid:
        detail = "at least one value is required" if not selected else ", ".join(invalid)
        raise argparse.ArgumentTypeError(f"invalid {name}: {detail}")
    return selected


def representation_map(traces: np.ndarray) -> dict[str, np.ndarray]:
    """Return positive-only and signed inputs on one physical time axis."""

    bundle = build_representations(traces)
    values = bundle.as_dict()
    values["signed_difference"] = signed_difference(traces)
    values["signed_innovation"] = signed_ar1_innovation(
        traces, gamma=bundle.gamma
    )
    return values


def _episodic_truth() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    first = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    second = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    return first | second, (first, second)


def episodic_dataset(seed: int, n_steps: int) -> dict[str, Any]:
    truth, sequence = _episodic_truth()
    config = DynamicSimulationConfig(
        n_episodes=3,
        min_rise_length=20,
        max_rise_length=28,
        rise_waveform_length=20,
        fall_to_rise_ratio_min=2.1,
        fall_to_rise_ratio_max=2.8,
        initial_activation_probability=0.7,
        edge_dropout_probability=0.0,
        adjacency_sequence=sequence,
        active_node_sequence=(
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.ones(5, dtype=bool),
        ),
        fall_noise_rate=0.05,
        fall_noise_scale=0.02,
        fall_state_mode="stochastic_independent",
    )
    dataset = simulate_calcium_dataset(
        truth,
        n_steps=n_steps,
        gamma=0.88,
        noise_std=0.04,
        shared_noise_std=0.0,
        spontaneous_rate=0.02,
        transmission_probability=0.85,
        random_state=seed,
        simulator_mode="episodic_dynamic",
        dynamic_config=config,
    )
    return {
        "truth": dataset.union_adjacency,
        "traces": dataset.fluorescence,
        "input_digest": static_input_digest(
            dataset.union_adjacency, dataset.fluorescence
        ),
        "simulation_kind": "episodic",
        "condition": "episodic_dynamic",
    }


def _fit(
    algorithm: str,
    values: np.ndarray,
    *,
    max_lag: int,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if algorithm == "pcmciplus":
        result = PCMCIPlusAdapter(
            tau_max=max_lag,
            run_kwargs={
                "tau_min": 0,
                "pc_alpha": alpha,
                "fdr_method": "fdr_bh",
            },
        ).fit(values)
        native_adjacency = result.lagged_adjacency()
        adjacency = project_significant_lagged_directed_graph(
            result.graph,
            result.p_matrix,
            alpha,
        )
        marks = Counter(str(item) for item in result.graph.ravel() if str(item))
        metadata = {
            "conditional_independence_test": result.cond_ind_test,
            "raw_mark_counts": json.dumps(dict(sorted(marks.items()))),
            "projection": "bh_filtered_directed_lagged_links",
            "native_retained_edges": int(np.count_nonzero(native_adjacency)),
            "multiple_testing": (
                "Tigramite fdr_bh adjusts p_matrix; scored lagged projection "
                "also requires adjusted p <= alpha"
            ),
            "assumptions": " | ".join(result.assumptions),
        }
        arrays = {
            "graph": result.graph,
            "p_matrix": result.p_matrix,
            "val_matrix": result.val_matrix,
            "native_adjacency": native_adjacency,
            "adjacency": adjacency,
        }
        return adjacency, metadata, arrays
    if algorithm == "var-granger":
        result = VARGrangerAdapter(
            max_lag=max_lag,
            alpha=alpha,
            fdr=True,
        ).fit(values)
        metadata = {
            "conditional_independence_test": "nested VAR omnibus F-test",
            "projection": "none",
            "companion_spectral_radius": result.companion_spectral_radius,
            "var_stable": result.stable,
            "assumptions": " | ".join(result.assumptions),
        }
        arrays = {
            "adjacency": result.adjacency,
            "p_values": result.p_values,
            "scores": result.scores,
            "best_lags": result.best_lags,
            "coefficients": result.coefficients,
            "companion_spectral_radius": np.asarray(
                result.companion_spectral_radius
            ),
        }
        return result.adjacency, metadata, arrays
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _simulation_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        if "static" in args.simulation_kinds:
            for condition in args.conditions:
                matched = matched_static_dataset(
                    run_index=0,
                    condition=condition,
                    seed=seed,
                    n_steps=args.n_steps,
                    n_seeds=max(args.n_seeds, 2),
                )
                specs.append(
                    {
                        **matched,
                        "simulation_kind": "static",
                        "condition": condition,
                        "seed": seed,
                    }
                )
        if "episodic" in args.simulation_kinds:
            specs.append({**episodic_dataset(seed, args.n_steps), "seed": seed})
    return specs


def _run_simulations(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    specs = _simulation_specs(args)
    total = len(specs) * len(args.representations)
    rows: list[dict[str, Any]] = []
    completed = 0
    for spec in specs:
        represented = representation_map(np.asarray(spec["traces"], dtype=float))
        for representation in args.representations:
            unit = (
                f"{spec['simulation_kind']}|{spec['condition']}|{spec['seed']}|"
                f"{args.algorithm}|{representation}"
            )
            cached = store.load_rows(unit) if args.resume else None
            if cached is not None:
                rows.extend(cached)
                completed += 1
                continue
            adjacency, method_metadata, arrays = _fit(
                args.algorithm,
                represented[representation],
                max_lag=args.max_lag,
                alpha=args.alpha,
            )
            recovery = edge_recovery(spec["truth"], adjacency)
            artifact = args.output_dir / "artifacts" / f"{store.digest}__{completed}.npz"
            _atomic_npz(
                artifact,
                **arrays,
                truth=np.asarray(spec["truth"], dtype=bool),
                input_digest=np.asarray(spec["input_digest"]),
            )
            row = {
                "dataset": "simulations",
                "simulation_kind": spec["simulation_kind"],
                "condition": spec["condition"],
                "seed": spec["seed"],
                "algorithm": args.algorithm,
                "representation": representation,
                "n_rois": int(np.asarray(spec["traces"]).shape[0]),
                "n_timepoints": int(np.asarray(spec["traces"]).shape[1]),
                "max_lag": args.max_lag,
                "alpha": args.alpha,
                "precision": recovery.precision,
                "recall": recovery.recall,
                "false_positive_rate": recovery.false_positive_rate,
                "orientation_accuracy": recovery.orientation_accuracy,
                "retained_edges": int(np.count_nonzero(adjacency)),
                "input_digest": spec["input_digest"],
                "artifact_path": str(artifact),
                **method_metadata,
            }
            store.save_rows(unit, [row])
            rows.append(row)
            completed += 1
            print(format_progress(completed, total, label="Baseline fits"), flush=True)
    store.finish(completed_units=total, total_units=total)
    return rows


def _run_motorneurons(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    records = load_motoneuron_records(
        args.data_file,
        fluo_types=args.fluo_types,
        recordings=args.recordings,
    )
    total = len(records) * len(args.representations)
    rows: list[dict[str, Any]] = []
    completed = 0
    for record in records:
        represented = representation_map(record["traces"])
        for representation in args.representations:
            unit = (
                f"{record['fluo_type']}|{record['recording']}|{args.algorithm}|"
                f"{representation}"
            )
            cached = store.load_rows(unit) if args.resume else None
            if cached is not None:
                rows.extend(cached)
                completed += 1
                continue
            adjacency, method_metadata, arrays = _fit(
                args.algorithm,
                represented[representation],
                max_lag=args.max_lag,
                alpha=args.alpha,
            )
            artifact = args.output_dir / "artifacts" / f"{store.digest}__{completed}.npz"
            _atomic_npz(
                artifact,
                **arrays,
                input_digest=np.asarray(record["input_digest"]),
            )
            row = {
                "dataset": "motorneurons",
                "fluo_type": record["fluo_type"],
                "recording": record["recording"],
                "fish": record["fish"],
                "trial": record["trial"],
                "algorithm": args.algorithm,
                "representation": representation,
                "n_rois": int(record["traces"].shape[0]),
                "n_timepoints": int(record["traces"].shape[1]),
                "max_lag": args.max_lag,
                "alpha": args.alpha,
                "input_digest": record["input_digest"],
                "artifact_path": str(artifact),
                **graph_summary(adjacency.astype(float), record["mid"], binary=True),
                **method_metadata,
            }
            store.save_rows(unit, [row])
            rows.append(row)
            completed += 1
            print(format_progress(completed, total, label="Baseline fits"), flush=True)
    store.finish(completed_units=total, total_units=total)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--simulation-kinds", default="static,episodic")
    parser.add_argument("--conditions", default="native,shared_input")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--recordings", default="F3T1,F3T2,F5T2")
    parser.add_argument("--fluo-types", default="dff,f_smooth")
    args = parser.parse_args()
    args.representations = _parse_choices(
        args.representations, REPRESENTATIONS, "representation"
    )
    args.simulation_kinds = _parse_choices(
        args.simulation_kinds, SIMULATION_KINDS, "simulation kind"
    )
    args.conditions = tuple(
        item.strip() for item in args.conditions.split(",") if item.strip()
    )
    args.recordings = tuple(
        item.strip() for item in args.recordings.split(",") if item.strip()
    )
    args.fluo_types = tuple(
        item.strip() for item in args.fluo_types.split(",") if item.strip()
    )
    if args.max_lag < 1 or args.n_seeds < 1 or args.n_steps < 100:
        raise SystemExit("lags/seeds must be positive and n-steps at least 100")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must lie in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    config = {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "representations": list(args.representations),
        "max_lag": args.max_lag,
        "alpha": args.alpha,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "n_steps": args.n_steps,
        "simulation_kinds": list(args.simulation_kinds),
        "conditions": list(args.conditions),
        "data_file": str(args.data_file),
        "recordings": list(args.recordings),
        "fluo_types": list(args.fluo_types),
        "schema_version": 2,
    }
    store = JsonUnitCheckpointStore(
        args.output_dir,
        namespace=f"{args.dataset}_{args.algorithm}",
        config=config,
    )
    store.initialize(resume=args.resume)
    rows = (
        _run_simulations(args, store)
        if args.dataset == "simulations"
        else _run_motorneurons(args, store)
    )
    _write_csv(args.output_dir / "baseline_rows.csv", rows)
    summary = {
        "status": "complete",
        "config": config,
        "n_rows": len(rows),
        "input_digest_count": len({row["input_digest"] for row in rows}),
        "interpretation": {
            "pcmciplus": (
                "Causal-sufficiency benchmark; raw contemporaneous marks are "
                "preserved and scoring uses BH-filtered lagged directed links."
            ),
            "var-granger": (
                "Conditional predictive baseline from nested linear VAR F-tests; "
                "it is not an intervention graph, and fit stability is reported."
            ),
            "signed_inputs": (
                "Signed representations retain negative samples, but c-GC-style "
                "absolute dependence does not identify an excitatory/inhibitory sign."
            ),
        },
        "outputs": ["baseline_rows.csv", "artifacts/*.npz"],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
