"""Run causal learners on matched full-axis project datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    DynamicSimulationConfig,
    LPCMCIAdapter,
    PCMCIPlusAdapter,
    VARGrangerAdapter,
    array_input_digest,
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
from examples.dynamic_extensions import (  # noqa: E402
    dynamic_conditions,
    dynamic_truth_graphs,
)
from examples.run_empirical_null_controls import (  # noqa: E402
    CASE_FILES,
    load_case_traces,
)
from examples.simulation_baselines import matched_static_dataset  # noqa: E402


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger", "lpcmci")
DATASETS = ("simulations", "motorneurons")
SIMULATION_KINDS = ("static", "episodic", "dynamic-extension", "confounding")
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
DEFAULT_CASE_DATA_DIR = Path("data/motoneurons")


def _parse_choices(value: str, choices: Sequence[str], name: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(choices))
    if not selected or invalid:
        detail = (
            "at least one value is required" if not selected else ", ".join(invalid)
        )
        raise argparse.ArgumentTypeError(f"invalid {name}: {detail}")
    return selected


def representation_map(traces: np.ndarray) -> dict[str, np.ndarray]:
    """Return positive-only and signed inputs on one physical time axis."""

    bundle = build_representations(traces)
    values = bundle.as_dict()
    values["signed_difference"] = signed_difference(traces)
    values["signed_innovation"] = signed_ar1_innovation(traces, gamma=bundle.gamma)
    return values


def load_case_records(
    data_dir: Path,
    *,
    cases: Sequence[str],
    recordings: Sequence[str],
) -> list[dict[str, Any]]:
    """Load explicitly named preprocessing cases for a matched comparison."""

    records = load_case_traces(
        data_dir,
        cases=tuple(cases),
        recordings=set(recordings),
    )
    expected = {(case, recording) for case in cases for recording in recordings}
    actual = {(str(row["case"]), str(row["recording"])) for row in records}
    missing = sorted(expected - actual)
    if missing:
        labels = ", ".join(f"{case}/{recording}" for case, recording in missing)
        raise FileNotFoundError(f"requested motoneuron units are missing: {labels}")
    for record in records:
        record["fluo_type"] = str(record["case"])
        record["input_digest"] = array_input_digest(record["traces"])
    return records


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


def dynamic_extension_datasets(seed: int, n_steps: int) -> tuple[dict[str, Any], ...]:
    """Generate the declared dynamic stress conditions on one physical time axis."""

    rise_truth, rise_sequence, fall_sequence = dynamic_truth_graphs()
    datasets: list[dict[str, Any]] = []
    for condition in dynamic_conditions(rise_sequence, fall_sequence):
        dataset = simulate_calcium_dataset(
            rise_truth,
            n_steps=n_steps,
            gamma=condition.gamma,
            noise_std=condition.noise_std,
            shared_noise_std=condition.shared_noise_std,
            spontaneous_rate=condition.spontaneous_rate,
            transmission_probability=condition.transmission_probability,
            random_state=seed,
            simulator_mode=condition.simulator_mode,
            dynamic_config=condition.dynamic_config,
        )
        fall_truth = dataset.fall_union_adjacency
        if fall_truth is None:
            fall_truth = np.zeros_like(dataset.union_adjacency, dtype=bool)
        union_truth = dataset.union_adjacency | fall_truth
        datasets.append(
            {
                "truth": union_truth,
                "truth_targets": {
                    "union": union_truth,
                    "rise": dataset.union_adjacency,
                    "fall": fall_truth,
                },
                "traces": dataset.fluorescence,
                "input_digest": static_input_digest(union_truth, dataset.fluorescence),
                "simulation_kind": "dynamic-extension",
                "condition": condition.name,
                "gamma": condition.gamma,
            }
        )
    return tuple(datasets)


def confounding_datasets(seed: int, n_steps: int) -> tuple[dict[str, Any], ...]:
    """Generate matched observed, shared-noise, and hidden-driver conditions."""

    observed_truth = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    specifications = (
        ("native", observed_truth, 0.0, None),
        ("shared_observation_noise", observed_truth, 0.12, None),
    )
    datasets: list[dict[str, Any]] = []
    for condition, truth, shared_noise_std, latent_targets in specifications:
        dataset = simulate_calcium_dataset(
            truth,
            n_steps=n_steps,
            gamma=0.88,
            noise_std=0.04,
            shared_noise_std=shared_noise_std,
            spontaneous_rate=0.02,
            transmission_probability=0.85,
            random_state=seed,
        )
        datasets.append(
            {
                "truth": observed_truth,
                "traces": dataset.fluorescence,
                "input_digest": static_input_digest(
                    observed_truth, dataset.fluorescence
                ),
                "simulation_kind": "confounding",
                "condition": condition,
                "shared_noise_std": shared_noise_std,
                "latent_driver_targets": latent_targets,
            }
        )

    latent_index = observed_truth.shape[0]
    extended_truth = np.zeros((latent_index + 1, latent_index + 1), dtype=bool)
    extended_truth[:latent_index, :latent_index] = observed_truth
    latent_targets = (1, 3)
    extended_truth[latent_index, list(latent_targets)] = True
    latent_dataset = simulate_calcium_dataset(
        extended_truth,
        n_steps=n_steps,
        gamma=0.88,
        noise_std=0.04,
        shared_noise_std=0.0,
        spontaneous_rate=0.02,
        transmission_probability=0.85,
        random_state=seed,
    )
    observed_traces = latent_dataset.fluorescence[:latent_index]
    datasets.append(
        {
            "truth": observed_truth,
            "traces": observed_traces,
            "input_digest": static_input_digest(observed_truth, observed_traces),
            "simulation_kind": "confounding",
            "condition": "latent_common_driver",
            "shared_noise_std": 0.0,
            "latent_driver_targets": ",".join(str(item) for item in latent_targets),
        }
    )
    return tuple(datasets)


def _fit(
    algorithm: str,
    values: np.ndarray,
    *,
    max_lag: int,
    alpha: float,
    n_surrogates: int = 0,
    random_state: int | None = None,
    n_pasts: int | None = None,
    tau: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if algorithm in {"cgc", "cgc-star"}:
        result = CausalisedGC(
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            alpha=alpha,
            fdr=True,
            event_mode="physical",
            method=algorithm,
            simulation=False,
            random_state=random_state,
            n_pasts=n_pasts,
            tau=tau,
        ).fit(values)
        effective_n_pasts = max_lag if n_pasts is None else n_pasts
        metadata = {
            "conditional_independence_test": (
                "absolute unconditional and residualized lagged correlation"
            ),
            "projection": "bh_filtered_directed_lagged_links",
            "multiple_testing": (
                "finite-sample add-one permutation probabilities with BH across "
                "ordered pairs"
            ),
            "n_surrogates": n_surrogates,
            "n_pasts": effective_n_pasts,
            "tested_lag": tau if tau is not None else "1..max_lag",
            "assumptions": (
                "causal structure relation on the supplied full-axis input under "
                "the c-GC/c-GC* assumptions | "
                "stationarity over the fitted interval | absolute scores do not "
                "identify excitatory or inhibitory sign"
            ),
        }
        arrays = {
            "adjacency": result.adjacency,
            "scores": result.scores,
            "retained_scores": result.retained_scores,
            "p_values": result.p_values,
            "best_lags": result.best_lags,
        }
        return result.adjacency, metadata, arrays
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
    if algorithm == "lpcmci":
        result = LPCMCIAdapter(
            tau_max=max_lag,
            run_kwargs={"tau_min": 0, "pc_alpha": alpha},
        ).fit(values)
        adjacency = result.lossy_lagged_skeleton()
        marks = Counter(str(item) for item in result.graph.ravel() if str(item))
        metadata = {
            "conditional_independence_test": result.cond_ind_test,
            "raw_mark_counts": json.dumps(dict(sorted(marks.items()))),
            "projection": "lossy_lagged_pag_skeleton",
            "multiple_testing": "native LPCMCI PAG selection at declared pc_alpha",
            "assumptions": " | ".join(result.assumptions),
            "estimand": "lagged_skeleton",
        }
        arrays = {
            "graph": result.graph,
            "p_matrix": result.p_matrix,
            "val_matrix": result.val_matrix,
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
            "companion_spectral_radius": np.asarray(result.companion_spectral_radius),
        }
        return result.adjacency, metadata, arrays
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _stable_seed(base_seed: int, unit: str) -> int:
    digest = hashlib.sha256(unit.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big") % 1_000_000


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
        if "dynamic-extension" in args.simulation_kinds:
            available = {
                dataset["condition"]: dataset
                for dataset in dynamic_extension_datasets(seed, args.n_steps)
            }
            unknown = sorted(set(args.conditions) - set(available))
            if unknown:
                raise ValueError(
                    "unsupported dynamic-extension conditions: " + ", ".join(unknown)
                )
            specs.extend({**available[name], "seed": seed} for name in args.conditions)
        if "confounding" in args.simulation_kinds:
            available = {
                dataset["condition"]: dataset
                for dataset in confounding_datasets(seed, args.n_steps)
            }
            unknown = sorted(set(args.conditions) - set(available))
            if unknown:
                raise ValueError(
                    "unsupported confounding conditions: " + ", ".join(unknown)
                )
            specs.extend({**available[name], "seed": seed} for name in args.conditions)
    return specs


class SimulationFitTask(NamedTuple):
    artifact_index: int
    unit: str
    spec: dict[str, Any]
    representation: str
    n_pasts: int | None


SimulationFitResult = tuple[str, list[dict[str, Any]]]


def _run_simulation_fit_task(
    task: SimulationFitTask,
    *,
    algorithm: str,
    max_lag: int,
    alpha: float,
    n_surrogates: int,
    seed: int,
    cgc_tau: int | None,
    output_dir: Path,
    config_digest: str,
) -> SimulationFitResult:
    print(f"[{algorithm}] worker starting baseline fit: {task.unit}", flush=True)
    represented = representation_map(np.asarray(task.spec["traces"], dtype=float))
    values = represented[task.representation]
    adjacency, method_metadata, arrays = _fit(
        algorithm,
        values,
        max_lag=max_lag,
        alpha=alpha,
        n_surrogates=n_surrogates,
        random_state=_stable_seed(seed, task.unit),
        n_pasts=task.n_pasts,
        tau=cgc_tau,
    )
    artifact = (
        output_dir / "artifacts" / f"{config_digest}__{task.artifact_index}.npz"
    )
    _atomic_npz(
        artifact,
        **arrays,
        truth=np.asarray(task.spec["truth"], dtype=bool),
        input_digest=np.asarray(task.spec["input_digest"]),
        input_values=np.asarray(values, dtype=float),
    )
    truth_targets = task.spec.get("truth_targets", {"union": task.spec["truth"]})
    unit_rows: list[dict[str, Any]] = []
    for truth_target, truth in truth_targets.items():
        recovery = edge_recovery(np.asarray(truth, dtype=bool), adjacency)
        truth_skeleton = np.asarray(truth, dtype=bool)
        truth_skeleton = truth_skeleton | truth_skeleton.T
        adjacency_skeleton = np.asarray(adjacency, dtype=bool)
        adjacency_skeleton = adjacency_skeleton | adjacency_skeleton.T
        skeleton_recovery = edge_recovery(truth_skeleton, adjacency_skeleton)
        f1 = (
            0.0
            if recovery.precision + recovery.recall == 0.0
            else 2.0
            * recovery.precision
            * recovery.recall
            / (recovery.precision + recovery.recall)
        )
        skeleton_f1 = (
            0.0
            if skeleton_recovery.precision + skeleton_recovery.recall == 0.0
            else 2.0
            * skeleton_recovery.precision
            * skeleton_recovery.recall
            / (skeleton_recovery.precision + skeleton_recovery.recall)
        )
        directed_metrics: dict[str, float | None] = {
            "precision": recovery.precision,
            "recall": recovery.recall,
            "false_positive_rate": recovery.false_positive_rate,
            "orientation_accuracy": recovery.orientation_accuracy,
            "f1": f1,
        }
        if algorithm == "lpcmci":
            directed_metrics = {
                "precision": None,
                "recall": None,
                "false_positive_rate": None,
                "orientation_accuracy": None,
                "f1": None,
            }
        unit_rows.append(
            {
                "dataset": "simulations",
                "simulation_kind": task.spec["simulation_kind"],
                "condition": task.spec["condition"],
                "seed": task.spec["seed"],
                "algorithm": algorithm,
                "representation": task.representation,
                "truth_target": truth_target,
                "truth_edges": int(np.count_nonzero(truth)),
                "n_rois": int(np.asarray(task.spec["traces"]).shape[0]),
                "n_timepoints": int(np.asarray(task.spec["traces"]).shape[1]),
                "max_lag": max_lag,
                "n_pasts": task.n_pasts,
                "alpha": alpha,
                **directed_metrics,
                "skeleton_precision": skeleton_recovery.precision,
                "skeleton_recall": skeleton_recovery.recall,
                "skeleton_false_positive_rate": (
                    skeleton_recovery.false_positive_rate
                ),
                "skeleton_f1": skeleton_f1,
                "retained_edges": int(np.count_nonzero(adjacency)),
                "input_digest": task.spec["input_digest"],
                "shared_noise_std": task.spec.get("shared_noise_std"),
                "latent_driver_targets": task.spec.get("latent_driver_targets"),
                "artifact_path": str(artifact),
                **method_metadata,
            }
        )
    return task.unit, unit_rows


def _iter_simulation_fit_results(
    worker: Callable[[SimulationFitTask], SimulationFitResult],
    tasks: Sequence[SimulationFitTask],
    *,
    n_jobs: int,
) -> Iterator[SimulationFitResult]:
    if not tasks:
        return
    if n_jobs == 1:
        yield from map(worker, tasks)
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        yield from executor.map(worker, tasks, chunksize=1)


def _run_simulations(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    specs = _simulation_specs(args)
    cgc_depths = args.cgc_depths if args.algorithm in {"cgc", "cgc-star"} else (None,)
    tasks: list[SimulationFitTask] = []
    for spec in specs:
        for representation in args.representations:
            for n_pasts in cgc_depths:
                depth_label = "native" if n_pasts is None else str(n_pasts)
                unit = (
                    f"{spec['simulation_kind']}|{spec['condition']}|{spec['seed']}|"
                    f"{args.algorithm}|{representation}|n_pasts={depth_label}"
                )
                tasks.append(
                    SimulationFitTask(
                        artifact_index=len(tasks),
                        unit=unit,
                        spec=spec,
                        representation=representation,
                        n_pasts=n_pasts,
                    )
                )
    total = len(tasks)
    rows_by_unit: dict[str, list[dict[str, Any]]] = {}
    pending_tasks: list[SimulationFitTask] = []
    for task in tasks:
        cached = store.load_rows(task.unit) if args.resume else None
        if cached is None:
            pending_tasks.append(task)
        else:
            rows_by_unit[task.unit] = cached
    completed = len(rows_by_unit)
    worker = partial(
        _run_simulation_fit_task,
        algorithm=args.algorithm,
        max_lag=args.max_lag,
        alpha=args.alpha,
        n_surrogates=args.n_surrogates,
        seed=args.seed,
        cgc_tau=args.cgc_tau,
        output_dir=args.output_dir,
        config_digest=store.digest,
    )
    for unit, unit_rows in _iter_simulation_fit_results(
        worker,
        pending_tasks,
        n_jobs=args.n_jobs,
    ):
        store.save_rows(unit, unit_rows)
        rows_by_unit[unit] = unit_rows
        completed += 1
        print(
            format_progress(completed, total, label="Baseline fits"),
            flush=True,
        )
    rows = [row for task in tasks for row in rows_by_unit[task.unit]]
    store.finish(completed_units=total, total_units=total)
    return rows


def _run_motorneurons(
    args: argparse.Namespace,
    store: JsonUnitCheckpointStore,
) -> list[dict[str, Any]]:
    records = (
        load_case_records(
            args.case_data_dir,
            cases=args.cases,
            recordings=args.recordings,
        )
        if args.cases
        else load_motoneuron_records(
            args.data_file,
            fluo_types=args.fluo_types,
            recordings=args.recordings,
        )
    )
    cgc_depths = args.cgc_depths if args.algorithm in {"cgc", "cgc-star"} else (None,)
    total = len(records) * len(args.representations) * len(cgc_depths)
    rows: list[dict[str, Any]] = []
    completed = 0
    for record in records:
        represented = representation_map(record["traces"])
        for representation in args.representations:
            for n_pasts in cgc_depths:
                depth_label = "native" if n_pasts is None else str(n_pasts)
                unit = (
                    f"{record['fluo_type']}|{record['recording']}|{args.algorithm}|"
                    f"{representation}|n_pasts={depth_label}"
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
                    n_surrogates=args.n_surrogates,
                    random_state=_stable_seed(args.seed, unit),
                    n_pasts=n_pasts,
                    tau=args.cgc_tau,
                )
                artifact = (
                    args.output_dir / "artifacts" / f"{store.digest}__{completed}.npz"
                )
                _atomic_npz(
                    artifact,
                    **arrays,
                    input_digest=np.asarray(record["input_digest"]),
                    input_values=np.asarray(represented[representation], dtype=float),
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
                    "n_pasts": n_pasts,
                    "alpha": args.alpha,
                    "input_digest": record["input_digest"],
                    "artifact_path": str(artifact),
                    **graph_summary(
                        adjacency.astype(float), record["mid"], binary=True
                    ),
                    **method_metadata,
                }
                store.save_rows(unit, [row])
                rows.append(row)
                completed += 1
                print(
                    format_progress(completed, total, label="Baseline fits"),
                    flush=True,
                )
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
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=1000,
        help="Circular-shift replicates for c-GC/c-GC*; ignored by other learners.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Independent simulation fits to run concurrently",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument(
        "--cgc-depths",
        default="",
        help=(
            "Comma-separated conditioning depths for c-GC/c-GC*. An empty value "
            "uses max-lag. Other learners retain their native conditioning."
        ),
    )
    parser.add_argument(
        "--cgc-tau",
        type=int,
        default=None,
        help=(
            "Optional single tested lag for c-GC/c-GC*. Leave unset to test "
            "lags 1..max-lag. Use 1 with max-lag=1 for the matched benchmark."
        ),
    )
    parser.add_argument("--simulation-kinds", default="static,episodic")
    parser.add_argument("--conditions", default="native,shared_input")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--case-data-dir", type=Path, default=DEFAULT_CASE_DATA_DIR)
    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Comma-separated preprocessing cases from the A--D case dictionaries. "
            "When supplied, these replace --data-file/--fluo-types as the input."
        ),
    )
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
    args.cgc_depths = tuple(
        int(item.strip()) for item in args.cgc_depths.split(",") if item.strip()
    ) or (args.max_lag,)
    args.recordings = tuple(
        item.strip() for item in args.recordings.split(",") if item.strip()
    )
    args.fluo_types = tuple(
        item.strip() for item in args.fluo_types.split(",") if item.strip()
    )
    args.cases = tuple(
        item.strip().upper() for item in args.cases.split(",") if item.strip()
    )
    invalid_cases = sorted(set(args.cases) - set(CASE_FILES))
    if invalid_cases:
        raise SystemExit(f"unsupported preprocessing cases: {', '.join(invalid_cases)}")
    if args.max_lag < 1 or args.n_seeds < 1 or args.n_steps < 100:
        raise SystemExit("lags/seeds must be positive and n-steps at least 100")
    if min(args.cgc_depths) < args.max_lag:
        raise SystemExit("every c-GC conditioning depth must be at least max-lag")
    if args.cgc_tau is not None and not 1 <= args.cgc_tau <= args.max_lag:
        raise SystemExit("--cgc-tau must lie between 1 and max-lag")
    if args.n_surrogates < 1 and args.algorithm in {"cgc", "cgc-star"}:
        raise SystemExit("--n-surrogates must be positive for c-GC/c-GC*")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must lie in (0, 1)")
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.dataset == "simulations":
        input_source = "seeded_simulation"
    elif args.cases:
        input_source = "case_dictionaries"
    else:
        input_source = "combined_dataframe"
    config = {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "representations": list(args.representations),
        "max_lag": args.max_lag,
        "cgc_depths": (
            list(args.cgc_depths) if args.algorithm in {"cgc", "cgc-star"} else []
        ),
        "cgc_tau": (args.cgc_tau if args.algorithm in {"cgc", "cgc-star"} else None),
        "alpha": args.alpha,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "n_steps": args.n_steps,
        "simulation_kinds": list(args.simulation_kinds),
        "conditions": list(args.conditions),
        "input_source": input_source,
        "data_file": (
            str(args.data_file)
            if args.dataset == "motorneurons" and not args.cases
            else None
        ),
        "case_data_dir": (
            str(args.case_data_dir)
            if args.dataset == "motorneurons" and args.cases
            else None
        ),
        "cases": list(args.cases) if args.dataset == "motorneurons" else [],
        "recordings": list(args.recordings),
        "fluo_types": (
            list(args.fluo_types)
            if args.dataset == "motorneurons" and not args.cases
            else []
        ),
        "schema_version": 6,
    }
    if args.algorithm in {"cgc", "cgc-star"}:
        config.update(
            {
                "n_surrogates": args.n_surrogates,
                "seed": args.seed,
                "permutation_probability": "add-one",
            }
        )
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
            "cgc": (
                "c-GC/c-GC* operate on the same complete time-axis representation "
                "as the other learners in this runner; no pair-specific event "
                "selection is applied."
            ),
            "pcmciplus": (
                "Causal-sufficiency benchmark; raw contemporaneous marks are "
                "preserved and scoring uses BH-filtered lagged directed links."
            ),
            "var-granger": (
                "Conditional predictive baseline from nested linear VAR F-tests; "
                "it is not an intervention graph, and fit stability is reported."
            ),
            "lpcmci": (
                "Latent-confounding sensitivity method; the raw PAG is preserved. "
                "Only its explicitly lossy lagged skeleton is used for common "
                "skeleton-level recovery summaries."
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
