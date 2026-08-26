"""Run resumable OASIS and LPCMCI baselines on motoneuron recordings.

OASIS is a fluorescence-to-event preprocessing baseline, not a causal learner.
Its spike and denoised outputs are passed to matched c-GC/c-GC* estimators.
LPCMCI retains its raw PAG, p-value, and test-statistic tensors; the separate
lagged skeleton is explicitly lossy and has no orientation interpretation. The
input records come from the exact combined dataframe read by both motoneuron
c-GC notebooks; per-record array digests make that contract auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from calcium_transient_rising_flank import (
    CausalisedGC,
    LPCMCIAdapter,
    OASISDeconvolver,
    array_input_digest,
    build_representations,
    graph_summary,
)
from calcium_transient_rising_flank.checkpointing import format_progress


DEFAULT_DATA_FILE = Path(
    "data/motoneurons/df_motorneurons_F3T1_F3T2_F5T2.pkl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/empirical_baselines")
FLUO_TYPE_CHOICES = ("dff", "f_smooth")
COMPONENT_CHOICES = ("oasis", "lpcmci")
REPRESENTATION_CHOICES = (
    "full",
    "deconvolved",
    "rise",
    "fall",
    "fall_residual",
)
OASIS_OUTPUT_CHOICES = ("spikes", "denoised")
CGC_METHOD_CHOICES = ("cgc", "cgc-star")
PROGRESS_FILE = "progress.json"
OASIS_PARTIAL_FILE = "oasis_preprocessing_rows.partial.csv"
GRAPH_PARTIAL_FILE = "graph_summary_rows.partial.csv"


def _parse_choices(
    value: str,
    *,
    choices: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    invalid = sorted(set(selected) - set(choices))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported {name}: {', '.join(invalid)}"
        )
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"duplicate {name} values are not allowed")
    return selected


def _parse_recordings(value: str) -> tuple[str, ...]:
    recordings = tuple(item.strip() for item in value.split(",") if item.strip())
    if not recordings:
        raise argparse.ArgumentTypeError("at least one recording is required")
    if len(set(recordings)) != len(recordings):
        raise argparse.ArgumentTypeError("duplicate recordings are not allowed")
    return recordings


def _clean_traces(raw: Any, *, expected_cells: int | None = None) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected two-dimensional traces, got {values.shape}")
    if (
        expected_cells is not None
        and values.shape[0] != expected_cells
        and values.shape[1] == expected_cells
    ):
        values = values.T
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] > values.shape[1]:
        raise ValueError(
            f"trace matrix looks transposed or too short: {values.shape}"
        )
    return values


def _array_digest(values: np.ndarray) -> str:
    return array_input_digest(np.asarray(values, dtype=float))


def load_motoneuron_records(
    data_file: Path,
    *,
    fluo_types: Sequence[str],
    recordings: Sequence[str],
) -> list[dict[str, Any]]:
    """Load and deduplicate records exactly as the c-GC notebooks do."""

    dataframe = pd.read_pickle(data_file)
    if not hasattr(dataframe, "iterrows"):
        raise TypeError(f"expected a pandas DataFrame, got {type(dataframe)!r}")
    trace_column = (
        "Trial"
        if "Trial" in dataframe.columns
        else "Trace"
        if "Trace" in dataframe.columns
        else None
    )
    required = {"Fish", "fluo", "fluo_type", "mid"}
    missing_columns = required.difference(dataframe.columns)
    if trace_column is None:
        missing_columns.add("Trial/Trace")
    if missing_columns:
        raise KeyError(
            "motoneuron dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    requested_recordings = set(recordings)
    requested_fluo_types = set(fluo_types)
    loaded: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for _, row in dataframe.iterrows():
        fluo_type = str(row["fluo_type"])
        fish = int(row["Fish"])
        trial = int(row[trace_column])
        recording = f"F{fish}T{trial}"
        key = (fish, trial, fluo_type)
        if (
            key in seen
            or fluo_type not in requested_fluo_types
            or recording not in requested_recordings
        ):
            continue
        seen.add(key)
        expected_cells = (
            int(row["n_cells"]) if "n_cells" in dataframe.columns else None
        )
        values = _clean_traces(row["fluo"], expected_cells=expected_cells)
        loaded.append(
            {
                "fluo_type": fluo_type,
                "recording": recording,
                "fish": fish,
                "trial": trial,
                "traces": values,
                "mid": int(row["mid"]),
                "input_digest": _array_digest(values),
            }
        )

    actual = {(row["fluo_type"], row["recording"]) for row in loaded}
    expected = {
        (fluo_type, recording)
        for fluo_type in fluo_types
        for recording in recordings
    }
    missing = sorted(expected - actual)
    if missing:
        labels = ", ".join(
            f"{fluo_type}/{recording}" for fluo_type, recording in missing
        )
        raise FileNotFoundError(f"requested motoneuron units are missing: {labels}")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in value.lower()
    ).strip("-")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _valid_unit_artifact(
    path: Path,
    *,
    unit: str,
    input_digest: str,
) -> bool:
    if not path.is_file():
        return False
    if unit.startswith("oasis-transform|"):
        required = {
            "denoised",
            "spikes",
            "baseline",
            "penalty_lambda",
            "ar_params_json",
            "input_digest",
        }
    elif unit.startswith("oasis-graph|"):
        required = {
            "adjacency",
            "retained_scores",
            "p_values",
            "best_lags",
            "input_digest",
        }
    elif unit.startswith("lpcmci|"):
        required = {
            "graph",
            "p_matrix",
            "val_matrix",
            "lossy_lagged_skeleton",
            "input_digest",
        }
    else:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return required.issubset(payload.files) and (
                str(np.asarray(payload["input_digest"]).item()) == input_digest
            )
    except (OSError, ValueError):
        return False


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _coerce(value: str | None) -> Any:
    if value in {None, ""}:
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def _stable_seed(base_seed: int, unit: str) -> int:
    digest = hashlib.sha256(unit.encode()).digest()
    return base_seed + int.from_bytes(digest[:4], "big") % 1_000_000


def _oasis_transform_unit(record: dict[str, Any]) -> str:
    return f"oasis-transform|{record['fluo_type']}|{record['recording']}"


def _oasis_graph_unit(
    record: dict[str, Any],
    *,
    method: str,
    oasis_output: str,
) -> str:
    return (
        f"oasis-graph|{method}|{oasis_output}|"
        f"{record['fluo_type']}|{record['recording']}"
    )


def _lpcmci_unit(record: dict[str, Any], *, representation: str) -> str:
    return f"lpcmci|{representation}|{record['fluo_type']}|{record['recording']}"


def _oasis_artifact_path(output_dir: Path, record: dict[str, Any]) -> Path:
    return output_dir / "raw_oasis" / (
        f"{_slug(record['fluo_type'])}__{_slug(record['recording'])}.npz"
    )


def _oasis_graph_artifact_path(
    output_dir: Path,
    record: dict[str, Any],
    *,
    method: str,
    oasis_output: str,
) -> Path:
    return output_dir / "oasis_graphs" / (
        f"{_slug(record['fluo_type'])}__{_slug(record['recording'])}__"
        f"{_slug(method)}__{_slug(oasis_output)}.npz"
    )


def _pag_artifact_path(
    output_dir: Path,
    record: dict[str, Any],
    *,
    representation: str,
) -> Path:
    return output_dir / "raw_pag" / (
        f"{_slug(record['fluo_type'])}__{_slug(record['recording'])}__"
        f"{_slug(representation)}.npz"
    )


def _pag_mark_counts(graph: np.ndarray) -> str:
    counts = Counter(str(value) for value in np.asarray(graph).ravel() if str(value))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _lagged_p_descriptives(
    p_matrix: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float | None]:
    values = np.asarray(p_matrix, dtype=float)[:, :, 1:]
    if values.size == 0:
        return {"lagged_p_mean": None, "lagged_p_le_alpha_fraction": None}
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    eligible = np.broadcast_to(off_diagonal[:, :, None], values.shape)
    selected = values[eligible]
    return {
        "lagged_p_mean": float(np.mean(selected)),
        "lagged_p_le_alpha_fraction": float(np.mean(selected <= alpha)),
    }


def _representation(
    traces: np.ndarray,
    name: str,
) -> np.ndarray:
    bundle = build_representations(traces)
    return bundle.as_dict()[name]


def run_oasis_transform(
    record: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    result = OASISDeconvolver(
        backend_kwargs={"penalty": 1, "optimize_g": 0}
    ).fit(record["traces"])
    artifact = _oasis_artifact_path(output_dir, record)
    _atomic_write_npz(
        artifact,
        denoised=result.denoised,
        spikes=result.spikes,
        baseline=result.baseline,
        penalty_lambda=result.penalty_lambda,
        ar_params_json=np.asarray(json.dumps(result.ar_params)),
        input_digest=np.asarray(record["input_digest"]),
    )
    positive = result.spikes > 0
    return {
        "unit": _oasis_transform_unit(record),
        "component": "oasis",
        "fluo_type": record["fluo_type"],
        "recording": record["recording"],
        "fish": record["fish"],
        "trial": record["trial"],
        "n_rois": int(result.spikes.shape[0]),
        "n_timepoints": int(result.spikes.shape[1]),
        "positive_event_count": int(np.count_nonzero(positive)),
        "positive_event_fraction": float(np.mean(positive)),
        "median_baseline": float(np.median(result.baseline)),
        "median_penalty_lambda": float(np.median(result.penalty_lambda)),
        "input_digest": record["input_digest"],
        "input_contract": "motoneuron-cgc-dataframe-v1",
        "artifact_path": str(artifact),
        "output_semantics": "OASIS spikes and denoised fluorescence",
    }


def run_oasis_graph(
    record: dict[str, Any],
    *,
    output_dir: Path,
    method: str,
    oasis_output: str,
    max_lag: int,
    n_surrogates: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    source_path = _oasis_artifact_path(output_dir, record)
    with np.load(source_path, allow_pickle=False) as source:
        values = np.asarray(source[oasis_output], dtype=float)
    unit = _oasis_graph_unit(record, method=method, oasis_output=oasis_output)
    result = CausalisedGC(
        max_lag=max_lag,
        n_surrogates=n_surrogates,
        alpha=alpha,
        fdr=True,
        event_mode="physical",
        method=method,
        simulation=False,
        random_state=_stable_seed(seed, unit),
    ).fit(values)
    artifact = _oasis_graph_artifact_path(
        output_dir,
        record,
        method=method,
        oasis_output=oasis_output,
    )
    _atomic_write_npz(
        artifact,
        adjacency=np.asarray(result.adjacency, dtype=bool),
        retained_scores=np.asarray(result.retained_scores, dtype=float),
        p_values=np.asarray(result.p_values, dtype=float),
        best_lags=np.asarray(result.best_lags, dtype=int),
        input_digest=np.asarray(record["input_digest"]),
    )
    return {
        "unit": unit,
        "component": "oasis",
        "fluo_type": record["fluo_type"],
        "recording": record["recording"],
        "fish": record["fish"],
        "trial": record["trial"],
        "method": method,
        "representation": f"oasis_{oasis_output}",
        "output_semantics": "directed predictive adjacency after OASIS preprocessing",
        "projection": "none",
        "artifact_path": str(artifact),
        "alpha": alpha,
        "fdr": True,
        "n_estimator_surrogates": n_surrogates,
        "max_lag": max_lag,
        "input_digest": record["input_digest"],
        "input_contract": "motoneuron-cgc-dataframe-v1",
        **graph_summary(result.retained_scores, record["mid"], binary=False),
    }


def run_lpcmci(
    record: dict[str, Any],
    *,
    output_dir: Path,
    representation: str,
    tau_max: int,
    pc_alpha: float,
    progress: bool = False,
) -> dict[str, Any]:
    values = _representation(record["traces"], representation)
    started_at = time.perf_counter()
    if progress:
        print(
            "[lpcmci] entering Tigramite "
            f"(representation={representation}, rois={values.shape[0]}, "
            f"timepoints={values.shape[1]}, tau_max={tau_max}, "
            f"pc_alpha={pc_alpha})",
            flush=True,
        )
    result = LPCMCIAdapter(
        tau_max=tau_max,
        run_kwargs={"tau_min": 0, "pc_alpha": pc_alpha},
        verbosity=1 if progress else 0,
    ).fit(values)
    skeleton = result.lossy_lagged_skeleton()
    artifact = _pag_artifact_path(
        output_dir,
        record,
        representation=representation,
    )
    _atomic_write_npz(
        artifact,
        graph=result.graph,
        p_matrix=result.p_matrix,
        val_matrix=result.val_matrix,
        lossy_lagged_skeleton=skeleton,
        input_digest=np.asarray(record["input_digest"]),
    )
    if progress:
        print(
            f"[lpcmci] fit finished in {time.perf_counter() - started_at:.1f}s; "
            f"saved {artifact}",
            flush=True,
        )
    skeleton_summary = graph_summary(
        skeleton.astype(float),
        record["mid"],
        binary=True,
    )
    return {
        "unit": _lpcmci_unit(record, representation=representation),
        "component": "lpcmci",
        "fluo_type": record["fluo_type"],
        "recording": record["recording"],
        "fish": record["fish"],
        "trial": record["trial"],
        "method": "lpcmci",
        "representation": representation,
        "output_semantics": "raw PAG with separate lossy lagged skeleton",
        "projection": "lossy_lagged_pag_skeleton",
        "orientation_interpretation": "none for the skeleton projection",
        "artifact_path": str(artifact),
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        "conditional_independence_test": result.cond_ind_test,
        "input_digest": record["input_digest"],
        "input_contract": "motoneuron-cgc-dataframe-v1",
        "pag_mark_counts": _pag_mark_counts(result.graph),
        "skeleton_w_ic": skeleton_summary["w_ic"],
        "skeleton_edge_density": skeleton_summary["edge_density"],
        "skeleton_retained_edges": skeleton_summary["retained_edges"],
        **_lagged_p_descriptives(result.p_matrix, alpha=pc_alpha),
    }


def _expected_units(
    records: Sequence[dict[str, Any]],
    *,
    components: Sequence[str],
    representations: Sequence[str],
    oasis_outputs: Sequence[str],
    cgc_methods: Sequence[str],
) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for record in records:
        if "oasis" in components:
            expected[_oasis_transform_unit(record)] = _oasis_artifact_path(
                Path("."), record
            )
            for method in cgc_methods:
                for oasis_output in oasis_outputs:
                    expected[
                        _oasis_graph_unit(
                            record,
                            method=method,
                            oasis_output=oasis_output,
                        )
                    ] = _oasis_graph_artifact_path(
                        Path("."),
                        record,
                        method=method,
                        oasis_output=oasis_output,
                    )
        if "lpcmci" in components:
            for representation in representations:
                expected[_lpcmci_unit(record, representation=representation)] = (
                    _pag_artifact_path(
                        Path("."),
                        record,
                        representation=representation,
                    )
                )
    return expected


def _recover_completed_units(
    *,
    output_dir: Path,
    expected: dict[str, Path],
    oasis_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in (*oasis_rows, *graph_rows):
        grouped.setdefault(str(row.get("unit")), []).append(dict(row))
    unknown = set(grouped) - set(expected)
    if unknown:
        raise SystemExit(
            "resume partial rows contain unexpected units: "
            + ", ".join(sorted(unknown))
        )
    complete: set[str] = set()
    for unit, relative_artifact in expected.items():
        rows = grouped.get(unit, [])
        if len(rows) == 1 and _valid_unit_artifact(
            output_dir / relative_artifact,
            unit=unit,
            input_digest=str(rows[0].get("input_digest")),
        ):
            complete.add(unit)
    for unit in tuple(complete):
        if not unit.startswith("oasis-graph|"):
            continue
        _, _, _, fluo_type, recording = unit.split("|", maxsplit=4)
        transform_unit = f"oasis-transform|{fluo_type}|{recording}"
        if transform_unit not in complete:
            complete.remove(unit)
    recovered_oasis = [row for row in oasis_rows if row.get("unit") in complete]
    recovered_graph = [row for row in graph_rows if row.get("unit") in complete]
    return recovered_oasis, recovered_graph, complete


def _write_progress(
    output_dir: Path,
    *,
    config: dict[str, Any],
    completed: set[str],
    expected_count: int,
    status: str,
    active_unit: str | None = None,
) -> None:
    _atomic_write_text(
        output_dir / PROGRESS_FILE,
        json.dumps(
            {
                "status": status,
                "config": config,
                "completed_units": sorted(completed),
                "completed_unit_count": len(completed),
                "expected_unit_count": expected_count,
                "active_unit": active_unit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fluo-types", default=",".join(FLUO_TYPE_CHOICES))
    parser.add_argument("--recordings", default="F3T1,F3T2,F5T2")
    parser.add_argument("--components", default="oasis,lpcmci")
    parser.add_argument(
        "--representations",
        default="full,deconvolved,rise,fall,fall_residual",
    )
    parser.add_argument("--oasis-outputs", default="spikes,denoised")
    parser.add_argument("--cgc-methods", default="cgc,cgc-star")
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--n-cgc-surrogates", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lpcmci-tau-max", type=int, default=3)
    parser.add_argument("--lpcmci-pc-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fluo_types = _parse_choices(
        args.fluo_types,
        choices=FLUO_TYPE_CHOICES,
        name="fluorescence type",
    )
    recordings = _parse_recordings(args.recordings)
    components = _parse_choices(
        args.components,
        choices=COMPONENT_CHOICES,
        name="component",
    )
    representations = _parse_choices(
        args.representations,
        choices=REPRESENTATION_CHOICES,
        name="representation",
    )
    oasis_outputs = _parse_choices(
        args.oasis_outputs,
        choices=OASIS_OUTPUT_CHOICES,
        name="OASIS output",
    )
    cgc_methods = _parse_choices(
        args.cgc_methods,
        choices=CGC_METHOD_CHOICES,
        name="c-GC method",
    )
    if args.max_lag < 1 or args.lpcmci_tau_max < 1:
        raise SystemExit("lag limits must be positive")
    if args.n_cgc_surrogates < 1:
        raise SystemExit("--n-cgc-surrogates must be positive")
    if not 0.0 < args.alpha < 1.0 or not 0.0 < args.lpcmci_pc_alpha < 1.0:
        raise SystemExit("alpha values must lie in (0, 1)")

    records = load_motoneuron_records(
        args.data_file,
        fluo_types=fluo_types,
        recordings=recordings,
    )
    config = {
        "resume_schema_version": 2,
        "input_contract": "motoneuron-cgc-dataframe-v1",
        "data_file": str(args.data_file),
        "data_file_sha256": _sha256(args.data_file),
        "fluo_types": list(fluo_types),
        "recordings": list(recordings),
        "components": list(components),
        "representations": list(representations),
        "oasis_outputs": list(oasis_outputs),
        "cgc_methods": list(cgc_methods),
        "max_lag": args.max_lag,
        "n_cgc_surrogates": args.n_cgc_surrogates,
        "alpha": args.alpha,
        "lpcmci_tau_max": args.lpcmci_tau_max,
        "lpcmci_pc_alpha": args.lpcmci_pc_alpha,
        "seed": args.seed,
        "oasis": {"penalty": 1, "optimize_g": 0, "kinetics": "auto_ar1"},
        "pag_projection_policy": "raw PAG is primary; lagged skeleton is lossy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILE
    oasis_partial = args.output_dir / OASIS_PARTIAL_FILE
    graph_partial = args.output_dir / GRAPH_PARTIAL_FILE
    oasis_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    expected_relative = _expected_units(
        records,
        components=components,
        representations=representations,
        oasis_outputs=oasis_outputs,
        cgc_methods=cgc_methods,
    )
    expected_count = len(expected_relative)
    previous_active_unit: str | None = None
    print(
        f"[plan] {expected_count} checkpoint units; output={args.output_dir}",
        flush=True,
    )
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("config") != config:
            raise SystemExit("resume configuration does not match the saved run")
        previous_active_unit = progress.get("active_unit")
        oasis_rows, graph_rows, completed = _recover_completed_units(
            output_dir=args.output_dir,
            expected=expected_relative,
            oasis_rows=_read_csv(oasis_partial),
            graph_rows=_read_csv(graph_partial),
        )
        print(
            f"[resume] loaded and validated {len(completed)}/{expected_count} "
            "complete checkpoint units; completed units will be skipped",
            flush=True,
        )
        if previous_active_unit and previous_active_unit not in completed:
            print(
                f"[resume] previous run stopped during {previous_active_unit}; "
                "that in-flight fit will restart",
                flush=True,
            )
    elif args.resume:
        print(
            f"[resume] no saved progress found at {progress_path}; starting a new run",
            flush=True,
        )
    else:
        print("[resume] disabled; saved completed units will not be loaded", flush=True)

    print(
        format_progress(len(completed), expected_count, label="Overall units"),
        flush=True,
    )

    _write_csv(
        args.output_dir / "input_manifest.csv",
        [
            {
                "fluo_type": record["fluo_type"],
                "recording": record["recording"],
                "fish": record["fish"],
                "trial": record["trial"],
                "n_rois": int(record["traces"].shape[0]),
                "n_timepoints": int(record["traces"].shape[1]),
                "input_digest": record["input_digest"],
                "input_contract": "motoneuron-cgc-dataframe-v1",
            }
            for record in records
        ],
    )

    _write_progress(
        args.output_dir,
        config=config,
        completed=completed,
        expected_count=expected_count,
        status="running",
    )
    for record in records:
        if "oasis" in components:
            transform_unit = _oasis_transform_unit(record)
            if transform_unit not in completed:
                print(
                    f"[unit] starting {len(completed) + 1}/{expected_count}: "
                    f"{transform_unit}",
                    flush=True,
                )
                _write_progress(
                    args.output_dir,
                    config=config,
                    completed=completed,
                    expected_count=expected_count,
                    status="running",
                    active_unit=transform_unit,
                )
                oasis_rows.append(
                    run_oasis_transform(record, output_dir=args.output_dir)
                )
                completed.add(transform_unit)
                _write_csv(oasis_partial, oasis_rows)
                _write_progress(
                    args.output_dir,
                    config=config,
                    completed=completed,
                    expected_count=expected_count,
                    status="running",
                )
                print(
                    format_progress(
                        len(completed), expected_count, label="Overall units"
                    )
                    + f" | completed {transform_unit}",
                    flush=True,
                )
            for method in cgc_methods:
                for oasis_output in oasis_outputs:
                    unit = _oasis_graph_unit(
                        record,
                        method=method,
                        oasis_output=oasis_output,
                    )
                    if unit in completed:
                        continue
                    print(
                        f"[unit] starting {len(completed) + 1}/{expected_count}: {unit}",
                        flush=True,
                    )
                    _write_progress(
                        args.output_dir,
                        config=config,
                        completed=completed,
                        expected_count=expected_count,
                        status="running",
                        active_unit=unit,
                    )
                    graph_rows.append(
                        run_oasis_graph(
                            record,
                            output_dir=args.output_dir,
                            method=method,
                            oasis_output=oasis_output,
                            max_lag=args.max_lag,
                            n_surrogates=args.n_cgc_surrogates,
                            alpha=args.alpha,
                            seed=args.seed,
                        )
                    )
                    completed.add(unit)
                    _write_csv(graph_partial, graph_rows)
                    _write_progress(
                        args.output_dir,
                        config=config,
                        completed=completed,
                        expected_count=expected_count,
                        status="running",
                    )
                    print(
                        format_progress(
                            len(completed), expected_count, label="Overall units"
                        )
                        + f" | completed {unit}",
                        flush=True,
                    )

        if "lpcmci" in components:
            for representation in representations:
                unit = _lpcmci_unit(record, representation=representation)
                if unit in completed:
                    continue
                print(
                    f"[unit] starting {len(completed) + 1}/{expected_count}: {unit}",
                    flush=True,
                )
                _write_progress(
                    args.output_dir,
                    config=config,
                    completed=completed,
                    expected_count=expected_count,
                    status="running",
                    active_unit=unit,
                )
                graph_rows.append(
                    run_lpcmci(
                        record,
                        output_dir=args.output_dir,
                        representation=representation,
                        tau_max=args.lpcmci_tau_max,
                        pc_alpha=args.lpcmci_pc_alpha,
                        progress=True,
                    )
                )
                completed.add(unit)
                _write_csv(graph_partial, graph_rows)
                _write_progress(
                    args.output_dir,
                    config=config,
                    completed=completed,
                    expected_count=expected_count,
                    status="running",
                )
                print(
                    format_progress(
                        len(completed), expected_count, label="Overall units"
                    )
                    + f" | completed {unit}",
                    flush=True,
                )

    _write_csv(args.output_dir / "oasis_preprocessing_rows.csv", oasis_rows)
    _write_csv(args.output_dir / "graph_summary_rows.csv", graph_rows)
    interpretation: dict[str, str] = {
        "matched_input": (
            "input digests identify the exact deduplicated records used by the "
            "motoneuron c-GC and c-GC* notebooks"
        ),
        "empirical": "descriptive only; recordings have no directed ground truth",
    }
    if "oasis" in components:
        interpretation["oasis"] = (
            "preprocessing baseline; no empirical event ground truth"
        )
    if "lpcmci" in components:
        interpretation["lpcmci"] = (
            "raw PAG retained; lagged skeleton is lossy and unoriented"
        )
    summary = {
        "status": "complete",
        "config": config,
        "n_oasis_rows": len(oasis_rows),
        "n_graph_rows": len(graph_rows),
        "interpretation": interpretation,
        "outputs": (
            (["oasis_preprocessing_rows.csv", "raw_oasis/*.npz"] if oasis_rows else [])
            + ["graph_summary_rows.csv", "input_manifest.csv"]
            + (["oasis_graphs/*.npz"] if "oasis" in components else [])
            + (["raw_pag/*.npz"] if "lpcmci" in components else [])
        ),
    }
    _atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed=completed,
        expected_count=expected_count,
        status="complete",
    )
    print(
        format_progress(expected_count, expected_count, label="Overall units")
        + f" | complete; summary={args.output_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
