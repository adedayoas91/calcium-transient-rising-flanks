"""Create the strict analysis bundle for the temporal-resolvability map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


REGIMES = (
    "clean_homogeneous",
    "independent_stress",
    "shared_noise_stress",
    "shared_heterogeneous",
)
LAYERS = (
    "native_latent_events",
    "sampled_latent_events",
    "sampled_noiseless_calcium",
    "sampled_noisy_fluorescence",
)
PRIMARY_LAYER = "sampled_noisy_fluorescence"
PRIMARY_DEADBAND = 0
PRIMARY_METRICS = (
    "true_edge_coverage",
    "unconditional_direction_accuracy",
    "candidate_density",
)
SEED_METRICS = PRIMARY_METRICS + (
    "directional_pair_fraction",
    "ambiguous_pair_fraction",
    "unmatched_pair_fraction",
    "nonedge_exclusion",
    "heldout_directional_replication",
    "median_observed_lag_absolute_error",
)
CONTRASTS = (
    ("clean_homogeneous", "independent_stress", "independent_noise"),
    ("independent_stress", "shared_noise_stress", "shared_noise"),
    ("shared_noise_stress", "shared_heterogeneous", "decay_heterogeneity"),
)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"no rows found in {path}")
    integer_fields = {
        "seed",
        "fold",
        "acquisition_phase",
        "acquisition_phase_count",
        "native_delay_frames",
        "downsample",
        "deadband_frames",
    }
    optional_numeric = set(SEED_METRICS) | {"observed_delay_frames"}
    rows: list[dict[str, Any]] = []
    for row in raw:
        item: dict[str, Any] = dict(row)
        for field in integer_fields:
            item[field] = int(item[field])
        for field in optional_numeric:
            item[field] = _optional_float(item[field])
        rows.append(item)
    return rows


def read_cells(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"no cells found in {path}")
    integer_fields = {
        "native_delay_frames",
        "downsample",
        "deadband_frames",
        "n_seeds",
        "seed_pass_count",
    }
    numeric_fields = {
        "observed_delay_frames",
        "true_edge_coverage_mean",
        "true_edge_coverage_sd",
        "unconditional_direction_accuracy_mean",
        "unconditional_direction_accuracy_sd",
        "candidate_density_mean",
        "candidate_density_sd",
        "seed_pass_fraction",
    }
    boolean_fields = {
        "passes_mean_gates",
        "passes_seed_replication",
        "passes_adjacency",
        "viable",
    }
    cells: list[dict[str, Any]] = []
    for row in raw:
        item: dict[str, Any] = dict(row)
        for field in integer_fields:
            item[field] = int(item[field])
        for field in numeric_fields:
            item[field] = _optional_float(item[field])
        for field in boolean_fields:
            item[field] = item[field].lower() == "true"
        cells.append(item)
    return cells


def seed_cell_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average folds and acquisition phases before treating seeds as units."""

    keys = (
        "regime",
        "signal_layer",
        "native_delay_frames",
        "downsample",
        "observed_delay_frames",
        "deadband_frames",
        "seed",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item = dict(zip(keys, key))
        for metric in SEED_METRICS:
            values = [
                float(member[metric])
                for member in members
                if member[metric] is not None
            ]
            item[metric] = None if not values else float(np.mean(values))
        output.append(item)
    return output


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    random_state: int,
    n_bootstrap: int = 10_000,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("values must be a non-empty finite vector")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, data.size, size=(n_bootstrap, data.size))
    means = np.mean(data[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def exact_sign_flip_pvalue(values: np.ndarray, chunk_size: int = 65_536) -> float:
    """Two-sided exact paired sign-flip p-value for a mean difference."""

    differences = np.asarray(values, dtype=float)
    if (
        differences.ndim != 1
        or differences.size == 0
        or differences.size > 24
        or not np.isfinite(differences).all()
    ):
        raise ValueError("values must contain 1 to 24 finite paired differences")
    observed = abs(float(np.mean(differences)))
    total = 1 << differences.size
    exceedances = 0
    bit_positions = np.arange(differences.size, dtype=np.uint64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        indices = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 2.0 * ((indices >> bit_positions) & 1).astype(float) - 1.0
        statistics = np.abs(np.mean(signs * differences[None, :], axis=1))
        exceedances += int(np.count_nonzero(statistics >= observed - 1e-15))
    return exceedances / total


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(values)
    adjusted_ordered = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * values[index])
        adjusted_ordered[rank] = min(running, 1.0)
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted.tolist()


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _cohen_dz(values: np.ndarray) -> float:
    sd = _sample_sd(values)
    if sd == 0.0:
        return 0.0 if float(np.mean(values)) == 0.0 else float("inf")
    return float(np.mean(values) / sd)


def descriptive_rows(
    seed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    primary = [
        row
        for row in seed_rows
        if row["signal_layer"] == PRIMARY_LAYER
        and int(row["deadband_frames"]) == PRIMARY_DEADBAND
    ]
    index = 0
    group_keys = sorted(
        {
            (
                str(row["regime"]),
                int(row["native_delay_frames"]),
                int(row["downsample"]),
                float(row["observed_delay_frames"]),
            )
            for row in primary
        }
    )
    for regime, native_delay, downsample, observed_delay in group_keys:
        members = [
            row
            for row in primary
            if row["regime"] == regime
            and row["native_delay_frames"] == native_delay
            and row["downsample"] == downsample
        ]
        for metric in PRIMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in members])
            low, high = bootstrap_mean_ci(
                values,
                random_state=20260821 + index,
            )
            index += 1
            output.append(
                {
                    "regime": regime,
                    "native_delay_frames": native_delay,
                    "downsample": downsample,
                    "observed_delay_frames": observed_delay,
                    "metric": metric,
                    "n_seeds": values.size,
                    "mean": float(np.mean(values)),
                    "sd": _sample_sd(values),
                    "median": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _grid_seed_values(
    seed_rows: Sequence[dict[str, Any]],
    *,
    regime: str,
    layer: str,
    deadband: int,
    metric: str,
) -> dict[int, float]:
    groups: dict[int, list[float]] = {}
    for row in seed_rows:
        if (
            row["regime"] == regime
            and row["signal_layer"] == layer
            and int(row["deadband_frames"]) == deadband
            and row[metric] is not None
        ):
            groups.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in groups.items()}


def layer_summary_rows(
    seed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    index = 0
    for layer in LAYERS:
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                list(
                    _grid_seed_values(
                        seed_rows,
                        regime="clean_homogeneous",
                        layer=layer,
                        deadband=0,
                        metric=metric,
                    ).values()
                )
            )
            low, high = bootstrap_mean_ci(
                values,
                random_state=20260921 + index,
            )
            index += 1
            output.append(
                {
                    "regime": "clean_homogeneous",
                    "signal_layer": layer,
                    "metric": metric,
                    "n_seeds": values.size,
                    "mean_across_grid": float(np.mean(values)),
                    "sd_across_seeds": _sample_sd(values),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def inferential_rows(
    seed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        family: list[dict[str, Any]] = []
        for contrast_index, (left, right, label) in enumerate(CONTRASTS):
            left_values = _grid_seed_values(
                seed_rows,
                regime=left,
                layer=PRIMARY_LAYER,
                deadband=PRIMARY_DEADBAND,
                metric=metric,
            )
            right_values = _grid_seed_values(
                seed_rows,
                regime=right,
                layer=PRIMARY_LAYER,
                deadband=PRIMARY_DEADBAND,
                metric=metric,
            )
            seeds = sorted(set(left_values) & set(right_values))
            differences = np.asarray(
                [right_values[seed] - left_values[seed] for seed in seeds]
            )
            low, high = bootstrap_mean_ci(
                differences,
                random_state=91 + 20 * metric_index + contrast_index,
            )
            family.append(
                {
                    "family": metric,
                    "contrast": label,
                    "left_regime": left,
                    "right_regime": right,
                    "difference_direction": "right_minus_left",
                    "n_seeds": differences.size,
                    "mean_difference": float(np.mean(differences)),
                    "sd_difference": _sample_sd(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "cohen_dz": _cohen_dz(differences),
                    "p_exact": exact_sign_flip_pvalue(differences),
                }
            )
        adjusted = holm_adjust([float(row["p_exact"]) for row in family])
        output.extend(
            {**row, "p_holm": p_holm}
            for row, p_holm in zip(family, adjusted)
        )
    return output


def _cell_matrix(
    cells: Sequence[dict[str, Any]],
    *,
    regime: str,
    layer: str,
    metric: str,
) -> tuple[np.ndarray, list[int], list[int], list[dict[str, Any]]]:
    selected = [
        cell
        for cell in cells
        if cell["regime"] == regime
        and cell["signal_layer"] == layer
        and int(cell["deadband_frames"]) == PRIMARY_DEADBAND
    ]
    native_delays = sorted({int(cell["native_delay_frames"]) for cell in selected})
    downsamples = sorted({int(cell["downsample"]) for cell in selected})
    lookup = {
        (int(cell["native_delay_frames"]), int(cell["downsample"])): cell
        for cell in selected
    }
    matrix = np.asarray(
        [
            [float(lookup[(delay, sample)][metric]) for sample in downsamples]
            for delay in native_delays
        ]
    )
    return matrix, native_delays, downsamples, selected


def plot_resolution_maps(cells: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    image = None
    for axis, regime in zip(axes.flat, REGIMES):
        accuracy, delays, samples, selected = _cell_matrix(
            cells,
            regime=regime,
            layer=PRIMARY_LAYER,
            metric="unconditional_direction_accuracy_mean",
        )
        density, _, _, _ = _cell_matrix(
            cells,
            regime=regime,
            layer=PRIMARY_LAYER,
            metric="candidate_density_mean",
        )
        image = axis.imshow(
            accuracy,
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
            aspect="auto",
            origin="lower",
        )
        viable = {
            (int(cell["native_delay_frames"]), int(cell["downsample"]))
            for cell in selected
            if cell["viable"]
        }
        for row_index, delay in enumerate(delays):
            for column_index, sample in enumerate(samples):
                value = accuracy[row_index, column_index]
                color = "white" if value < 0.48 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"A {value:.2f}\nD {density[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=color,
                )
                if (delay, sample) in viable:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#D55E00",
                            linewidth=2.4,
                        )
                    )
        axis.set_xticks(range(len(samples)), samples)
        axis.set_yticks(range(len(delays)), delays)
        axis.set_xlabel("Downsampling factor")
        axis.set_ylabel("Native edge delay (frames)")
        axis.set_title(regime.replace("_", " "), fontsize=10)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
        colorbar.set_label("Unconditional direction accuracy")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_loss_decomposition(cells: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.25), constrained_layout=True)
    image = None
    for axis, layer in zip(axes, LAYERS):
        means, delays, samples, _ = _cell_matrix(
            cells,
            regime="clean_homogeneous",
            layer=layer,
            metric="unconditional_direction_accuracy_mean",
        )
        sds, _, _, _ = _cell_matrix(
            cells,
            regime="clean_homogeneous",
            layer=layer,
            metric="unconditional_direction_accuracy_sd",
        )
        image = axis.imshow(
            means,
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
            aspect="auto",
            origin="lower",
        )
        for row_index in range(len(delays)):
            for column_index in range(len(samples)):
                value = means[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}\n±{sds[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if value < 0.48 else "black",
                )
        axis.set_xticks(range(len(samples)), samples)
        axis.set_yticks(range(len(delays)), delays)
        axis.set_xlabel("Downsampling factor")
        axis.set_title(layer.replace("_", " "), fontsize=8.5)
    axes[0].set_ylabel("Native edge delay (frames)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_label("Unconditional direction accuracy")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def write_analysis_report(
    path: Path,
    cells: Sequence[dict[str, Any]],
    layers: Sequence[dict[str, Any]],
    inference: Sequence[dict[str, Any]],
    gates: dict[str, Any],
) -> None:
    primary = [
        cell
        for cell in cells
        if cell["signal_layer"] == PRIMARY_LAYER
        and int(cell["deadband_frames"]) == PRIMARY_DEADBAND
    ]
    viable = sorted(
        (cell for cell in primary if cell["viable"]),
        key=lambda item: (
            REGIMES.index(str(item["regime"])),
            int(item["native_delay_frames"]),
            int(item["downsample"]),
        ),
    )
    seed_counts = {int(cell["n_seeds"]) for cell in primary}
    if len(seed_counts) != 1:
        raise ValueError("primary cells must use one common seed count")
    n_seeds = seed_counts.pop()
    native_delays = sorted({int(cell["native_delay_frames"]) for cell in primary})
    downsamples = sorted({int(cell["downsample"]) for cell in primary})
    grid_cell_count = len(native_delays) * len(downsamples)
    viable_regimes = sorted({str(cell["regime"]) for cell in viable})
    regime_text = ", ".join(name.replace("_", " ") for name in viable_regimes)
    if viable:
        key_finding = (
            f"{len(viable)} of {len(primary)} primary fluorescence cells passed "
            f"the locked conjunctive gate. Passing cells occurred in: {regime_text}."
        )
        if viable_regimes == ["clean_homogeneous"]:
            key_finding += (
                " No cell with stronger independent noise, shared noise, or "
                "heterogeneous decay passed."
            )
        key_finding += (
            " The three-state screen is therefore conditionally feasible as a "
            "candidate-pruning pre-screen only within the passing synthetic "
            "upper-bound conditions."
        )
    else:
        key_finding = (
            f"None of the {len(primary)} primary fluorescence cells passed the "
            "locked conjunctive gate, so this screen is not feasible under the "
            "tested conditions."
        )
    mean_only_failures = [
        cell
        for cell in primary
        if cell["passes_mean_gates"] and not cell["passes_seed_replication"]
    ]
    if mean_only_failures:
        example = sorted(
            mean_only_failures,
            key=lambda cell: (
                REGIMES.index(str(cell["regime"])),
                int(cell["native_delay_frames"]),
                int(cell["downsample"]),
            ),
        )[0]
        replication_sentence = (
            "The passing cells are not described by a single delay-to-frame "
            f"ratio. For example, native delay {example['native_delay_frames']} "
            f"at downsampling {example['downsample']} passes the mean thresholds "
            f"but only {example['seed_pass_count']}/{example['n_seeds']} seeds "
            "pass jointly, so it fails the seed-replication gate. This is why "
            "the map and conjunctive seed criterion are more informative than "
            "a global one-frame cutoff."
        )
    else:
        replication_sentence = (
            "No cell passed the mean thresholds while failing the seed-replication "
            "gate in this run; the gate remains retained to prevent unstable means "
            "from defining the usable region in future runs."
        )
    layer_lookup = {
        (str(row["signal_layer"]), str(row["metric"])): row for row in layers
    }
    lines = [
        "# Temporal-resolvability map: strict analysis",
        "",
        "## Analysis question",
        "",
        f"Under what acquisition and observation conditions can the three-state rising-flank screen retain true directions, orient them correctly, and reduce the candidate family before any c-GC fit? The independent unit is the simulation seed (n = {n_seeds}). Cross-fit folds and all acquisition-phase offsets are averaged within seed.",
        "",
        "## Key finding",
        "",
        key_finding,
        "",
        "## Passing region",
        "",
        "| regime | native delay | downsample | observed delay | coverage | unconditional accuracy | density | passing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in viable:
        lines.append(
            f"| {cell['regime']} | {cell['native_delay_frames']} | "
            f"{cell['downsample']} | {_fmt(float(cell['observed_delay_frames']))} | "
            f"{_fmt(float(cell['true_edge_coverage_mean']))} | "
            f"{_fmt(float(cell['unconditional_direction_accuracy_mean']))} | "
            f"{_fmt(float(cell['candidate_density_mean']))} | "
            f"{cell['seed_pass_count']}/{cell['n_seeds']} |"
        )
    if not viable:
        lines.append("| _none_ | | | | | | | |")
    lines.extend(
        [
            "",
            replication_sentence,
            "",
            "## Where information is lost",
            "",
            f"Clean-regime values below average the complete {len(native_delays)} × {len(downsamples)} delay/downsampling grid; uncertainty is across {n_seeds} seed-level grid averages.",
            "",
            "| signal layer | accuracy mean ± SD | 95% CI | candidate density |",
            "|---|---:|---:|---:|",
        ]
    )
    for layer in LAYERS:
        accuracy = layer_lookup[(layer, "unconditional_direction_accuracy")]
        density = layer_lookup[(layer, "candidate_density")]
        lines.append(
            f"| {layer} | {_fmt(float(accuracy['mean_across_grid']))} ± "
            f"{_fmt(float(accuracy['sd_across_seeds']))} | "
            f"[{_fmt(float(accuracy['ci95_low']))}, "
            f"{_fmt(float(accuracy['ci95_high']))}] | "
            f"{_fmt(float(density['mean_across_grid']))} |"
        )
    lines.extend(
        [
            "",
            "Native latent events preserve the imposed order. Point sampling loses many impulse events, while noiseless calcium integration recovers much of the ordering. Observation noise then reduces accuracy and increases ambiguity. These layer comparisons localize measurability loss; they are not additional hypothesis tests or causal evidence.",
            "",
            "## Sequential stress contrasts",
            "",
            f"Contrasts average the {grid_cell_count} primary grid cells within seed and report right-minus-left changes. Exact paired sign-flip p-values are Holm-adjusted within each three-contrast metric family.",
            "",
            "| metric | added factor | mean change | 95% CI | dz | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in inference:
        lines.append(
            f"| {row['family']} | {row['contrast']} | "
            f"{_fmt(float(row['mean_difference']))} | "
            f"[{_fmt(float(row['ci95_low']))}, {_fmt(float(row['ci95_high']))}] | "
            f"{_fmt(float(row['cohen_dz']))} | {_fmt_p(float(row['p_holm']))} |"
        )
    lines.extend(
        [
            "",
            "## Decision and evidence boundary",
            "",
            (
                f"Proceed only to a small c-GC follow-up restricted to the {len(viable)} viable coordinates ({regime_text}), with the three-state mask treated as a screening prior rather than ground-truth causality. Do not use the rule as a universal hard direction constraint, and do not promote it to noisy empirical recordings on the strength of this experiment."
                if viable
                else "Do not proceed to a c-GC follow-up with this screen because no tested coordinate passed the locked feasibility gate."
            ),
            "",
            "The simulator is a deterministic source-triggered five-node chain with no spontaneous events or transmission failures. It measures temporal observability under known propagation; it does not establish causal identification, handle reciprocal edges, model ROI-specific rise kernels, or demonstrate graph-recovery improvement. Deadband 1 remains descriptive sensitivity only.",
            "",
            f"Prespecified recommendation: `{gates['recommendation']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stats_appendix(
    path: Path,
    inference: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
) -> None:
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    if not seeds:
        raise ValueError("seed rows cannot be empty")
    seed_text = (
        f"{seeds[0]}–{seeds[-1]}"
        if seeds == list(range(seeds[0], seeds[-1] + 1))
        else ", ".join(map(str, seeds))
    )
    grid_cells = {
        (int(row["native_delay_frames"]), int(row["downsample"]))
        for row in seed_rows
        if row["signal_layer"] == PRIMARY_LAYER
        and int(row["deadband_frames"]) == PRIMARY_DEADBAND
    }
    lines = [
        "# Statistical appendix",
        "",
        f"The independent repeated-measure unit is the random seed (n = {len(seeds)}, seeds {seed_text}). Two cross-fit episode folds and all q acquisition offsets for downsampling factor q are averaged within seed. Sequential regime contrasts then average the {len(grid_cells)} locked delay/downsampling cells within each seed, preserving common-random-number pairing.",
        "",
        f"Metrics are bounded and often concentrated at thresholds, so inference does not rely on normality. Each right-minus-left regime contrast uses a two-sided exact paired sign-flip randomization test over all 2^{len(seeds)} assignments, a percentile seed-bootstrap 95% confidence interval with 10,000 resamples, and Cohen's dz. Holm correction controls family-wise error separately for the three contrasts of each metric. Descriptive cell intervals use the same 10,000-resample seed bootstrap.",
        "",
        "| family | contrast | n | mean diff | SD diff | 95% CI | dz | exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inference:
        lines.append(
            f"| {row['family']} | {row['contrast']} | {row['n_seeds']} | "
            f"{_fmt(float(row['mean_difference']))} | "
            f"{_fmt(float(row['sd_difference']))} | "
            f"[{_fmt(float(row['ci95_low']))}, {_fmt(float(row['ci95_high']))}] | "
            f"{_fmt(float(row['cohen_dz']))} | "
            f"{_fmt_p(float(row['p_exact']))} | "
            f"{_fmt_p(float(row['p_holm']))} |"
        )
    lines.extend(
        [
            "",
            "The inferential contrasts estimate sequential additions in the locked nested regimes; they are not factorial main effects or interactions. Grid cells are deliberately not treated as independent replicates. The pass/fail map is governed by prespecified effect thresholds and seed replication, not p-values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure_catalog(
    path: Path,
    cells: Sequence[dict[str, Any]],
) -> None:
    primary = [
        cell
        for cell in cells
        if cell["signal_layer"] == PRIMARY_LAYER
        and int(cell["deadband_frames"]) == PRIMARY_DEADBAND
    ]
    seed_counts = {int(cell["n_seeds"]) for cell in primary}
    if len(seed_counts) != 1:
        raise ValueError("primary cells must use one common seed count")
    n_seeds = seed_counts.pop()
    viable = [cell for cell in primary if cell["viable"]]
    viable_regimes = sorted({str(cell["regime"]) for cell in viable})
    regime_text = ", ".join(name.replace("_", " ") for name in viable_regimes)
    observation = (
        f"{len(viable)} cells pass in {regime_text}; all other cells fail."
        if viable
        else "No primary cell passes the locked feasibility gate."
    )
    path.write_text(
        f"""# Figure catalog

## Figure 1 — `figures/figure-01-primary-resolution-map.pdf`

- Purpose: identify where the primary noisy-fluorescence screen satisfies the temporal-direction requirements before c-GC.
- Data: seed means after averaging two cross-fit folds and all acquisition phases, n = {n_seeds} seeds per cell; deadband 0.
- Display: color is unconditional direction accuracy; each cell prints accuracy (A) and total three-state candidate density (D); orange outlines mark cells that also pass coverage, seed replication, and adjacency gates.
- Observation: {observation}
- Interpretation: useful pruning requires both sufficiently stable onset order and low enough ambiguity, not merely true-edge admission.
- Implication: any c-GC follow-up should be limited to the outlined viable cells.
- Caveat: the source-triggered chain is an upper-bound measurability test, not causal identification.

## Figure 2 — `figures/figure-02-information-loss.pdf`

- Purpose: localize whether failures arise in latent timing, sampling, calcium convolution, or observation noise.
- Data: clean homogeneous regime, deadband 0, n = {n_seeds} seed means per cell.
- Display: color and annotations show mean unconditional direction accuracy; annotations also show sample SD across seeds.
- Observation: native latent order is perfectly measurable; point sampling removes impulses, noiseless calcium integration restores much of the order, and fluorescence noise creates additional losses.
- Interpretation: the observed boundary is a measurement/extraction boundary rather than failure of the imposed latent propagation order.
- Implication: better event extraction may expand the feasible region, but the current rule should not be generalized beyond observed passing cells.
- Caveat: sampled impulse events are a diagnostic layer and do not mimic the temporal integration of a physical camera exposure.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/validation_results/temporal_resolvability_map"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis-output"
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_dir / "resolvability_rows.csv")
    cells = read_cells(args.input_dir / "resolvability_cells.csv")
    seed_rows = seed_cell_rows(rows)
    descriptive = descriptive_rows(seed_rows)
    layers = layer_summary_rows(seed_rows)
    inference = inferential_rows(seed_rows)
    gates = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))[
        "gates"
    ]
    _write_csv(output_dir / "seed-cell-summary.csv", seed_rows)
    _write_csv(output_dir / "descriptive-summary.csv", descriptive)
    _write_csv(output_dir / "layer-summary.csv", layers)
    _write_csv(output_dir / "inferential-tests.csv", inference)
    plot_resolution_maps(cells, figures / "figure-01-primary-resolution-map")
    plot_loss_decomposition(cells, figures / "figure-02-information-loss")
    write_analysis_report(
        output_dir / "analysis-report.md",
        cells,
        layers,
        inference,
        gates,
    )
    write_stats_appendix(output_dir / "stats-appendix.md", inference, seed_rows)
    write_figure_catalog(output_dir / "figure-catalog.md", cells)
    print(output_dir)


if __name__ == "__main__":
    main()
