"""Cross-fitted robust/soft temporal-prior ablation for rising flanks.

The temporal prior is learned from disjoint rise episodes. Every scientific
arm is then derived from one unrestricted held-out c-GC p-value matrix, so the
comparison changes only the hypothesis policy. A second mask-aware core fit is
used solely to measure computational pruning; its discoveries are not scored.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    DynamicSimulationConfig,
    benjamini_hochberg,
    build_representations,
    build_temporal_prior,
    downsample_dataset,
    edge_recovery,
    finite_sample_permutation_p_values,
    rise_flank_candidate_pairs,
    simulate_calcium_dataset,
    weighted_benjamini_hochberg,
)
from calcium_transient_rising_flank.estimators import extract_rise_flank_runs


DEFAULT_OUTPUT = Path("outputs/validation_results/rise_prior_ablation")
ARMS = (
    "unrestricted",
    "naive_hard",
    "robust_hard",
    "soft_prior",
    "density_matched_random",
    "oracle_hard",
)


@dataclass(frozen=True)
class ObservationCondition:
    name: str
    noise_std: float
    shared_noise_std: float
    downsample: int
    null_graph: bool = False


CONDITIONS = (
    ObservationCondition("clean", noise_std=0.01, shared_noise_std=0.0, downsample=1),
    ObservationCondition(
        "lowrate_shared",
        noise_std=0.04,
        shared_noise_std=0.03,
        downsample=2,
    ),
    ObservationCondition(
        "null_lowrate_shared",
        noise_std=0.04,
        shared_noise_std=0.03,
        downsample=2,
        null_graph=True,
    ),
)


def truth_graphs() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return the prespecified alternating five-node dynamic graphs."""

    first = np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    second = np.array(
        [
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    return first | second, (first, second)


def crossfit_episode_folds(n_episodes: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return two folds that preserve both alternating graph states per half."""

    if n_episodes != 8:
        raise ValueError("the locked cross-fit requires exactly eight episodes")
    first = (0, 1, 4, 5)
    second = (2, 3, 6, 7)
    return ((first, second), (second, first))


def episode_segments(dataset: Any, episode_ids: Iterable[int]) -> np.ndarray:
    """Label selected rise windows and exclude every other physical frame."""

    segments = np.full(dataset.fluorescence.shape[1], -1, dtype=int)
    for episode_id in episode_ids:
        episode = dataset.episodes[int(episode_id)]
        segments[episode.rise_start : episode.rise_stop] = int(episode_id)
    return segments


def episode_truth(dataset: Any, episode_ids: Iterable[int]) -> np.ndarray:
    """Return the union of directed edges active in selected episodes."""

    truth = np.zeros_like(dataset.adjacency, dtype=bool)
    for episode_id in episode_ids:
        truth |= np.asarray(dataset.episodes[int(episode_id)].adjacency, dtype=bool)
    np.fill_diagonal(truth, False)
    return truth


def event_indices_in_segments(
    rise: np.ndarray,
    segment_ids: np.ndarray,
    *,
    min_run_samples: int,
) -> tuple[np.ndarray, ...]:
    """Select retained rising-flank runs that lie in held-out rise episodes."""

    masked = np.where(segment_ids[np.newaxis, :] >= 0, rise, 0.0)
    return extract_rise_flank_runs(masked, min_run_samples).event_indices


def density_matched_random_mask(mask: np.ndarray, random_state: int) -> np.ndarray:
    """Sample an off-diagonal mask with the same number of eligible directions."""

    candidate = np.asarray(mask, dtype=bool)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("mask must be square")
    eligible = np.argwhere(~np.eye(candidate.shape[0], dtype=bool))
    count = int(np.count_nonzero(candidate & ~np.eye(candidate.shape[0], dtype=bool)))
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(len(eligible), size=count, replace=False)
    random_mask = np.zeros_like(candidate)
    for source, target in eligible[chosen]:
        random_mask[source, target] = True
    return random_mask


def arm_adjacencies(
    p_values: np.ndarray,
    *,
    alpha: float,
    naive_mask: np.ndarray,
    robust_mask: np.ndarray,
    hypothesis_weights: np.ndarray,
    random_mask: np.ndarray,
    oracle_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply all ablation policies to one shared held-out p-value matrix."""

    return {
        "unrestricted": benjamini_hochberg(p_values, alpha),
        "naive_hard": benjamini_hochberg(
            p_values, alpha, eligible_mask=naive_mask
        ),
        "robust_hard": benjamini_hochberg(
            p_values, alpha, eligible_mask=robust_mask
        ),
        "soft_prior": weighted_benjamini_hochberg(
            p_values, hypothesis_weights, alpha
        ),
        "density_matched_random": benjamini_hochberg(
            p_values, alpha, eligible_mask=random_mask
        ),
        "oracle_hard": benjamini_hochberg(
            p_values, alpha, eligible_mask=oracle_mask
        ),
    }


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _mask_recall(mask: np.ndarray, truth: np.ndarray) -> float:
    true_count = int(np.count_nonzero(truth))
    if true_count == 0:
        return 1.0
    return float(np.count_nonzero(np.asarray(mask, dtype=bool) & truth) / true_count)


def _candidate_density(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(values, False)
    denominator = values.shape[0] * (values.shape[0] - 1)
    return float(np.count_nonzero(values) / denominator)


def _fit_estimator(
    rise: np.ndarray,
    event_indices: tuple[np.ndarray, ...],
    segment_ids: np.ndarray,
    *,
    n_surrogates: int,
    random_state: int,
    candidate_mask: np.ndarray | None = None,
) -> tuple[Any, float]:
    estimator = CausalisedGC(
        max_lag=1,
        tau=1,
        n_pasts=3,
        n_surrogates=n_surrogates,
        alpha=0.05,
        fdr=True,
        event_mode="physical",
        method="cgc",
        random_state=random_state,
    )
    started = perf_counter()
    result = estimator.fit(
        rise,
        event_indices=event_indices,
        segment_ids=segment_ids,
        candidate_adjacency=candidate_mask,
    )
    return result, perf_counter() - started


def _dynamic_config(sequence: tuple[np.ndarray, ...]) -> DynamicSimulationConfig:
    return DynamicSimulationConfig(
        n_episodes=8,
        min_rise_length=24,
        max_rise_length=24,
        rise_waveform_length=12,
        fall_to_rise_ratio_min=2.1,
        fall_to_rise_ratio_max=2.2,
        propagation_delay=1,
        initial_activation_probability=0.7,
        edge_dropout_probability=0.0,
        edge_addition_probability=0.0,
        source_dropout_probability=0.0,
        source_recruitment_probability=0.0,
        fall_noise_rate=0.03,
        fall_noise_scale=0.02,
        adjacency_sequence=sequence,
    )


def _simulate_condition(
    condition: ObservationCondition,
    seed: int,
    *,
    n_steps: int,
) -> tuple[Any, float]:
    union, sequence = truth_graphs()
    if condition.null_graph:
        union = np.zeros_like(union)
        sequence = tuple(np.zeros_like(graph) for graph in sequence)
    gamma = 0.9
    dataset = simulate_calcium_dataset(
        union,
        n_steps=n_steps,
        gamma=gamma,
        noise_std=condition.noise_std,
        shared_noise_std=condition.shared_noise_std,
        spontaneous_rate=0.04,
        transmission_probability=0.9,
        random_state=seed,
        simulator_mode="episodic_dynamic",
        dynamic_config=_dynamic_config(sequence),
    )
    return downsample_dataset(dataset, condition.downsample), gamma ** condition.downsample


def _screen_masks(
    rise: np.ndarray,
    screen_segments: np.ndarray,
    *,
    deadband: int,
    max_onset_lag: int,
    min_run_samples: int,
) -> tuple[Any, np.ndarray]:
    prior = build_temporal_prior(
        rise,
        screen_segments,
        min_run_samples=min_run_samples,
        max_onset_lag=max_onset_lag,
        timing_deadband=deadband,
        minimum_decisive_support=2,
        consistency_threshold=0.75,
        beta_prior_concentration=1.0,
    )
    screen_rise = np.where(screen_segments[np.newaxis, :] >= 0, rise, 0.0)
    naive = rise_flank_candidate_pairs(
        screen_rise,
        min_run_samples=min_run_samples,
        min_lag=1,
        max_lag=max_onset_lag,
        min_overlap_samples=1,
        min_overlap_fraction=0.5,
    ).candidate_adjacency
    return prior, naive


def run_ablation(
    *,
    seeds: Sequence[int],
    conditions: Sequence[ObservationCondition],
    deadbands: Sequence[int],
    n_steps: int,
    n_surrogates: int,
    alpha: float = 0.05,
    tolerance: float = 0.02,
    min_run_samples: int = 2,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run the locked episode-cross-fitted ablation and return long-form rows."""

    if n_surrogates < 1:
        raise ValueError("n_surrogates must be positive")
    if not seeds or not conditions or not deadbands:
        raise ValueError("seeds, conditions, and deadbands must be non-empty")
    rows: list[dict[str, Any]] = []
    folds = crossfit_episode_folds(8)
    for condition in conditions:
        for seed in seeds:
            dataset, sampled_gamma = _simulate_condition(
                condition, seed, n_steps=n_steps
            )
            rise = build_representations(
                dataset.fluorescence,
                tolerance=tolerance,
                gamma=sampled_gamma,
            ).rise
            for fold, (screen_ids, test_ids) in enumerate(folds):
                screen_segments = episode_segments(dataset, screen_ids)
                test_segments = episode_segments(dataset, test_ids)
                test_events = event_indices_in_segments(
                    rise,
                    test_segments,
                    min_run_samples=min_run_samples,
                )
                truth = episode_truth(dataset, test_ids)
                raw_result, raw_seconds = _fit_estimator(
                    rise,
                    test_events,
                    test_segments,
                    n_surrogates=n_surrogates,
                    random_state=seed * 100 + fold,
                )
                p_values = finite_sample_permutation_p_values(
                    raw_result.p_values,
                    n_surrogates,
                )
                for deadband in deadbands:
                    max_onset_lag = max(1, 2 // condition.downsample)
                    prior, naive_mask = _screen_masks(
                        rise,
                        screen_segments,
                        deadband=deadband,
                        max_onset_lag=max_onset_lag,
                        min_run_samples=min_run_samples,
                    )
                    random_mask = density_matched_random_mask(
                        prior.robust_hard_mask,
                        random_state=seed * 10_000 + fold * 10 + deadband,
                    )
                    discoveries = arm_adjacencies(
                        p_values,
                        alpha=alpha,
                        naive_mask=naive_mask,
                        robust_mask=prior.robust_hard_mask,
                        hypothesis_weights=prior.hypothesis_weights,
                        random_mask=random_mask,
                        oracle_mask=truth,
                    )
                    _, pruned_seconds = _fit_estimator(
                        rise,
                        test_events,
                        test_segments,
                        n_surrogates=n_surrogates,
                        random_state=seed * 100 + fold,
                        candidate_mask=prior.robust_hard_mask,
                    )
                    speedup = (
                        float(raw_seconds / pruned_seconds)
                        if pruned_seconds > 0
                        else float("inf")
                    )
                    masks = {
                        "unrestricted": ~np.eye(truth.shape[0], dtype=bool),
                        "naive_hard": naive_mask,
                        "robust_hard": prior.robust_hard_mask,
                        "soft_prior": ~np.eye(truth.shape[0], dtype=bool),
                        "density_matched_random": random_mask,
                        "oracle_hard": truth,
                    }
                    for arm in ARMS:
                        recovery = edge_recovery(truth, discoveries[arm])
                        selected = recovery.true_positives + recovery.false_positives
                        fdp = recovery.false_positives / selected if selected else 0.0
                        rows.append(
                            {
                                "condition": condition.name,
                                "is_null": condition.null_graph,
                                "seed": int(seed),
                                "fold": fold,
                                "screen_episode_ids": ";".join(map(str, screen_ids)),
                                "test_episode_ids": ";".join(map(str, test_ids)),
                                "deadband": int(deadband),
                                "arm": arm,
                                "precision": recovery.precision,
                                "recall": recovery.recall,
                                "f1": _f1(recovery.precision, recovery.recall),
                                "false_positive_rate": recovery.false_positive_rate,
                                "false_discovery_proportion": fdp,
                                "orientation_accuracy": recovery.orientation_accuracy,
                                "true_positives": recovery.true_positives,
                                "false_positives": recovery.false_positives,
                                "false_negatives": recovery.false_negatives,
                                "discoveries": selected,
                                "truth_edges": int(np.count_nonzero(truth)),
                                "candidate_density": _candidate_density(masks[arm]),
                                "mask_recall": _mask_recall(masks[arm], truth),
                                "raw_fit_seconds": raw_seconds,
                                "robust_pruned_fit_seconds": pruned_seconds,
                                "robust_pruning_speedup": speedup,
                                "robust_candidate_density": _candidate_density(
                                    prior.robust_hard_mask
                                ),
                                "robust_decisive_votes": int(
                                    np.count_nonzero(prior.decisive_episode_counts)
                                ),
                            }
                        )
            if progress:
                print(f"completed condition={condition.name} seed={seed}", flush=True)
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate seed/fold rows without pooling distinct deadbands or arms."""

    metrics = (
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_discovery_proportion",
        "orientation_accuracy",
        "candidate_density",
        "mask_recall",
        "raw_fit_seconds",
        "robust_pruned_fit_seconds",
        "robust_pruning_speedup",
        "robust_candidate_density",
    )
    groups: dict[tuple[str, bool, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            bool(row["is_null"]),
            int(row["deadband"]),
            str(row["arm"]),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (condition, is_null, deadband, arm), members in sorted(groups.items()):
        item: dict[str, Any] = {
            "condition": condition,
            "is_null": is_null,
            "deadband": deadband,
            "arm": arm,
            "n_rows": len(members),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_median"] = float(np.median(values))
        summary.append(item)
    return summary


def paired_contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute arm-minus-unrestricted contrasts within each held-out fit."""

    indexed = {
        (
            row["condition"],
            row["seed"],
            row["fold"],
            row["deadband"],
            row["arm"],
        ): row
        for row in rows
    }
    metrics = ("f1", "recall", "false_positive_rate", "orientation_accuracy")
    contrasts = []
    group_keys = sorted(
        {
            (row["condition"], row["seed"], row["fold"], row["deadband"])
            for row in rows
        }
    )
    for condition, seed, fold, deadband in group_keys:
        baseline = indexed[(condition, seed, fold, deadband, "unrestricted")]
        for arm in ARMS[1:]:
            row = indexed[(condition, seed, fold, deadband, arm)]
            contrast: dict[str, Any] = {
                "condition": condition,
                "seed": seed,
                "fold": fold,
                "deadband": deadband,
                "arm": arm,
            }
            for metric in metrics:
                contrast[f"delta_{metric}"] = float(row[metric]) - float(
                    baseline[metric]
                )
            contrasts.append(contrast)
    return contrasts


def evaluate_gates(
    rows: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the prespecified feasibility thresholds to held-out results."""

    nonnull = [row for row in rows if not row["is_null"]]
    null_rows = [row for row in rows if row["is_null"]]
    nonnull_contrasts = [
        row for row in contrasts if not str(row["condition"]).startswith("null_")
    ]

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    f1_delta = {
        arm: mean(
            [
                float(row["delta_f1"])
                for row in nonnull_contrasts
                if row["arm"] == arm
            ]
        )
        for arm in ("robust_hard", "soft_prior")
    }
    clean_hard_recall_loss = -mean(
        [
            float(row["delta_recall"])
            for row in nonnull_contrasts
            if row["arm"] == "robust_hard" and row["condition"] == "clean"
        ]
    )
    hard_mask_recall = {
        condition: mean(
            [
                float(row["mask_recall"])
                for row in nonnull
                if row["arm"] == "robust_hard" and row["condition"] == condition
            ]
        )
        for condition in ("clean", "lowrate_shared")
    }
    null_fpr_delta = mean(
        [
            float(row["delta_false_positive_rate"])
            for row in contrasts
            if str(row["condition"]).startswith("null_")
            and row["arm"] in {"robust_hard", "soft_prior"}
        ]
    )
    density = mean(
        [
            float(row["robust_candidate_density"])
            for row in nonnull
            if row["arm"] == "robust_hard"
        ]
    )
    speedup = float(
        np.median(
            [
                float(row["robust_pruning_speedup"])
                for row in nonnull
                if row["arm"] == "robust_hard"
            ]
        )
    )
    hard_pass = (
        f1_delta["robust_hard"] >= 0.05
        and clean_hard_recall_loss <= 0.05
        and hard_mask_recall["clean"] >= 0.95
        and hard_mask_recall["lowrate_shared"] >= 0.90
    )
    soft_pass = f1_delta["soft_prior"] >= 0.05
    safety_pass = null_fpr_delta <= 0.02
    efficiency_pass = density <= 0.60 and speedup >= 1.67
    if safety_pass and hard_pass:
        recommendation = "proceed_robust_hard_and_soft"
    elif safety_pass and soft_pass:
        recommendation = "proceed_soft_only"
    else:
        recommendation = "no_go_for_causal_restriction"
    return {
        "thresholds": {
            "minimum_f1_gain": 0.05,
            "maximum_clean_recall_loss": 0.05,
            "minimum_clean_mask_recall": 0.95,
            "minimum_stress_mask_recall": 0.90,
            "maximum_null_fpr_increase": 0.02,
            "maximum_candidate_density": 0.60,
            "minimum_pruning_speedup": 1.67,
        },
        "observed": {
            "mean_f1_delta": f1_delta,
            "clean_hard_recall_loss": clean_hard_recall_loss,
            "hard_mask_recall": hard_mask_recall,
            "mean_null_fpr_delta": null_fpr_delta,
            "mean_robust_candidate_density": density,
            "median_pruning_speedup": speedup,
        },
        "passes": {
            "robust_hard": hard_pass,
            "soft_prior": soft_pass,
            "null_safety": safety_pass,
            "efficiency": efficiency_pass,
        },
        "recommendation": recommendation,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(summary: Sequence[dict[str, Any]], gates: dict[str, Any]) -> str:
    lines = [
        "# Robust/soft temporal-prior ablation",
        "",
        "Temporal priors were learned from four rise episodes and evaluated on four disjoint episodes, with the folds swapped. All scientific arms use the same held-out c-GC p-value matrix; the mask-aware refit is timing-only.",
        "",
        f"Recommendation: `{gates['recommendation']}`.",
        "",
        "## Prespecified gate observations",
        "",
        "```json",
        json.dumps(gates["observed"], indent=2, sort_keys=True),
        "```",
        "",
        "## Mean recovery by condition, deadband, and arm",
        "",
        "| condition | deadband | arm | F1 | recall | FPR | mask recall | candidate density |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {condition} | {deadband} | {arm} | {f1_mean:.3f} | "
            "{recall_mean:.3f} | {false_positive_rate_mean:.3f} | "
            "{mask_recall_mean:.3f} | {candidate_density_mean:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Lead/lag order is evaluated here as an external temporal prior, not as proof of causation. Masked-out directions are untested hypotheses, not established absences. Deadband 0 retains one-frame ordering; deadband 1 treats one-frame differences as ambiguous and is expected to be conservative at the simulated one-frame propagation delay.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_ints, default=tuple(range(1, 9)))
    parser.add_argument("--deadbands", type=_parse_ints, default=(0, 1))
    parser.add_argument("--n-steps", type=int, default=640)
    parser.add_argument("--n-surrogates", type=int, default=499)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-run-samples", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_ablation(
        seeds=args.seeds,
        conditions=CONDITIONS,
        deadbands=args.deadbands,
        n_steps=args.n_steps,
        n_surrogates=args.n_surrogates,
        alpha=args.alpha,
        tolerance=args.tolerance,
        min_run_samples=args.min_run_samples,
        progress=True,
    )
    summary = summarize_rows(rows)
    contrasts = paired_contrasts(rows)
    gates = evaluate_gates(rows, contrasts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "ablation_rows.csv", rows)
    _write_csv(args.output_dir / "ablation_summary.csv", summary)
    _write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    payload = {
        "config": {
            "seeds": list(args.seeds),
            "deadbands": list(args.deadbands),
            "n_steps": args.n_steps,
            "n_surrogates": args.n_surrogates,
            "alpha": args.alpha,
            "tolerance": args.tolerance,
            "min_run_samples": args.min_run_samples,
            "conditions": [asdict(condition) for condition in CONDITIONS],
            "screening": {
                "episode_crossfit": True,
                "minimum_decisive_support": 2,
                "consistency_threshold": 0.75,
                "beta_prior_concentration": 1.0,
            },
            "inference": {
                "method": "cgc",
                "event_mode": "physical",
                "tau": 1,
                "n_pasts": 3,
                "finite_sample_plus_one": True,
            },
        },
        "gates": gates,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _report_markdown(summary, gates), encoding="utf-8"
    )
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
