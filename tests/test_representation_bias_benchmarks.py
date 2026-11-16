import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load(
    "run_representation_bias_benchmarks",
    "examples/run_representation_bias_benchmarks.py",
)
analysis = _load(
    "analyze_representation_bias_benchmarks",
    "examples/analyze_representation_bias_benchmarks.py",
)


class RepresentationBiasBenchmarkTests(unittest.TestCase):
    def test_cross_falsification_is_not_treated_as_a_rise_graph_contrast(self) -> None:
        self.assertEqual(
            analysis.CROSS_ROLES,
            ("rise_graph_context", "fall_residual_to_rise_falsification"),
        )

    def test_validation_rejects_unexpected_role_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected test_role labels"):
            analysis._assert_labels(
                [{"test_role": "unregistered_role"}],
                "test_role",
                analysis.CROSS_ROLES,
            )

    def test_signed_effect_conditions_are_deterministic_and_declared(self) -> None:
        for condition in runner.SIGNED_CONDITIONS:
            first = runner.signed_effect_dataset(condition, seed=7, n_steps=200)
            second = runner.signed_effect_dataset(condition, seed=7, n_steps=200)
            with self.subTest(condition=condition):
                np.testing.assert_allclose(first["traces"], second["traces"])
                np.testing.assert_array_equal(first["truth"], second["truth"])
                self.assertEqual(first["trace_digest"], second["trace_digest"])
                self.assertGreater(np.count_nonzero(first["excitatory_mask"]), 0)

        mixed = runner.signed_effect_dataset(
            "mixed_excitation_inhibition", seed=7, n_steps=200
        )
        rebound = runner.signed_effect_dataset(
            "post_inhibitory_rebound", seed=7, n_steps=200
        )
        self.assertGreater(np.count_nonzero(mixed["inhibitory_mask"]), 0)
        self.assertEqual(mixed["max_lag"], 1)
        self.assertGreater(np.count_nonzero(rebound["rebound_mask"]), 0)
        self.assertEqual(rebound["max_lag"], 2)
        self.assertTrue(np.all(rebound["lag2_weights"][rebound["rebound_mask"]] > 0.0))
        self.assertTrue(np.all(rebound["lag1_weights"][rebound["rebound_mask"]] < 0.0))

    def test_signed_and_rectified_innovations_are_exact_pairs(self) -> None:
        dataset = runner.signed_effect_dataset(
            "post_inhibitory_rebound", seed=3, n_steps=240
        )
        represented = runner._signed_representations(dataset)

        np.testing.assert_allclose(
            represented["deconvolved_rectified"],
            np.maximum(represented["signed_innovation"], 0.0),
        )
        self.assertGreater(np.count_nonzero(represented["signed_innovation"] < 0.0), 0)
        self.assertEqual(
            np.count_nonzero(represented["deconvolved_rectified"] < 0.0), 0
        )

    def test_cross_fit_rejects_unmatched_source_and_target_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal shape"):
            runner._fit_cross_graph(
                "var-granger",
                np.zeros((2, 100)),
                np.zeros((3, 100)),
                alpha=0.05,
                n_surrogates=9,
                random_state=1,
            )

    def test_var_cross_fit_returns_only_the_source_target_block(self) -> None:
        rng = np.random.default_rng(4)
        sources = rng.normal(size=(3, 180))
        outcomes = rng.normal(size=(3, 180))

        adjacency, metadata, arrays = runner._fit_cross_graph(
            "var-granger",
            sources,
            outcomes,
            alpha=0.05,
            n_surrogates=9,
            random_state=1,
        )

        self.assertEqual(adjacency.shape, (3, 3))
        self.assertEqual(arrays["combined_p_values"].shape, (6, 6))
        self.assertFalse(np.any(np.diag(adjacency)))
        self.assertEqual(metadata["cross_fit"], "joint_2N_system_cross_block_bh_family")

    def test_motor_contrasts_do_not_assign_inferential_p_values(self) -> None:
        rows = [
            {
                "algorithm": "cgc",
                "condition": "observed",
                "case": "C",
                "recording": "F3T1",
                "representation": "deconvolved_rectified",
                "edge_density": "0.1",
            },
            {
                "algorithm": "cgc",
                "condition": "observed",
                "case": "C",
                "recording": "F3T1",
                "representation": "signed_innovation",
                "edge_density": "0.2",
            },
        ]

        contrasts = analysis._paired_contrasts(
            rows,
            unit_keys=("case", "recording"),
            group_keys=("algorithm", "condition"),
            label_key="representation",
            reference="deconvolved_rectified",
            comparator="signed_innovation",
            metrics=("edge_density",),
            inferential=False,
        )

        self.assertEqual(len(contrasts), 1)
        self.assertAlmostEqual(contrasts[0]["mean_delta"], 0.1)
        self.assertIsNone(contrasts[0]["p_value"])
        self.assertIsNone(contrasts[0]["holm_p_value"])

    def test_publication_analysis_rejects_underresolved_cgc_surrogates(self) -> None:
        row = {
            "algorithm": "cgc",
            "n_rois": "5",
            "alpha": "0.05",
            "n_surrogates": "9",
        }

        with self.assertRaisesRegex(ValueError, "first-rank BH resolution"):
            analysis._assert_permutation_resolution([row])
        analysis._assert_permutation_resolution([row], allow_underresolved=True)

    def test_checkpoint_config_tracks_random_seed_and_empirical_inputs(self) -> None:
        args = SimpleNamespace(
            dataset="motorneurons",
            algorithm="cgc",
            n_seeds=20,
            seed_start=1,
            n_steps=1500,
            n_surrogates=6000,
            alpha=0.05,
            seed=11,
            signed_conditions=runner.SIGNED_CONDITIONS,
            cases=("C",),
            recordings=("F3T1",),
            case_data_dir=Path("data/motoneurons"),
        )
        first_records = [
            {"fluo_type": "C", "recording": "F3T1", "input_digest": "first"}
        ]
        second_records = [
            {"fluo_type": "C", "recording": "F3T1", "input_digest": "second"}
        ]

        first = runner._checkpoint_config(args, first_records)
        args.seed = 12
        changed_seed = runner._checkpoint_config(args, first_records)
        changed_input = runner._checkpoint_config(args, second_records)

        self.assertNotEqual(first["random_seed"], changed_seed["random_seed"])
        self.assertNotEqual(first["input_fingerprint"], changed_input["input_fingerprint"])
        self.assertEqual(first["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
