import importlib.util
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "run_fast_causal_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("run_fast_causal_baselines", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


class FastCausalBaselineScriptTests(unittest.TestCase):
    def test_load_case_records_uses_declared_case_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            traces_c = np.arange(80, dtype=float).reshape(4, 20)
            traces_d = traces_c / 2
            for filename, payload in (
                ("dff_corrected_dict.pkl", {(3, 1): traces_c}),
                ("dff_smoothed_dict.pkl", {(3, 1): traces_d}),
                ("middle_removed_dict.pkl", {(3, 1): 2}),
            ):
                with (data_dir / filename).open("wb") as handle:
                    pickle.dump(payload, handle)

            records = script.load_case_records(
                data_dir,
                cases=("C", "D"),
                recordings=("F3T1",),
            )

        self.assertEqual([record["fluo_type"] for record in records], ["C", "D"])
        np.testing.assert_array_equal(records[0]["traces"], traces_c)
        np.testing.assert_array_equal(records[1]["traces"], traces_d)
        self.assertNotEqual(records[0]["input_digest"], records[1]["input_digest"])

    def test_representation_map_retains_signed_samples(self) -> None:
        traces = np.array([[0.0, 1.0, 0.7, 0.1], [0.0, 0.2, 0.5, 0.3]])

        represented = script.representation_map(traces)

        self.assertEqual(set(represented), set(script.REPRESENTATIONS))
        self.assertLess(float(np.min(represented["signed_difference"])), 0.0)
        self.assertLess(float(np.min(represented["signed_innovation"])), 0.0)
        self.assertTrue(np.all(represented["rise"] >= 0.0))

    def test_episodic_dataset_is_deterministic_and_has_truth(self) -> None:
        first = script.episodic_dataset(seed=4, n_steps=400)
        second = script.episodic_dataset(seed=4, n_steps=400)

        np.testing.assert_array_equal(first["truth"], second["truth"])
        np.testing.assert_allclose(first["traces"], second["traces"])
        self.assertGreater(np.count_nonzero(first["truth"]), 0)
        self.assertEqual(first["input_digest"], second["input_digest"])

    def test_dynamic_extension_datasets_are_matched_and_deterministic(self) -> None:
        first = script.dynamic_extension_datasets(seed=4, n_steps=360)
        second = script.dynamic_extension_datasets(seed=4, n_steps=360)

        self.assertEqual(len(first), 3)
        self.assertEqual(
            {dataset["condition"] for dataset in first},
            {
                "dynamic_a_noncausal_fall",
                "dynamic_b_hybrid_causal_fall_overlap",
                "dynamic_c_hybrid_causal_fall_kinetic_misspecification",
            },
        )
        for observed, repeated in zip(first, second):
            self.assertEqual(observed["input_digest"], repeated["input_digest"])
            np.testing.assert_allclose(observed["traces"], repeated["traces"])
            self.assertEqual(set(observed["truth_targets"]), {"union", "rise", "fall"})
            np.testing.assert_array_equal(
                observed["truth_targets"]["union"],
                observed["truth_targets"]["rise"] | observed["truth_targets"]["fall"],
            )

    def test_confounding_datasets_include_an_unobserved_common_driver(self) -> None:
        first = script.confounding_datasets(seed=8, n_steps=300)
        second = script.confounding_datasets(seed=8, n_steps=300)

        self.assertEqual(
            {dataset["condition"] for dataset in first},
            {"native", "shared_observation_noise", "latent_common_driver"},
        )
        latent = next(
            dataset
            for dataset in first
            if dataset["condition"] == "latent_common_driver"
        )
        self.assertEqual(latent["traces"].shape, (5, 300))
        self.assertEqual(latent["latent_driver_targets"], "1,3")
        for observed, repeated in zip(first, second):
            np.testing.assert_allclose(observed["traces"], repeated["traces"])
            self.assertEqual(observed["input_digest"], repeated["input_digest"])

    def test_var_granger_fit_returns_auditable_arrays(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(3, 300))
        values[1, 1:] += values[0, :-1]

        adjacency, metadata, arrays = script._fit(
            "var-granger", values, max_lag=1, alpha=0.05
        )

        self.assertEqual(adjacency.shape, (3, 3))
        self.assertIn("p_values", arrays)
        self.assertIn("coefficients", arrays)
        self.assertIn("companion_spectral_radius", arrays)
        self.assertIn("var_stable", metadata)
        self.assertEqual(metadata["projection"], "none")

    def test_cgc_fit_uses_full_axis_and_finite_sample_permutations(self) -> None:
        result = type(
            "Result",
            (),
            {
                "adjacency": np.array([[False, True], [False, False]]),
                "scores": np.array([[0.0, 0.8], [0.1, 0.0]]),
                "retained_scores": np.array([[0.0, 0.8], [0.0, 0.0]]),
                "p_values": np.array([[1.0, 0.01], [0.8, 1.0]]),
                "best_lags": np.array([[0, 1], [1, 0]]),
            },
        )()
        with patch.object(script, "CausalisedGC") as estimator:
            estimator.return_value.fit.return_value = result
            adjacency, metadata, arrays = script._fit(
                "cgc-star",
                np.ones((2, 20)),
                max_lag=1,
                alpha=0.05,
                n_surrogates=6000,
                random_state=17,
                n_pasts=3,
                tau=1,
            )

        self.assertTrue(adjacency[0, 1])
        self.assertIn("retained_scores", arrays)
        self.assertEqual(metadata["n_surrogates"], 6000)
        self.assertEqual(
            estimator.call_args.kwargs,
            {
                "max_lag": 1,
                "n_surrogates": 6000,
                "alpha": 0.05,
                "fdr": True,
                "event_mode": "physical",
                "method": "cgc-star",
                "simulation": False,
                "random_state": 17,
                "n_pasts": 3,
                "tau": 1,
            },
        )
        self.assertEqual(metadata["n_pasts"], 3)
        self.assertEqual(metadata["tested_lag"], 1)
        self.assertIn("causal structure relation", metadata["assumptions"])

    def test_pcmciplus_fit_uses_requested_alpha_and_bh(self) -> None:
        graph = np.full((2, 2, 2), "", dtype=object)
        result = type(
            "Result",
            (),
            {
                "graph": graph,
                "p_matrix": np.ones((2, 2, 2)),
                "val_matrix": np.zeros((2, 2, 2)),
                "cond_ind_test": "ParCorr",
                "assumptions": (),
                "lagged_adjacency": lambda self: np.zeros((2, 2), dtype=bool),
            },
        )()
        with patch.object(script, "PCMCIPlusAdapter") as adapter:
            adapter.return_value.fit.return_value = result
            script._fit("pcmciplus", np.ones((2, 20)), max_lag=2, alpha=0.01)

        self.assertEqual(
            adapter.call_args.kwargs["run_kwargs"],
            {
                "tau_min": 0,
                "pc_alpha": 0.01,
                "fdr_method": "fdr_bh",
            },
        )

    def test_lpcmci_fit_preserves_pag_and_declares_lossy_skeleton(self) -> None:
        graph = np.full((2, 2, 2), "", dtype=object)
        result = type(
            "Result",
            (),
            {
                "graph": graph,
                "p_matrix": np.ones((2, 2, 2)),
                "val_matrix": np.zeros((2, 2, 2)),
                "cond_ind_test": "ParCorr",
                "assumptions": ("latent confounding allowed",),
                "lossy_lagged_skeleton": lambda self: np.zeros((2, 2), dtype=bool),
            },
        )()
        with patch.object(script, "LPCMCIAdapter") as adapter:
            adapter.return_value.fit.return_value = result
            adjacency, metadata, arrays = script._fit(
                "lpcmci", np.ones((2, 20)), max_lag=1, alpha=0.05
            )

        self.assertFalse(np.any(adjacency))
        self.assertIn("graph", arrays)
        self.assertEqual(metadata["estimand"], "lagged_skeleton")
        self.assertEqual(metadata["projection"], "lossy_lagged_pag_skeleton")

    def test_cli_separates_cgc_conditioning_depth_from_tested_lag(self) -> None:
        argv = [
            "run_fast_causal_baselines.py",
            "--dataset",
            "simulations",
            "--algorithm",
            "cgc",
            "--output-dir",
            "unused",
            "--max-lag",
            "1",
            "--cgc-tau",
            "1",
            "--cgc-depths",
            "1,2,3",
            "--n-jobs",
            "4",
        ]
        with patch("sys.argv", argv):
            args = script.parse_args()

        self.assertEqual(args.cgc_tau, 1)
        self.assertEqual(args.cgc_depths, (1, 2, 3))
        self.assertEqual(args.n_jobs, 4)

    def test_parallel_map_preserves_task_order(self) -> None:
        tasks = (
            script.SimulationFitTask(0, "first", {}, "rise", None),
            script.SimulationFitTask(1, "second", {}, "fall", None),
        )

        def worker(task):
            return task.unit, [{"unit": task.unit}]

        results = list(
            script._iter_simulation_fit_results(worker, tasks, n_jobs=1)
        )

        self.assertEqual([unit for unit, _ in results], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
