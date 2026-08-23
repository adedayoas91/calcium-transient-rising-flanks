import unittest
from collections import namedtuple
from unittest.mock import patch

import numpy as np

from calcium_transient_rising_flank.baselines import (
    LPCMCIAdapter,
    OASISDeconvolver,
    project_lossy_lagged_pag_skeleton,
)
from calcium_transient_rising_flank.sensitivity import PartialAncestralGraph


class BaselineAdapterTests(unittest.TestCase):
    def test_oasis_adapter_preserves_roi_time_orientation_and_backend_outputs(
        self,
    ) -> None:
        traces = np.array([[0.0, 1.0, 0.8], [0.2, 0.1, 0.4]])
        calls: list[np.ndarray] = []
        result_type = namedtuple("DeconvolveResult", ["c", "s", "b", "g", "lam"])

        def fake_backend(row: np.ndarray, **_: object) -> object:
            calls.append(row.copy())
            if np.allclose(row, traces[0]):
                return result_type(
                    c=row + 1.0,
                    s=np.array([0.0, 1.0, 0.0]),
                    b=0.25,
                    g=(0.95,),
                    lam=0.01,
                )
            return (
                row + 2.0,
                np.array([1.0, 0.0, 1.0]),
                0.5,
                np.array([1.6, -0.64]),
                0.02,
            )

        result = OASISDeconvolver(backend=fake_backend).fit(traces)

        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0], traces[0])
        np.testing.assert_allclose(calls[1], traces[1])
        self.assertEqual(result.denoised.shape, traces.shape)
        self.assertEqual(result.spikes.shape, traces.shape)
        np.testing.assert_allclose(result.baseline, [0.25, 0.5])
        self.assertEqual(result.ar_params, ((0.95,), (1.6, -0.64)))
        np.testing.assert_allclose(result.penalty_lambda, [0.01, 0.02])

    def test_oasis_adapter_lazy_import_error_mentions_optional_extra(self) -> None:
        with patch(
            "calcium_transient_rising_flank.baselines.importlib.import_module",
            side_effect=ImportError("missing optional dependency"),
        ):
            with self.assertRaisesRegex(ImportError, "deconvolution"):
                OASISDeconvolver().fit(np.ones((2, 4)))

    def test_lpcmci_adapter_preserves_raw_pag_and_transposes_for_tigramite(
        self,
    ) -> None:
        traces = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        captured: dict[str, object] = {}
        graph = np.full((2, 2, 3), "", dtype=object)
        graph[0, 1, 0] = "o-o"
        graph[0, 1, 1] = "-->"
        graph[1, 0, 2] = "<->"
        p_matrix = np.zeros((2, 2, 3), dtype=float)
        val_matrix = np.ones((2, 2, 3), dtype=float)

        def dataframe_factory(data: np.ndarray) -> dict[str, np.ndarray]:
            captured["dataframe_input"] = data.copy()
            return {"data": data}

        class FakeCondIndTest:
            pass

        class FakeEstimator:
            def __init__(self, dataframe: object, cond_ind_test: object, verbosity: int):
                captured["dataframe"] = dataframe
                captured["cond_ind_test"] = cond_ind_test
                captured["verbosity"] = verbosity

            def run_lpcmci(self, **kwargs: object) -> dict[str, np.ndarray]:
                captured["run_kwargs"] = dict(kwargs)
                return {
                    "graph": graph,
                    "p_matrix": p_matrix,
                    "val_matrix": val_matrix,
                }

        result = LPCMCIAdapter(
            tau_max=2,
            run_kwargs={"pc_alpha": 0.1},
            verbosity=3,
            dataframe_factory=dataframe_factory,
            cond_ind_test_factory=FakeCondIndTest,
            estimator_factory=FakeEstimator,
        ).fit(traces)

        self.assertIsInstance(result, PartialAncestralGraph)
        np.testing.assert_allclose(captured["dataframe_input"], traces.T)
        self.assertEqual(captured["run_kwargs"], {"tau_max": 2, "pc_alpha": 0.1})
        self.assertEqual(result.cond_ind_test, "FakeCondIndTest")
        self.assertIs(result.raw_pag, result.graph)
        self.assertEqual(result.endpoint_marks.shape, (2, 2, 3))
        self.assertEqual(result.graph[0, 1, 1], "-->")
        self.assertEqual(result.graph[1, 0, 2], "<->")
        np.testing.assert_array_equal(
            result.lossy_lagged_skeleton(),
            np.array([[False, True], [True, False]], dtype=bool),
        )
        np.testing.assert_array_equal(
            project_lossy_lagged_pag_skeleton(result.graph),
            np.array([[False, True], [True, False]], dtype=bool),
        )

    def test_lpcmci_adapter_lazy_import_error_mentions_optional_extra(self) -> None:
        with patch(
            "calcium_transient_rising_flank.baselines.importlib.import_module",
            side_effect=ImportError("missing optional dependency"),
        ):
            with self.assertRaisesRegex(ImportError, "pag"):
                LPCMCIAdapter(tau_max=1).fit(np.ones((2, 4)))

    def test_lpcmci_adapter_rejects_duplicate_tau_max(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not override tau_max"):
            LPCMCIAdapter(tau_max=1, run_kwargs={"tau_max": 2}).fit(
                np.ones((2, 4))
            )


if __name__ == "__main__":
    unittest.main()
