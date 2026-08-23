import json
import tempfile
import unittest
from pathlib import Path

from calcium_transient_rising_flank.checkpointing import JsonUnitCheckpointStore


class JsonUnitCheckpointStoreTests(unittest.TestCase):
    def test_initialize_rejects_resume_when_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            original = JsonUnitCheckpointStore(
                output_dir,
                "temporal_resolvability",
                {"seed": 1, "n_steps": 64},
            )
            original.initialize(resume=False)

            mismatched = JsonUnitCheckpointStore(
                output_dir,
                "temporal_resolvability",
                {"seed": 2, "n_steps": 64},
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot resume with settings that differ",
            ):
                mismatched.initialize(resume=True)

    def test_load_rows_ignores_corrupt_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonUnitCheckpointStore(
                Path(directory),
                "temporal_resolvability",
                {"seed": 1},
            )
            store.initialize(resume=False)
            unit_path = store._unit_path("regime|delay=1|seed=1")
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text("{not-json", encoding="utf-8")

            self.assertIsNone(store.load_rows("regime|delay=1|seed=1"))

    def test_load_rows_keeps_last_complete_unit_when_temporary_file_remains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonUnitCheckpointStore(
                Path(directory),
                "motorneuron_temporal_screen",
                {"case": "D"},
            )
            store.initialize(resume=False)
            expected = [{"recording": "F1T1", "status": "ok"}]
            store.save_rows("D|F1T1", expected)

            unit_path = store._unit_path("D|F1T1")
            stray_temporary = unit_path.with_name(f".{unit_path.name}.999.tmp")
            stray_temporary.write_text(
                json.dumps({"status": "running", "rows": []}),
                encoding="utf-8",
            )

            self.assertEqual(store.load_rows("D|F1T1"), expected)

    def test_load_rows_ignores_checkpoint_with_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonUnitCheckpointStore(
                Path(directory),
                "empirical_stability",
                {"method": "cgc"},
            )
            store.initialize(resume=False)
            unit_path = store._unit_path("cgc|D|F1T1")
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "unit_id": "cgc|D|F1T1",
                        "config_digest": "wrong-digest",
                        "rows": [{"status": "ok"}],
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(store.load_rows("cgc|D|F1T1"))


if __name__ == "__main__":
    unittest.main()
