import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).parents[1] / "examples" / "run_reviewer_revision_campaign.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_reviewer_revision_campaign", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reviewer campaign script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewerRevisionCampaignScriptTests(unittest.TestCase):
    def test_all_long_running_stage_commands_include_resume(self) -> None:
        script = _load_script_module()
        stages = script.build_stages(
            python="python",
            output_root=Path("outputs/example"),
            seeds="1,2",
            dynamic_n_seeds=2,
            n_surrogates=10,
        )
        self.assertEqual(tuple(stage.name for stage in stages), script.CAMPAIGN_STAGES)
        for stage in stages:
            self.assertIn("--resume", stage.command)

    def test_only_complete_summary_is_skippable(self) -> None:
        script = _load_script_module()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"status": "running"}))
            self.assertFalse(script._summary_is_complete(path))
            path.write_text(json.dumps({"status": "complete"}))
            self.assertTrue(script._summary_is_complete(path))

    def test_stage_parser_rejects_unknown_names(self) -> None:
        script = _load_script_module()
        with self.assertRaises(Exception):
            script._parse_stage_names("not-a-stage")


if __name__ == "__main__":
    unittest.main()
