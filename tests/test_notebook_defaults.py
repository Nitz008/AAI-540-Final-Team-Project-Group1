"""Exercise every default notebook cell with no data and AWS APIs prohibited."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookDefaultsTest(unittest.TestCase):
    def test_default_run_is_complete_and_never_calls_aws(self):
        folder = ROOT / ".local_checks"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            directory = Path(temporary)
            config = json.loads((ROOT / "configs/sagemaker.example.json").read_text())
            config["data"]["processed_dir"] = str(directory / "absent-data")
            config["reports_dir"] = str(directory / "reports")
            config_path = directory / "local.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            notebook = json.loads((ROOT / "notebooks/02_sagemaker_training_evaluation.ipynb").read_text(encoding="utf-8"))
            namespace = {"__name__": "__notebook_check__"}
            previous_cwd = Path.cwd()
            try:
                os.chdir(ROOT)
                with patch.dict(os.environ, {"AAI540_CONFIG": str(config_path)}), \
                     patch("botocore.client.BaseClient._make_api_call", side_effect=AssertionError("AWS called by default notebook")) as aws_api, \
                     redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"), namespace)
                    aws_api.assert_not_called()
                self.assertFalse(namespace["RUN_AWS"])
                self.assertIsNone(namespace["baseline_artifacts"])
                self.assertIsNone(namespace["comparison_artifacts"])
                self.assertIsNone(namespace["final_report"])
                self.assertIsNone(namespace["policy_path"])
                self.assertFalse(namespace["RUN_FINAL_TEST"])
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
