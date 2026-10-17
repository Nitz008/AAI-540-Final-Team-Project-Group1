"""Run the complete local notebook on synthetic data, including its final test."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modeling_fixtures import create_handoff

ROOT = Path(__file__).resolve().parents[1]


class NotebookModelingTest(unittest.TestCase):
    def test_baseline_policy_comparison_and_explicit_final_test(self):
        (ROOT / ".local_checks").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".local_checks") as temporary:
            directory = Path(temporary)
            config = json.loads((ROOT / "configs/sagemaker.example.json").read_text())
            config["data"].update(processed_dir=str(directory / "processed"), minimum_class_rows=2)
            config["reports_dir"] = str(directory / "reports")
            config["compute"].update(max_train_rows=0, n_jobs=1)
            config_path = directory / "local.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            create_handoff(config)
            notebook = json.loads((ROOT / "notebooks/02_sagemaker_training_evaluation.ipynb").read_text(encoding="utf-8"))
            namespace = {"__name__": "__synthetic_notebook_check__"}
            previous = Path.cwd()
            try:
                os.chdir(ROOT)
                with patch.dict(os.environ, {"AAI540_CONFIG": str(config_path), "MPLBACKEND": "Agg"}), \
                     patch("botocore.client.BaseClient._make_api_call", side_effect=AssertionError("AWS API attempted")), \
                     redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] != "code":
                            continue
                        source = "".join(cell["source"])
                        # A test-specific opt-in against a private synthetic fixture.
                        for name in ("RUN_BASELINE", "RUN_COMPARISON", "RUN_FINAL_TEST", "POLICY_READY"):
                            source = source.replace(f"{name} = False", f"{name} = True")
                        source = source.replace("MAX_FPR = None", "MAX_FPR = 0.1")
                        source = source.replace('DECISION_NOTE = ""', 'DECISION_NOTE = "Synthetic baseline review for code validation only."')
                        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
                self.assertEqual(namespace["baseline_artifacts"]["report"]["phase"], "baseline")
                comparison = namespace["comparison_artifacts"]["report"]
                self.assertEqual({row["name"] for row in comparison["family_comparison"]},
                                 {"logistic_regression", "random_forest", "xgboost"})
                self.assertEqual(namespace["final_report"]["phase"], "final_test")
                self.assertFalse(namespace["final_report"]["reserved_production_loaded"])
                final_directory = namespace["REPORT_DIR"] / "final_test"
                self.assertEqual(len(list(final_directory.glob("*.png"))), 3)
                self.assertIsNone(namespace["context"])
            finally:
                os.chdir(previous)
                import matplotlib.pyplot as plt
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
