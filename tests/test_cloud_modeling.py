"""Check account portability and train/test isolation without cloud mutations."""
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from malware_mlops.aws_workflow import (collect_training_artifacts, download_handoff, extract_training_artifacts,
    load_config, run_baseline_training, run_model_comparison, upload_training_inputs, validate_handoff)
from modeling_fixtures import create_handoff


class ModelingCloudTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".local_checks").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".local_checks")
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict(os.environ, {"AWS_EC2_METADATA_DISABLED": "true", "AWS_ACCESS_KEY_ID": "testing",
                                              "AWS_SECRET_ACCESS_KEY": "testing", "SAGEMAKER_SUPPRESS_V2_WARNING": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        blocked_api = patch("botocore.client.BaseClient._make_api_call", side_effect=AssertionError("AWS API called during offline test"))
        blocked_api.start()
        self.addCleanup(blocked_api.stop)
        self.folder = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/sagemaker.example.json", {
            "member": "tester", "run_id": "offline", "reports_dir": str(self.folder / "reports"),
            "source_dir": str(ROOT / "sagemaker_src"), "aws": {"bucket": "test-output-bucket"},
            "data": {"processed_dir": str(self.folder / "processed"), "minimum_class_rows": 2,
                     "handoff_s3_uri": "s3://test-handoff-bucket/processed"}})

    def test_default_validation_never_opens_test_or_reserve(self):
        _, splits = create_handoff(self.config)
        (splits / "test.csv").write_text("INVALID TEST MUST NOT BE READ YET")
        checked = validate_handoff(self.config)
        self.assertEqual(list(checked["splits"]), ["train", "validation"])
        self.assertNotIn("test", checked["paths"])
        with self.assertRaises(ValueError):
            validate_handoff(self.config, include_test=True)

    def test_exact_s3_transfers_and_explicit_test_opt_in(self):
        create_handoff(self.config)
        context = {"s3": Mock()}
        for action in (download_handoff, upload_training_inputs):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                action(self.config, context)
        self.assertEqual(context["s3"].mock_calls, [])
        self.config["cloud_execution"] = True
        download_handoff(self.config, context)
        keys = [call.args[1] for call in context["s3"].download_file.call_args_list]
        self.assertEqual(len(keys), 4)
        self.assertFalse(any(key.endswith("/test.csv") or "reserved_production" in key for key in keys))
        context["s3"].reset_mock()
        download_handoff(self.config, context, include_test=True)
        keys = [call.args[1] for call in context["s3"].download_file.call_args_list]
        self.assertTrue(any(key.endswith("/test.csv") for key in keys))
        self.assertFalse(any("reserved_production" in key for key in keys))
        uploaded = upload_training_inputs(self.config, context)
        self.assertEqual(set(uploaded), {"train", "validation", "schema"})

    def test_real_training_class_minimum_and_duplicate_ids(self):
        _, splits = create_handoff(self.config)
        self.config["data"]["minimum_class_rows"] = 10000
        with self.assertRaisesRegex(ValueError, "10,000"):
            validate_handoff(self.config)
        self.config["data"]["minimum_class_rows"] = 2
        path = splits / "test.csv"
        path.write_text(path.read_text().replace("synthetic-250", "synthetic-0"))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_handoff(self.config, include_test=True)

    def test_managed_baseline_sdk_contract_and_source_requirements(self):
        self.config["cloud_execution"] = True
        inputs = {name: f"s3://test-output-bucket/inputs/{name}" for name in ("train", "validation", "schema")}
        estimator = Mock()
        with patch("sagemaker.sklearn.estimator.SKLearn", return_value=estimator) as factory, \
             patch("malware_mlops.aws_workflow.resolve_image_uri", return_value="regional-image:1.4-2-py312"):
            run_baseline_training(self.config, {"role_arn": "arn:aws:iam::123456789012:role/TestRole",
                "region": "us-east-1", "sagemaker_session": Mock()}, inputs)
        arguments = factory.call_args.kwargs
        self.assertEqual(arguments["hyperparameters"]["phase"], "baseline")
        self.assertEqual(arguments["hyperparameters"]["n-jobs"], self.config["compute"]["n_jobs"])
        self.assertTrue((Path(arguments["source_dir"]) / "requirements.txt").exists())
        self.assertEqual(set(estimator.fit.call_args.kwargs["inputs"]), set(inputs))
        with self.assertRaisesRegex(ValueError, "excluded"):
            run_baseline_training(self.config, {}, dict(inputs, test="s3://forbidden/test"))

    def test_collect_refuses_foreign_job_and_archive_rejects_links(self):
        estimator = Mock()
        estimator.latest_training_job.name = "another-member-job"
        client = Mock()
        with self.assertRaisesRegex(ValueError, "current member"):
            collect_training_artifacts(self.config, {"sagemaker": client}, estimator, "baseline")
        client.describe_training_job.assert_not_called()
        archive = self.folder / "artifact.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            entry = tarfile.TarInfo("model.joblib")
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../outside"
            handle.addfile(entry)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            extract_training_artifacts(archive, self.folder / "extracted")


if __name__ == "__main__":
    unittest.main()
