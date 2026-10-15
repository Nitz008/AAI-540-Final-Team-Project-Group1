"""Serialize the real SDK DAG with all AWS traffic blocked and S3 uploads stubbed."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PipelineSDKTest(unittest.TestCase):
    def test_serialized_dag_preserves_holdout_gate_and_serving_contracts(self):
        from malware_mlops.aws_workflow import build_pipeline, load_config

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "AWS_EC2_METADATA_DISABLED": "true", "AWS_ACCESS_KEY_ID": "testing",
                "AWS_SECRET_ACCESS_KEY": "testing", "SAGEMAKER_SUPPRESS_V2_WARNING": "1",
            }))
            import boto3
            from sagemaker.fw_utils import UploadedCode
            from sagemaker.workflow.pipeline_context import PipelineSession

            folder = ROOT / ".local_checks"
            folder.mkdir(exist_ok=True)
            temporary = stack.enter_context(tempfile.TemporaryDirectory(dir=folder))
            config = load_config(ROOT / "configs/sagemaker.example.json", {
                "cloud_execution": True, "member": "sdk-test", "run_id": "offline",
                "aws": {"bucket": "offline-test-bucket"}, "reports_dir": temporary,
                "source_dir": str(ROOT / "sagemaker_src"),
            })
            boto = boto3.Session(region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing")
            session = PipelineSession(boto_session=boto, default_bucket="offline-test-bucket")
            context = {"pipeline_session": session, "role_arn": "arn:aws:iam::123456789012:role/OfflineRole", "region": "us-east-1"}
            inputs = {name: f"s3://offline-test-bucket/inputs/{name}/{filename}" for name, filename in {
                "train": "train.csv", "validation": "validation.csv", "test": "test.csv", "schema": "feature_schema.json",
            }.items()}
            for target in ("sagemaker.estimator.tar_and_upload_dir", "sagemaker.fw_utils.tar_and_upload_dir"):
                stack.enter_context(patch(target, side_effect=lambda *args, **kwargs: UploadedCode(
                    "s3://offline-test-bucket/code/sourcedir.tar.gz", Path(kwargs.get("script", "train.py")).name)))
            for method in ("upload", "upload_string_as_file_body"):
                stack.enter_context(patch(f"sagemaker.s3.S3Uploader.{method}", return_value="s3://offline-test-bucket/code/upload"))
            stack.enter_context(patch.object(session, "upload_data", return_value="s3://offline-test-bucket/code/upload"))
            stack.enter_context(patch.object(session, "default_bucket", return_value="offline-test-bucket"))
            stack.enter_context(patch.object(session, "expand_role", side_effect=lambda role: role))
            stack.enter_context(patch("sagemaker.telemetry.telemetry_logging._send_telemetry_request"))
            api = stack.enter_context(patch("botocore.client.BaseClient._make_api_call",
                                            side_effect=AssertionError("AWS API attempted during offline test")))
            definition = json.loads(build_pipeline(config, context, inputs).definition())
            api.assert_not_called()

            steps = {step["Name"]: step for step in definition["Steps"]}
            train = steps["TrainCandidates"]["Arguments"]
            self.assertEqual({channel["ChannelName"] for channel in train["InputDataConfig"]},
                             {"train", "validation", "schema"})
            self.assertNotIn("reserved_production", json.dumps(definition))
            self.assertTrue(train["AlgorithmSpecification"]["TrainingImage"].endswith(":1.4-2-py312"))
            processing = steps["EvaluateTemporalTest"]["Arguments"]
            self.assertEqual(processing["AppSpecification"]["ImageUri"], train["AlgorithmSpecification"]["TrainingImage"])
            gate = steps["QualityGate"]["Arguments"]
            self.assertEqual(len(gate["Conditions"]), 4)
            self.assertIn("ForceFail", json.dumps(gate["Conditions"]))
            self.assertEqual(gate["ElseSteps"][0]["Type"], "Fail")
            registration = next(step for step in gate["IfSteps"] if step["Type"] == "RegisterModel")["Arguments"]
            self.assertEqual(registration["ModelApprovalStatus"], "PendingManualApproval")
            serving = registration["InferenceSpecification"]["Containers"][0]
            self.assertEqual(serving["Environment"]["SAGEMAKER_PROGRAM"], "inference.py")
            self.assertEqual(serving["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"], "/opt/ml/model/code")
            repack = next(step for step in gate["IfSteps"] if "RepackModel" in step["Name"])
            self.assertEqual(repack["Type"], "Training")
            self.assertEqual(serving["ModelDataUrl"]["Get"], f"Steps.{repack['Name']}.ModelArtifacts.S3ModelArtifacts")
            self.assertEqual(repack["Arguments"]["AlgorithmSpecification"]["TrainingImage"], serving["Image"])
            self.assertEqual(serving["Image"], train["AlgorithmSpecification"]["TrainingImage"])
            self.assertIn("text/csv", registration["InferenceSpecification"]["SupportedResponseMIMETypes"])


if __name__ == "__main__":
    unittest.main()
