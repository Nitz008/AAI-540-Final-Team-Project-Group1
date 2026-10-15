"""Offline checks for data isolation, guardrails and real-metric publication."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malware_mlops.aws_workflow import (
    cleanup_owned_resources, download_handoff, ensure_feature_group, load_config,
    prepare_batch_input, publish_observability, upload_handoff, validate_handoff,
    validate_batch_output,
)
from malware_mlops.config import resource_name, run_prefix
from malware_mlops.pipeline import ensure_model_package_group


class CloudWorkflowTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".local_checks").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".local_checks")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/sagemaker.example.json", {
            "member": "tester", "run_id": "unit-test-1", "reports_dir": str(self.directory / "reports"),
            "data": {"processed_dir": str(self.directory / "processed")},
            "aws": {"bucket": "test-existing-bucket"},
        })

    def handoff(self):
        root = Path(self.config["data"]["processed_dir"])
        splits = root / "temporal_splits_v1"
        splits.mkdir(parents=True)
        schema = {"model_feature_columns": ["feature_a", "feature_b"], "fit_scope": "model_train_only",
                  "fit_end_week": "2024-06-23"}
        schema_bytes = json.dumps(schema).encode()
        (root / "feature_schema.json").write_bytes(schema_bytes)
        manifest = {"feature_schema": {"sha256": hashlib.sha256(schema_bytes).hexdigest()}, "splits": {}}
        for split, start, end, partition in [
            ("train", "2024-06-23", "2024-06-29", "train"),
            ("validation", "2024-07-01", "2024-07-07", "train"),
            ("test", "2024-09-22", "2024-09-28", "test"),
        ]:
            with (splits / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["record_id", "label", "source_partition", "source_week_start", "source_week_end", "feature_a", "feature_b"])
                for label in (0, 1):
                    writer.writerow([f"{split}-{label}", label, partition, start, end, label + 0.5, label - 0.5])
            manifest["splits"][split] = {"row_count": 2}
        (splits / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # A malformed reserve proves this file is never opened by the ML workflow.
        (splits / "reserved_production.csv").write_text("DO NOT READ", encoding="utf-8")
        return root, splits

    def test_cloud_is_disabled_and_aws_clients_are_not_called(self):
        context = {"s3": Mock(), "sagemaker": Mock(), "cloudwatch": Mock()}
        for function, extra in [(upload_handoff, []), (download_handoff, []),
                                (ensure_feature_group, [{}]), (publish_observability, [{}])]:
            with self.subTest(function=function.__name__), self.assertRaisesRegex(RuntimeError, "disabled"):
                function(self.config, context, *extra)
        for client in context.values():
            self.assertEqual(client.mock_calls, [])

    def test_validate_and_upload_exclude_production_reserve(self):
        self.handoff()
        checked = validate_handoff(self.config)
        self.assertEqual(list(checked["splits"]), ["train", "validation", "test"])
        self.config["cloud_execution"] = True
        context = {"s3": Mock()}
        uploaded = upload_handoff(self.config, context, checked)
        self.assertEqual(context["s3"].upload_file.call_count, 5)
        self.assertNotIn("reserved_production", repr(context["s3"].mock_calls))
        self.assertIn("/tester/unit-test-1/", uploaded["train"])

    def test_download_is_exact_named_objects(self):
        self.config["cloud_execution"] = True
        self.config["data"]["handoff_s3_uri"] = "s3://existing-source/processed"
        context = {"s3": Mock()}
        download_handoff(self.config, context)
        self.assertEqual(context["s3"].download_file.call_count, 5)
        self.assertNotIn("reserved_production", repr(context["s3"].mock_calls))

    def test_overlapping_ids_rejected(self):
        _, folder = self.handoff()
        path = folder / "test.csv"
        path.write_text(path.read_text().replace("test-0", "train-0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_handoff(self.config)

    def test_fingerprint_and_feature_order_checked(self):
        root, _ = self.handoff()
        path = root / "feature_schema.json"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_handoff(self.config)

    def update_schema(self, changes):
        root = Path(self.config["data"]["processed_dir"])
        schema_path = root / "feature_schema.json"
        schema = json.loads(schema_path.read_text())
        schema.update(changes)
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        manifest_path = root / "temporal_splits_v1" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["feature_schema"]["sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_legacy_preprocessing_rejected_before_upload(self):
        self.handoff()
        self.update_schema({"fit_scope": "official_train_partition", "fit_end_week": None})
        with self.assertRaisesRegex(ValueError, "Legacy preprocessing"):
            validate_handoff(self.config)
        self.config["data"]["allow_legacy_preprocessing"] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = validate_handoff(self.config)
        self.assertTrue(caught)
        self.assertTrue(result["preprocessing_warnings"])

    def test_cutoff_must_include_train_and_precede_validation(self):
        self.handoff()
        for cutoff in ("2024-06-22", "2024-07-01"):
            self.update_schema({"fit_end_week": cutoff})
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "cutoff"):
                validate_handoff(self.config)

    def test_batch_outputs_require_valid_scores_and_matching_counts(self):
        path = self.directory / "predictions.csv"
        path.write_text("0.2,0\n\n0.9,1\n", encoding="utf-8")
        self.assertEqual(validate_batch_output(path, 2), 2)
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_batch_output(path, 3)
        for invalid in ("1.2,1", "nan,0", "0.3,2", "0.5,1,unexpected"):
            path.write_text(invalid, encoding="utf-8")
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "probability"):
                validate_batch_output(path, 1)

    def test_registry_group_creation_and_denied_errors(self):
        class ServiceError(Exception):
            def __init__(self, code):
                self.response = {"Error": {"Code": code, "Message": "test"}}
        self.config["cloud_execution"] = True
        client = Mock()
        client.describe_model_package_group.side_effect = ServiceError("ResourceNotFound")
        client.create_model_package_group.return_value = {"ModelPackageGroupArn": "arn:test:group"}
        result = ensure_model_package_group(self.config, {"sagemaker": client})
        self.assertTrue(result["created"])
        self.assertEqual(client.create_model_package_group.call_args.kwargs["Tags"][2], {"Key": "RunId", "Value": self.config["run_id"]})
        client.reset_mock()
        client.describe_model_package_group.side_effect = ServiceError("AccessDeniedException")
        with self.assertRaises(ServiceError):
            ensure_model_package_group(self.config, {"sagemaker": client})
        client.create_model_package_group.assert_not_called()

    def test_registry_group_reuse_requires_ownership(self):
        self.config["cloud_execution"] = True
        client = Mock()
        client.describe_model_package_group.return_value = {"ModelPackageGroupArn": "arn:test:group"}
        client.list_tags.return_value = {"Tags": [{"Key": "Project", "Value": "another-project"}]}
        with self.assertRaisesRegex(ValueError, "ownership"):
            ensure_model_package_group(self.config, {"sagemaker": client})
        client.create_model_package_group.assert_not_called()

    def test_batch_excludes_headers_labels_and_reserve(self):
        self.handoff()
        result = prepare_batch_input(self.config, self.directory / "batch.csv", rows=2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual((self.directory / "batch.csv").read_text().splitlines()[0], "0.5,-0.5")
        with self.assertRaisesRegex(ValueError, "reserve"):
            prepare_batch_input(self.config, self.directory / "bad.csv", source_split="reserved_production")

    def test_dashboard_uses_only_actual_measured_metrics(self):
        self.config["cloud_execution"] = True
        cloudwatch = Mock()
        cloudwatch.put_dashboard.return_value = {"DashboardValidationMessages": []}
        result = publish_observability(self.config, {"cloudwatch": cloudwatch, "region": "us-east-1"},
                                       {"classification_metrics": {"recall": {"value": 0.75}, "fpr": {"value": 0.01}}},
                                       {"training": "actual-job-name"}, {"features": {"a": {"psi": 0.3}}})
        sent = cloudwatch.put_metric_data.call_args.kwargs["MetricData"]
        self.assertEqual({item["MetricName"] for item in sent}, {"recall", "fpr", "max_psi"})
        self.assertIn("actual-job-name/algo-1", cloudwatch.put_dashboard.call_args.kwargs["DashboardBody"])
        self.assertEqual(result["namespace"], "AAI540/MalwareDetection")

    def test_permission_error_is_not_reported_as_success(self):
        self.config["cloud_execution"] = True
        cloudwatch = Mock()
        cloudwatch.put_metric_data.side_effect = PermissionError("lab policy")
        with self.assertRaisesRegex(PermissionError, "lab policy"):
            publish_observability(self.config, {"cloudwatch": cloudwatch, "region": "us-east-1"},
                                  {"classification_metrics": {"recall": {"value": 0.75}}})
        cloudwatch.put_dashboard.assert_not_called()

    def test_batch_dashboard_discovers_real_instance_ids(self):
        self.config["cloud_execution"] = True
        cloudwatch = Mock()
        cloudwatch.put_dashboard.return_value = {}
        cloudwatch.get_paginator.return_value.paginate.return_value = [{"Metrics": [
            {"Dimensions": [{"Name": "Host", "Value": "actual-batch/i-real-instance"}]},
            {"Dimensions": [{"Name": "Host", "Value": "other-batch/i-unrelated-instance"}]},
        ]}]
        result = publish_observability(self.config, {"cloudwatch": cloudwatch, "region": "us-east-1"},
                                       {"classification_metrics": {"recall": {"value": 0.75}}},
                                       {"batch": "actual-batch"})
        body = cloudwatch.put_dashboard.call_args.kwargs["DashboardBody"]
        self.assertIn("actual-batch/i-real-instance", body)
        self.assertNotIn("unrelated-instance", body)
        self.assertNotIn("DiskUtilization", body)
        self.assertEqual(result["pending_job_metrics"], [])

    def test_batch_dashboard_reports_delayed_metrics_without_fabrication(self):
        self.config["cloud_execution"] = True
        cloudwatch = Mock()
        cloudwatch.put_dashboard.return_value = {}
        cloudwatch.get_paginator.return_value.paginate.return_value = [{"Metrics": []}]
        result = publish_observability(self.config, {"cloudwatch": cloudwatch, "region": "us-east-1"},
                                       {"classification_metrics": {"recall": {"value": 0.75}}},
                                       {"batch": "actual-batch"})
        self.assertEqual(result["pending_job_metrics"], ["actual-batch"])
        self.assertNotIn("actual-batch/algo-1", cloudwatch.put_dashboard.call_args.kwargs["DashboardBody"])

    def test_cleanup_refuses_another_run(self):
        self.config["cloud_execution"] = True
        sm = Mock()
        with self.assertRaisesRegex(ValueError, "matching"):
            cleanup_owned_resources(self.config, {"sagemaker": sm}, {}, "another-run")
        with self.assertRaisesRegex(ValueError, "outside"):
            cleanup_owned_resources(self.config, {"sagemaker": sm}, {"feature_group_name": "shared-group"}, self.config["run_id"])
        self.assertEqual(sm.mock_calls, [])

    def test_names_change_even_for_long_timestamp_run_ids(self):
        alternate = deepcopy(self.config)
        self.config["run_id"], alternate["run_id"] = "20261015T000000-abcdef", "20261015T000000-abcdee"
        self.assertNotEqual(resource_name(self.config, "pipeline"), resource_name(alternate, "pipeline"))
        self.assertTrue(run_prefix(self.config).endswith(self.config["run_id"]))


if __name__ == "__main__":
    unittest.main()
