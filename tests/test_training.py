"""Synthetic behavioral tests; no real malware performance is claimed."""

from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sagemaker_src"))
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from evaluate import extract_model
from inference import input_fn, output_fn, predict_fn
from training import (FAMILIES, candidate_estimators, choose_threshold, compare_candidates,
                      evaluate_bundle, fit_baseline, report_hash, serialized_size,
                      validate_features, validate_policy, validate_training_contract, weighted_score)


def synthetic_frames():
    x, y = make_classification(n_samples=180, n_features=4, n_informative=3,
                               n_redundant=0, random_state=540)
    schema = {"model_feature_columns": [f"f{i}" for i in range(4)], "fit_scope": "model_train_only",
              "fit_end_week": "2024-06-23"}
    frame = pd.DataFrame(x, columns=schema["model_feature_columns"])
    frame["label"], frame["record_id"] = y, [f"id{i}" for i in range(len(y))]
    train, validation, test = frame.iloc[:100].copy(), frame.iloc[100:140].copy(), frame.iloc[140:].copy()
    train["source_week_start"] = "2024-06-23"
    validation["source_week_start"] = "2024-06-30"
    test["source_week_start"] = "2024-09-22"
    return train, validation, test, schema


def approved_policy(baseline):
    return {"approved": True, "max_fpr": .1,
            "weights": {"recall": .30, "fpr": .25, "pr_auc": .15, "f1": .10,
                        "latency": .10, "model_size": .05, "interpretability": .05},
            "latency_reference_ms": 20., "model_size_reference_mb": 20.,
            "interpretability": {"logistic_regression": 1., "random_forest": .5, "xgboost": .3},
            "baseline_report_sha256": baseline["report_sha256"],
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_note": "Synthetic-test decision after baseline: allow 10% FPR for this fixture only."}


PROVENANCE = {"train": "a" * 64, "validation": "b" * 64}


class ThresholdTests(unittest.TestCase):
    def test_zero_fpr_uses_whole_tied_groups(self):
        result = choose_threshold([1, 0, 1, 0], [.9, .7, .7, .2], max_fpr=0)
        self.assertAlmostEqual(result["threshold"], .9)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["fpr"], 0)

    def test_all_negative_point_is_finite_and_available(self):
        result = choose_threshold([0, 1], [1., 1.], max_fpr=0)
        self.assertTrue(np.isfinite(result["threshold"]))
        self.assertGreater(result["threshold"], 1.)
        self.assertEqual(result["recall"], 0)

    def test_fpr_constraint_and_best_recall_match_exhaustive_candidates(self):
        random = np.random.RandomState(540)
        for maximum in [0, .01, .1, .5, 1]:
            y = np.array([0, 1] * 50)
            scores = np.round(random.uniform(size=100), 1)
            result = choose_threshold(y, scores, maximum)
            feasible = []
            for threshold in [np.nextafter(scores.max(), np.inf), *np.unique(scores)]:
                predictions = scores >= threshold
                fpr = predictions[y == 0].mean()
                if fpr <= maximum:
                    feasible.append((predictions[y == 1].mean(), fpr))
            self.assertLessEqual(result["fpr"], maximum)
            self.assertEqual(result["recall"], max(row[0] for row in feasible))

    def test_invalid_constraints_or_scores_rejected(self):
        for maximum, scores in [(-.1, [.1, .9]), (1.1, [.1, .9]), (.01, [np.nan, .9])]:
            with self.assertRaises(ValueError):
                choose_threshold([0, 1], scores, maximum)


class TrainingContractTests(unittest.TestCase):
    def test_legacy_rejected_by_default_and_warned_on_explicit_override(self):
        train, validation, _, schema = synthetic_frames()
        del schema["fit_scope"]
        with self.assertRaisesRegex(ValueError, "Legacy preprocessing"):
            validate_training_contract(train, validation, schema)
        with warnings.catch_warnings(record=True) as found:
            warnings.simplefilter("always")
            validate_training_contract(train, validation, schema, allow_legacy=True)
            self.assertTrue(found)

    def test_temporal_overlap_and_duplicate_ids_rejected(self):
        train, validation, _, schema = synthetic_frames()
        validate_training_contract(train, validation, schema)
        validation.loc[validation.index[0], "source_week_start"] = "2024-06-23"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            validate_training_contract(train, validation, schema)
        validation["source_week_start"] = "2024-06-30"
        validation.loc[validation.index[0], "record_id"] = train["record_id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            validate_training_contract(train, validation, schema)

    def test_nonfinite_and_missing_features_rejected(self):
        train, _, _, schema = synthetic_frames()
        train.loc[train.index[0], "f0"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_features(train, schema["model_feature_columns"])
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_features(train.drop(columns="f0"), schema["model_feature_columns"])


class ModelLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation, cls.test, cls.schema = synthetic_frames()
        cls.train["detection_ratio"] = cls.train["label"]
        cls.baseline_bundle, cls.baseline = fit_baseline(cls.train, cls.validation, cls.schema, PROVENANCE)
        cls.policy = approved_policy(cls.baseline)
        cls.bundle, cls.report = compare_candidates(cls.train, cls.validation, cls.schema,
                                                   cls.policy, cls.baseline, PROVENANCE, n_jobs=1)

    def test_exact_three_families_and_fixed_two_trials_each(self):
        candidates = candidate_estimators(n_jobs=1)
        self.assertEqual(set(family for _, family, _ in candidates), set(FAMILIES))
        self.assertEqual(len(candidates), 6)
        for family in FAMILIES:
            self.assertEqual(sum(f == family for _, f, _ in candidates), 2)
        self.assertEqual(self.baseline["phase"], "baseline")
        self.assertEqual(self.baseline["selected_model"], "logistic_regression")
        self.assertNotIn("selection_policy", self.baseline)
        self.assertEqual(self.baseline["threshold"], .5)
        self.assertEqual(len(self.baseline["threshold_sweep"]), 7)

    def test_reports_store_real_measurements_and_only_schema_features(self):
        self.assertEqual(self.bundle["estimator"].n_features_in_, 4)
        self.assertEqual(len(self.report["family_comparison"]), 3)
        for candidate in self.report["candidates"]:
            self.assertGreater(candidate["model_size_bytes"], 0)
            self.assertGreaterEqual(candidate["prediction_latency"]["p95_ms"], 0)
            self.assertEqual(len(candidate["validation"]) - 1, 9)
            self.assertLessEqual(candidate["validation"]["fpr"], self.policy["max_fpr"])
        selected = next(row for row in self.report["candidates"] if row["trial_id"] == self.report["selected_trial"])
        self.assertEqual(selected["model_size_bytes"], serialized_size(self.bundle))
        self.assertEqual(self.report["report_sha256"], report_hash(self.report))
        json.dumps(self.report, allow_nan=False)

    def test_policy_must_follow_reviewed_baseline(self):
        for changes in [{"approved": False}, {"max_fpr": None}, {"baseline_report_sha256": "c" * 64},
                        {"approved_at_utc": "2020-01-01T00:00:00+00:00"}, {"decision_note": ""},
                        {"latency_reference_ms": 0}]:
            policy = dict(self.policy, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_policy(policy, self.baseline)
        policy = deepcopy(self.policy)
        policy["weights"]["recall"] = .9
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_policy(policy)

    def test_schema_or_data_changes_reject_before_comparison_fit(self):
        with patch("training.candidate_estimators") as candidates:
            with self.assertRaisesRegex(ValueError, "identical schema"):
                compare_candidates(self.train, self.validation, self.schema, self.policy, self.baseline,
                                   {"train": "c" * 64, "validation": "b" * 64})
            candidates.assert_not_called()

    def test_weight_directions_and_fpr_eligibility(self):
        metrics = {"recall": .8, "fpr": .05, "pr_auc": .9, "f1": .7}
        score, components = weighted_score(metrics, "logistic_regression", 10, 1000000, self.policy)
        self.assertAlmostEqual(components["fpr"], .5)
        for changed, latency, size in [(dict(metrics, recall=.9), 10, 1000000),
                                        (dict(metrics, fpr=.01), 10, 1000000),
                                        (metrics, 1, 1000000), (metrics, 10, 1000)]:
            improved, _ = weighted_score(changed, "logistic_regression", latency, size, self.policy)
            self.assertGreater(improved, score)
        rejected, _ = weighted_score(dict(metrics, fpr=.2), "logistic_regression", 10, 1000, self.policy)
        self.assertIsNone(rejected)
        zero_policy = dict(self.policy, max_fpr=0)
        _, components = weighted_score(dict(metrics, fpr=0), "logistic_regression", 10, 1000, zero_policy)
        self.assertEqual(components["fpr"], 1.)

    def test_final_test_requires_frozen_winner_and_future_weeks(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            evaluate_bundle(self.bundle, self.test, self.schema, self.report)
        with self.assertRaisesRegex(ValueError, "comparison winner"):
            evaluate_bundle(self.baseline_bundle, self.test, self.schema, self.baseline, True)
        overlapping = self.test.copy()
        overlapping["source_week_start"] = "2024-06-30"
        with self.assertRaisesRegex(ValueError, "strictly after"):
            evaluate_bundle(self.bundle, overlapping, self.schema, self.report, True)
        threshold = self.bundle["threshold"]
        report_before = deepcopy(self.report)
        model_before = serialized_size(self.bundle)
        evaluation, scores = evaluate_bundle(self.bundle, self.test, self.schema, self.report, True)
        self.assertEqual(evaluation["threshold"], threshold)
        self.assertEqual(model_before, serialized_size(self.bundle))
        self.assertEqual(report_before, self.report)
        self.assertFalse(evaluation["reserved_production_loaded"])
        self.assertEqual(len(scores), len(self.test))
        self.assertIn("value", evaluation["classification_metrics"]["recall"])

    def test_frozen_policy_change_or_modified_report_is_rejected(self):
        changed = deepcopy(self.report)
        changed["selection_policy"]["max_fpr"] = .2
        with self.assertRaisesRegex(ValueError, "modified"):
            evaluate_bundle(self.bundle, self.test, self.schema, changed, True)
        changed["report_sha256"] = report_hash(changed)
        with self.assertRaisesRegex(ValueError, "policy changed"):
            evaluate_bundle(self.bundle, self.test, self.schema, changed, True)

    def test_multiline_csv_inference_preserves_schema_order_and_json_output(self):
        bundle = self.baseline_bundle
        values = input_fn("1,2,3,4\n-1,-2,-3,-4\n", "text/csv")
        predictions = predict_fn(values, bundle)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(json.loads(output_fn(predictions, "application/json"))["predictions"], predictions)
        self.assertEqual(len(output_fn(predictions, "text/csv").strip().splitlines()), 2)
        self.assertAlmostEqual(predictions[0]["probability"], bundle["estimator"].predict_proba(values)[0, 1])
        shuffled = self.validation[self.schema["model_feature_columns"]].iloc[:2][list(reversed(self.schema["model_feature_columns"]))]
        self.assertTrue(np.allclose(validate_features(shuffled, self.schema["model_feature_columns"]),
                                    self.validation[self.schema["model_feature_columns"]].iloc[:2].to_numpy()))
        with self.assertRaises(ValueError):
            predict_fn(input_fn("1,2\n", "text/csv"), bundle)
        for body in ["nan,2,3,4", "header,2,3,4"]:
            with self.assertRaises(ValueError):
                input_fn(body, "text/csv")


class ArtifactTests(unittest.TestCase):
    def test_tar_extraction_ignores_traversal_and_rejects_required_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with tarfile.open(root / "model.tar.gz", "w:gz") as handle:
                for name in ["model.joblib", "feature_schema.json", "../outside"]:
                    info = tarfile.TarInfo(name)
                    info.size = 2
                    handle.addfile(info, io.BytesIO(b"{}"))
            extracted = extract_model(root)
            self.assertTrue((extracted / "model.joblib").exists())
            self.assertFalse((root / "outside").exists())
            with tarfile.open(root / "model.tar.gz", "w:gz") as handle:
                info = tarfile.TarInfo("model.joblib")
                info.type = tarfile.SYMTYPE
                info.linkname = "../secret"
                handle.addfile(info)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_model(root)


if __name__ == "__main__":
    unittest.main()
