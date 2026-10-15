"""Synthetic behavioral tests; these are not evidence of real malware quality."""

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
from sklearn.linear_model import LogisticRegression

from evaluate import extract_model
from inference import input_fn, output_fn, predict_fn
from monitor import build_reference, compare_features
from training import (choose_threshold, evaluate_bundle, load_frame, schema_hash,
                      train_candidates, validate_features, validate_training_contract)


def synthetic_frames():
    x, y = make_classification(n_samples=160, n_features=4, n_informative=3,
                               n_redundant=0, random_state=540)
    schema = {"model_feature_columns": [f"f{i}" for i in range(4)], "fit_scope": "model_train_only",
              "fit_end_week": "2024-06-23"}
    frame = pd.DataFrame(x, columns=schema["model_feature_columns"])
    frame["label"], frame["record_id"] = y, [f"id{i}" for i in range(len(y))]
    train, validation, test = frame.iloc[:100].copy(), frame.iloc[100:130].copy(), frame.iloc[130:].copy()
    train["source_week_start"] = "2024-06-23"
    validation["source_week_start"] = "2024-06-30"
    test["source_week_start"] = "2024-09-22"
    return train, validation, test, schema


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
            candidates = [np.nextafter(scores.max(), np.inf), *np.unique(scores)]
            feasible = []
            for threshold in candidates:
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

    def test_training_uses_only_schema_inputs_and_evaluation_freezes_threshold(self):
        train, validation, test, schema = synthetic_frames()
        # A sensitive metadata column must not enter the model.
        train["detection_ratio"] = train["label"]
        with patch("training.candidate_estimators", return_value=[("baseline", LogisticRegression(random_state=540))]):
            bundle, report = train_candidates(train, validation, schema)
        self.assertEqual(bundle["estimator"].n_features_in_, 4)
        threshold = bundle["threshold"]
        evaluation, scores = evaluate_bundle(bundle, test, schema)
        self.assertEqual(evaluation["threshold"], threshold)
        self.assertEqual(bundle["threshold"], threshold)
        self.assertEqual(len(scores), len(test))
        self.assertIn("value", evaluation["classification_metrics"]["recall"])
        self.assertIn("2024-09-22", evaluation["weekly_metrics"])
        self.assertFalse(report["test_used_for_selection"])
        changed = dict(schema, extra_policy="changed")
        with self.assertRaisesRegex(ValueError, "schema"):
            evaluate_bundle(bundle, test, changed)
        json.dumps(evaluation, allow_nan=False)


class InferenceAndMonitoringTests(unittest.TestCase):
    def test_multiline_csv_inference_matches_model_and_json_output(self):
        train, validation, _, schema = synthetic_frames()
        with patch("training.candidate_estimators", return_value=[("baseline", LogisticRegression(random_state=540))]):
            bundle, _ = train_candidates(train, validation, schema)
        body = "1,2,3,4\n-1,-2,-3,-4\n"
        values = input_fn(body, "text/csv")
        predictions = predict_fn(values, bundle)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(json.loads(output_fn(predictions, "application/json"))["predictions"], predictions)
        self.assertEqual(len(output_fn(predictions, "text/csv").strip().splitlines()), 2)
        self.assertAlmostEqual(predictions[0]["probability"], bundle["estimator"].predict_proba(values)[0, 1])
        with self.assertRaises(ValueError):
            predict_fn(input_fn("1,2\n", "text/csv"), bundle)
        for body in ["nan,2,3,4", "header,2,3,4"]:
            with self.assertRaises(ValueError):
                input_fn(body, "text/csv")

    def test_drift_detects_shift_and_missing_features(self):
        frame = pd.DataFrame({"f0": np.linspace(-1, 1, 1000), "f1": np.ones(1000)})
        reference = build_reference(frame, ["f0", "f1"])
        unchanged = compare_features(reference, frame)
        self.assertAlmostEqual(unchanged["features"]["f0"]["psi"], 0)
        shifted = frame.copy()
        shifted["f0"] += 10
        report = compare_features(reference, shifted.drop(columns="f1"))
        self.assertGreater(report["features"]["f0"]["psi"], .2)
        self.assertEqual(report["features"]["f0"]["out_of_range_fraction"], 1)
        self.assertTrue(report["features"]["f1"]["missing_column"])
        json.dumps(report, allow_nan=False)

    def test_tar_extraction_ignores_traversal_and_rejects_required_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with tarfile.open(root / "model.tar.gz", "w:gz") as handle:
                for name in ["model.joblib", "feature_schema.json", "../outside"]:
                    data = b"{}"
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    handle.addfile(info, io.BytesIO(data))
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
