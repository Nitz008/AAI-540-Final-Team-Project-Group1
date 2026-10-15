"""Regression coverage for temporal normalization and official-partition dedup."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from malware_data.feature_prep import (FeaturePrepConfig, fit_preprocessor,
                                        transform_partition, transform_record)


def sample(key, value):
    return {"sha256": key, "label": 1, "file_type": "Win64", "general": {"size": value, "entropy": value},
            "strings": {"numstrings": value, "avlength": value, "printables": value, "entropy": value},
            "section": {"sections": [{}]}, "imports": {"kernel32": []}, "exports": [], "richheader": [],
            "pefilewarnings": [], "authenticode": {"num_certs": 0, "self_signed": 0}}


class FeatureCutoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = FeaturePrepConfig(self.root, self.root / "output", self.root / "schema.json")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, partition, start, records):
        folder = self.root / partition
        folder.mkdir(exist_ok=True)
        path = folder / f"Win64_{start}_{start}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_validation_extremes_do_not_change_fit_but_ids_still_exclude_test_overlap(self):
        self.write("train", "2024-06-23", [sample("first", 2), sample("second", 4)])
        baseline, _ = fit_preprocessor(self.config)
        self.write("train", "2024-06-30", [sample("future", 1e12)])
        schema, keys = fit_preprocessor(self.config)
        self.assertEqual(schema["fitted_features"], baseline["fitted_features"])
        self.assertEqual(schema["fit_counters"]["unique_rows_fitted"], 2)
        self.assertEqual(schema["fit_end_week"], "2024-06-23")
        self.assertEqual(schema["fit_scope"], "model_train_only")
        self.assertIn("future", keys)
        self.assertEqual(transform_record(sample("future", 1e12), schema), transform_record(sample("future", 1e12), baseline))
        self.write("test", "2024-09-22", [sample("future", 1), sample("new", 5)])
        report = transform_partition(self.config, schema, "test", self.root / "test.csv", keys)
        self.assertEqual(report["train_test_overlap_skipped"], 1)
        self.assertEqual(report["rows_written"], 1)

    def test_unknown_week_refuses_unsafe_fit(self):
        folder = self.root / "train"
        folder.mkdir()
        (folder / "unknown.jsonl").write_text(json.dumps(sample("a", 3)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no source week"):
            fit_preprocessor(self.config)


if __name__ == "__main__":
    unittest.main()
