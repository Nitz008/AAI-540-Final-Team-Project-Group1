"""SageMaker SKLearn entry point: only train, validation, and schema channels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import joblib

from monitor import build_reference
from training import (frame_stats, load_frame, sample_training, train_candidates,
                      validate_training_contract)


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation-dir", type=Path, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--schema-dir", type=Path, default=os.environ.get("SM_CHANNEL_SCHEMA", "/opt/ml/input/data/schema"))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--seed", type=int, default=540)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--max-train-rows", type=int, default=250000, help="Deterministic stratified CPU cap; 0 uses all training rows.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--allow-legacy-preprocessing", type=boolean, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_train_rows < 0:
        raise ValueError("max-train-rows must be nonnegative.")
    schema_path = args.schema_dir / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    train = load_frame(args.train_dir / "train.csv", schema)
    validation = load_frame(args.validation_dir / "validation.csv", schema)
    validate_training_contract(train, validation, schema, args.allow_legacy_preprocessing)
    available_train = frame_stats(train)
    train = sample_training(train, args.max_train_rows, args.seed)
    bundle, report = train_candidates(train, validation, schema, args.seed, args.max_fpr, args.n_jobs)
    report.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "available_train": available_train, "max_train_rows": args.max_train_rows,
                   "legacy_preprocessing_override": args.allow_legacy_preprocessing,
                   "preprocessing_fit_scope": schema.get("fit_scope", "legacy_unknown")})
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_dir / "model.joblib", compress=3)
    shutil.copyfile(schema_path, args.model_dir / "feature_schema.json")
    (args.model_dir / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    reference = build_reference(train, bundle["feature_columns"])
    (args.model_dir / "reference_histograms.json").write_text(json.dumps(reference, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("training_report.json", "reference_histograms.json"):
        shutil.copyfile(args.model_dir / name, args.output_dir / name)
    print(json.dumps({"selected_model": report["selected_model"], "threshold": report["threshold"],
                      "validation": next(row["validation"] for row in report["candidates"] if row["name"] == report["selected_model"])}))


if __name__ == "__main__":
    main()
