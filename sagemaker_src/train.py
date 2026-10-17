"""Two-phase modeling entry point for local use and SageMaker Training jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import joblib

from training import compare_candidates, file_sha256, fit_baseline, load_frame, report_hash


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
    parser.add_argument("--phase", choices=["baseline", "compare"], default="baseline")
    parser.add_argument("--policy-dir", type=Path, default=os.environ.get("SM_CHANNEL_POLICY", "/opt/ml/input/data/policy"))
    parser.add_argument("--baseline-dir", type=Path, default=os.environ.get("SM_CHANNEL_BASELINE", "/opt/ml/input/data/baseline"))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--seed", type=int, default=540)
    parser.add_argument("--max-train-rows", type=int, default=250000, help="Deterministic stratified CPU cap; 0 uses all training rows.")
    parser.add_argument("--n-jobs", type=int, default=2, help="CPU worker count for RF/XGBoost; positive integer.")
    parser.add_argument("--allow-legacy-preprocessing", type=boolean, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_train_rows < 0 or args.n_jobs < 1:
        raise ValueError("max-train-rows must be nonnegative and n-jobs positive.")
    schema_path = args.schema_dir / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    train_path, validation_path = args.train_dir / "train.csv", args.validation_dir / "validation.csv"
    train, validation = load_frame(train_path, schema), load_frame(validation_path, schema)
    provenance = {"train": file_sha256(train_path), "validation": file_sha256(validation_path)}
    if args.phase == "baseline":
        bundle, report = fit_baseline(train, validation, schema, provenance, args.seed,
                                      args.max_train_rows, args.allow_legacy_preprocessing)
        report_name = "baseline_report.json"
    else:
        policy = json.loads((args.policy_dir / "selection_policy.json").read_text(encoding="utf-8"))
        baseline_report = json.loads((args.baseline_dir / "baseline_report.json").read_text(encoding="utf-8"))
        bundle, report = compare_candidates(train, validation, schema, policy, baseline_report, provenance,
                                             args.seed, args.max_train_rows, args.n_jobs,
                                             args.allow_legacy_preprocessing)
        report_name = "training_report.json"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_dir / "model.joblib", compress=3)
    shutil.copyfile(schema_path, args.model_dir / "feature_schema.json")
    report.update({"preprocessing_fit_scope": schema.get("fit_scope", "legacy_unknown"),
                   "source_sha256": {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")},
                   "model_artifact_sha256": file_sha256(args.model_dir / "model.joblib")})
    report["report_sha256"] = report_hash(report)
    (args.model_dir / report_name).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.model_dir.resolve() != args.output_dir.resolve():
        shutil.copyfile(args.model_dir / report_name, args.output_dir / report_name)
    if args.phase == "compare":
        (args.model_dir / "selection_policy.json").write_text(json.dumps(report["selection_policy"], indent=2, allow_nan=False), encoding="utf-8")
        if args.model_dir.resolve() != args.output_dir.resolve():
            shutil.copyfile(args.model_dir / "selection_policy.json", args.output_dir / "selection_policy.json")
    print(json.dumps({"phase": args.phase, "selected_model": report["selected_model"],
                      "threshold": report["threshold"], "report": str(args.model_dir / report_name)}))


if __name__ == "__main__":
    main()
