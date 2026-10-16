"""Final temporal evaluation in the notebook runtime; preserves model and threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np

from training import evaluate_bundle, file_sha256, load_frame, report_hash


ARTIFACT_NAMES = {"model.joblib", "feature_schema.json", "training_report.json", "selection_policy.json", "baseline_report.json"}


def extract_model(model_dir: Path) -> Path:
    """Extract only flat allowlisted regular artifacts; reject unsafe archives."""
    archive = model_dir / "model.tar.gz"
    if not archive.exists():
        if (model_dir / "model.joblib").is_file():
            return model_dir
        raise FileNotFoundError(f"Neither model.tar.gz nor model.joblib found in {model_dir}")
    destination = model_dir / "extracted"
    destination.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            # SageMaker also packages code/ for serving; ignore every other path.
            if member.name not in ARTIFACT_NAMES:
                continue
            if not member.isfile() or member.size > 512 * 1024 * 1024 or member.name in found:
                raise ValueError(f"Unsafe or duplicate model artifact: {member.name}")
            found.add(member.name)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("Unreadable model artifact.")
            with source, (destination / member.name).open("wb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    if not {"model.joblib", "feature_schema.json"}.issubset(found):
        raise ValueError("Model archive is missing required artifacts.")
    return destination


def save_plots(frame, probabilities, threshold: float, output_dir: Path) -> list[str]:
    """Save final-test graphics when matplotlib is installed in this runtime."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    labels = frame["label"].to_numpy()
    if len(np.unique(labels)) == 2:
        for name, display in [("roc_curve", RocCurveDisplay), ("precision_recall_curve", PrecisionRecallDisplay)]:
            figure, axis = plt.subplots(figsize=(6, 5))
            display.from_predictions(labels, probabilities, ax=axis)
            axis.set_title(f"Temporal held-out test: {name.replace('_', ' ')}")
            figure.tight_layout()
            figure.savefig(output_dir / f"{name}.png", dpi=150)
            plt.close(figure)
            plots.append(f"{name}.png")
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(labels, probabilities >= threshold, labels=[0, 1],
                                            display_labels=["Benign", "Malware"], ax=axis, colorbar=False)
    axis.set_title(f"Held-out test; frozen threshold {threshold:.4g}")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    plots.append("confusion_matrix.png")
    return plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default="reports/model_training/final_test")
    parser.add_argument("--confirm-final-test", nargs="?", const=True, default=False,
                        type=lambda value: str(value).lower() == "true",
                        help="Explicitly confirm the winner/policy were frozen before opening test.csv.")
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("--confirm-final-test is required; complete baseline review and comparison first.")
    model_dir = extract_model(args.model_dir)
    training_report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    if training_report.get("report_sha256") != report_hash(training_report):
        raise ValueError("Frozen training report was modified.")
    if training_report.get("model_artifact_sha256") != file_sha256(model_dir / "model.joblib"):
        raise ValueError("Model artifact changed after comparison.")
    # joblib/pickle is executable: the input artifact must be from a trusted job.
    bundle = joblib.load(model_dir / "model.joblib")
    schema = json.loads((args.schema_dir / "feature_schema.json").read_text(encoding="utf-8"))
    frame = load_frame(args.test_dir / "test.csv", schema)
    report, probabilities = evaluate_bundle(bundle, frame, schema, training_report, confirm_final_test=True)
    report["test_data_sha256"] = file_sha256(args.test_dir / "test.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["graphics"] = save_plots(frame, probabilities, bundle["threshold"], args.output_dir)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["classification_metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
