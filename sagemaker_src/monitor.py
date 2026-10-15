"""Custom feature drift monitoring, distinct from native SageMaker Model Monitor.

PSI uses training-only reference quantile bins. This operates on normalized
features: raw missingness imputed by feature preparation cannot be reconstructed.
Optional delayed labeled performance reuses the frozen validation threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def histogram(values: np.ndarray, finite_edges: list[float]) -> np.ndarray:
    edges = np.asarray(finite_edges, dtype=float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
    return counts.astype(float)


def build_reference(frame: pd.DataFrame, columns: list[str], bins: int = 10) -> dict:
    reference = {"monitor_type": "custom_population_stability_index", "reference_rows": len(frame),
                 "reference_scope": "model training subset only", "features": {},
                 "missingness_note": "Measures missing normalized inputs; cannot recover raw missingness after mean imputation."}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"No finite reference values: {column}")
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
        if edges.size < 2:
            edges = np.array([finite[0] - 0.5, finite[0] + 0.5])
        counts = histogram(finite, edges.tolist())
        reference["features"][column] = {
            "edges": edges.tolist(), "fractions": (counts / counts.sum()).tolist(),
            "minimum": float(finite.min()), "maximum": float(finite.max()),
            "missing_fraction": float(1 - finite.size / values.size),
        }
    return reference


def compare_features(reference: dict, frame: pd.DataFrame, psi_alert: float = 0.2) -> dict:
    if frame.empty:
        raise ValueError("No observations supplied for monitoring.")
    report = {"monitor_type": "custom_population_stability_index", "observed_rows": len(frame),
              "reference_rows": reference["reference_rows"], "psi_alert_threshold": psi_alert,
              "features": {}, "alerts": [], "interpretation": "PSI is a descriptive drift heuristic, not a statistical significance test or proof of performance loss.",
              "missingness_note": reference["missingness_note"]}
    for name, spec in reference["features"].items():
        if name not in frame:
            report["features"][name] = {"missing_column": True, "psi": None}
            report["alerts"].append(f"{name}: missing column")
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        counts = histogram(values, spec["edges"])
        actual = counts / max(counts.sum(), 1)
        # Add a small fixed smoothing mass to handle empty bins stably.
        expected = np.maximum(np.array(spec["fractions"]), 1e-6)
        actual = np.maximum(actual, 1e-6)
        expected, actual = expected / expected.sum(), actual / actual.sum()
        psi = float(np.sum((actual - expected) * np.log(actual / expected))) if finite_mask.any() else None
        out_of_range = finite_mask & ((values < spec["minimum"]) | (values > spec["maximum"]))
        item = {"psi": psi, "missing_fraction": float(1 - finite_mask.mean()),
                "reference_missing_fraction": spec["missing_fraction"],
                "out_of_range_fraction": float(out_of_range.mean()), "observed_fractions": actual.tolist()}
        report["features"][name] = item
        if psi is not None and psi >= psi_alert:
            report["alerts"].append(f"{name}: PSI >= {psi_alert}")
        if item["missing_fraction"] > spec["missing_fraction"]:
            report["alerts"].append(f"{name}: missing fraction increased")
    report["alert_count"] = len(report["alerts"])
    return report


def save_drift_plot(report: dict, output_dir: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    values = [(name, item["psi"]) for name, item in report["features"].items() if item.get("psi") is not None]
    figure, axis = plt.subplots(figsize=(9, max(3, len(values) * 0.28)))
    axis.barh([name for name, _ in values], [value for _, value in values], color="#276b8c")
    axis.axvline(report["psi_alert_threshold"], color="#b44040", linestyle="--", label="Heuristic alert level")
    axis.set_xlabel("Population stability index (PSI)")
    axis.set_title("Custom normalized-feature drift report")
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "feature_drift.png", dpi=150)
    plt.close(figure)
    return "feature_drift.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default="/opt/ml/processing/monitoring")
    parser.add_argument("--model-dir", type=Path, help="Optional trusted model artifacts for delayed labeled performance.")
    parser.add_argument("--psi-alert", type=float, default=0.2)
    args = parser.parse_args()
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_path.read_text(encoding="utf-8"))
    if list(reference["features"]) != schema["model_feature_columns"]:
        raise ValueError("Monitoring reference and feature schema do not match.")
    frame = pd.read_csv(args.data_path)
    report = compare_features(reference, frame, args.psi_alert)
    if args.model_dir:
        import joblib
        from training import evaluate_bundle
        bundle = joblib.load(args.model_dir / "model.joblib")
        if "label" not in frame or not frame["label"].isin([0, 1]).all():
            raise ValueError("Delayed labeled performance requires nonmissing binary labels.")
        report["delayed_labeled_performance"], _ = evaluate_bundle(bundle, frame, schema)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["graphic"] = save_drift_plot(report, args.output_dir)
    (args.output_dir / "monitoring_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"alert_count": report["alert_count"], "observed_rows": report["observed_rows"]}))


if __name__ == "__main__":
    main()
