"""Reusable, account-independent binary malware training and evaluation logic.

Only approved schema columns enter estimators. Model/threshold selection uses
validation labels; held-out test and reserved-production data never enter here.
"""

from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


LINEAGE_COLUMNS = ["record_id", "label", "source_partition", "source_week_start", "source_week_end"]
FAMILIES = ("logistic_regression", "random_forest", "xgboost")
WEIGHT_COMPONENTS = ("recall", "fpr", "pr_auc", "f1", "latency", "model_size", "interpretability")
DEPLOYMENT_COMPLEXITY = {
    "logistic_regression": "Low: compact linear coefficients; requires this exact preprocessing schema.",
    "random_forest": "Medium: tree ensemble with larger artifact and CPU inference overhead.",
    "xgboost": "Medium/high: boosted trees require the pinned XGBoost runtime and matching feature order.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_hash(report: dict) -> str:
    return schema_hash({key: value for key, value in report.items() if key != "report_sha256"})


def environment_provenance() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    if packages["xgboost"] == "not installed":
        try:
            packages["xgboost"] = importlib.metadata.version("xgboost-cpu")
            packages["xgboost_distribution"] = "xgboost-cpu"
        except importlib.metadata.PackageNotFoundError:
            pass
    else:
        packages["xgboost_distribution"] = "xgboost"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def parameter_snapshot(estimator) -> dict:
    """Strict-JSON model parameters, including XGBoost's NaN missing sentinel."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value
    return clean(estimator.get_params())


def schema_hash(schema: dict) -> str:
    """Fingerprint logical schema content independently of whitespace."""
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_columns(schema: dict) -> list[str]:
    columns = schema.get("model_feature_columns", [])
    if not columns or len(set(columns)) != len(columns) or set(columns) & set(LINEAGE_COLUMNS):
        raise ValueError("Schema must contain unique approved model_feature_columns without lineage/target columns.")
    return list(columns)


def load_frame(path: Path, schema: dict, require_labels: bool = True) -> pd.DataFrame:
    columns = feature_columns(schema)
    frame = pd.read_csv(path, dtype={"record_id": str})
    validate_features(frame, columns)
    if require_labels:
        if "label" not in frame or frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path}: labels must be nonmissing 0 or 1.")
    if frame.empty:
        raise ValueError(f"{path}: no rows.")
    if "record_id" in frame and (frame["record_id"].isna().any() or frame["record_id"].duplicated().any()):
        raise ValueError(f"{path}: missing or duplicate record IDs.")
    return frame


def validate_features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required features: {sorted(absent)}")
    try:
        values = frame[columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every model feature must be numeric.") from exc
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Every model feature must be finite; prepare/impute raw features first.")
    return values


def validate_training_contract(train: pd.DataFrame, validation: pd.DataFrame, schema: dict,
                               allow_legacy: bool = False) -> None:
    """Fail closed on old full-official-train normalization unless explicitly opted in."""
    cutoff = schema.get("fit_end_week")
    strict = schema.get("fit_scope") == "model_train_only" and bool(cutoff)
    if not strict:
        message = ("Legacy preprocessing schema may include validation-week statistics. Regenerate features with "
                   "prepare_features.py --fit-end-week 2024-06-23 and recreate temporal splits.")
        if not allow_legacy:
            raise ValueError(message + " Explicit --allow-legacy-preprocessing true is required to override.")
        warnings.warn(message + " Results are exploratory and are not a leakage-safe final evaluation.", RuntimeWarning)
    else:
        date.fromisoformat(cutoff)
        for name, frame in [("train", train), ("validation", validation)]:
            if "source_week_start" not in frame or frame["source_week_start"].isna().any():
                raise ValueError(f"{name}: source_week_start is required to verify temporal separation.")
            for week in frame["source_week_start"].unique():
                date.fromisoformat(str(week))
        if train["source_week_start"].max() > cutoff or validation["source_week_start"].min() <= cutoff:
            raise ValueError("Schema cutoff must include all model-training weeks and precede every validation week.")
    for name, frame in [("train", train), ("validation", validation)]:
        if set(frame["label"].unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both binary classes.")
    if "record_id" in train and "record_id" in validation:
        if set(train["record_id"]) & set(validation["record_id"]):
            raise ValueError("Train/validation IDs overlap.")


def choose_threshold(y_true, probabilities, max_fpr: float = 0.01) -> dict:
    """Maximize recall subject to validation FPR, considering entire tied groups.

    A finite threshold above the maximum score represents the all-negative
    operating point. It is necessary when every nonempty prediction set violates
    the requested FPR, including a zero-FPR constraint.
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1.")
    if y.ndim != 1 or y.size != scores.size or not y.size or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned scores and both binary classes.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    # Last element of each tied score group; predictions use score >= threshold.
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), y.size - 1]
    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp
    positives, negatives = int(y.sum()), int((y == 0).sum())
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    recalls = np.r_[0.0, tp / positives]
    fprs = np.r_[0.0, fp / negatives]
    eligible = np.flatnonzero(fprs <= max_fpr)
    best = min(eligible, key=lambda index: (-recalls[index], fprs[index], -thresholds[index]))
    return {"threshold": float(thresholds[best]), "recall": float(recalls[best]),
            "fpr": float(fprs[best]), "max_fpr": float(max_fpr),
            "negative_count": negatives, "positive_count": positives,
            "fpr_resolution": 1.0 / negatives}


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "pr_auc": float(average_precision_score(y, scores)) if both else None,
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "fpr": float(fp / (tn + fp)) if tn + fp else None,
        "accuracy": float(accuracy_score(y, predicted)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def frame_stats(frame: pd.DataFrame) -> dict:
    result = {"rows": len(frame), "class_counts": {str(int(k)): int(v) for k, v in frame["label"].value_counts().items()}}
    if "source_week_start" in frame:
        result["first_week"] = str(frame["source_week_start"].min())
        result["last_week"] = str(frame["source_week_start"].max())
    return result


def sample_training(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Deterministic class-stratified subset for manageable CPU training cost."""
    if max_rows < 0:
        raise ValueError("max_train_rows must be nonnegative.")
    if max_rows == 0 or len(frame) <= max_rows:
        return frame
    if max_rows < 2:
        raise ValueError("max_train_rows must be zero (unlimited) or at least two.")
    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(frame, train_size=max_rows, random_state=seed, stratify=frame["label"])
    return sampled


def baseline_estimator(seed: int = 540):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=600, random_state=seed)


def candidate_estimators(seed: int = 540, n_jobs: int = 2) -> list[tuple[str, str, object]]:
    """Two fixed settings per required family, fitted on training rows only.

    XGBoost uses CPU histogram trees, fixed rounds, and no test/validation early
    stopping. It is imported only in the comparison phase.
    """
    from xgboost import XGBClassifier
    candidates = []
    for c in (0.1, 1.0):
        candidates.append((f"logistic_regression_c{c:g}", "logistic_regression",
                           LogisticRegression(C=c, class_weight="balanced", max_iter=600, random_state=seed)))
    for depth in (12, 20):
        candidates.append((f"random_forest_depth{depth}", "random_forest",
                           RandomForestClassifier(n_estimators=160, max_depth=depth, min_samples_leaf=2,
                                                  max_features="sqrt", class_weight="balanced_subsample",
                                                  n_jobs=n_jobs, random_state=seed)))
    for depth in (4, 6):
        candidates.append((f"xgboost_depth{depth}", "xgboost",
                           XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=0.05,
                                         subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                                         objective="binary:logistic", eval_metric="logloss",
                                         tree_method="hist", device="cpu", n_jobs=n_jobs, random_state=seed)))
    return candidates


def benchmark_prediction(estimator, x_validation: np.ndarray, seed: int = 540, sample_rows: int = 32) -> dict:
    """Actual single-record predict_proba timing, excluding request/network costs."""
    indices = np.random.RandomState(seed).choice(len(x_validation), min(sample_rows, len(x_validation)), replace=False)
    sample = x_validation[indices]
    for row in sample[:min(3, len(sample))]:
        estimator.predict_proba(row.reshape(1, -1))
    elapsed = []
    for row in sample:
        start = time.perf_counter()
        estimator.predict_proba(row.reshape(1, -1))
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(elapsed, 50)), "p95_ms": float(np.percentile(elapsed, 95)),
            "sample_rows": len(sample), "sample_seed": seed, "warmup_calls": min(3, len(sample)),
            "scope": "Single-record predict_proba on this host; excludes serialization, network, and endpoint overhead.",
            "reproducibility_note": "Measured timing varies with machine load; reruns can change a latency-weighted ranking."}


def serialized_size(bundle: dict) -> int:
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.tell()


def _bundle(estimator, family: str, trial_id: str, threshold: float, schema: dict, phase: str) -> dict:
    return {"estimator": estimator, "selected_model": family, "selected_trial": trial_id,
            "threshold": threshold, "feature_columns": feature_columns(schema),
            "schema_sha256": schema_hash(schema), "phase": phase}


def _base_report(train, validation, schema, provenance, seed, max_train_rows) -> dict:
    if set(provenance) != {"train", "validation"} or any(not isinstance(value, str) or len(value) != 64 for value in provenance.values()):
        raise ValueError("Provenance requires train and validation file SHA256 hashes.")
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "schema_sha256": schema_hash(schema),
            "data_sha256": dict(provenance), "seed": seed, "max_train_rows": max_train_rows,
            "train": frame_stats(train), "validation": frame_stats(validation),
            "environment": environment_provenance(), "test_used_for_selection": False,
            "reserved_production_loaded": False}


def fit_baseline(train: pd.DataFrame, validation: pd.DataFrame, schema: dict, provenance: dict,
                 seed: int = 540, max_train_rows: int = 250000, allow_legacy: bool = False) -> tuple[dict, dict]:
    """Fit only Logistic Regression, before any user-selected operational budget."""
    validate_training_contract(train, validation, schema, allow_legacy)
    available = frame_stats(train)
    sampled = sample_training(train, max_train_rows, seed)
    x_train = validate_features(sampled, feature_columns(schema))
    x_validation = validate_features(validation, feature_columns(schema))
    estimator = baseline_estimator(seed)
    start = time.perf_counter()
    estimator.fit(x_train, sampled["label"].to_numpy())
    fit_seconds = time.perf_counter() - start
    probabilities = estimator.predict_proba(x_validation)[:, 1]
    bundle = _bundle(estimator, "logistic_regression", "baseline_c1", 0.5, schema, "baseline")
    report = _base_report(sampled, validation, schema, provenance, seed, max_train_rows)
    sweep = []
    for budget in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        point = choose_threshold(validation["label"], probabilities, budget)
        sweep.append({"max_fpr": budget, "threshold": point["threshold"], "operating_point": point,
                      "validation": classification_metrics(validation["label"], probabilities, point["threshold"])})
    report.update({"phase": "baseline", "selected_model": "logistic_regression", "threshold": 0.5,
                   "parameters": parameter_snapshot(estimator), "available_train": available,
                   "validation_metrics": classification_metrics(validation["label"], probabilities, 0.5),
                   "threshold_sweep": sweep, "fit_seconds": fit_seconds,
                   "prediction_latency": benchmark_prediction(estimator, x_validation, seed),
                   "model_size_bytes": serialized_size(bundle), "deployment_complexity": DEPLOYMENT_COMPLEXITY["logistic_regression"],
                   "legacy_preprocessing_override": allow_legacy,
                   "interpretation": "Review this validation baseline before approving an FPR budget and score weights. No final model or final-test claim is made."})
    report["report_sha256"] = report_hash(report)
    return bundle, report


def validate_policy(policy: dict, baseline_report: dict | None = None) -> dict:
    """Validate an explicit, baseline-bound decision before any comparison fit."""
    if policy.get("approved") is not True:
        raise ValueError("Selection policy must be explicitly approved after reviewing the baseline.")
    budget = policy.get("max_fpr")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Approved policy requires a finite max_fpr between zero and one.")
    weights = policy.get("weights", {})
    if set(weights) != set(WEIGHT_COMPONENTS):
        raise ValueError(f"Policy weights must contain exactly {WEIGHT_COMPONENTS}.")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Policy weights must be finite nonnegative numbers.")
    if not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Policy weights must sum to one.")
    for key in ("latency_reference_ms", "model_size_reference_mb"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    ratings = policy.get("interpretability", {})
    if set(ratings) != set(FAMILIES):
        raise ValueError("Interpretability requires one documented prior rating for each of the three model families.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1 for value in ratings.values()):
        raise ValueError("Interpretability ratings must be finite values between zero and one.")
    if baseline_report is not None:
        if baseline_report.get("phase") != "baseline" or baseline_report.get("report_sha256") != report_hash(baseline_report):
            raise ValueError("Baseline report is missing, modified, or not a baseline phase report.")
        if policy.get("baseline_report_sha256") != baseline_report["report_sha256"]:
            raise ValueError("Policy must reference the reviewed baseline_report_sha256.")
        try:
            approved = datetime.fromisoformat(policy.get("approved_at_utc", "").replace("Z", "+00:00"))
            baseline_time = datetime.fromisoformat(baseline_report["generated_at_utc"])
            if approved.tzinfo is None or approved < baseline_time:
                raise ValueError("Approval must follow the baseline and include a timezone.")
        except (TypeError, ValueError) as error:
            raise ValueError("approved_at_utc must be a timezone-aware timestamp at or after baseline generation.") from error
        if not isinstance(policy.get("decision_note"), str) or not policy["decision_note"].strip():
            raise ValueError("Document the post-baseline FPR budget/weight decision in decision_note.")
    # Copy so a caller cannot mutate the frozen policy through shared references.
    return json.loads(json.dumps(policy, allow_nan=False))


def weighted_score(metrics: dict, family: str, latency_p95_ms: float, model_size_bytes: int, policy: dict) -> tuple[float | None, dict]:
    """Score only FPR-eligible trials. Every component has higher-is-better direction."""
    validate_policy(policy)
    if not np.isfinite(latency_p95_ms) or latency_p95_ms < 0 or not np.isfinite(model_size_bytes) or model_size_bytes <= 0:
        raise ValueError("Measured latency must be finite/nonnegative and serialized size positive.")
    budget, fpr = float(policy["max_fpr"]), metrics["fpr"]
    if fpr is None or not np.isfinite(fpr) or fpr > budget:
        return None, {}
    fpr_component = 1.0 if budget == 0 else max(0.0, 1.0 - fpr / budget)
    components = {"recall": metrics["recall"], "fpr": fpr_component, "pr_auc": metrics["pr_auc"], "f1": metrics["f1"],
                  "latency": 1.0 / (1.0 + latency_p95_ms / policy["latency_reference_ms"]),
                  "model_size": 1.0 / (1.0 + model_size_bytes / (1024 * 1024) / policy["model_size_reference_mb"]),
                  "interpretability": policy["interpretability"][family]}
    if any(value is None or not np.isfinite(value) or not 0 <= value <= 1 for value in components.values()):
        raise ValueError("Scoring components must be finite and between zero and one.")
    return float(sum(policy["weights"][key] * value for key, value in components.items())), components


def compare_candidates(train: pd.DataFrame, validation: pd.DataFrame, schema: dict,
                       policy: dict, baseline_report: dict, provenance: dict,
                       seed: int = 540, max_train_rows: int = 250000, n_jobs: int = 2,
                       allow_legacy: bool = False) -> tuple[dict, dict]:
    """Train/validation-only comparison, after reviewing the baseline and policy."""
    policy = validate_policy(policy, baseline_report)
    validate_training_contract(train, validation, schema, allow_legacy)
    if baseline_report["schema_sha256"] != schema_hash(schema) or baseline_report["data_sha256"] != provenance:
        raise ValueError("Baseline and comparison must use identical schema and train/validation file hashes.")
    if baseline_report["seed"] != seed or baseline_report["max_train_rows"] != max_train_rows:
        raise ValueError("Baseline and comparison seed/sample cap must match.")
    available = frame_stats(train)
    sampled = sample_training(train, max_train_rows, seed)
    columns = feature_columns(schema)
    x_train, x_validation = validate_features(sampled, columns), validate_features(validation, columns)
    comparison, fitted = [], {}
    for trial_id, family, estimator in candidate_estimators(seed, n_jobs):
        start = time.perf_counter()
        estimator.fit(x_train, sampled["label"].to_numpy())
        fit_seconds = time.perf_counter() - start
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        point = choose_threshold(validation["label"], probabilities, policy["max_fpr"])
        bundle = _bundle(estimator, family, trial_id, point["threshold"], schema, "compare")
        # Bind artifacts to the frozen policy/data before measuring serialized bytes.
        bundle.update({"policy_sha256": schema_hash(policy), "data_sha256": dict(provenance),
                       "validation_last_week": baseline_report["validation"]["last_week"],
                       "baseline_report_sha256": baseline_report["report_sha256"]})
        metrics = classification_metrics(validation["label"], probabilities, point["threshold"])
        latency = benchmark_prediction(estimator, x_validation, seed)
        size = serialized_size(bundle)
        score, components = weighted_score(metrics, family, latency["p95_ms"], size, policy)
        row = {"trial_id": trial_id, "name": family, "parameters": parameter_snapshot(estimator), "validation": metrics,
               "operating_point": point, "fit_seconds": fit_seconds, "prediction_latency": latency,
               "model_size_bytes": size, "model_size_mb": size / (1024 * 1024),
               "eligible": score is not None, "weighted_score": score, "score_components": components,
               "deployment_complexity": DEPLOYMENT_COMPLEXITY[family]}
        comparison.append(row)
        fitted[trial_id] = bundle
    eligible = [row for row in comparison if row["eligible"]]
    if not eligible:
        raise ValueError("No candidate meets the approved validation FPR constraint.")
    ranking = lambda row: (-row["weighted_score"], -row["validation"]["recall"], row["validation"]["fpr"], row["trial_id"])
    selected = min(eligible, key=ranking)
    bundle = fitted[selected["trial_id"]]
    report = _base_report(sampled, validation, schema, provenance, seed, max_train_rows)
    report.update({"phase": "compare", "selected_model": selected["name"], "selected_trial": selected["trial_id"],
                   "selected_parameters": selected["parameters"], "threshold": bundle["threshold"],
                   "available_train": available, "candidates": comparison,
                   "family_comparison": [min([row for row in eligible if row["name"] == family], key=ranking) for family in FAMILIES],
                   "selection_policy": policy, "policy_sha256": schema_hash(policy),
                   "baseline_report_sha256": baseline_report["report_sha256"],
                   "legacy_preprocessing_override": allow_legacy,
                   "selection_policy_note": "Maximize weighted validation score among FPR-eligible trials; ties use recall, FPR, then trial ID. No test-based tuning or early stopping.",
                   "interpretability_note": "Interpretability ratings are explicit subjective priors supplied before comparison, not measured model performance.",
                   "model_size_scope": "Compressed joblib bundle bytes, excluding schema/report/source archive overhead.",
                   "caveat": "Empirical validation FPR is not a production guarantee; latency is host-specific. Do not adjust this policy after final-test inspection."})
    report["report_sha256"] = report_hash(report)
    return bundle, report


def evaluate_bundle(bundle: dict, frame: pd.DataFrame, schema: dict, training_report: dict | None = None,
                    confirm_final_test: bool = False) -> tuple[dict, np.ndarray]:
    """Evaluate a frozen comparison winner; explicit confirmation and chronology required."""
    if not confirm_final_test:
        raise ValueError("Final test requires explicit confirm_final_test=True after freezing the comparison winner.")
    if not training_report or training_report.get("phase") != "compare" or bundle.get("phase") != "compare":
        raise ValueError("Final test requires a frozen comparison winner and its training_report.json.")
    if training_report.get("report_sha256") != report_hash(training_report):
        raise ValueError("Frozen training report was modified.")
    for key in ("selected_model", "selected_trial", "threshold", "schema_sha256", "policy_sha256", "data_sha256", "baseline_report_sha256"):
        if bundle.get(key) != training_report.get(key):
            raise ValueError(f"Frozen bundle/report mismatch: {key}")
    if bundle["schema_sha256"] != schema_hash(schema) or bundle["feature_columns"] != feature_columns(schema):
        raise ValueError("Evaluation schema does not match the trained model.")
    if schema_hash(parameter_snapshot(bundle["estimator"])) != schema_hash(training_report["selected_parameters"]):
        raise ValueError("Selected model parameters changed after validation selection.")
    if schema_hash(training_report["selection_policy"]) != bundle["policy_sha256"]:
        raise ValueError("Selection policy changed after comparison.")
    if "source_week_start" not in frame or frame["source_week_start"].isna().any():
        raise ValueError("Final-test source_week_start is required for temporal validation.")
    for week in frame["source_week_start"].unique():
        date.fromisoformat(str(week))
    if frame["source_week_start"].min() <= training_report["validation"]["last_week"]:
        raise ValueError("Final-test weeks must be strictly after all validation weeks.")
    probabilities = bundle["estimator"].predict_proba(validate_features(frame, bundle["feature_columns"]))[:, 1]
    metrics = classification_metrics(frame["label"], probabilities, bundle["threshold"])
    report = {"phase": "final_test", "classification_metrics": {name: {"value": value} for name, value in metrics.items() if name != "confusion_matrix"},
              "confusion_matrix": metrics["confusion_matrix"], "threshold": bundle["threshold"],
              "schema_sha256": bundle["schema_sha256"], "policy_sha256": bundle["policy_sha256"],
              "training_report_sha256": training_report["report_sha256"], "sample_statistics": frame_stats(frame),
              "preprocessing_fit_scope": schema.get("fit_scope", "legacy_unknown"),
              "legacy_preprocessing_used": not (schema.get("fit_scope") == "model_train_only" and bool(schema.get("fit_end_week"))),
              "threshold_source": "validation only; frozen before held-out test evaluation", "weekly_metrics": {},
              "reserved_production_loaded": False}
    report["evaluation_status"] = "exploratory_legacy_preprocessing" if report["legacy_preprocessing_used"] else "strict_temporal_test"
    for week in sorted(frame["source_week_start"].unique()):
        mask = (frame["source_week_start"] == week).to_numpy()
        report["weekly_metrics"][str(week)] = classification_metrics(frame.loc[mask, "label"], probabilities[mask], bundle["threshold"])
    return report, probabilities
