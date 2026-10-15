"""Reusable, account-independent binary malware training and evaluation logic.

Only approved schema columns enter estimators. Model/threshold selection uses
validation labels; held-out test and reserved-production data never enter here.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


LINEAGE_COLUMNS = ["record_id", "label", "source_partition", "source_week_start", "source_week_end"]


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
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    if max_rows < 2:
        raise ValueError("max_train_rows must be zero (unlimited) or at least two.")
    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(frame, train_size=max_rows, random_state=seed, stratify=frame["label"])
    return sampled


def candidate_estimators(seed: int, n_jobs: int = -1) -> list[tuple[str, object]]:
    """Fixed small search; no adaptive or held-out-test optimization."""
    return [
        ("logistic_regression", LogisticRegression(C=1.0, class_weight="balanced", max_iter=600, random_state=seed)),
        ("random_forest", RandomForestClassifier(n_estimators=160, max_depth=18, min_samples_leaf=2,
                                                 max_features="sqrt", class_weight="balanced_subsample",
                                                 n_jobs=n_jobs, random_state=seed)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31,
                                                                 learning_rate=0.05, l2_regularization=1.0,
                                                                 early_stopping=False, random_state=seed)),
    ]


def train_candidates(train: pd.DataFrame, validation: pd.DataFrame, schema: dict,
                     seed: int = 540, max_fpr: float = 0.01, n_jobs: int = -1) -> tuple[dict, dict]:
    columns = feature_columns(schema)
    x_train, x_validation = validate_features(train, columns), validate_features(validation, columns)
    y_train, y_validation = train["label"].to_numpy(), validation["label"].to_numpy()
    comparison, fitted = [], {}
    for name, estimator in candidate_estimators(seed, n_jobs):
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        operating_point = choose_threshold(y_validation, probabilities, max_fpr)
        metrics = classification_metrics(y_validation, probabilities, operating_point["threshold"])
        comparison.append({"name": name, "parameters": estimator.get_params(), "validation": metrics,
                           "operating_point": operating_point})
        fitted[name] = estimator
    selected = min(comparison, key=lambda row: (-row["validation"]["recall"], row["validation"]["fpr"],
                                               -row["validation"]["pr_auc"]))
    bundle = {"estimator": fitted[selected["name"]], "threshold": selected["operating_point"]["threshold"],
              "feature_columns": columns, "schema_sha256": schema_hash(schema), "selected_model": selected["name"]}
    report = {"selected_model": selected["name"], "threshold": bundle["threshold"], "seed": seed,
              "max_fpr": max_fpr, "schema_sha256": bundle["schema_sha256"], "candidates": comparison,
              "train": frame_stats(train), "validation": frame_stats(validation),
              "selection_policy": "Highest validation recall at FPR <= max_fpr, then lowest FPR, then highest PR AUC; stable candidate order breaks remaining ties.",
              "test_used_for_selection": False, "reserved_production_loaded": False,
              "caveat": "Empirical validation FPR is not a production guarantee; inspect benign count and temporal test results."}
    return bundle, report


def evaluate_bundle(bundle: dict, frame: pd.DataFrame, schema: dict) -> tuple[dict, np.ndarray]:
    if bundle["schema_sha256"] != schema_hash(schema) or bundle["feature_columns"] != feature_columns(schema):
        raise ValueError("Evaluation schema does not match the trained model.")
    probabilities = bundle["estimator"].predict_proba(validate_features(frame, bundle["feature_columns"]))[:, 1]
    metrics = classification_metrics(frame["label"], probabilities, bundle["threshold"])
    report = {"classification_metrics": {name: {"value": value} for name, value in metrics.items() if name != "confusion_matrix"},
              "confusion_matrix": metrics["confusion_matrix"], "threshold": bundle["threshold"],
              "schema_sha256": bundle["schema_sha256"], "sample_statistics": frame_stats(frame),
              "threshold_source": "validation only; frozen before held-out test evaluation", "weekly_metrics": {}}
    if "source_week_start" in frame:
        for week in sorted(frame["source_week_start"].unique()):
            mask = (frame["source_week_start"] == week).to_numpy()
            report["weekly_metrics"][str(week)] = classification_metrics(frame.loc[mask, "label"], probabilities[mask], bundle["threshold"])
    return report, probabilities
