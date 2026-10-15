"""SageMaker inference contract for already-prepared static numeric features.

text/csv requests have no header or labels and follow model_feature_columns.
JSON requests are {"instances": [[feature1, feature2, ...], ...]}.
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np


def model_fn(model_dir):
    # Only load artifacts produced by this project's trusted training job.
    bundle = joblib.load(Path(model_dir) / "model.joblib")
    if not {"estimator", "threshold", "feature_columns"}.issubset(bundle):
        raise ValueError("Invalid model bundle.")
    return bundle


def input_fn(request_body, request_content_type):
    body = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    content_type = request_content_type.split(";", 1)[0].strip().lower()
    try:
        if content_type == "text/csv":
            values = np.loadtxt(io.StringIO(body), delimiter=",", ndmin=2)
        elif content_type == "application/json":
            payload = json.loads(body)
            if not isinstance(payload, dict) or set(payload) != {"instances"}:
                raise ValueError("JSON must contain only an instances matrix.")
            values = np.asarray(payload["instances"], dtype=float)
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric feature payload: {exc}") from exc
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite two-dimensional feature matrix.")
    return values


def predict_fn(input_data, model):
    values = np.asarray(input_data, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model["feature_columns"]) or not np.isfinite(values).all():
        raise ValueError(f"Expected {len(model['feature_columns'])} finite numeric features per row in schema order.")
    probabilities = model["estimator"].predict_proba(values)[:, 1]
    return [{"probability": float(p), "label": int(p >= model["threshold"])} for p in probabilities]


def output_fn(prediction, accept):
    accept = (accept or "application/json").split(";", 1)[0].strip().lower()
    if accept == "application/json":
        return json.dumps({"predictions": prediction}, allow_nan=False)
    if accept == "text/csv":
        return "".join(f"{row['probability']:.12g},{row['label']}\n" for row in prediction)
    raise ValueError(f"Unsupported response content type: {accept}")
