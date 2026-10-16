"""Small synthetic handoff for behavioral checks; never real malware evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def create_handoff(config):
    import pandas as pd
    from sklearn.datasets import make_classification

    root = Path(config["data"]["processed_dir"])
    split_dir = root / "temporal_splits_v1"
    split_dir.mkdir(parents=True)
    schema = {"model_feature_columns": [f"f{i}" for i in range(4)], "fit_scope": "model_train_only",
              "fit_end_week": "2024-06-23"}
    raw_schema = json.dumps(schema).encode()
    (root / "feature_schema.json").write_bytes(raw_schema)
    x, y = make_classification(n_samples=320, n_features=4, n_informative=3, n_redundant=0, random_state=540)
    manifest = {"feature_schema": {"sha256": hashlib.sha256(raw_schema).hexdigest()}, "splits": {}}
    for name, start, stop, week, end in [
        ("train", 0, 180, "2024-06-23", "2024-06-29"),
        ("validation", 180, 250, "2024-06-30", "2024-07-06"),
        ("test", 250, 320, "2024-09-22", "2024-09-28"),
    ]:
        frame = pd.DataFrame(x[start:stop], columns=schema["model_feature_columns"])
        frame.insert(0, "source_week_end", end)
        frame.insert(0, "source_week_start", week)
        frame.insert(0, "source_partition", "test" if name == "test" else "train")
        frame.insert(0, "label", y[start:stop])
        frame.insert(0, "record_id", [f"synthetic-{i}" for i in range(start, stop)])
        frame.to_csv(split_dir / f"{name}.csv", index=False)
        manifest["splits"][name] = {"row_count": len(frame), "class_distribution": {
            str(label): int((frame["label"] == label).sum()) for label in (0, 1)}}
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (split_dir / "reserved_production.csv").write_text("INVALID RESERVE MUST NEVER BE READ", encoding="utf-8")
    return root, split_dir


def policy_after_baseline(config, baseline):
    from copy import deepcopy
    policy = deepcopy(config["selection"])
    policy.update(approved=True, max_fpr=0.1, baseline_report_sha256=baseline["report_sha256"],
                  approved_at_utc=datetime.now(timezone.utc).isoformat(), decision_note="Synthetic test decision after baseline review.")
    return policy
