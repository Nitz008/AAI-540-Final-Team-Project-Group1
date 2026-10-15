"""CLI for fit/transform feature preparation; it never trains a model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from malware_data.feature_prep import load_feature_prep_config, make_feature_prep_config, prepare_features
from malware_data.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit static transforms on train and prepare leakage-safe EMBER2024 features.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "feature_prep.example.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--schema-path")
    parser.add_argument("--fit-end-week", help="Inclusive model-training week-start cutoff; default 2024-06-23.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    values = load_feature_prep_config(args.config)
    config = make_feature_prep_config(values, data_root=args.data_root, output_dir=args.output_dir, schema_path=args.schema_path, fit_end_week=args.fit_end_week)
    report = prepare_features(config)
    logging.getLogger(__name__).info("Feature preparation complete: %s", report["partitions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
