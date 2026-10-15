"""Explicit AWS launcher used by a teammate or the optional OIDC workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malware_mlops.aws_workflow import (
    collect_evaluation, create_aws_context, download_handoff, ensure_feature_group,
    export_execution_report, load_config, publish_observability, run_pipeline,
    upload_handoff, validate_handoff,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sagemaker.local.json"))
    parser.add_argument("--execute", action="store_true", help="Explicitly enable paid AWS jobs")
    parser.add_argument("--download", action="store_true", help="Fetch only schema, manifest, train/validation/test")
    parser.add_argument("--force-failure", action="store_true", help="Exercise FailStep without altering quality thresholds")
    parser.add_argument("--expect-failure", action="store_true", help="Pass only for a deliberate RejectCandidate failure")
    parser.add_argument("--wait", action="store_true", help="Wait for completion and collect measured evidence")
    args = parser.parse_args()
    if args.expect_failure and not (args.force_failure and args.wait):
        parser.error("--expect-failure requires --force-failure --wait")
    if not args.execute:
        parser.error("Pass --execute after configuring your AWS account and reviewing costs")
    overrides = {"cloud_execution": True}
    aws_overrides = {key: os.environ[name] for key, name in {
        "region": "AWS_REGION", "bucket": "AAI540_BUCKET", "role_arn": "AAI540_EXECUTION_ROLE_ARN"
    }.items() if os.environ.get(name)}
    if aws_overrides:
        overrides["aws"] = aws_overrides
    if os.environ.get("AAI540_MEMBER"):
        overrides["member"] = os.environ["AAI540_MEMBER"]
    if os.environ.get("AAI540_HANDOFF_URI"):
        overrides["data"] = {"handoff_s3_uri": os.environ["AAI540_HANDOFF_URI"]}
    config = load_config(args.config, overrides)
    context = create_aws_context(config)
    print(json.dumps({"identity": context["identity"], "region": context["region"], "run_id": config["run_id"]}))
    if args.download:
        download_handoff(config, context)
    checked = validate_handoff(config)
    inputs = upload_handoff(config, context, checked)
    ensure_feature_group(config, context, checked)
    execution = run_pipeline(config, context, inputs, force_failure=args.force_failure)
    print(json.dumps({"pipeline_execution_arn": execution.arn}))
    if not args.wait:
        return
    wait_error = None
    try:
        execution.wait(delay=30, max_attempts=360)
    except Exception as error:
        wait_error = error
    # A timeout, permission problem, or infrastructure failure never counts as
    # the intended rejection-path evidence.
    folder = Path(config["reports_dir"]) / config["run_id"] / ("forced_failure" if args.force_failure else "candidate")
    evidence = export_execution_report(execution, folder)
    state = evidence["execution"]["PipelineExecutionStatus"]
    if args.expect_failure:
        rejected = any(step.get("StepName") == "RejectCandidate" and step.get("StepStatus") == "Failed"
                       for step in evidence["steps"])
        if state != "Failed" or not rejected:
            raise RuntimeError(f"Expected deliberate RejectCandidate failure, observed {state}; see {folder}") from wait_error
        print("Verified intentional rejection-path execution. Thresholds were unchanged.")
    elif state != "Succeeded":
        raise RuntimeError(f"Pipeline did not succeed ({state}); inspect saved execution evidence at {folder}") from wait_error
    collected = collect_evaluation(config, context, execution)
    publish_observability(config, context, collected["report"], job_names=collected["job_names"])


if __name__ == "__main__":
    main()
