# Reproduce baseline, model comparison, and final evaluation

Use `notebooks/02_sagemaker_training_evaluation.ipynb` on the [`marco-model-training-evaluation` branch](https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1/tree/marco-model-training-evaluation). The workflow covers ML modeling and a model handoff. It does not create deployment or monitoring infrastructure. Each teammate uses their own environment, existing S3 bucket, and allowed SageMaker execution role.

## 1. Clone and install

From a SageMaker JupyterLab terminal:

```bash
git clone --branch marco-model-training-evaluation --single-branch https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1.git
cd AAI-540-Final-Team-Project-Group1
python -m pip install -r requirements-sagemaker.txt
cp configs/sagemaker.example.json configs/sagemaker.local.json
jupyter lab notebooks/02_sagemaker_training_evaluation.ipynb
```

Use a Python 3.12 kernel with the installed dependencies; restart it if older packages are already imported. The notebook resolves its repository root and uses reusable code in `sagemaker_src`. The dependency set uses SageMaker SDK v2 and XGBoost 3.0.5. CPU-only XGBoost is available for x86_64 Linux/Windows, as described in [the official installation guide](https://xgboost.readthedocs.io/en/release_3.0.0/install.html). Installation requires permitted network access; managed training installs the extra dependency from `sagemaker_src/requirements.txt` before running the entry point.

The AWS image and notebook dependency versions must remain compatible when loading a trusted serialized model. As checked October 15, 2026, the selected scikit-learn Python 3.12 image `1.4-2-py312` reaches end of patch support on October 30, 2026 according to [AWS's support policy](https://docs.aws.amazon.com/en_en/sagemaker/latest/dg/pre-built-containers-support-policy.html). Review current regional availability/support before later use and test any replacement. The SDK `py3` selector is separate from the explicit Python 3.12 image tag; do not substitute an unverified container because its label looks similar.

## 2. Configure paths and your AWS account

Use the notebook's configuration cell and `configs/sagemaker.example.json`. Keep personal settings in the ignored `configs/sagemaker.local.json`; in PowerShell, copy it with `Copy-Item -LiteralPath configs/sagemaker.example.json -Destination configs/sagemaker.local.json`. Do not include credentials in that JSON or notebook outputs. Local debugging needs downloaded prepared data and installed packages; it does not require starting a managed job.

| Configuration field | Value to supply or review |
|---|---|
| `member` | Short non-sensitive identifier for your own run namespace. |
| `aws.region`, `aws.profile` | Region and optional local AWS profile; `null` profile uses normal notebook/lab providers. |
| `aws.bucket`, `aws.role_arn`, `aws.prefix` | Existing output bucket, allowed execution role, and project prefix. |
| `data.handoff_s3_uri` | The matching shared processed-data version supplied by the data owner. |
| `data.processed_dir`, `data.split_dir` | Local handoff layout, normally the README paths. |
| `data.minimum_class_rows` | Initially 10,000 in each class in the actual training CSV, before optional fitting subsampling. |
| `data.allow_legacy_preprocessing` | Keep `false` for the final leakage-safe experiment. |
| `compute.training_instance_type` | Account-permitted CPU training instance. |
| `compute.max_train_rows`, `compute.seed` | Initially 250,000 and 540; `0` cap uses all training rows. Keep baseline/comparison settings equal. |
| `compute.image_uri` / `compute.image_tag` | Reviewed full regional override, or documented default `1.4-2-py312` tag. |
| `selection` | Leave unapproved with no FPR until the baseline-dependent decision stage. |

Keep the loaded run ID and configuration for the entire experiment; managed output prefixes include member and run ID to avoid mixing teammates' artifacts.

For AWS operations, verify the active identity:

```bash
aws sts get-caller-identity
aws configure get region
```

Use the notebook role, an AWS profile/SSO session, or Learner Lab temporary credentials through the normal AWS provider chain. Temporary access also needs its session token. Confirm access to the input bucket/prefix and your output bucket, which may have different owners, and choose an output bucket in the job region. Needed permissions cover scoped S3 reads/writes, SageMaker training describe/create/stop, and passing the existing execution role. The helper does not create IAM roles, buckets, or administrator permissions.

Learner Lab can restrict roles, instance families, quotas, network access, or SageMaker APIs. Use the supplied permitted role and instance types. If a job or package install is denied, retain the sanitized error and resolve the specific permission, network, image, or quota issue with the account owner. A local run is useful evidence of modeling behavior but must not be described as a successful managed AWS job. Protect private S3 artifacts, encryption settings, credentials, and sensitive file identifiers.

## 3. Obtain the corrected train/validation handoff

Initial modeling needs these matching artifacts:

```text
data/processed/ember2024_win64_static_v1/
    feature_schema.json
    temporal_splits_v1/
        train.csv
        validation.csv
        split_manifest.json
```

Set `DOWNLOAD_HANDOFF=True` only when downloading the handoff from S3 is necessary. Default helpers download and validate train/validation/schema/manifest only. Test data is deferred until the explicit final-test stage, and `reserved_production.csv` is excluded throughout. Existing local test files are not permission to use them during baseline or comparison. The shared manifest can describe test/reserve counts without opening their CSVs.

Validate schema fingerprint and exact columns/order, numeric finite features, binary labels, minimum training class counts, non-overlapping record IDs, source partitions, and chronology. The full actual train CSV must have at least `data.minimum_class_rows` observations per class before model-fitting subsampling; validation must have both classes. Synthetic fixtures with a lower configured minimum remain code tests. Use only `model_feature_columns` as `X`; exclude `record_id`, `label`, `source_partition`, and source-week metadata. Expected preprocessing declares `fit_scope: "model_train_only"` and inclusive week-start cutoff `2024-06-23`, covering June 23–29. The 12 later official-training weeks are validation and cannot contribute to normalization.

If the old handoff was fitted over all 52 official-training weeks, the data owner must regenerate it from raw data:

```bash
python scripts/prepare_features.py --fit-end-week 2024-06-23
python scripts/create_temporal_splits.py
```

Publish the resulting schema, CSVs, processing report, manifest, and metadata together under a fresh S3 prefix. Do not edit an old schema to claim a new cutoff or mix files from different versions. The explicit legacy compatibility override is exploratory only and must remain disclosed in every resulting report. See [feature preparation](feature_preparation.md) and [temporal strategy](temporal_split_strategy.md).

Where raw JSONL files are unavailable locally, existing processed splits (`data/processed/ember2024_reconstructed_v1/`) serve as an exploratory approximation reconstructed by reversing standardization and fitting replacement scaling moments on the training partition (source weeks through 2024-06-23) without an independent per-row cutoff filter. Original missing-value indicators were lost and estimated via a three-feature zero proxy. Exact reproducibility of the original raw pipeline cannot be claimed.

## 4. Run the baseline first

The notebook defaults to the following controls:

| Control | Default and purpose |
|---|---|
| `EXECUTION_MODE` | `"local"`; choose `"sagemaker"` for managed baseline/comparison jobs. |
| `RUN_AWS` | `False`; explicitly enable cloud operations after account setup. |
| `DOWNLOAD_HANDOFF` | `False`; optionally fetch named handoff artifacts. |
| `RUN_BASELINE` | `False`; run the standalone Logistic Regression baseline. |
| `POLICY_READY` | `False`; record the post-baseline policy decision only after reviewing tradeoffs. |
| `MAX_FPR` | `None`; no acceptable FPR is assumed to have been approved. |
| `DECISION_NOTE` | Empty; explain the selected FPR budget and tradeoffs. |
| `RUN_COMPARISON` | `False`; compare the three model families using the frozen policy. |
| `RUN_FINAL_TEST` | `False`; separately opt into final test only after a winner is frozen. |

Enable `RUN_BASELINE` after validating the train/validation handoff. Local mode runs the shared training entry point in the kernel environment; SageMaker mode launches a paid training job with train, validation, and schema channels only. Inspect its actual report and validation threshold table/curves. The baseline does not determine the final model and does not inspect test labels. Counts, parameters, data/schema provenance, and runtime versions should be visible with the baseline results.

## 5. Decide and freeze the selection policy

Use baseline validation behavior to choose `MAX_FPR` and write `DECISION_NOTE`. The budget is an operating assumption about benign false alerts; a threshold with an observed FPR below the limit does not guarantee future production FPR. Review benign sample count and false-positive resolution. Do not choose the limit after inspecting the project test results.

The example proposes RFC Option A: recall 30%, FPR 25%, PR-AUC 15%, F1 10%, local latency 10%, model size 5%, and interpretability 5%. The proposed latency/size reference values and LR/RF/XGBoost interpretability priors are explicit configurable assumptions. Precision, ROC-AUC, accuracy, and qualitative deployment complexity remain reported outside that weighted formula. Confirm the selected weights before comparison; the example is not prior team approval.

Set `POLICY_READY=True` only after the baseline review and decision. The notebook saves `selection_policy.json` with an approved flag, FPR limit, weights, rationale, canonical baseline-report SHA-256, and a timestamp no earlier than baseline generation. The comparison validates that link. Keep the policy and baseline report together; do not reuse a policy for a different data/schema version or change weights after seeing final test behavior.

## 6. Compare Logistic Regression, Random Forest, and XGBoost

Enable `RUN_COMPARISON` with the frozen policy and baseline report. The bounded search evaluates two predefined trials per model family, fits estimators only on train, and selects thresholds/trials using validation. Review raw metrics and normalized score components alongside the final weighted score. Check the report's actual available/used rows if a training cap is configured. The selected estimator and threshold remain frozen without refitting on validation.

Inspect recall, FPR, precision, F1, PR-AUC, ROC-AUC, accuracy, confusion counts, local `predict_proba` p50/p95 timing, serialized bundle size, and interpretability/deployment assumptions. Timing uses up to 32 seeded validation rows, three warmup calls, then one single-record call per selected row. It is environment-specific model computation, not a SageMaker endpoint response-time measurement. Use the same benchmark settings across candidates and retain measured row count, warmup count, seed, platform/package versions and the managed instance type when applicable. Bundle size is compressed `joblib` bytes, with schema/report/source overhead excluded. Curves and charts must be generated from actual outputs, not filled with example results.

Managed runs return model artifacts and reports through your output bucket. Download only your own trusted training artifact and use the allowlisted extraction helper. Preserve the model bundle, schema, baseline/comparison reports, policy, chosen parameters, hashes, source commit, seeds, versions, and actual training job name/artifact URI. `joblib` deserialization is executable, so do not load arbitrary shared model archives.

## 7. Optional final test

Only after the winner, threshold, and policy are saved should you enable `RUN_FINAL_TEST`. The workflow then requests/downloads the six-week project test subset, validates its chronology/schema/IDs against the earlier handoff, and evaluates the frozen model. `sagemaker_src/evaluate.py` requires an explicit `--confirm-final-test true` opt-in. Evaluation runs in the current local or SageMaker notebook kernel; this scope does not create an additional processing job or upload test data to a training job.

Retain the final report and plots with test counts, confusion matrix, recall/FPR, precision/F1, PR-AUC/ROC-AUC, accuracy, weekly behavior, threshold, and provenance. Describe the dataset as an **official-test-derived temporal subset**, because the official test period is split between project test and reserve. Do not retune a model, alter the threshold/weights, or open the reserve in response to the test result. Missing real artifacts mean results are pending; no score or cloud success should be inferred from source code or synthetic checks.

## 8. Hand off artifacts and stop idle compute

Give teammates `model.joblib`, `feature_schema.json`, `baseline_report.json`, `training_report.json`, `selection_policy.json`, optional `evaluation.json` and plots, source/data fingerprints, runtime versions, and actual job/S3 identifiers. The baseline and comparison model directories are separate; hand off the selected comparison model for downstream work. Keep data, credentials, personal account JSON, and large generated artifacts out of Git. This handoff supports the separate integration workstreams; it does not claim an endpoint, registry, Feature Store, or monitoring service has been created.

Managed training jobs terminate after completion, but notebook instances/Studio apps and retained S3 objects can continue to incur charges. Stop idle notebook compute after saving outputs, check for any interrupted training job, and apply the team's reviewed artifact-retention policy. Do not remove a shared handoff or another member's data. This workflow does not provision persistent inference resources needing teardown.

## Troubleshooting

| Symptom | Next action |
|---|---|
| Legacy preprocessing rejected | Regenerate features/splits with the 40-week fit boundary; preserve version consistency. |
| Schema hash, columns, or row counts disagree | Obtain a complete matching handoff instead of changing metadata to fit the files. |
| Minimum class count rejected | Inspect actual count scope and dataset version; synthetic downsampled data is not the real-data requirement. |
| Comparison rejects policy | Run/review baseline first and save its linked post-baseline policy with budget, weights and rationale. |
| XGBoost install/import fails | Check pinned package/platform support, Python environment, permitted network access and job logs. |
| AWS identity/role denied or token expired | Refresh temporary credentials and verify the intended role/account and scoped `iam:PassRole`. |
| Instance, quota, image or S3 access rejected | Use an allowed regional combination and verify bucket/key permissions; retain actual error evidence. |
| Final evaluation refuses to run | Confirm explicit final-test opt-in and a trusted saved comparison winner/policy. |
| Score or latency differs across environments | Compare source/data/package versions, seed, row caps and timing settings; report measured context. |
