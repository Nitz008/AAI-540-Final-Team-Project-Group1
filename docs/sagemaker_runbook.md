# Run the training branch in your own SageMaker account

This runbook uses [`marco-model-training-evaluation`](https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1/tree/marco-model-training-evaluation) and `notebooks/02_sagemaker_training_evaluation.ipynb`. Each member supplies their own AWS identity, region, execution role, and existing S3 bucket. Account-specific values and credentials are not distributed with the notebook. The default notebook switches leave AWS resource creation disabled.

## 1. Clone and prepare a kernel

In a SageMaker Studio JupyterLab terminal, or a configured Linux environment:

```bash
git clone --branch marco-model-training-evaluation --single-branch https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1.git
cd AAI-540-Final-Team-Project-Group1
python -m pip install -r requirements-sagemaker.txt
cp configs/sagemaker.example.json configs/sagemaker.local.json
```

On Windows PowerShell, replace the last command with:

```powershell
Copy-Item -LiteralPath configs/sagemaker.example.json -Destination configs/sagemaker.local.json
```

Use a **Python 3.12** kernel where dependencies were installed and run from the repository root; the notebook resolves its repository path if opened from `notebooks/`. Restart the kernel after dependency installation if an old SageMaker SDK is already imported. This branch uses SageMaker Python SDK **v2** (`>=2.257,<3`), constrained in `requirements-sagemaker.txt`; a preinstalled SDK v3 requires a separate kernel/environment or the pinned installation.

The default job image is the region-specific AWS scikit-learn tag `1.4-2-py312`, listed in [AWS available images](https://aws.github.io/deep-learning-containers/reference/available_images/#scikit-learn). The SDK's `compute.py_version` selector remains `py3`; it is not the image's Python interpreter version. The helper resolves the regional ECR repository and supplies the Python 3.12 tag explicitly, or accepts `compute.image_uri` as a reviewed override. As checked October 15, 2026, AWS lists **October 30, 2026** as this image's end of patch support in [the support policy](https://docs.aws.amazon.com/en_en/sagemaker/latest/dg/pre-built-containers-support-policy.html). Review supported images before later operational use and test any replacement against training, evaluation, and serving together. Local synthetic checks do not establish that a particular regional image was pulled successfully.

## 2. Verify credentials and account controls

```bash
aws sts get-caller-identity
aws configure get region
```

In SageMaker, use the assigned notebook/Studio role through the AWS credential provider chain. Locally, use an AWS profile, IAM Identity Center session, or temporary Learner Lab credentials supplied outside the repo. For temporary credentials, the session token is required as well as the access key and secret. Expired lab credentials must be refreshed in the runtime environment. Never paste credentials into notebook cells, configuration files, committed outputs, or screenshots.

Choose an existing bucket in the job region and an execution role your identity is permitted to pass to SageMaker. Needed actions depend on enabled sections: S3 read/write for input/output prefixes; SageMaker training, processing, pipeline and registry APIs; `iam:PassRole` for the execution role; Feature Store creation/runtime ingestion when enabled; CloudWatch dashboard/metric operations when enabled; batch transform and model APIs for inference. Offline Feature Store may also need S3, KMS, and catalog permissions. Review [SageMaker data protection](https://docs.aws.amazon.com/sagemaker/latest/dg/data-protection.html), bucket public-access settings, encryption, and retention before uploading.

AWS Learner Lab often limits `iam:CreateRole`, `iam:PassRole`, allowed instance types, Feature Store, Glue, pipelines, or dashboards. Use the existing allowed lab role rather than attempting to create a new administrator role. If a required action is denied, record the action, role, service, and sanitized error in the final evidence; ask the lab/instructor for the supported role or permitted service. A disabled optional section, local plot, or synthetic test must not be described as successful AWS operation. Request a permitted region/instance or quota adjustment before retrying a rejected job; do not assume notebook access implies every service is available.

## 3. Configure your run

Edit the ignored `configs/sagemaker.local.json`. Required replacements include:

| Field | Meaning |
|---|---|
| `member` | Short non-sensitive identifier, using letters, digits, or hyphens; avoids collisions between members. |
| `aws.region` | Region for the bucket and SageMaker jobs. |
| `aws.profile` | Local AWS profile name, or `null` for notebook/lab credentials. |
| `aws.bucket` | Your existing output bucket, without `s3://`. |
| `aws.role_arn` | Allowed SageMaker execution-role ARN. The context helper may resolve an assigned role, but an explicit account value is easiest to audit. |
| `aws.prefix` | Your project output prefix. Run outputs also include member and run ID. |
| `data.handoff_s3_uri` | Shared processed-data prefix supplied by the data owner. This can be a different authorized bucket from your output bucket. |
| `data.processed_dir` | Local prepared-data root, default `data/processed/ember2024_win64_static_v1`. |
| `data.allow_legacy_preprocessing` | Keep `false` for the final leakage-safe experiment. |
| `quality.max_fpr` | Validation operating budget and test release gate, initially `0.01`. |
| `quality.min_recall`, `quality.min_roc_auc` | Test acceptance gates, initially `0.50` and `0.80`. These are proposed requirements, not attained scores. |
| `compute.*_instance_type` | Allowed training, processing, and batch instance types. |
| `compute.framework_version`, `compute.py_version` | Defaults `1.4-2` and SDK selector `py3`; explicit resolved AWS image uses tag `1.4-2-py312`. |
| `compute.image_uri` | Optional full regional compatible image URI; keep `null` to use the documented resolver. |
| `feature_store.enabled`, `feature_store.sample_rows` | Enable a bounded training-only feature group demonstration with online and offline storage, initially 100 rows. |
| `feature_store.disable_glue_table_creation` | Defaults to `true`; offline S3 delivery still occurs, while optional catalog/Athena work needs extra permissions and setup. |
| `cloud_execution` | Keep `false` until intentionally running the AWS sections. |

Leave `run_id` as `null` to generate a unique ID once per configuration load. Reuse the loaded configuration object throughout one notebook run so resources and evidence refer to the same ID. Do not change the quality gates after inspecting a test result and call it a fresh evaluation. For configuration merging from code, `load_config(path, overrides={...})` accepts nested overrides; values may be supplied from environment variables without saving personal account settings in Git.

The notebook supports `AAI540_CONFIG` for a different local config path and reads `AWS_REGION`, `AAI540_BUCKET`, `SAGEMAKER_ROLE_ARN`, `AAI540_HANDOFF_S3_URI`, and `AAI540_MEMBER` as overrides. These account identifiers are configuration, not credentials. AWS credentials remain with the normal provider chain. For example, a terminal launched before Jupyter can export the bucket and role, or PowerShell can set `$env:AAI540_BUCKET` and `$env:SAGEMAKER_ROLE_ARN`; the kernel must inherit those values.

## 4. Acquire and validate the handoff

The data owner must publish the following **matching version** of the artifacts:

```text
feature_schema.json
temporal_splits_v1/
    train.csv
    validation.csv
    test.csv
    split_manifest.json
```

The notebook's optional `DOWNLOAD_HANDOFF` section calls `download_handoff` for these named files only. It omits `reserved_production.csv` and raw JSONL. The manifest can describe all four splits while no reserve records are downloaded. If the files are already present, leave `DOWNLOAD_HANDOFF=False` and validate them locally. Reading a shared handoff needs permission on its bucket and any KMS key, as well as credentials in your own account.

Before any cloud training, `validate_handoff` checks the schema hash against the split manifest, exact columns and feature order, binary classes, finite values, row counts, unique identifiers across train/validation/test, source partitions, and chronological order. The strict trainer also verifies the preprocessing fit boundary. Expected schema declarations are `fit_scope: "model_train_only"` and `fit_end_week: "2024-06-23"`, the start of the final model-training week. Every model-input matrix follows `model_feature_columns`; label and lineage columns must never enter inference payloads.

**If the old handoff fails:** ask the data owner to regenerate, rather than modifying the schema declaration. From a machine with the raw data, run:

```bash
python scripts/prepare_features.py --fit-end-week 2024-06-23
python scripts/create_temporal_splits.py
```

The corrected preparer fits on the first 40 weeks and transforms later periods with those saved statistics. Publish the new schema, CSVs, reports and manifest together under a fresh versioned S3 prefix; update `data.handoff_s3_uri` accordingly. The raw data is about 31.24 GiB locally, so regeneration is best performed once by the data owner. The explicit legacy override only supports exploratory compatibility and must be disclosed; it does not satisfy the final leakage-safe protocol.

## 5. Run the notebook in stages

Open `notebooks/02_sagemaker_training_evaluation.ipynb` and run its setup/configuration cells. Inspect configuration and handoff validation output first. The notebook separates optional paid operations with these switches:

| Switch | Operation when enabled |
|---|---|
| `RUN_AWS` | Establish account context, upload selected input channels, and create/start the training pipeline. Sets the loaded config's cloud guard. |
| `DOWNLOAD_HANDOFF` | Fetch only the named non-reserve handoff files. Requires AWS execution context. |
| `RUN_LOCAL_TRAINING` | Debug the same training/evaluation entry points locally using downloaded data. Creates no SageMaker training job and does not satisfy AWS operation evidence. |
| `RUN_FEATURE_STORE` | Create a real feature group and ingest a bounded sample of training rows. |
| `RUN_FAILURE_DEMO` | Start a separately labeled forced quality-gate failure execution. |
| `RUN_BATCH` | Run paid batch inference from the registered package. |
| `APPROVE_MODEL` | Explicitly approve a chosen package before a production-oriented batch run. Review that version first. |
| `PUBLISH_DASHBOARD` | Publish custom metrics and create/update the account CloudWatch dashboard. |
| `RUN_CLEANUP` | Invoke the documented cleanup of tracked run resources after inventory review. |

Enable only the sections you intend to execute. A real train/evaluate pipeline can create multiple paid compute jobs; Feature Store, S3, dashboards, logs, and notebook apps can also accrue costs. Allow jobs to finish before restarting the kernel or changing configuration. The CLI offers the same reusable path through `scripts/run_sagemaker_pipeline.py`; use `python scripts/run_sagemaker_pipeline.py --help` for its current arguments.

The model compares logistic regression, random forest, and histogram gradient boosting, fitting estimators on train and selecting candidate/threshold on validation. The entry point's default deterministic stratified cap is 250,000 training rows, with available/used counts in `training_report.json`; `--max-train-rows 0` uses all training rows when compute allows. The evaluation job then uses the frozen model on test. Review the candidate comparison, selected threshold, sample counts, test metrics and weekly plots. A failing test gate is a valid outcome; keep the report and diagnose the limitation using an appropriate new experiment rather than silently loosening the gate or consuming the reserve.

## 6. Validate each required AWS component

**Pipeline and registry:** save the execution report and actual SageMaker execution ARN. In the account console, show the successful Train → Evaluate → Condition → Register path when the real result passes. Inspect the package ARN, evaluation report reference, and `PendingManualApproval` state. Run the deliberate failure control separately with `RUN_FAILURE_DEMO=True` and show its condition/Fail path and reason. Label it as a forced failure in narration and filenames. If the normal run fails on quality or permissions, report that outcome; do not substitute a fabricated success screenshot.

**Feature Store:** enable the training-only sample and display the feature group's schema, record identifier, event time, and online/offline stores. The helper verifies a sample record through the online runtime API; show the offline S3 destination and actual delivery after propagation separately. The default disables Glue table creation to reduce lab/catalog dependencies; it does not disable offline S3 delivery. The sample demonstrates managed storage and does not replace S3 CSV training inputs. [AWS Feature Store concepts](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-concepts.html) explains the identifier and event-time contract. Record permissions/availability limitations honestly if the lab cannot create the feature group.

**Batch deployment:** use the package returned by the passing registry step. Review and explicitly approve a production-bound package; any pending-package demonstration must be clearly identified as a demonstration. The helper builds a features-only payload from the test split, preserving the saved schema order and removing targets, hashes, and lineage. Record transform job name/status and S3 output URI. Display actual scores/decisions and output row count, keeping source labels separate from the serving request. These predictions are an evaluation demonstration on known test records, not live production results.

**Monitoring and dashboard:** generate a custom PSI/drift report using the saved training-reference histograms and the demonstration batch. Include delayed-label metrics only when matching labels are actually available. Display counts, alerts, scope, and the associated run/job timestamps; drift alone does not establish an accuracy drop. Missingness here concerns normalized inputs; raw missingness was already imputed and cannot be recovered from these files. Publish it through the observability helper and open the actual CloudWatch dashboard with its custom metrics and available job/service metrics. Empty panels can indicate no emitted metric, wrong region/dimensions, propagation delay, or missing permissions; a dashboard JSON definition is not proof that metric samples arrived.

## 7. Retain evidence and clean up

Keep a sanitized run record with commit SHA, schema/manifest versions and hashes, effective configuration, account/region, execution IDs, actual model scores, package ARN/approval state, batch URI, drift JSON URI, feature-group name, and dashboard name. Save screenshots/clips only after verifying the shown account and timestamps. Do not commit large datasets, downloaded model archives, session credentials, or generated notebook outputs containing sensitive information. Local reports remain under the configured ignored reports directory; durable evidence belongs in the team's restricted S3 output prefix and final reviewed submission.

Review the cleanup inventory before enabling `RUN_CLEANUP`. The helper requires the exact current run ID and checks tracked resources so unrelated team resources are not targeted. Wait for active training, processing, and transform jobs to finish or stop them deliberately. Delete disposable registered-model entities, feature groups, and dashboards where the helper supports them; inspect and remove any remaining run-scoped resources manually. S3 outputs and retained logs may intentionally remain after compute cleanup. Delete those only after the team has retained the required evidence and reviewed retention/access needs. Do not delete shared raw data, handoffs, package groups, or another member's resources.

Stop idle SageMaker notebook instances or Studio apps through the service console after saving work. Review S3 object lifecycle, CloudWatch log retention, offline Feature Store S3 files, and any Glue metadata separately; deleting a feature group does not imply that every historical storage object was removed. No persistent real-time endpoint is part of the default workflow. For the final video and submission checklist, use [demo outline](demo_outline.md) and [rubric coverage](rubric_coverage.md).

## Troubleshooting

| Symptom | Action |
|---|---|
| `ExpiredToken` / identity mismatch | Refresh the lab/SSO session, confirm `sts get-caller-identity`, and recreate the boto session. |
| `AccessDenied` / role cannot be passed | Verify the intended identity, execution role and scoped permissions; record lab limitations. |
| S3/KMS denied or wrong region | Verify handoff/output bucket region, prefix access and KMS key policy; the two buckets can have different owners. |
| Missing handoff or schema hash mismatch | Obtain a complete matching artifact version; do not mix an old manifest with new features. |
| Legacy preprocessing rejected | Regenerate features/splits with the corrected 40-week fit boundary. |
| No approved instance quota or container image unavailable | Select an allowed regional instance/container combination and consult current AWS support; retain the error as evidence if unavailable. |
| Pipeline rejected genuine model quality | Preserve metrics, discuss model limitations, and design a new training/validation experiment without reserve leakage. |
| Feature Store/Glue unavailable in lab | Use the supported configuration if permitted; otherwise document the missing rubric evidence and request instructor guidance. |
| No metric on dashboard | Verify namespace, dimensions, region, publishing result, and time range; wait for normal service propagation. |
| Notebook dependency conflict | Install the pinned requirements into a clean kernel/environment and restart before imports. |
