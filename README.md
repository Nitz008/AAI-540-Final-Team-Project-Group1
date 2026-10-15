# Adaptive Static Malware Detection for Emerging Threats

AAI-540 MLOps Final Team Project - Group 1

This repository implements a reproducible ML system for benign-versus-malicious static malware classification using the Win64 portion of EMBER2024. The original validation, EDA, feature preparation, and chronological splits are extended on `marco-model-training-evaluation` with model comparison, validation-only threshold selection, test evaluation, and an account-portable AWS SageMaker workflow. Cloud components must be run in your account to produce deployment evidence; no measured model scores or successful AWS runs are claimed by the source code alone.

## SageMaker quick start

Start with [02_sagemaker_training_evaluation.ipynb](notebooks/02_sagemaker_training_evaluation.ipynb). It walks through configuration, S3 handoff validation, managed training and evaluation, a conditional model registry gate, Feature Store, batch inference, custom drift reports, CloudWatch dashboards, and cleanup. Paid AWS actions are disabled by default. Each teammate uses an existing bucket and execution role in their own account; credentials come from the normal AWS credential provider chain.

```bash
git clone --branch marco-model-training-evaluation https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1.git
cd AAI-540-Final-Team-Project-Group1
python -m pip install -r requirements-sagemaker.txt
cp configs/sagemaker.example.json configs/sagemaker.local.json
# Edit the local JSON for your region, bucket, execution role, member, and run ID.
jupyter lab notebooks/02_sagemaker_training_evaluation.ipynb
```

In PowerShell, replace `cp` with `Copy-Item` if needed. Use the [SageMaker runbook](docs/sagemaker_runbook.md) for exact account setup, S3 paths, IAM requirements, instance quotas, and execution steps. Existing processed artifacts must be regenerated with this branch's train-window-only preprocessing before strict model training: the earlier schema fitted normalization on all 52 official training weeks, including the validation period. The notebook explains the explicit legacy reproduction option and its limitation.

- [ML system design](docs/ml_system_design.md): problem, impact, architecture, implementation rationale, security, bias, ethics, and operational risks.
- [Rubric coverage and evidence](docs/rubric_coverage.md): code paths and remaining account-run/team deliverables.
- [Video outline](docs/demo_outline.md): timed narration, equal participation, and evidence to record.

The managed DAG is **Train → Evaluate on test → Quality gate → Register (PendingManualApproval) or Fail**. Model and threshold selection use validation only. The newest reserved-production split remains untouched by this workflow. Batch deployment requires an approved registry package by default. Monitoring here is an implemented custom drift/performance report with CloudWatch integration, not a configured SageMaker Model Monitor schedule.

## Data pipeline contributions

The data-pipeline ownership area delivered in this repository includes:

- Raw EMBER2024 Win64 dataset setup, file inventory, schema inspection, and documentation.
- Streaming data validation for partitions, row/label counts, data types, missing values, class balance, temporal fields, and SHA-256 duplicate detection.
- EDA notebook and reusable EDA modules covering distributions, outliers, class comparisons, correlations, temporal patterns, and leakage review.
- Identification of a row-duplication issue in the Kaggle copy and SHA-256 deduplication safeguards in downstream processing.
- Leakage-aware static-feature preparation with train-only fitted transformations and a machine-readable feature schema.
- Strict chronological train, validation, test, and reserved-production datasets with split manifests and per-split metadata for model-training handoff.
- Local-first, reproducible documentation and an S3 artifact-handoff plan; no large dataset files are committed to Git.

## Repository contents

```text
configs/       Portable JSON configurations for validation, EDA, features, and splits
docs/          Dataset inventory, EDA, feature-preparation, and temporal-split documentation
notebooks/     EDA notebook and account-portable SageMaker system notebook
scripts/       CLI entry points for reproducible local pipeline stages
src/           Reusable data and AWS orchestration modules
sagemaker_src/ Managed training, evaluation, inference, and custom monitoring code
tests/         Synthetic data, leakage, inference, and cloud-contract checks
.github/       CI checks and optional account-configured AWS workflow
data/raw/      Local-only EMBER2024 source JSONL files; ignored by Git
data/processed/ Local-only prepared features and final splits; ignored by Git
reports/       Local-only validation and EDA reports; ignored by Git
```

## Dataset scope and integrity note

The local dataset is the **Win64** EMBER2024 subset only. It contains supplied weekly train and test JSONL files. The Kaggle copy contained repeated SHA-256 records, so validation, EDA sampling, feature preparation, and final split creation all explicitly deduplicate by SHA-256.

The dataset's official temporal layout is 52 training weeks followed by 12 test weeks. The final project uses a documented chronological hybrid strategy rather than a random or 40/10/10/40 split:

| Split | Dates | Role |
|---|---|---|
| Train | 2023-09-24 to 2024-06-29 | Future model fitting |
| Validation | 2024-06-30 to 2024-09-21 | Future selection/threshold work |
| Test | 2024-09-22 to 2024-11-02 | Unseen newer-threat evaluation |
| Reserved production | 2024-11-03 to 2024-12-14 | Final untouched newest-time evaluation |

The test and production-reserve periods derive from the supplied official test partition. Results must therefore be described as using an **official-test-derived temporal subset**, not an untouched full official EMBER2024 test evaluation. Details are in [docs/temporal_split_strategy.md](docs/temporal_split_strategy.md).

## Dataset for model-training teammates

The model-training handoff is the following local-only directory:

```text
data/processed/ember2024_win64_static_v1/
+-- feature_schema.json
+-- processing_report.json
+-- temporal_splits_v1/
    +-- train.csv
    +-- validation.csv
    +-- test.csv
    +-- reserved_production.csv
    +-- split_manifest.json
    +-- metadata/
        +-- train_metadata.json
        +-- validation_metadata.json
        +-- test_metadata.json
        +-- reserved_production_metadata.json
```

Use only the ordered columns in `feature_schema.json` → `model_feature_columns` as `X`. Do **not** use `record_id`, `label`, `source_partition`, `source_week_start`, or `source_week_end` as model features. `label` is the binary target; the remaining non-feature columns are lineage metadata.

Large raw and processed data must not be committed to Git. Share these artifacts through S3, not GitHub.

## Local setup and pipeline

Run commands from the repository root (`project/`).

```powershell
python -m pip install -r requirements.txt
```

The reusable scripts use only the Python standard library. The requirements file supplies notebook visualization packages.

```powershell
# Optional only when raw data is available locally
python scripts/run_validation.py
python scripts/run_eda.py
python scripts/prepare_features.py
python scripts/create_temporal_splits.py
```

Open `notebooks/01_ember2024_eda.ipynb` in Jupyter after its required artifacts are present. The optional generation cells default to `False`; they do not rerun large raw-data jobs unless intentionally enabled.

## S3 handoff for AWS Learner Lab

Each teammate running the SageMaker notebook needs their own authorized AWS session, execution role, and existing S3 bucket. The data pipeline remains local-first. Do not commit credentials, access keys, per-account configuration, or large data files. Cross-account access to another teammate's handoff bucket must be granted by that bucket's owner, or the data must be copied into your own bucket first.

### 1. Configure AWS access

In the AWS Learner Lab terminal or configured local terminal, confirm the active identity:

```bash
aws sts get-caller-identity
```

If the CLI is not already configured, use the Learner Lab-provided temporary credentials and region. Do not place credentials in this repository.

### 2. Choose a bucket and upload only required artifacts

Replace `YOUR-BUCKET-NAME` with the S3 bucket created by the AWS owner. The following uploads the model-training handoff and the reports needed to view the submitted notebook without rerunning raw-data processing:

```bash
aws s3 sync data/processed/ember2024_win64_static_v1/ \
  s3://YOUR-BUCKET-NAME/ember2024/processed/ember2024_win64_static_v1/

aws s3 sync reports/validation/ \
  s3://YOUR-BUCKET-NAME/ember2024/reports/validation/

aws s3 sync reports/eda/ \
  s3://YOUR-BUCKET-NAME/ember2024/reports/eda/
```

This intentionally does **not** upload `data/raw/` by default. The raw Win64 JSONL data is large (~33.5 GB locally) and is only needed if someone must rerun validation, EDA sampling, or feature preparation from source.

If raw-data reproducibility is required and the bucket has adequate storage, upload it separately:

```bash
aws s3 sync data/raw/ember2024/ \
  s3://YOUR-BUCKET-NAME/ember2024/raw/ember2024/
```

### 3. Download artifacts in another environment

From a SageMaker notebook instance, EC2 machine, or another authorized environment, recreate the expected local layout:

```bash
aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/processed/ember2024_win64_static_v1/ \
  data/processed/ember2024_win64_static_v1/

aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/reports/validation/ \
  reports/validation/

aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/reports/eda/ \
  reports/eda/
```

With these downloaded artifacts, the submitted notebook can display its completed validation/EDA results, feature schema, and final split metadata without the raw dataset. Raw files are necessary only for optional regeneration cells.

### Recommended S3 prefix layout

```text
s3://YOUR-BUCKET-NAME/ember2024/
+-- processed/ember2024_win64_static_v1/
+-- reports/validation/
+-- reports/eda/
+-- raw/ember2024/                 # Optional; large and immutable
```

Keep raw data immutable. When a later pipeline version is created, write it under a new versioned processed prefix rather than overwriting `ember2024_win64_static_v1`.

## Documentation

- [Dataset inventory](docs/dataset_inventory.md)
- [Validation instructions](docs/dataset_validation.md)
- [EDA instructions](docs/eda.md)
- [Feature preparation](docs/feature_preparation.md)
- [Final temporal split strategy](docs/temporal_split_strategy.md)

## Local verification

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
```

Tests use synthetic records and AWS stubs; they do not download EMBER2024, create cloud resources, or establish production accuracy. Notebook outputs are intentionally cleared for sharing. After a real run, save charts, reports, execution ARNs, registry version, and batch outputs as evidence in your private versioned S3 prefix. Review outputs before committing any executed notebook. No teammate identities or contributions are fabricated; each member must commit their own work and participate in the recording.
