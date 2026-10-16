# Adaptive Static Malware Detection for Emerging Threats

AAI-540 MLOps Final Team Project - Group 1

This repository contains the reproducible data pipeline and ML modeling workflow for benign-versus-malicious classification of EMBER2024 Win64 static features. The `marco-model-training-evaluation` branch consumes the prepared data handoff to establish a Logistic Regression baseline, compare Logistic Regression, Random Forest, and XGBoost, and evaluate a frozen model on a newer temporal test subset.

## ML modeling on this branch

The project has three members: **Gaurav SS, Marco Gonzalez Ortiz, and Nitendra Tiwari**. This branch focuses on Marco's model-training and evaluation workstream, using the data pipeline and handoff documented below. Its scope follows the project RFC and the [team tracker](https://docs.google.com/document/d/1acXxwMhhT72I4s_3FafBtHH58gxTVbYuXE2U2JbwsPE/edit?tab=t.0).

Start with [02_sagemaker_training_evaluation.ipynb](notebooks/02_sagemaker_training_evaluation.ipynb) and the [SageMaker modeling runbook](docs/sagemaker_runbook.md). The notebook supports local debugging and managed SageMaker training in each teammate's own account. An existing S3 bucket, authorized execution role, region, and normal AWS credential provider are used; no credentials are stored in source.

```bash
git clone --branch marco-model-training-evaluation https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1.git
cd AAI-540-Final-Team-Project-Group1
python -m pip install -r requirements-sagemaker.txt
cp configs/sagemaker.example.json configs/sagemaker.local.json
jupyter lab notebooks/02_sagemaker_training_evaluation.ipynb
```

Use `Copy-Item` for the copy command in PowerShell. Edit the ignored local configuration before starting paid jobs. The default notebook starts no AWS jobs and downloads no final-test data.

The modeling sequence is:

1. Validate the prepared train/validation handoff and its schema; check real class counts and temporal separation.
2. Fit the Logistic Regression benchmark on training rows and inspect validation threshold tradeoffs.
3. Choose and record the FPR budget and model-selection policy after baseline review. The RFC's proposed security weights remain configurable and require an explicit recorded decision.
4. Compare and tune Logistic Regression, Random Forest, and XGBoost using train/validation only. Report classification metrics, measured local inference latency, artifact size, and documented interpretability and deployment tradeoffs.
5. Freeze the winning model, threshold, policy, and data provenance, then explicitly run final test evaluation once. Leave the newest reserved-production split untouched.

The outputs are model artifacts, validation comparisons, final-test reports and plots, and a reproducible handoff for the teammates integrating deployment and operations. The notebook labels timing as a local CPU inference measurement; it does not claim endpoint latency. No measured real-data scores or successful AWS jobs are supplied with unexecuted source code.

- [ML modeling design](docs/ml_system_design.md)
- [Scope and source alignment](docs/modeling_scope.md)
- [SageMaker modeling runbook](docs/sagemaker_runbook.md)

**Preprocessing compatibility:** older handoffs normalized using all 52 official training weeks, including validation. This branch fits transformations only through the final training week starting `2024-06-23`. The data owner must regenerate and publish a matching schema, CSV files, and split manifest before strict comparison; an explicit exploratory legacy override does not repair leakage or qualify the result as a final experiment.

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
configs/       Portable data and per-account modeling configuration examples
docs/          Data documentation, modeling design, and SageMaker runbook
notebooks/     EDA notebook and separate model-training/evaluation notebook
scripts/       Data preparation and notebook validation entry points
src/           Reusable data and account-neutral modeling helpers
sagemaker_src/ Model fitting, comparison, final evaluation, and inference contract
tests/         Synthetic modeling and data-isolation checks
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

Each member running managed model training needs an authorized session and resources in their own AWS account. The data pipeline is local-first. Do not commit credentials, access keys, bucket names containing personal information, or large data files.

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

For initial modeling, download only the schema, manifest, train, and validation files. The notebook's `download_handoff` helper performs these named downloads. The equivalent CLI filter is:

```bash
aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/processed/ember2024_win64_static_v1/ \
  data/processed/ember2024_win64_static_v1/ \
  --exclude "*" --include "feature_schema.json" \
  --include "temporal_splits_v1/split_manifest.json" \
  --include "temporal_splits_v1/train.csv" --include "temporal_splits_v1/validation.csv"

```

The final-test stage downloads `test.csv` only after model selection is frozen. Keep `reserved_production.csv` outside the modeling environment. Optional reports for the separate EDA notebook can be downloaded with:

```bash

aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/reports/validation/ \
  reports/validation/

aws s3 sync s3://YOUR-BUCKET-NAME/ember2024/reports/eda/ \
  reports/eda/
```

The modeling notebook consumes prepared features without the raw dataset. The separate EDA notebook can display completed results from the optional reports; its regeneration cells require raw files.

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

## Modeling checks

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
```

Tests use synthetic data and stub AWS calls. Final model quality must be measured on the actual approved data. Large datasets, account configuration, generated reports, and model files remain outside Git.
