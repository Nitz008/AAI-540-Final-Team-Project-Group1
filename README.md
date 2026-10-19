# Adaptive Static Malware Detection for Emerging Threats

**AAI-540 MLOps Final Team Project — Group 1**
**Team Members:** Gaurav SS, Marco Gonzalez Ortiz, Nitendra Tiwari
**Institution:** University of San Diego
**Repository:** [AAI-540-Final-Team-Project-Group1](https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1)

---

## Executive Summary & Deliverables Guide

This repository contains the complete MLOps engineering codebase, exploratory analysis, candidate model comparison, test evaluation, and system design documentation for benign-versus-malicious classification of Windows 64-bit (Win64) executables based on the EMBER2024 static feature benchmark.

### Final Course Deliverables (Module 7)

| Deliverable | Description | Key Repository Artifacts |
|---|---|---|
| **Deliverable #1: ML System Design Document** | Comprehensive technical design document covering problem framing, architecture, data engineering, feature contracts, model evaluation, deployment, monitoring, CI/CD, security/ethics, and implementation evidence. | - [`docs/ml_system_design.md`](docs/ml_system_design.md)<br/>- [`docs/AAI540_ML_System_Design_Document_Group1.docx`](docs/AAI540_ML_System_Design_Document_Group1.docx) |
| **Deliverable #2: Video Demonstration Script & Outline** | 10–13 minute presentation script and slide outline detailing business context, architecture, live codebase walkthrough, MLOps operations, and speaker allocations. | - [`docs/video_demonstration_script_and_outline.md`](docs/video_demonstration_script_and_outline.md) |
| **Deliverable #3: Full Codebase & Reproducible Artifacts** | Clean, modular Python codebase with passing automated unit tests, validated notebooks, feature schemas, and reports. | - [`notebooks/`](notebooks/)<br/>- [`tests/`](tests/)<br/>- [`src/`](src/) & [`sagemaker_src/`](sagemaker_src/)<br/>- [`reports/exploratory_local/`](reports/exploratory_local/) |

---

## Notebooks and Execution Workflows

The repository organizes analysis and modeling into explicit, reproducible notebooks and cloud training artifacts:

### 1. `notebooks/01_ember2024_eda.ipynb`
- **Purpose:** Exploratory Data Analysis of the EMBER2024 Win64 dataset.
- **Contents:** Feature distribution plots, correlation heatmaps, class balance distributions, and temporal collection timelines.
- **Status:** Validated; all visual outputs, tables, and figures are fully preserved. Expensive raw data regeneration cells default to `False`.
- **Data Source:** Analyzes `data/processed/ember2024_reconstructed_v1/`.

### 2. `notebooks/02_sagemaker_training_evaluation_local_exploratory_label_fix.ipynb`
- **Purpose:** Local, reproducible candidate model comparison and temporal test evaluation.
- **Methodology & Exploratory Random Forest Metrics:**
  - Evaluates Logistic Regression, Random Forest, and XGBoost on a stratified 100,000-record training sample from the reconstructed approximation.
  - Compares candidates on the full 120,000-record validation partition (Random Forest achieved top validation PR-AUC: 0.9912).
  - Sweeps validation probabilities to select an operating threshold ($\tau = 0.790775$) enforcing a strict **False Positive Rate budget $\le 1.0\%$**.
  - Performs a single, frozen evaluation on the held-out 59,995-record temporal test partition:
    - **Validation PR-AUC:** 0.9912
    - **Selected Threshold:** 0.790775 (targeting validation FPR $\le$ 1%)
    - **Test Precision:** 98.71% (299 false positives out of 29,995 benign files)
    - **Test Recall:** 76.51% (22,954 true positives out of 30,000 malware files)
    - **Test F1-Score:** 86.21%
    - **Test ROC-AUC:** 0.9875 | **Test PR-AUC:** 0.9879
- **Status:** Fully executable locally; verified and consistent with saved reports in `reports/exploratory_local/`.

### 3. `notebooks/02_sagemaker_training_evaluation.ipynb` & SageMaker Baseline Training
- **Purpose:** Guarded Amazon SageMaker production training and evaluation workflow.
- **Safety Policy:** All cloud execution switches remain disabled by default (`RUN_AWS = False`, `DOWNLOAD_HANDOFF = False`, `RUN_BASELINE = False`, `RUN_COMPARISON = False`, `RUN_FINAL_TEST = False`). Zero AWS charges are incurred during default notebook execution.
- **Separate SageMaker Baseline Logistic-Regression Evaluation:**
  - An exploratory baseline Logistic Regression training job was run on SageMaker using the preprocessed handoff and legacy preprocessing compatibility.
  - Cloud artifacts are retained and represent a separate exploratory experiment:
    - **Threshold:** 0.5
    - **ROC-AUC:** 0.9449563
    - **PR-AUC:** 0.9442056
    - **Recall:** 0.89015
    - **Precision:** 0.8705907
    - **F1-Score:** 0.8802617
    - **False-Positive Rate:** 0.1323167
    - **Accuracy:** 0.8789167
    - **Specificity:** 0.8676833
    - **Balanced Accuracy:** 0.8789167
- **Note on Metric Separation:** The local notebook Random Forest evaluation and the SageMaker baseline Logistic Regression job are distinct experiments. Their attributions, model types, operating thresholds, and experimental contexts are preserved separately and not interchanged.

---

## Dataset Scope, Provenance & Integrity Disclosures

- **Target Binary Task:** Supervised classification where `0` = Benign and `1` = Malicious Windows 64-bit Portable Executables (PE) using EMBER2024-derived static features.
- **Verified Temporal Partitions:**
  - **Training:** 400,000 rows (200,000 benign / 200,000 malicious); source weeks 2023-09-24 through 2024-06-23 (40 weeks).
  - **Validation:** 120,000 rows (60,000 benign / 60,000 malicious); source weeks 2024-06-30 through 2024-09-15 (12 weeks).
  - **Test:** 59,995 rows (29,995 benign / 30,000 malicious); source weeks 2024-09-22 through 2024-10-27 (6 weeks).
  - **Reserved Production:** 59,998 rows (29,998 benign / 30,000 malicious); source weeks 2024-11-03 through 2024-12-08 (6 weeks, held untouched).
- **Data Provenance & Reconstruction Caveats:**
  - Original raw EMBER2024 JSONL files (~33.5 GB) were not available in the local workspace for this finalization.
  - The processed CSV files in `data/processed/ember2024_reconstructed_v1/` are reconstructed approximations derived by reversing previously stored standardization transformations and fitting replacement scaling moments on the reconstructed training split.
  - The reconstruction script fits replacement scaling moments on the supplied training split but does not independently enforce a per-row source-week cutoff. The verified training split's latest source week is 2024-06-23. Neither validation, test, nor reserved-production splits were used to fit the replacement scaling moments.
  - Original missing-value indicators were lost during earlier preprocessing. A three-feature zero-pattern proxy (`section_count`, `authenticode_num_certs`, `authenticode_self_signed`) was used to estimate missingness and may misclassify legitimate zero values.
  - Because of these reconstruction and missingness limitations, exact reproducibility of the original preprocessing pipeline cannot be claimed. Model metrics reflect exploratory results from the respective experiments and demonstrate the MLOps workflow rather than an exact raw-data regeneration or a production-certified deployment.
  - The reserved-production split was preserved untouched as a holdout and is not presented as a completed production validation exercise.

---

## Local Verification and Automated Testing

The repository relies on Python 3.12 and established unit tests.

### 1. Run Automated Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```
**Outcome:** 25/25 unit tests pass cleanly:
```text
Ran 25 tests in 10.693s
OK
```

### 2. Validate Notebook Abstract Syntax Trees (AST)
```powershell
.\.venv\Scripts\python.exe scripts\validate_notebooks.py
```
**Outcome:** All committed notebooks possess valid structure and compilable code cells.

---

## Repository Structure

```text
configs/        Configuration templates for local and SageMaker execution
docs/           Complete ML System Design Document, Runbook, Video Script & Strategy Docs
notebooks/      Validated EDA and model training & evaluation notebooks
reports/        Saved local evaluation metrics, threshold sweeps, and comparison CSVs
sagemaker_src/  Model training, evaluation, and inference contract (inference.py)
scripts/        Reconstruction scripts, validation entry points, and notebook validators
src/            Modular data processing, logging, temporal splitting, and validation libraries
tests/          Automated unit tests for data isolation, schemas, and notebook execution
data/processed/ Reconstructed temporal splits (train, validation, test, reserved_production)
```

---

## Remaining User-Facing Tasks Before Submission

1. **Record Video Demonstration:**
   Using [`docs/video_demonstration_script_and_outline.md`](docs/video_demonstration_script_and_outline.md), record a 10–13 minute screencast in MP4 format divided among all three teammates (Gaurav SS, Marco Gonzalez Ortiz, Nitendra Tiwari).
2. **Submit Group Deliverables via Canvas:**
   One team delegate submits all three files simultaneously:
   - Deliverable #1: `docs/AAI540_ML_System_Design_Document_Group1.docx` (or PDF export)
   - Deliverable #2: Video file (`.mp4`) + Video Transcript / Outline (`docs/video_demonstration_script_and_outline.md`)
   - Deliverable #3: Link to Team GitHub Codebase
3. **Submit Individual Peer Evaluations:**
   Each team member individually completes the Module 7 Peer Evaluation form on Canvas.
