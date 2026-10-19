# Deliverable #2: ML System Operation Validation — Video Demonstration Script & Outline

**Course:** AAI-540 — Machine Learning Operations (MLOps)
**Project:** Adaptive Static Malware Detection for Emerging Threats
**Team:** Group 1 — Gaurav SS, Marco Gonzalez Ortiz, Nitendra Tiwari
**Target Video Duration:** 11–13 Minutes (Maximum: 15 minutes)
**Deliverable Format:** MP4 Screencast with Voice Narration + This Script & Outline

---

## Executive Overview & Recording Guidelines

This document provides the complete, presentation-ready script, visual storyboard, and slide outline for Group 1's Module 7 video demonstration. The presentation is structured into three equal segments (~3.5 to 4 minutes each) to satisfy course teamwork and presentation requirements.

### Technical & Production Checklist
1. **Software:** Use Zoom, OBS Studio, or Screencast-O-Matic.
2. **Resolution:** 1080p (1920x1080) at 30/60 fps.
3. **Audio:** High-quality microphone; test audio levels before recording.
4. **Visual Layout:** Fullscreen screencast of repository files, running terminal tests, Jupyter notebooks with output plots, and architecture diagrams.
5. **Export Format:** `.mp4` file, named according to team guidelines (e.g., `Final-Project_Team-1_Deliverable-2.mp4`).

---

## Storyboard and Team Agenda

| Segment | Speaker | Topic & Scope | Time | Primary Screencast Asset |
|---|---|---|---|---|
| **Segment 1** | **Nitendra Tiwari** | Business Problem, SOC Triage Need, System Architecture & Data Engineering | 00:00 – 03:45 | `docs/ml_system_design.md` Architecture Diagram & `notebooks/01_ember2024_eda.ipynb` |
| **Segment 2** | **Marco Gonzalez Ortiz**| Feature Extraction, Data Integrity/Leakage, Model Comparison & Test Metrics | 03:45 – 07:30 | `notebooks/02_sagemaker_training_evaluation_local_exploratory_label_fix.ipynb` & Reports |
| **Segment 3** | **Gaurav SS** | MLOps Operational Architecture: Feature Store, Model Registry, Endpoint, Monitoring & CI/CD DAG | 07:30 – 11:30 | Terminal (`unittest` execution), `sagemaker_src/inference.py`, CI/CD DAG Diagrams |

---

## Detailed Speaking Script and Visual Directives

### Segment 1: Business Motivation, Architecture & Data Engineering
**Speaker:** Nitendra Tiwari
**Time Allocation:** [00:00 – 03:45] (~3 minutes 45 seconds)
**Visual Cue:** Title Slide -> System Architecture Diagram (`docs/ml_system_design.md`) -> `notebooks/01_ember2024_eda.ipynb`.

#### Spoken Script:
> *"Hello everyone. Welcome to the Final Project demonstration for AAI-540 MLOps, presented by Group 1: Gaurav SS, Marco Gonzalez Ortiz, and myself, Nitendra Tiwari. Our project addresses an urgent cybersecurity challenge: **Adaptive Static Malware Detection for Emerging Threats**.*
>
> *(Switch screencast to `docs/ml_system_design.md` architecture diagram)*
>
> *"In modern Security Operations Centers, analysts face tens of thousands of newly compiled Windows binaries daily. Traditional signature matching fails against recompiled or obfuscated malware, while full dynamic sandbox detonation takes minutes per file and is computationally prohibitive. Our mission was to design and validate an MLOps-ready static malware triage engine that inspects PE file structural metadata and string statistics in milliseconds, filtering high-confidence files and prioritizing suspicious threats for human analysts.*
>
> *(Switch screencast to `notebooks/01_ember2024_eda.ipynb`)*
>
> *"To ensure academic and engineering rigor, our pipeline consumes the Windows 64-bit partition of the EMBER2024 benchmark. We established strict data integrity safeguards:*
> - *First, we implemented SHA-256 deduplication to prevent repeated sample leakage.*
> - *Second, we enforced strict chronological splitting across 64 collection weeks: Training across 40 source weeks from 2023-09-24 through 2024-06-23 (400,000 files), Validation across 12 source weeks from 2024-06-30 through 2024-09-15 (120,000 files), Test across 6 source weeks from 2024-09-22 through 2024-10-27 (59,995 files), and 6 source weeks from 2024-11-03 through 2024-12-08 held untouched in Reserved Production.*
>
> *"As shown here in our validated EDA notebook, our data distributions show balanced 50/50 class representations across all temporal windows. Notice that raw JSONL feature files were not available in the local workspace. Instead, our data pipeline used an exploratory reconstruction approximation where replacement scaling moments were fitted on the training split, keeping partitions isolated while transparently disclosing the limitations of the missingness proxy.*
>
> *"Now, I'll hand over to Marco to walk through our feature pipeline, candidate modeling, and test performance."*

---

### Segment 2: Feature Engineering, Model Selection & Test Evaluation
**Speaker:** Marco Gonzalez Ortiz
**Time Allocation:** [03:45 – 07:30] (~3 minutes 45 seconds)
**Visual Cue:** `data/processed/ember2024_reconstructed_v1/feature_schema_reconstructed.json` -> `notebooks/02_sagemaker_training_evaluation_local_exploratory_label_fix.ipynb` (Cells 9–17).

#### Spoken Script:
> *"Thank you, Nitendra. Moving to feature engineering and model evaluation:*
>
> *(Show `feature_schema_reconstructed.json` in VS Code / Jupyter)*
>
> *"Our model operates on a compact contract of 13 numeric PE features. These include file size, Byte Shannon entropy, string counts, string entropy, PE section counts, imported DLL counts, export counts, Visual Studio rich header tools, parser warnings, and digital certificate indicators. Highly skewed distributions are log-transformed, and continuous features are standardized using replacement moments fitted on the reconstructed training partition, whose source weeks run through June 23, 2024. The reconstruction script does not independently enforce a per-row cutoff filter, but neither validation nor test rows were used to fit the scaler.*
>
> *(Switch screencast to `02_sagemaker_training_evaluation_local_exploratory_label_fix.ipynb` Cell 11 & Table)*
>
> *"In our local candidate comparison, we evaluated three distinct algorithms on a stratified 100,000-record training sample:*
> 1. *Logistic Regression as an interpretable linear baseline (Validation PR-AUC: 0.9494)*
> 2. *XGBoost as an advanced gradient boosted tree (Validation PR-AUC: 0.9891)*
> 3. *Random Forest as a bagged ensemble of 160 trees (Validation PR-AUC: 0.9912)*
>
> *"Random Forest achieved the highest ranking quality. But in a security operations center, raw accuracy is secondary to controlling false alarms. An analyst cannot investigate hundreds of false positives.*
>
> *(Scroll to Cell 13 & Validation Diagnostics plots in Cell 15)*
>
> *"We therefore conducted a validation probability sweep with an explicit operational constraint: **False Positive Rate must not exceed 1.0%**. The optimal threshold discovered was **0.790775**, achieving 86.58% recall and 98.86% precision on validation data.*
>
> *(Scroll to Cell 17 Test Metrics Table)*
>
> *"We then performed a single, frozen evaluation on the held-out temporal test split containing 59,995 unseen newer binaries. As you can see from our verified report:*
> - *ROC-AUC reached **0.9875** and PR-AUC reached **0.9879**.*
> - *Precision was outstanding at **98.71%**, producing only 299 false positives out of nearly 30,000 benign files.*
> - *Recall was **76.51%**, successfully intercepting 22,954 malware binaries.*
> - *F1-Score was **86.21%**.*
>
> *"In addition, as documented in our system design, a separate managed baseline Logistic Regression job was executed on Amazon SageMaker with explicit legacy-preprocessing disclosure, achieving a validation ROC-AUC of 0.9450, PR-AUC of 0.9442, recall of 89.02%, and FPR of 13.23% at threshold 0.50. These are two distinct experiments and their metrics are preserved separately.*
>
> *"We disclose that these metrics were computed against the reconstructed dataset approximation, which proves our MLOps workflow while maintaining honesty about data provenance. Now, Gaurav will present our operational MLOps infrastructure, monitoring, and CI/CD."*

---

### Segment 3: Operational Architecture, MLOps Governance & CI/CD
**Speaker:** Gaurav SS
**Time Allocation:** [07:30 - 11:30] (~4 minutes)
**Visual Cue:** PowerShell terminal showing local validation -> `sagemaker_src/inference.py` -> proposed CI/CD and AWS architecture diagrams in `docs/ml_system_design.md`.

#### Spoken Script:
> *"Thank you, Marco. I will conclude by demonstrating the engineering controls we have verified and explaining how the system is designed to be operationalized in AWS SageMaker.*
>
> *(Switch screencast to PowerShell Terminal.)*
>
> *"First, let's validate our local engineering foundation. We ran our automated unit-test suite and notebook validator using Python 3.12. The verified results were 25 unit tests passed, with zero failures, and all three notebooks passed structural and code-cell validation.*
>
> *(Show the recorded validation results. If rerunning the commands, use the repository's documented virtual-environment interpreter.)*
>
> *"These tests cover project-specific data isolation, schema and temporal-split safeguards, and other engineering checks. They are local software tests; they do not demonstrate that a cloud deployment or production monitoring system is active.*
>
> *(Switch to `sagemaker_src/inference.py` and the architecture document.)*
>
> *"Our inference module defines the model-serving contract: it validates the 13 input features in the expected schema order, obtains probability predictions, and applies the configured threshold of 0.790775. The reported model results use a reconstructed dataset approximation, not a regenerated official raw EMBER2024 dataset. The threshold and metrics should therefore be interpreted within that limitation.*
>
> *"Our proposed AWS deployment architecture has several operational components:*
>
> *1. **Feature Store:** A versioned feature group can provide consistent features for training and inference. The feature group and its retrieval latency have not been verified as deployed for this project.*
>
> *2. **Model Registry:** A registry can retain versioned model packages and evaluation metadata, with an approval gate before release. Registration and approval of a project model have not been demonstrated.*
>
> *3. **Inference:** SageMaker real-time endpoints or Batch Transform could support online triage or scheduled scanning. Neither a live endpoint nor a completed batch inference job is claimed here.*
>
> *4. **Monitoring:** CloudWatch dashboards and SageMaker monitoring capabilities could track operational metrics, data quality, and drift. The 20 ms latency value is a design target, not a measured service-level result.*
>
> *(Switch to the CI/CD DAG diagram in `docs/ml_system_design.md`.)*
>
> *"The design document describes a staged release pipeline with validation gates. In the intended successful path, code and data checks pass, the candidate meets the agreed evaluation criteria, and model approval precedes deployment. In the failure path, a failed test, temporal-leakage check, or false-positive-rate gate should stop progression. These diagrams describe the intended workflow; we have not verified an active eight-stage AWS deployment pipeline or its successful and failed executions.*
>
> *"To summarize, we have validated the local engineering foundation, documented the inference contract and proposed MLOps architecture, and disclosed the limitations of our reconstructed data. AWS Feature Store, Model Registry, endpoints, monitoring dashboards, and deployment automation remain deployment work rather than demonstrated live components. This distinction is important for reproducibility, security, and responsible malware detection.*
>
> *"Thank you for your time. We welcome your questions."*

---
---

## Technical Appendix: Screencast Key Visual Assets

1. **Architecture Overview:** `docs/ml_system_design.md` (Mermaid diagrams)
2. **EDA Visual Evidence:** `notebooks/01_ember2024_eda.ipynb` (Cells 4, 8, 10, 12)
3. **Model Training & Test Report:** `notebooks/02_sagemaker_training_evaluation_local_exploratory_label_fix.ipynb` (Cells 11, 13, 15, 17)
4. **Local Verification Terminal Command:**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover tests -v
   .\.venv\Scripts\python.exe scripts\validate_notebooks.py
   ```
5. **Inference Contract Code:** `sagemaker_src/inference.py`
