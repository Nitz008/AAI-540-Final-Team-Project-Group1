# Final project requirement and evidence map

Branch: [`marco-model-training-evaluation`](https://github.com/dodo0-0/AAI-540-Final-Team-Project-Group1/tree/marco-model-training-evaluation).

**Runnable implementation** means code/documentation is provided for the requirement. **Needs AWS execution** means the team must run it in an authorized account and retain actual outputs. **Human delivery** means a recording, review, contribution, or submission step remains. No real model score, AWS success, video, or final PDF is implied by this map.

## Deliverable 1: technical ML design

| Requirement | Repository location | Status / evidence to collect |
|---|---|---|
| Clearly defined ML problem, 1–2 paragraphs | `docs/ml_system_design.md` → Problem statement | Runnable documentation: two paragraphs define binary static Win64 triage and scope. |
| Goal-linked impact measures, 1–2 paragraphs | Design → Goals and impact measurement | Runnable documentation: two paragraphs connect recall/FPR and review workload; replace unobserved metrics with real results only after execution. |
| Security checklist, sensitive data, bias and ethics | Design → Security checklist; Bias, sensitive data, and ethical risks | Implementation/configuration/operational status distinguished; account owner must verify AWS controls and reviewer decisions. |
| Data Sources, 2–3 paragraphs | Design → Data Sources; `docs/dataset_inventory.md` | Two substantive paragraphs, local derivative/provenance caveats and scope included. |
| Data Engineering, 2–3 paragraphs | Design → Data Engineering; `src/malware_data`; split configs | Two paragraphs grounded in validation, deduplication and strict chronology. |
| Feature Engineering, 2–3 paragraphs | Design → Feature Engineering; `docs/feature_preparation.md`; feature schema | Two paragraphs, approved/excluded fields and corrected preprocessing fit boundary. |
| Model Training & Evaluation, 2–3 paragraphs | Design → Model Training & Evaluation; `sagemaker_src`; notebook 02 | Runnable implementation; real candidate comparison and held-out metrics need dataset/AWS execution. |
| Model Deployment, 2–3 paragraphs | Design → Model Deployment; `src/malware_mlops` | Runnable registry and registered-model batch path; account evidence and manual review remain. |
| Model Monitoring, 2–3 paragraphs | Design → Model Monitoring; `sagemaker_src/monitor.py`; AWS observability helpers | Runnable custom drift/delayed-label reports and dashboard path; distinguish from scheduled managed Model Monitor. |
| CI/CD, 2–3 paragraphs | Design → CI/CD; `.github/workflows` | Runnable local CI and optional account-configured AWS workflow; real GitHub/AWS execution evidence remains. |
| Repository link and coherent technical audience | Design opening; code references and diagrams | Documentation provided; team review against final artifacts required. |
| Final standardized PDF | Export reviewed design as `Final-Project_Team-1_Deliverable-1.pdf` | Human delivery: PDF export and visual review remain. |

## Deliverable 2: operation and demonstration

| Required screen/narration | Implementation/location | Evidence still required |
|---|---|---|
| Business use case and architecture | Design problem statement and Mermaid diagrams; notebook 02 | Human delivery: narrate scope/goals and readable architecture. |
| SageMaker implementation | `notebooks/02_sagemaker_training_evaluation.ipynb`; `scripts/run_sagemaker_pipeline.py`; `src/malware_mlops` | Needs AWS execution: account-local real jobs, statuses, identifiers and S3 reports. |
| Feature Store and feature groups | Optional training-only sample via AWS helpers | Needs AWS execution: group schema, store configuration, ingestion and read/delivery proof. Permission denial is a documented limitation, not success evidence. |
| Infrastructure monitoring dashboard | CloudWatch publishing/dashboard helpers | Needs AWS execution: actual timestamped dashboard and job/service/custom metrics. |
| Model/data monitoring reports | `sagemaker_src/monitor.py`; notebook plots | Needs dataset/AWS execution: genuine drift report and labeled metrics when available, with scope disclosed. |
| CI/CD DAG successful state | SageMaker Train → Evaluate → Condition → Register; GitHub CI | Needs AWS/GitHub execution: successful DAG with real qualifying test report and real CI checks. |
| CI/CD DAG failed state | Forced-failure parameter → Fail step | Needs AWS execution: distinct deliberately forced-failed execution and honest narration. A local diagram is insufficient. |
| Model Registry | Passing pipeline registers `PendingManualApproval` package | Needs AWS execution: package ARN/version, linked metrics, and actual approval state. |
| Batch inference or endpoint output | Registered-model batch transform; default creates no endpoint | Needs AWS execution: completed transform, actual output URI, scores/decisions and row count. |
| Future work, difficulties and risks | Design risk/monitoring sections; `docs/demo_outline.md` | Human delivery: actual encountered problems, wider evaluation, automation and security limitations. |
| Planned narration/outline/transcript | `docs/demo_outline.md` | Outline provided; replace placeholders with team names and observed results. |
| Equal team presentation and clear audio/video | Demo timing plan | Human delivery: assign equal speaking time, rehearse, record, edit, review. |
| Duration | Demo targets 10 minutes | Human delivery: resolve 10–15-minute body vs ≤10-minute highest rubric rating with instructor. |
| Final MP4 | `Final-Project_Team-1_Deliverable-2.mp4` | Human delivery: no recording is created by this branch. |

## Deliverable 3: eight codebase requirements

| # | Requirement | Repository support | Status / remaining evidence |
|---|---|---|---|
| 1 | Clean professional GitHub code; `.ipynb` notebooks | Organized `src`, `sagemaker_src`, scripts/configs/docs, notebooks 01 and 02 | Runnable implementation; branch must be committed/pushed and reviewed by team. |
| 2 | Useful clean goal-directed code/comments | Reusable parsing, feature, temporal, training, serving and AWS modules | Runnable implementation and tests; maintainers review final diff. |
| 3 | Data stored in S3 and documented | Handoff/output configuration; runbook transfer and versioning instructions | Needs AWS execution: actual restricted bucket/prefix and retained manifests. The repo does not upload data until enabled. |
| 4 | Explanatory graphics in notebooks | EDA notebook and training/evaluation/architecture cells in notebook 02 | Runnable plotting; render with genuine local/cloud reports before recording. |
| 5 | Comprehensive ML system codebase | Upstream data pipeline plus training, eval, serving, registry, Feature Store and monitoring paths | Runnable implementation; account constraints and production extensions disclosed. |
| 6 | Code and design reinforce each other | Design links to exact code paths; runbook and evidence map | Documentation provided; update final design with actual execution IDs, metrics and limitations. |
| 7 | Every team member contributes | Branch workflow and scoped account configuration permit independent work | Human delivery: each member makes real code/docs/review contributions; do not manufacture authorship. |
| 8 | Commit history available for review | GitHub branch/repository link | Human delivery: commit/push meaningful contributions and retain history for instructor. |

## Final submission checklist

| Item | Owner / status |
|---|---|
| Real leakage-safe data version regenerated if needed | Data owner; confirm `fit_scope`, cutoff and manifest hash before final training. |
| Actual account runs, model scores and component evidence | Team AWS operators; still required until executed. |
| Final document checked for clarity, citations and alignment | Team review; similarity/Draft Coach review per course guidance. |
| Standardized design PDF, MP4, outline and repository reference | Team; source Markdown/notebook exist, final exports and video remain. |
| One delegate selected and all required files submitted together | Team human action; not automated. |
| Separate individual peer evaluations | Each member; human action through the course link. |

Use [the runbook](sagemaker_runbook.md) for execution and cleanup, and [the demo outline](demo_outline.md) for evidence capture and balanced narration. Retain honest partial evidence when a lab restriction prevents a component, and identify the requirement as unresolved until it is demonstrated or the instructor approves an alternative.
