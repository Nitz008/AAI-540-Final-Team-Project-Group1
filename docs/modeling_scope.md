# Modeling workstream: scope and source alignment

This branch supports the current request to implement **ML modeling only** in `marco-model-training-evaluation`. The attached RFC and the [Group 1 teamwork tracker](https://docs.google.com/document/d/1acXxwMhhT72I4s_3FafBtHH58gxTVbYuXE2U2JbwsPE/edit?tab=t.0) provide project context. Their full-system plans do not expand this branch into other members' implementation areas. The original DOCX and shared tracker are reference sources; this work does not edit either document or execute the tracker's sharing/template directions.

## Three-member team and existing ownership context

| Member | Work described in the tracker | Relationship to this branch |
|---|---|---|
| Gaurav SS | Owns the data pipeline: validation, EDA, feature preparation, temporal splits, metadata, and handoff documentation. | Modeling consumes that schema and prepared-data contract; necessary preprocessing boundary corrections are documented for regeneration. |
| Marco Gonzalez Ortiz | Reviews model-ready data/features, defines benchmark/evaluation approach, and describes initial model training/debugging and metric comparison. | The branch implements the reproducible baseline, candidate comparison, policy/threshold choice, final evaluation, and model handoff. |
| Nitendra Tiwari | Reviews feature engineering/data preparation, Feature Store planning, and monitoring/operational requirements. | Receives the selected model, schema, metrics, and provenance for subsequent integration; this branch does not claim completion of that work. |

These are descriptions from the tracker, not independent verification of particular jobs or contributions. The tracker provides no numeric model results that can be reused as measured results for this implementation. No additional members, role assignments, fabricated commits, or cloud achievements are inferred.

## Source decisions and implementation choices

| Topic | Source context | Modeling implementation |
|---|---|---|
| Classification target | RFC: supervised benign/malicious static-file detection for analyst triage. | Binary target `label`; maliciousness score and frozen-threshold decision. |
| Dataset minimum | RFC: at least 10,000 labeled observations per predicted class. | Validate/report actual real-data counts and the configured class-count scope. Small synthetic tests remain labeled as tests. |
| Feature contract | Tracker Module 4: 13 approved static aggregates; exclude hashes, labels, AV/time metadata and other leakage risks. | Preserve ordered schema columns and metadata exclusions. |
| Preprocessing | Tracker describes training-only normalization; the earlier code fitted all 52 supplied training weeks. | Fit only the first 40 model-training weeks through week start `2024-06-23`; regenerate old artifacts. |
| Temporal split | RFC initially reserves the official test partition; concrete Module 4 tracker and README adopt 40/12/6/6 weeks. | Preserve the later documented hybrid; identify final test as an official-test-derived subset and keep the newest six-week reserve untouched. |
| Models | RFC: Logistic Regression, Random Forest, and either XGBoost or LightGBM. | Choose XGBoost 3.0.5 as the documented alternative; compare the three required families with bounded trials. |
| Baseline and FPR budget | RFC: establish acceptable FPR after baseline evaluation; multiple thresholds. | Baseline uses validation threshold tradeoffs first. No FPR is preapproved; save an explicit decision before model comparison. |
| Selection | RFC pending Option A: seven weights 30/25/15/10/10/5/5; general text lists additional criteria. | Provide Option A as configurable proposal; record the approved policy. Report precision, ROC-AUC, accuracy, and qualitative deployment complexity alongside weighted criteria. |
| Latency and size | RFC requests predictive and operational comparison. | Measure local `predict_proba` p50/p95 latency and serialized bundle size. Do not call local timing endpoint latency. |
| Final test | RFC restricts training/tuning to train and validation. | Freeze winner/threshold/policy before separately enabled test download/evaluation; no retuning or reserve reads. |
| Reproducibility | RFC asks for parameters, data versions, metrics, and artifacts. | Save model/schema, baseline/comparison/final reports, policy, seeds, versions, source/data hashes, and actual account job identifiers when run. |
| Individual AWS accounts | Tracker Module 5 states Learner Lab accounts are individual and large artifacts remain outside Git. | Parameterize existing bucket, region, execution role, S3 handoff and local paths; use credential providers and explicit managed-job opt-in. |

## Work kept outside this branch

The RFC's wider system roadmap includes real-time deployment, Feature Store, Model Registry, monitoring/adaptation, dashboards, Terraform, and deployment CI/CD. These remain integration context rather than deliverables claimed by this modeling implementation. The handoff supplies the model and evidence those workstreams need. SageMaker training jobs, evaluation in the notebook kernel, and S3 artifact exchange remain in scope because they allow teammates to reproduce modeling in their own accounts.

The project sources also leave modeling policy choices open. XGBoost and the proposed Option A weights are implementation choices that remain visible and configurable; they are not described as historical approvals. The acceptable FPR, rationale, and final policy must be recorded after baseline review and before comparison/test. No source text is used to invent performance values, endorse a release, or imply that a real AWS run has happened.
