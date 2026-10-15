# Final project demonstration outline

Proposed deliverable: `Final-Project_Team-1_Deliverable-2.mp4`, with this document or an edited transcript supplied alongside it. Record actual AWS execution evidence from the branch; notebook cells, local diagrams, or synthetic metrics cannot establish that an AWS component operated successfully.

The assignment body requests 10–15 minutes, but the highest video rating states no more than 10 minutes. Plan a **10-minute** recording and confirm the instructor's interpretation before final export. Trim pauses and transitions instead of speeding up narration. Assign every member an equal speaking duration: `600 / number_of_members` seconds, adjusting the segment boundaries below. Replace `Speaker A/B/C/D` with actual member names and rebalance for the real team size; these placeholders are not a statement about the team roster.

## Preparation before recording

1. Complete one genuine AWS run and retain its identity, data version, reports, model package, transform outputs, Feature Store state, and CloudWatch dashboard. A real quality-gate success is required to show a successful registry path; if gates fail, explain that outcome and resolve the evidence gap before claiming completion.
2. Complete a separately labeled forced-failure pipeline run using the supplied parameter. Keep its execution ID distinct from the genuine run.
3. Open the notebook's architecture and results, SageMaker pipeline executions, feature group, model registry, batch transform output, monitoring JSON, and CloudWatch dashboard. Set consistent account/region and readable time ranges.
4. Remove secrets and unrelated personal data from the screen. Verify captions, microphone clarity, cursor visibility, and readable text at export resolution.
5. Rehearse handoffs and timing. Save the source recording and edited MP4. Write down the exact commit SHA and AWS execution IDs used in the demonstration.

## Ten-minute screen and narration plan

| Time | Suggested speaker | Screen/action | Narration points |
|---|---|---|---|
| 00:00–00:45 | A | Project title and design problem statement | Explain static Win64 malware triage, the cost of missed malware and benign false alarms, and why time-separated evaluation matters. |
| 00:45–01:30 | A | Architecture and split diagrams in notebook/design | Walk from deduplicated features and versioned S3 to training, evaluation, registry, batch, and monitoring. Distinguish model training from validation and untouched reserve. |
| 01:30–02:30 | A → B | Feature schema and actual Feature Store group | Show 13 approved features, source-week fit cutoff, group identifier/event-time fields, actual storage mode, sample ingestion/read evidence. Explain that S3 CSVs remain the training source. |
| 02:30–03:45 | B | Notebook comparisons and reports | Explain fixed candidates, train-only fit, validation threshold selection at the FPR budget, and actual test recall/FPR/AUC with sample counts. State the official-test-derived subset limitation. |
| 03:45–05:00 | B → C | Successful SageMaker pipeline DAG; failed DAG | Show real step statuses and job links. Identify the successful quality gate and registry step. Open the deliberately forced-failure execution and explain why its Fail step demonstrates the rejection control. |
| 05:00–06:15 | C | Model Registry and batch transform | Show package version, linked report, approval state, transform job, and actual S3 scores/decisions. Explain frozen threshold and ordered features-only inference input. |
| 06:15–07:30 | C → D | Monitoring report and CloudWatch dashboard | Show actual histogram PSI results, counts/alerts, delayed-label metrics where available, and timestamped infrastructure/custom metric panels. Distinguish drift from observed predictive quality. |
| 07:30–08:45 | D | GitHub CI and workflow/source | Show a real CI check and the reusable CLI/notebook, ignored account config, account-specific role/bucket, and OIDC deployment option if configured. Explain manual approval and repeatable teammate setup. |
| 08:45–10:00 | D | Limitations, contribution summary, cleanup evidence | Explain duplicate source data, corrected preprocessing boundary, possible lab restrictions actually encountered, drift/evasion risk, wider feature/subgroup evaluation, and future scheduling/rollback work. Summarize measured result and each member's real contribution; show idle resource cleanup plan. |

The suggested boundaries provide four 2.5-minute speaking blocks (A: 00:00–02:30; B: 02:30–05:00; C: 05:00–07:30; D: 07:30–10:00). Handoffs can retain a screen while the next person begins. For a different roster, reassign contiguous time blocks evenly and record the resulting names/times here before submission.

## Evidence-backed narration prompts

- “This run used commit **[SHA]**, schema **[version/hash]**, and AWS region **[region]**. These are observed account results.”
- “The selected candidate was **[model]**, with validation-selected threshold **[threshold]**. On **[test sample count]** later samples, recall was **[recall]** and benign FPR was **[FPR]**. These are from an official-test-derived subset.”
- “The successful execution is **[execution ID]**. This separate execution **[failure ID]** intentionally forces rejection; it is a control-path demonstration.”
- “Registry version **[package ARN/version]** is **[approval state]**. This transform job wrote **[row count]** predictions to **[URI]**.”
- “The monitoring report compares **[batch scope]** to train-only reference distributions. **[Observed alerts]** require investigation; unlabeled drift alone does not establish lost recall.”
- “The lab denied **[actual action, if any]**. We have **[actual evidence]** and identify **[missing evidence]** as unresolved.”

Replace every bracketed value from actual artifacts before recording. Omit claims for unavailable components rather than inserting demonstration numbers. If all required components cannot run in the lab, preserve the denial evidence and clarify the remaining work with the instructor.

## Final media and submission review

- The MP4 contains screencast and clear audio, is reviewed end to end, and includes readable required components without secrets or personal account information.
- Every member speaks an equal portion and commits their actual contribution. An edited outline/transcript is included.
- Standardized filenames are used for the design PDF, demonstration MP4, and outline; the repository/branch link is included in the design.
- A selected delegate submits the required files together because subsequent submissions overwrite earlier ones. Members submit the separate individual peer evaluation themselves.
- Document similarity review, captions, course recording guidance, and instructor timing clarification are human review steps; this repository does not certify that they were completed.
