# Final EMBER2024 temporal split strategy

## Decision

The project uses `ember2024_win64_temporal_hybrid_v1`, a four-way strictly chronological split. It intentionally does **not** apply the course 40/10/10/40 ratio because that would move official training weeks into a later production period and obscure EMBER2024's official temporal boundary.

| Final split | Source weeks | Source Week Range | Row Count | Source partition | Role |
|---|---:|---|---:|---|---|
| Training | 40 | 2023-09-24 through 2024-06-23 | 400,000 | Official train | Preprocessing fit and model fitting only |
| Validation | 12 | 2024-06-30 through 2024-09-15 | 120,000 | Official train | Candidate/threshold selection only |
| Test | 6 | 2024-09-22 through 2024-10-27 | 59,995 | Official test | Unseen newer-threat evaluation |
| Reserved production | 6 | 2024-11-03 through 2024-12-08 | 59,998 | Official test | Final untouched holdout; not evaluated |

This order maintains chronological separation across time windows. All rows are deduplicated by SHA-256 during feature preparation, and test records whose hash was already observed in train are removed before split creation.

The feature preparer's inclusive `fit_end_week: "2024-06-23"` refers to the start of the final model-training week, not June 23 as the last accepted day. In the ideal raw-data pipeline, all fitted normalization statistics come from the first 40 weeks, and the schema declares `fit_scope: "model_train_only"`.

### Reconstruction Caveat & Data Provenance
In the local workspace for this finalization, original raw JSONL files were not available. Existing processed CSV files (`data/processed/ember2024_reconstructed_v1/`) were reconstructed by reversing previously stored standardization transformations and fitting replacement scaling moments on the reconstructed training split. The reconstruction script fits replacement scaling moments on the supplied training split (whose source weeks run through 2024-06-23) but does not independently enforce a per-row source-week cutoff. Neither validation, test, nor reserved-production partitions were used to fit replacement scaling moments. Original missing-value indicators were lost and estimated via a three-feature zero proxy. Consequently, exact reproducibility of the original preprocessing pipeline cannot be claimed, and model metrics represent exploratory results.

## Important interpretation

The 12-week official EMBER2024 test partition is divided into a six-week project test and a six-week production reserve. Therefore, course results must be described as evaluation on an **official-test-derived temporal test subset**, not the untouched full official EMBER2024 test set.

## Generate handoff datasets

First regenerate prepared features so their non-feature lineage columns include the source week:

```powershell
python scripts/prepare_features.py
```

Then create the four final splits:

```powershell
python scripts/create_temporal_splits.py
```

The split directory contains `train.csv`, `validation.csv`, `test.csv`, `reserved_production.csv`, a shared `split_manifest.json`, and one metadata JSON document per split. Every metadata document includes row count, class distribution, source-partition/week counts, feature schema name/fingerprint/order, generation timestamp, and effective configuration.

Use only the feature columns named in the schema for training. `record_id`, `label`, `source_partition`, and source-week fields are metadata, not model features.

## Modeling boundaries

`sagemaker_src/training.py` fits the baseline and bounded candidate trials using `train.csv`. Baseline validation tradeoffs inform an explicit FPR budget and weighted policy before comparison; validation then selects the candidate and operating threshold. The winner, threshold, and policy are saved before the separately enabled `sagemaker_src/evaluate.py` step uses `test.csv`. Do not refit on validation, adjust the threshold using test labels, or keep changing candidates after seeing the test report while describing it as a fresh holdout evaluation.

The modeling notebook and helper defaults download only train/validation/schema/manifest; the test CSV is requested only for explicit final evaluation after selection is frozen. `reserved_production.csv` is omitted from every modeling download and job input and is never opened. A later reserve evaluation requires a separate plan. The current tracker confirms this hybrid split; the RFC's earlier full-official-test wording must not be used to describe these results.
