# EMBER2024 feature preparation

This stage creates a reusable static-feature dataset for the model-training teammate. It does not train, tune, evaluate, or select a model.

## Policy

`ember2024_win64_static_v1` uses 13 low-cardinality static PE metrics:

- General: `general_size`, `general_entropy`.
- Strings: `strings_numstrings`, `strings_avlength`, `strings_printables`, `strings_entropy`.
- PE structure: `section_count`, `import_library_count`, `export_count`, `richheader_count`, `pefilewarning_count`.
- Authenticode: `authenticode_num_certs`, `authenticode_self_signed`.

Count-like fields receive `log1p`; all approved fields are standardized using **model-training-only** fitted mean and standard deviation. Missing values are mean-imputed after standardization as `0.0`. The schema records every source path, transform, fit statistic, and missing-value count.

The default `fit_end_week` is `2024-06-23`, an inclusive **week-start** cutoff that includes June 23–29. The last 12 weeks of the supplied official `train` partition belong to model validation and must not contribute to fitted moments. Generated schemas declare `fit_scope: "model_train_only"` and the cutoff so downstream training can check this boundary.

The following are excluded from model inputs: target/taxonomy labels; hashes; temporal and VirusTotal metadata; Capa enrichment; and high-dimensional/dynamic nested structures. `general.is_pe` is excluded because it is nearly constant in this Win64 data. See the generated schema's `exclusion_policy` for the complete, machine-readable rationale.

## Run

From `project/`:

```powershell
python scripts/prepare_features.py
```

The command is intentionally ordered:

1. Fit transformations only on unique valid rows from the first 40 official-training weeks, through the week starting `2024-06-23`.
2. Transform and deduplicate `train` with the saved train statistics.
3. Transform and deduplicate `test` with those same statistics; when configured, discard test hashes already seen anywhere in the supplied official-training partition to prevent duplicate-based evaluation leakage.

It preserves the supplied temporal partitions and does not create a random split or a validation split.

## Existing handoff compatibility

Artifacts produced by the earlier implementation fitted scaling over all 52 supplied official-training weeks, including the 12 weeks later assigned to validation. They are unsuitable for the strict validation protocol even though no test rows contributed to scaling. Downloading them again does not fix the issue. The data owner must rerun feature preparation with the corrected cutoff and then rerun `scripts/create_temporal_splits.py`, keeping the new schema, CSVs, processing report, manifest, and metadata together under a new versioned S3 prefix.

The model trainer rejects schemas without the model-training-only fit declaration or the expected cutoff by default. An explicit legacy compatibility override is for diagnosing an old handoff and must be disclosed in resulting reports; it cannot establish a leakage-free final experiment. Do not edit an old schema to add the declaration without regenerating its values. See [the SageMaker runbook](sagemaker_runbook.md) for teammate setup and artifact checks.

### Local Workspace Reconstruction Caveat
In the local workspace for this project finalization, original raw JSONL files were not available. Existing processed CSV files (`data/processed/ember2024_reconstructed_v1/`) were reconstructed by reversing previously stored standardization transformations and fitting replacement scaling moments on the reconstructed training split. The reconstruction script fits replacement scaling moments on the supplied training split (whose source weeks run through 2024-06-23) but does not independently enforce a per-row source-week cutoff. Neither validation, test, nor reserved-production partitions were used to fit replacement scaling moments. Original missing-value indicators were lost during initial preprocessing and were estimated using a three-feature zero-pattern proxy (`section_count`, `authenticode_num_certs`, `authenticode_self_signed`). Exact reproducibility of the original raw-data preprocessing pipeline cannot be claimed, and model outputs must be presented as exploratory results.

## Outputs

All outputs are local-only under the configured `data/processed/...` directory and are ignored by Git:

- `feature_schema.json`: final feature names/order, source fields, transforms, train-only statistics, exclusion policy, and output contract.
- `train_features.csv` and `test_features.csv`: `record_id`, `label`, `source_partition`, `source_week_start`, `source_week_end`, then the model feature columns in the exact schema order. Week fields are lineage metadata, not model inputs.
- `processing_report.json`: row, invalid, duplicate, overlap, and output counts.

For model training, use only `model_feature_columns` from `feature_schema.json`; never pass `record_id`, `label`, `source_partition`, `source_week_start`, or `source_week_end` into a model. These files can later be copied to S3 unchanged, together with the schema and processing report.
