import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(
    r"C:\NitzDoc\USD\AAI_540_MLops\6-7FinalProject"
    r"\AAI-540-Final-Team-Project-Group1-main"
    r"\ember2024_win64_static_v1-20261017T205331Z-1-001"
    r"\ember2024_win64_static_v1"
)
SPLIT_DIR = SOURCE / "temporal_splits_v1"
SCHEMA_PATH = SOURCE / "feature_schema.json"

OUTPUT = Path("data/processed/ember2024_reconstructed_v1")
FEATURES = [
    "general_size", "general_entropy", "strings_numstrings",
    "strings_avlength", "strings_printables", "strings_entropy",
    "section_count", "import_library_count", "export_count",
    "richheader_count", "pefilewarning_count",
    "authenticode_num_certs", "authenticode_self_signed",
]
SPLITS = ["train", "validation", "test", "reserved_production"]
MISSING_PROXY_FEATURES = [
    "section_count", "authenticode_num_certs",
    "authenticode_self_signed",
]

def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )

    with SCHEMA_PATH.open(encoding="utf-8") as f:
        old_schema = json.load(f)

    fitted = old_schema["fitted_features"]
    frames = {}
    proxy_masks = {}
    recovered = {}

    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"
        df = pd.read_csv(path)
        required = set(FEATURES + ["label", "record_id"])
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

        frames[split] = df
        proxy_masks[split] = (
            df[MISSING_PROXY_FEATURES].eq(0).all(axis=1)
        )
        recovered[split] = pd.DataFrame(index=df.index)

        for feature in FEATURES:
            stats = fitted[feature]
            mean = float(stats["train_mean"])
            std = float(stats["train_standard_deviation"])
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid saved standard deviation: {feature}")

            # Reverse the prior standardization to recover the prior
            # transformed feature scale (not the original raw value).
            values = pd.to_numeric(df[feature], errors="raise").astype(float)
            base_values = values * std + mean

            if feature in MISSING_PROXY_FEATURES:
                base_values = base_values.mask(proxy_masks[split])

            recovered[split][feature] = base_values

    # Fit new moments using only reconstructed, non-proxy-missing
    # training values. Validation/test/reserved data do not fit the scaler.
    new_stats = {}
    for feature in FEATURES:
        train_values = recovered["train"][feature].dropna()
        if train_values.empty:
            raise ValueError(f"No training values available for {feature}")

        mean = float(train_values.mean())
        std = float(train_values.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reconstructed training scale: {feature}")

        new_stats[feature] = {
            "train_mean": mean,
            "train_standard_deviation": std,
            "train_nonmissing_count": int(train_values.count()),
            "train_missing_count_proxy": int(
                recovered["train"][feature].isna().sum()
            ),
        }

    OUTPUT.mkdir(parents=True)
    row_counts = {}

    for split in SPLITS:
        out = frames[split].copy()
        for feature in FEATURES:
            mean = new_stats[feature]["train_mean"]
            std = new_stats[feature]["train_standard_deviation"]
            values = recovered[split][feature]
            # Match the project's existing convention: impute missing
            # standardized values with zero after scaling.
            out[feature] = ((values - mean) / std).fillna(0.0)

        out.to_csv(OUTPUT / f"{split}.csv", index=False)
        row_counts[split] = len(out)

    new_schema = copy.deepcopy(old_schema)
    new_schema["reconstruction_notice"] = (
        "Reconstructed from previously standardized CSVs; not regenerated "
        "from raw EMBER records. The original missingness mask was lost. "
        "For section_count, authenticode_num_certs and "
        "authenticode_self_signed, rows where all three saved values equal "
        "zero were treated as a missingness proxy. Genuine zero-standardized "
        "observations may be misclassified. Recovered values are on the "
        "previous transformed scale, not the raw feature scale."
    )
    new_schema["fit_scope"] = "reconstructed_train_split_only"
    new_schema["fit_end_week"] = "2024-06-23"
    new_schema["fit_policy"] = (
        "Replacement scaling moments fitted only on the reconstructed "
        "training split. Validation, test and reserved production were "
        "transformed but not used to fit replacement moments."
    )
    new_schema["reconstructed_scaling_statistics"] = new_stats

    with (OUTPUT / "feature_schema_reconstructed.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(new_schema, f, indent=2)

    report = {
        "status": "RECONSTRUCTED_APPROXIMATION_NOT_RAW_REGENERATION",
        "source_schema": str(SCHEMA_PATH),
        "source_split_directory": str(SPLIT_DIR),
        "output_directory": str(OUTPUT),
        "split_row_counts": row_counts,
        "feature_count": len(FEATURES),
        "missingness_proxy_features": MISSING_PROXY_FEATURES,
        "missingness_proxy_counts": {
            split: int(mask.sum()) for split, mask in proxy_masks.items()
        },
        "scaler_fit_partition": "train only",
        "limitations": [
            "Original raw JSONL files were unavailable in the searched workspace.",
            "Inverse standardization recovers transformed-scale values, not raw values.",
            "The original missing-value mask was lost during earlier imputation.",
            "The shared zero pattern is a proxy and may misclassify genuine observations.",
            "Do not present these artifacts as an exact raw-data regeneration.",
        ],
    }
    with (OUTPUT / "reconstruction_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("Reconstruction completed.")
    print(f"Output: {OUTPUT.resolve()}")
    print("Rows:", row_counts)
    print("Proxy-missing counts:", report["missingness_proxy_counts"])
    print("Important: approximation only; see reconstruction_report.json")

if __name__ == "__main__":
    main()
