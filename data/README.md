# Data Directory

This directory contains data documentation and metadata.

## Raw Data

The original `creditcard.csv` is located at the **project root** (`../creditcard.csv`).  
The pipeline discovers it via `src/config.py → DATASET_PATH`. The raw CSV is never modified by the pipeline.

### Raw Dataset Characteristics
- Total records: 284,807 transactions
- Features: 30 input features (`Time`, `Amount`, `V1`–`V28`)
- Target: `Class` (0 = Legitimate: 284,315 | 1 = Fraudulent: 492)
- Raw fraud prevalence: 0.1727%

## Zero-Leakage Data Preparation & Deduplication

During exploratory audit, 1,081 duplicate transactions (1,854 total rows across duplicate groups) were identified in the raw dataset. Because identical feature records could otherwise cross the train, validation, and test splits (contaminating the held-out test evaluation), the pipeline applies deduplication (`src.data.deduplicate_dataset`) prior to splitting:

- Duplicate rows removed: 1,081 (1,062 legitimate, 19 fraudulent)
- Clean modeled transactions: 283,726 (283,253 legitimate, 473 fraudulent)
- Final fraud prevalence: 0.1667%
- Cross-partition record overlap: **0** across Train, Validation, and Test partitions.

## Preprocessing

Preprocessing (`StandardScaler` on `Amount`) is fit strictly on the training partition (`X_train`) and transformed identically on validation, test, and inference data via the serialized `Preprocessor` saved inside `artifacts/model_bundle.joblib`.
