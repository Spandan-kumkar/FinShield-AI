"""
FinShield AI — Data loading, validation, and splitting
=======================================================
Loads the raw CSV, validates expectations, and produces
reproducible stratified train / validation / test splits.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ── Loading & validation ─────────────────────────────────────────────

def discover_dataset(path: Path = DATASET_PATH) -> Path:
    """Return the resolved path to the CSV, raising if missing."""
    p = Path(path).resolve()
    if p.exists():
        return p
    # Check standard fallback locations
    fallbacks = [
        PROJECT_ROOT / "creditcard.csv",
        PROJECT_ROOT / "data" / "creditcard.csv",
        PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
    ]
    for fb in fallbacks:
        if fb.resolve().exists():
            return fb.resolve()
    raise FileNotFoundError(
        f"Dataset not found at {path}. "
        "Place the creditcard.csv file at the project root or in data/. "
        "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    path = discover_dataset(path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset from %s — shape %s", path.name, df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run a battery of checks and return a summary dict.
    Raises ValueError only if the data is fundamentally unsuitable.
    """
    report: dict = {}
    report["shape"] = df.shape
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    report["missing_values"] = int(df.isnull().sum().sum())
    report["duplicate_rows_all"] = int(df.duplicated().sum())

    # Check duplicates across input features only
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    report["duplicate_rows_features"] = int(df.duplicated(subset=feat_cols).sum())

    # Check for conflicting labels across duplicate feature vectors
    if TARGET_COL in df.columns and feat_cols:
        conflicts = int((df.groupby(feat_cols)[TARGET_COL].nunique() > 1).sum())
        report["conflicting_duplicate_groups"] = conflicts
    else:
        report["conflicting_duplicate_groups"] = 0

    num_cols = df.select_dtypes(include="number").columns
    inf_count = int(
        df[num_cols].apply(lambda c: np.isinf(c).sum()).sum()
    )
    report["infinite_values"] = inf_count

    # Target checks
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    report["unique_targets"] = sorted(df[TARGET_COL].unique().tolist())
    vc = df[TARGET_COL].value_counts()
    report["class_counts"] = {int(k): int(v) for k, v in vc.items()}
    report["fraud_prevalence_pct"] = round(
        df[TARGET_COL].mean() * 100, 6
    )

    # Feature checks
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    report["num_features"] = len(FEATURE_COLUMNS)

    _log_validation_report(report)
    return report


def _log_validation_report(report: dict) -> None:
    logger.info("─── Dataset Validation Report ───")
    logger.info("  Shape                   : %s", report["shape"])
    logger.info("  Missing values          : %d", report["missing_values"])
    logger.info("  Duplicate rows (all)    : %d", report["duplicate_rows_all"])
    logger.info("  Duplicate rows (feat)   : %d", report["duplicate_rows_features"])
    logger.info("  Conflicting label grps  : %d", report["conflicting_duplicate_groups"])
    logger.info("  Infinite values         : %d", report["infinite_values"])
    logger.info("  Unique targets          : %s", report["unique_targets"])
    logger.info("  Class counts            : %s", report["class_counts"])
    logger.info("  Fraud prevalence        : %.4f%%", report["fraud_prevalence_pct"])
    logger.info("  Num features            : %d", report["num_features"])


def deduplicate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate records to prevent cross-partition leakage.
    Returns the deduplicated DataFrame and an audit metadata dict.
    """
    initial_len = len(df)
    initial_fraud = int((df[TARGET_COL] == 1).sum()) if TARGET_COL in df.columns else 0
    initial_legit = initial_len - initial_fraud

    df_clean = df.drop_duplicates().reset_index(drop=True)

    final_len = len(df_clean)
    final_fraud = int((df_clean[TARGET_COL] == 1).sum()) if TARGET_COL in df_clean.columns else 0
    final_legit = final_len - final_fraud
    removed_total = initial_len - final_len
    removed_fraud = initial_fraud - final_fraud
    removed_legit = initial_legit - final_legit

    audit = {
        "initial_total": initial_len,
        "initial_fraud": initial_fraud,
        "initial_legitimate": initial_legit,
        "final_total": final_len,
        "final_fraud": final_fraud,
        "final_legitimate": final_legit,
        "removed_duplicates": removed_total,
        "removed_fraud": removed_fraud,
        "removed_legitimate": removed_legit,
        "final_fraud_prevalence_pct": round(final_fraud / final_len * 100, 4) if final_len > 0 else 0.0,
    }

    logger.info(
        "Deduplication: removed %d duplicate rows (%d legit, %d fraud). "
        "Remaining: %d rows (%.4f%% fraud).",
        removed_total, removed_legit, removed_fraud, final_len, audit["final_fraud_prevalence_pct"],
    )
    return df_clean, audit


# ── Feature / target separation ──────────────────────────────────────

def separate_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y). Ensures 'Class' is never in X and feature order
    matches FEATURE_COLUMNS.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()
    assert TARGET_COL not in X.columns, "Target leaked into features!"
    return X, y


# ── Stratified splitting ─────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Produce stratified train / val / test splits.

    Strategy: first split 70 / 30, then split the 30 into 15 / 15
    (i.e. 50 / 50 of the dev set).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1."

    dev_ratio = val_ratio + test_ratio          # 0.30
    val_share_of_dev = val_ratio / dev_ratio    # 0.50

    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y,
        test_size=dev_ratio,
        stratify=y,
        random_state=seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_dev, y_dev,
        test_size=(1 - val_share_of_dev),
        stratify=y_dev,
        random_state=seed,
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    _log_split_info(splits)
    return splits


def check_cross_partition_overlap(splits: dict) -> dict:
    """
    Verify whether any identical feature vectors cross partition boundaries.
    Returns counts of overlapping unique feature vectors.
    """
    train_tuples = set(map(tuple, splits["X_train"].values))
    val_tuples = set(map(tuple, splits["X_val"].values))
    test_tuples = set(map(tuple, splits["X_test"].values))

    overlap = {
        "train_val_overlap": len(train_tuples & val_tuples),
        "train_test_overlap": len(train_tuples & test_tuples),
        "val_test_overlap": len(val_tuples & test_tuples),
    }
    return overlap


def _log_split_info(splits: dict) -> None:
    for name in ("train", "val", "test"):
        X = splits[f"X_{name}"]
        y = splits[f"y_{name}"]
        fraud = int(y.sum())
        legit = len(y) - fraud
        logger.info(
            "  %-5s  total=%6d  legit=%6d  fraud=%3d  (%.3f%%)",
            name, len(y), legit, fraud, y.mean() * 100,
        )
