"""
Tests — data loading, validation, feature separation, and splitting.
Uses the real dataset for validation checks and small synthetic
fixtures for split-logic verification.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, RANDOM_SEED, TARGET_COL
from src.data import (
    load_dataset,
    separate_features_target,
    split_data,
    validate_dataset,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load the real dataset once for this module if available."""
    try:
        return load_dataset()
    except FileNotFoundError:
        pytest.skip(
            "creditcard.csv not found. Download the full 144 MB dataset "
            "from Kaggle to run raw dataset validation tests."
        )


@pytest.fixture
def small_df():
    """A tiny synthetic DataFrame with known properties."""
    rng = np.random.RandomState(42)
    n = 1000
    data = {col: rng.randn(n) for col in FEATURE_COLUMNS}
    data[TARGET_COL] = np.array([0] * 990 + [1] * 10)
    rng.shuffle(data[TARGET_COL])
    return pd.DataFrame(data)


# ── Dataset validation ───────────────────────────────────────────────

class TestDatasetValidation:
    def test_shape(self, raw_df):
        assert raw_df.shape == (284807, 31)

    def test_columns_present(self, raw_df):
        for col in FEATURE_COLUMNS + [TARGET_COL]:
            assert col in raw_df.columns

    def test_no_missing(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_target_values(self, raw_df):
        assert set(raw_df[TARGET_COL].unique()) == {0, 1}

    def test_class_counts(self, raw_df):
        vc = raw_df[TARGET_COL].value_counts()
        assert vc[0] == 284315
        assert vc[1] == 492

    def test_validate_report(self, raw_df):
        report = validate_dataset(raw_df)
        assert report["fraud_prevalence_pct"] == pytest.approx(0.1727, rel=0.01)
        assert report["num_features"] == 30
        assert report["duplicate_rows_all"] == 1081
        assert report["duplicate_rows_features"] == 1081
        assert report["conflicting_duplicate_groups"] == 0


# ── Deduplication & Leakage Prevention ────────────────────────────────

class TestDeduplication:
    def test_deduplicate_dataset_counts(self, raw_df):
        from src.data import deduplicate_dataset

        df_clean, audit = deduplicate_dataset(raw_df)
        assert len(df_clean) == 283726
        assert audit["removed_duplicates"] == 1081
        assert audit["removed_fraud"] == 19
        assert audit["removed_legitimate"] == 1062
        assert audit["final_fraud"] == 473
        assert audit["final_legitimate"] == 283253

    def test_zero_cross_partition_leakage(self, raw_df):
        from src.data import check_cross_partition_overlap, deduplicate_dataset

        df_clean, _ = deduplicate_dataset(raw_df)
        X, y = separate_features_target(df_clean)
        splits = split_data(X, y)
        overlap = check_cross_partition_overlap(splits)
        assert overlap["train_val_overlap"] == 0
        assert overlap["train_test_overlap"] == 0
        assert overlap["val_test_overlap"] == 0


# ── Feature / target separation ──────────────────────────────────────

class TestFeatureSeparation:
    def test_class_not_in_X(self, raw_df):
        X, y = separate_features_target(raw_df)
        assert TARGET_COL not in X.columns

    def test_y_dtype(self, raw_df):
        _, y = separate_features_target(raw_df)
        assert y.dtype in (np.int64, np.int32, int)

    def test_feature_count(self, raw_df):
        X, _ = separate_features_target(raw_df)
        assert X.shape[1] == 30

    def test_feature_order(self, raw_df):
        X, _ = separate_features_target(raw_df)
        assert list(X.columns) == FEATURE_COLUMNS


# ── Stratified splitting ─────────────────────────────────────────────

class TestSplitting:
    def test_no_overlap(self, small_df):
        X, y = separate_features_target(small_df)
        splits = split_data(X, y)
        train_idx = set(splits["X_train"].index)
        val_idx = set(splits["X_val"].index)
        test_idx = set(splits["X_test"].index)
        assert train_idx & val_idx == set()
        assert train_idx & test_idx == set()
        assert val_idx & test_idx == set()

    def test_total_preserved(self, small_df):
        X, y = separate_features_target(small_df)
        splits = split_data(X, y)
        total = (
            len(splits["X_train"])
            + len(splits["X_val"])
            + len(splits["X_test"])
        )
        assert total == len(X)

    def test_approximate_ratios(self, small_df):
        X, y = separate_features_target(small_df)
        splits = split_data(X, y)
        n = len(X)
        assert abs(len(splits["X_train"]) / n - 0.70) < 0.03
        assert abs(len(splits["X_val"]) / n - 0.15) < 0.03
        assert abs(len(splits["X_test"]) / n - 0.15) < 0.03

    def test_stratified_fraud_ratio(self, small_df):
        """Fraud ratio in each split should be close to original."""
        X, y = separate_features_target(small_df)
        overall_rate = y.mean()
        splits = split_data(X, y)
        for name in ("train", "val", "test"):
            rate = splits[f"y_{name}"].mean()
            assert abs(rate - overall_rate) < 0.02, (
                f"{name} fraud rate {rate:.4f} too far from {overall_rate:.4f}"
            )

    def test_reproducibility(self, small_df):
        X, y = separate_features_target(small_df)
        s1 = split_data(X, y, seed=RANDOM_SEED)
        s2 = split_data(X, y, seed=RANDOM_SEED)
        assert list(s1["X_train"].index) == list(s2["X_train"].index)

    def test_class_weight_calculation(self, small_df):
        X, y = separate_features_target(small_df)
        splits = split_data(X, y)
        y_train = splits["y_train"]
        n_legit = int((y_train == 0).sum())
        n_fraud = int((y_train == 1).sum())
        spw = n_legit / n_fraud
        assert spw > 1.0   # Should be substantially > 1 for imbalanced data
