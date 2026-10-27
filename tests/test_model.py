"""
Tests — model bundle, preprocessing, prediction, evaluation, and SHAP.
Uses saved artefacts where available; falls back to synthetic fixtures
for logic tests that don't require a trained model.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    EVALUATION_RESULTS_PATH,
    FEATURE_COLUMNS,
    MODEL_BUNDLE_PATH,
    TARGET_COL,
)
from src.metrics import compute_metrics
from src.predict import (
    clear_bundle_cache,
    load_model_bundle,
    predict_batch,
    predict_single,
)
from src.preprocessing import Preprocessor


# ── Skip if artefacts don't exist ────────────────────────────────────

requires_model = pytest.mark.skipif(
    not MODEL_BUNDLE_PATH.exists(),
    reason="Model bundle not found — run training first.",
)
requires_eval = pytest.mark.skipif(
    not EVALUATION_RESULTS_PATH.exists(),
    reason="Evaluation results not found — run training first.",
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_cache():
    clear_bundle_cache()
    yield
    clear_bundle_cache()


@pytest.fixture
def bundle():
    return load_model_bundle()


@pytest.fixture
def sample_transaction(bundle):
    """A synthetic transaction with all features set to zero."""
    return {col: 0.0 for col in bundle["feature_columns"]}


@pytest.fixture
def sample_fraud_like_transaction(bundle):
    """Load a real fraud-ish example from the dataset for SHAP testing."""
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "creditcard.csv")
    fraud_rows = df[df["Class"] == 1]
    row = fraud_rows.iloc[0]
    return {col: float(row[col]) for col in bundle["feature_columns"]}


# ── Preprocessor ─────────────────────────────────────────────────────

class TestPreprocessor:
    def test_fit_transform_shape(self):
        rng = np.random.RandomState(42)
        data = {c: rng.randn(100) for c in FEATURE_COLUMNS}
        df = pd.DataFrame(data)
        pp = Preprocessor()
        out = pp.fit_transform(df)
        assert out.shape == df.shape
        assert list(out.columns) == FEATURE_COLUMNS

    def test_transform_without_fit_raises(self):
        pp = Preprocessor()
        df = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
        with pytest.raises(RuntimeError, match="not been fitted"):
            pp.transform(df)

    def test_rejects_class_column(self):
        pp = Preprocessor()
        data = {c: [0.0] for c in FEATURE_COLUMNS}
        data["Class"] = [0]
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match="Class"):
            pp.fit(df)

    def test_consistency(self):
        """Same data produces same output."""
        rng = np.random.RandomState(0)
        data = {c: rng.randn(50) for c in FEATURE_COLUMNS}
        df = pd.DataFrame(data)
        pp = Preprocessor()
        pp.fit(df)
        out1 = pp.transform(df)
        out2 = pp.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


# ── Model bundle ─────────────────────────────────────────────────────

@requires_model
class TestModelBundle:
    def test_bundle_keys(self, bundle):
        required = {
            "model", "preprocessor", "threshold",
            "feature_columns", "model_version",
        }
        assert required.issubset(bundle.keys())

    def test_feature_columns_match(self, bundle):
        assert bundle["feature_columns"] == FEATURE_COLUMNS

    def test_threshold_in_range(self, bundle):
        t = bundle["threshold"]
        assert 0.0 < t < 1.0

    def test_preprocessor_is_fitted(self, bundle):
        assert bundle["preprocessor"].is_fitted


# ── Single prediction ────────────────────────────────────────────────

@requires_model
class TestSinglePrediction:
    def test_output_keys(self, sample_transaction):
        result = predict_single(sample_transaction)
        for key in (
            "predicted_class", "predicted_label",
            "fraud_probability", "threshold", "model_version",
        ):
            assert key in result

    def test_predicted_class_valid(self, sample_transaction):
        result = predict_single(sample_transaction)
        assert result["predicted_class"] in (0, 1)

    def test_probability_range(self, sample_transaction):
        result = predict_single(sample_transaction)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_threshold_applied(self, sample_transaction, bundle):
        result = predict_single(sample_transaction)
        expected_class = int(
            result["fraud_probability"] >= bundle["threshold"]
        )
        assert result["predicted_class"] == expected_class

    def test_rejects_class_in_input(self, sample_transaction):
        sample_transaction["Class"] = 0
        with pytest.raises(ValueError, match="Class"):
            predict_single(sample_transaction)

    def test_rejects_missing_feature(self):
        txn = {col: 0.0 for col in FEATURE_COLUMNS[:-1]}  # drop last
        with pytest.raises(ValueError, match="Missing"):
            predict_single(txn)

    def test_rejects_nan(self, sample_transaction):
        sample_transaction["Amount"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            predict_single(sample_transaction)

    def test_invalid_threshold_raises(self, sample_transaction, bundle):
        bad_bundle = dict(bundle)
        bad_bundle["threshold"] = -0.5
        with pytest.raises(ValueError, match="between 0 and 1"):
            predict_single(sample_transaction, bundle=bad_bundle)

        bad_bundle["threshold"] = None
        with pytest.raises(ValueError, match="missing required 'threshold'"):
            predict_single(sample_transaction, bundle=bad_bundle)


# ── Batch prediction ────────────────────────────────────────────────

@requires_model
class TestBatchPrediction:
    def test_basic_batch(self):
        data = {c: [0.0, 1.0, -1.0] for c in FEATURE_COLUMNS}
        df = pd.DataFrame(data)
        result = predict_batch(df)
        assert len(result) == 3
        assert "predicted_class" in result.columns
        assert "fraud_probability" in result.columns

    def test_class_column_stripped(self):
        data = {c: [0.0] for c in FEATURE_COLUMNS}
        data["Class"] = [1]
        df = pd.DataFrame(data)
        result = predict_batch(df)
        assert len(result) == 1

    def test_missing_column_raises(self):
        data = {c: [0.0] for c in FEATURE_COLUMNS[:-1]}
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match="Missing"):
            predict_batch(df)

    def test_row_order_preserved(self):
        data = {c: list(range(5)) for c in FEATURE_COLUMNS}
        df = pd.DataFrame(data)
        result = predict_batch(df)
        assert list(result["row_index"]) == [0, 1, 2, 3, 4]

    def test_predict_batch_csv(self, tmp_path):
        from src.predict import predict_batch_csv

        data = {c: [0.0, 1.5] for c in FEATURE_COLUMNS}
        data["Class"] = [0, 1]
        test_df = pd.DataFrame(data)
        csv_file = tmp_path / "test_batch.csv"
        test_df.to_csv(csv_file, index=False)

        res = predict_batch_csv(csv_file)
        assert len(res) == 2
        assert list(res["row_index"]) == [0, 1]
        assert "fraud_probability" in res.columns

    def test_single_and_batch_consistency(self, sample_transaction, bundle):
        single_res = predict_single(sample_transaction, bundle=bundle)
        batch_res = predict_batch(pd.DataFrame([sample_transaction]), bundle=bundle)
        assert abs(single_res["fraud_probability"] - batch_res.iloc[0]["fraud_probability"]) < 1e-6
        assert single_res["predicted_class"] == batch_res.iloc[0]["predicted_class"]


# ── Metrics ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        m = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert m["fraud_recall"] == 1.0
        assert m["fraud_precision"] == 1.0
        assert m["fraud_f1"] == 1.0
        assert m["fraud_missed"] == 0
        assert m["legitimate_incorrectly_flagged"] == 0
        assert "average_precision" in m
        assert "pr_auc" in m

    def test_all_missed(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.2, 0.3])
        m = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert m["fraud_recall"] == 0.0
        assert m["fraud_missed"] == 2

    def test_confusion_matrix_convention(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.3, 0.8])
        m = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        cm = m["confusion_matrix"]
        assert cm["TN"] == 1
        assert cm["FP"] == 1
        assert cm["FN"] == 1
        assert cm["TP"] == 1

    def test_zero_division_safety(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1])
        m = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert m["accuracy"] == 1.0
        assert m["fraud_recall"] == 0.0
        assert m["fraud_precision"] == 0.0
        assert m["false_negative_rate"] == 0.0


# ── Evaluation consistency ──────────────────────────────────────────

@requires_eval
class TestEvaluationConsistency:
    def test_eval_file_parseable(self):
        with open(EVALUATION_RESULTS_PATH) as f:
            data = json.load(f)
        assert "fraud_f1" in data
        assert "roc_auc" in data
        assert "average_precision" in data
        assert "pr_auc" in data
        assert "confusion_matrix" in data

    def test_eval_not_hardcoded(self):
        """Evaluation should be from the test split, not training."""
        with open(EVALUATION_RESULTS_PATH) as f:
            data = json.load(f)
        assert data.get("split") == "test"
        assert data.get("note", "").startswith("These metrics")

    def test_reports_match_bundle_threshold(self, bundle):
        with open(EVALUATION_RESULTS_PATH) as f:
            data = json.load(f)
        assert abs(data["threshold_used"] - bundle["threshold"]) < 1e-6


# ── SHAP ─────────────────────────────────────────────────────────────

@requires_model
class TestSHAP:
    def test_explain_returns_values(self, sample_transaction):
        from src.explain import explain_single

        result = explain_single(sample_transaction)
        assert "base_value" in result
        assert "shap_values" in result
        assert len(result["shap_values"]) == len(FEATURE_COLUMNS)

    def test_shap_features_match(self, sample_transaction):
        from src.explain import explain_single

        result = explain_single(sample_transaction)
        features_in_shap = [s["feature"] for s in result["shap_values"]]
        assert features_in_shap == FEATURE_COLUMNS

    def test_shap_top_contributors(self, sample_fraud_like_transaction):
        from src.explain import explain_single

        result = explain_single(sample_fraud_like_transaction, top_n=5)
        assert len(result["top_contributors"]) == 5

    def test_shap_prediction_matches(self, sample_transaction):
        """SHAP prediction_info must match predict_single."""
        from src.explain import explain_single

        shap_result = explain_single(sample_transaction)
        pred_result = predict_single(sample_transaction)
        assert shap_result["prediction_info"]["predicted_class"] == pred_result["predicted_class"]
        assert abs(
            shap_result["prediction_info"]["fraud_probability"]
            - pred_result["fraud_probability"]
        ) < 1e-4
