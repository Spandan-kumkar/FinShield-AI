"""
FinShield AI — Evaluation helpers
===================================
Stand-alone evaluation that loads the saved model bundle and evaluates
on the test split from the original data.  This module can regenerate
the test evaluation JSON without retraining.
"""

import json
import logging
import sys

import joblib
import numpy as np

from src.config import (
    EVALUATION_RESULTS_PATH,
    MODEL_BUNDLE_PATH,
    REPORTS_DIR,
)
from src.data import (
    deduplicate_dataset,
    load_dataset,
    separate_features_target,
    split_data,
)
from src.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model() -> dict:
    """
    Load the saved model bundle and re-evaluate on the test split.
    Returns the metrics dict and saves it to disk.
    """
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_BUNDLE_PATH}. "
            "Run 'python -m src.train' first."
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = bundle["threshold"]

    # Reproduce the exact clean test split
    df_raw = load_dataset()
    df_clean, _ = deduplicate_dataset(df_raw)
    X, y = separate_features_target(df_clean)
    splits = split_data(X, y)
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    X_test_pp = preprocessor.transform(X_test)
    y_prob = model.predict_proba(X_test_pp)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test.values, y_pred, y_prob, threshold)
    metrics["split"] = "test"
    metrics["model_experiment"] = bundle["experiment"]
    metrics["model_version"] = bundle["model_version"]
    metrics["note"] = (
        "These metrics are from the held-out test set and were NOT used "
        "for model or threshold selection."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation results saved → %s", EVALUATION_RESULTS_PATH)
    return metrics


if __name__ == "__main__":
    evaluate_model()
