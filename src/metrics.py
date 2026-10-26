"""
FinShield AI — Metrics helper
==============================
Compute a full suite of classification metrics for the fraud-detection
use case.  All functions expect binary labels and/or predicted
probabilities with Class 1 (fraud) as the positive class.
"""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Return a comprehensive metrics dict.

    Metric Terminology Note:
    -----------------------
    * Average Precision (AP): Computed via `average_precision_score(y_true, y_prob)`.
      Represents the weighted mean of precisions achieved at each threshold,
      weighted by the recall increase: AP = sum_n (R_n - R_{n-1}) * P_n.
    * PR-AUC: Computed via `auc(rec, prec)` using trapezoidal integration over
      the precision-recall curve coordinates.
    These two metrics are related but distinct; FinShield AI computes and reports both.

    Confusion-matrix convention
    ---------------------------
    sklearn.metrics.confusion_matrix returns:

        [[TN, FP],
         [FN, TP]]

    where rows are actual classes (0, 1) and columns are predicted classes (0, 1).
    * TN — legitimate correctly identified as legitimate
    * FP — legitimate incorrectly flagged as fraud
    * FN — fraud missed by the model (predicted legitimate)
    * TP — fraud correctly detected
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_fraud = int(tp + fn)
    total_legit = int(tn + fp)

    metrics: Dict[str, Any] = {}

    # ── Confusion matrix ─────────────────────────────────────────────
    metrics["confusion_matrix"] = {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
    }

    # ── Basic counts ─────────────────────────────────────────────────
    metrics["total_samples"] = int(len(y_true))
    metrics["total_fraud"] = total_fraud
    metrics["total_legitimate"] = total_legit
    metrics["fraud_detected"] = int(tp)
    metrics["fraud_missed"] = int(fn)
    metrics["legitimate_incorrectly_flagged"] = int(fp)

    # ── Rates ────────────────────────────────────────────────────────
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["fraud_precision"] = float(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    metrics["fraud_recall"] = float(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    metrics["fraud_f1"] = float(
        f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    metrics["specificity"] = float(tn / total_legit) if total_legit > 0 else 0.0
    metrics["false_positive_rate"] = float(fp / total_legit) if total_legit > 0 else 0.0
    metrics["false_negative_rate"] = float(fn / total_fraud) if total_fraud > 0 else 0.0
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(y_true, y_pred))
        if total_fraud > 0 and total_legit > 0
        else float(metrics["accuracy"])
    )

    # ── Probability-based ────────────────────────────────────────────
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if total_fraud > 0 and total_legit > 0 else 0.0
    metrics["average_precision"] = float(average_precision_score(y_true, y_prob)) if total_fraud > 0 else 0.0

    # Trapezoidal PR-AUC
    if total_fraud > 0:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = float(auc(rec, prec))
    else:
        metrics["pr_auc"] = 0.0

    # ── Metadata ─────────────────────────────────────────────────────
    metrics["threshold_used"] = float(threshold)

    return metrics
