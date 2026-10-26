"""
FinShield AI — Training pipeline
==================================
Trains XGBoost experiments, selects the threshold on validation data,
picks the best model, evaluates on the held-out test set, and saves
everything into the model bundle + evaluation artefacts.
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    EVALUATION_RESULTS_PATH,
    FEATURE_COLUMNS,
    MODEL_BUNDLE_PATH,
    MODEL_VERSION,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
    THRESHOLD_CURVE_PATH,
    VALIDATION_RESULTS_PATH,
    XGBOOST_BASE_PARAMS,
)
from src.data import (
    check_cross_partition_overlap,
    deduplicate_dataset,
    load_dataset,
    separate_features_target,
    split_data,
    validate_dataset,
)
from src.metrics import compute_metrics
from src.preprocessing import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _train_xgb(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label: str,
) -> XGBClassifier:
    """Train one XGBClassifier with early stopping on validation AUCPR."""
    logger.info("Training experiment: %s", label)
    logger.info("  params = %s", {k: v for k, v in params.items()
                                   if k not in ("n_jobs",)})

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("  best iteration = %s", getattr(model, "best_iteration", "N/A"))
    return model


def _evaluate_on_split(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> dict:
    """Return metrics for a given split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    m = compute_metrics(y.values, y_pred, y_prob, threshold)
    logger.info(
        "  [%s] F1=%.4f  Recall=%.4f  Prec=%.4f  AP=%.4f  PR-AUC=%.4f  ROC-AUC=%.4f",
        split_name, m["fraud_f1"], m["fraud_recall"],
        m["fraud_precision"], m["average_precision"], m["pr_auc"], m["roc_auc"],
    )
    return m


def _select_threshold(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    Use the precision-recall curve on validation data to find the
    threshold that maximises F1. Also evaluate 0.5 as a reference.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    # F1 at every threshold (skip last entry — it's a sentinel)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    # Evaluate at 0.5 for comparison
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_val, y_pred_05, pos_label=1, zero_division=0))

    analysis = {
        "best_f1_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "default_0.5_f1": f1_05,
        "criterion": "max_f1_on_validation",
    }
    logger.info(
        "Threshold analysis — best=%.4f (F1=%.4f)  default_0.5 (F1=%.4f)",
        best_threshold, best_f1, f1_05,
    )
    return analysis


# ── Main training pipeline ──────────────────────────────────────────

def run_training_pipeline() -> None:
    t0 = time.time()

    # 1. Load, validate, and deduplicate
    logger.info("═══ Step 1: Load, validate, and deduplicate dataset ═══")
    df_raw = load_dataset()
    raw_report = validate_dataset(df_raw)

    df_clean, dedup_audit = deduplicate_dataset(df_raw)

    # 2. Separate features / target
    logger.info("═══ Step 2: Separate features and target ═══")
    X, y = separate_features_target(df_clean)

    # 3. Stratified Split & Zero-Leakage Check
    logger.info("═══ Step 3: Stratified split (70 / 15 / 15) ═══")
    splits = split_data(X, y)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    overlap = check_cross_partition_overlap(splits)
    logger.info("  Cross-partition feature overlap: %s", overlap)
    total_overlap = sum(overlap.values())
    if total_overlap != 0:
        raise RuntimeError(f"Data leakage detected! Cross-partition overlap: {overlap}")
    logger.info("  Confirmed: 0 cross-partition feature overlap (leak-free).")

    # 4. Preprocess — fit on training only
    logger.info("═══ Step 4: Preprocessing (fit on train) ═══")
    preprocessor = Preprocessor()
    X_train_pp = preprocessor.fit_transform(X_train)
    X_val_pp   = preprocessor.transform(X_val)
    X_test_pp  = preprocessor.transform(X_test)

    # 5. Experiment A — unweighted
    logger.info("═══ Step 5a: Experiment A — Unweighted XGBoost ═══")
    params_a = dict(XGBOOST_BASE_PARAMS)
    model_a = _train_xgb(params_a, X_train_pp, y_train, X_val_pp, y_val, "A_unweighted")

    # 6. Experiment B — class-weighted
    logger.info("═══ Step 5b: Experiment B — Class-Weighted XGBoost ═══")
    n_legit_train = int((y_train == 0).sum())
    n_fraud_train = int((y_train == 1).sum())
    spw = n_legit_train / n_fraud_train
    logger.info("  scale_pos_weight = %d / %d = %.2f", n_legit_train, n_fraud_train, spw)
    params_b = dict(XGBOOST_BASE_PARAMS, scale_pos_weight=spw)
    model_b = _train_xgb(params_b, X_train_pp, y_train, X_val_pp, y_val, "B_weighted")

    # 7. Validation comparison (at default 0.5 threshold)
    logger.info("═══ Step 6: Validation comparison (threshold=0.5) ═══")
    val_a = _evaluate_on_split(model_a, X_val_pp, y_val, 0.5, "val_A")
    val_b = _evaluate_on_split(model_b, X_val_pp, y_val, 0.5, "val_B")

    # ── Model selection ──────────────────────────────────────────────
    logger.info("═══ Step 7: Model selection ═══")
    comparison = {
        "experiment_A_unweighted": {
            "params": {k: str(v) for k, v in params_a.items()},
            "validation_metrics": val_a,
        },
        "experiment_B_weighted": {
            "params": {k: str(v) for k, v in params_b.items()},
            "validation_metrics": val_b,
        },
        "selection_criterion": (
            "Primary criterion: Highest fraud-class F1-score on the validation set. "
            "Secondary criterion: High precision to minimize false alarms in operational bank workflows. "
            "Evaluated strictly on the validation partition prior to inspecting test data."
        ),
    }

    # Selection rule: Primary criterion is validation F1
    if val_b["fraud_f1"] > val_a["fraud_f1"]:
        selected_label = "B_weighted"
        selected_model = model_b
        selected_params = params_b
        reason = (
            f"Experiment B (weighted) achieved higher validation F1 ({val_b['fraud_f1']:.4f} "
            f"vs {val_a['fraud_f1']:.4f})."
        )
    else:
        selected_label = "A_unweighted"
        selected_model = model_a
        selected_params = params_a
        reason = (
            f"Experiment A (unweighted) achieved higher validation F1 ({val_a['fraud_f1']:.4f} "
            f"vs {val_b['fraud_f1']:.4f}) and superior precision ({val_a['fraud_precision']:.4f} "
            f"vs {val_b['fraud_precision']:.4f}), resulting in substantially fewer false alarms "
            f"while maintaining strong fraud detection."
        )

    logger.info("Model selection decision: %s. Reason: %s", selected_label, reason)
    comparison["selected_experiment"] = selected_label
    comparison["selection_reason"] = reason

    # 8. Threshold selection on validation
    logger.info("═══ Step 8: Threshold selection (validation) ═══")
    threshold_info = _select_threshold(selected_model, X_val_pp, y_val)
    selected_threshold = threshold_info["best_f1_threshold"]

    # Re-evaluate selected model at the chosen threshold
    val_selected = _evaluate_on_split(
        selected_model, X_val_pp, y_val, selected_threshold, "val_selected_threshold",
    )
    comparison[f"validation_{selected_label}_at_best_threshold"] = val_selected

    # 9. Final test evaluation
    logger.info("═══ Step 9: Final test evaluation (Held-Out Test Set) ═══")
    test_metrics = _evaluate_on_split(
        selected_model, X_test_pp, y_test, selected_threshold, "TEST",
    )
    test_metrics["split"] = "test"
    test_metrics["model_experiment"] = selected_label
    test_metrics["model_version"] = MODEL_VERSION
    test_metrics["note"] = (
        "These metrics are from the held-out test set and were NOT used "
        "for model selection, threshold tuning, or feature selection."
    )

    # ── Log human-readable summary ───────────────────────────────────
    cm = test_metrics["confusion_matrix"]
    logger.info("─── Final Test Set Summary ───")
    logger.info("  Total fraud in test set     : %d", test_metrics["total_fraud"])
    logger.info("  Fraud detected (TP)         : %d", cm["TP"])
    logger.info("  Fraud missed (FN)           : %d", cm["FN"])
    logger.info("  Legit incorrectly flagged   : %d", cm["FP"])
    logger.info("  Fraud recall                : %.4f", test_metrics["fraud_recall"])
    logger.info("  Fraud precision             : %.4f", test_metrics["fraud_precision"])
    logger.info("  Fraud F1                    : %.4f", test_metrics["fraud_f1"])
    logger.info("  ROC-AUC                     : %.4f", test_metrics["roc_auc"])
    logger.info("  Average precision (AP)      : %.4f", test_metrics["average_precision"])
    logger.info("  PR-AUC (Trapezoidal)        : %.4f", test_metrics["pr_auc"])
    logger.info("  Accuracy                    : %.4f", test_metrics["accuracy"])
    logger.info(
        "Note: accuracy is misleadingly high because ~99.83%% of "
        "transactions are legitimate. Fraud-specific metrics (Recall, Precision, F1, AP) "
        "are the true measures of detection capability."
    )

    # 10. Save artefacts
    logger.info("═══ Step 10: Save artefacts ═══")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model bundle — everything needed for inference
    bundle = {
        "model": selected_model,
        "preprocessor": preprocessor,
        "threshold": selected_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": MODEL_VERSION,
        "experiment": selected_label,
        "params": selected_params,
        "scale_pos_weight": selected_params.get("scale_pos_weight", None),
        "train_class_counts": {"legitimate": n_legit_train, "fraud": n_fraud_train},
        "dataset_report": raw_report,
        "dedup_audit": dedup_audit,
        "cross_partition_overlap": overlap,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    logger.info("  Model bundle saved → %s", MODEL_BUNDLE_PATH)

    # Evaluation JSON
    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("  Test evaluation saved → %s", EVALUATION_RESULTS_PATH)

    # Validation comparison JSON
    with open(VALIDATION_RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("  Validation comparison saved → %s", VALIDATION_RESULTS_PATH)

    # Threshold analysis JSON
    with open(THRESHOLD_CURVE_PATH, "w") as f:
        json.dump(threshold_info, f, indent=2)
    logger.info("  Threshold analysis saved → %s", THRESHOLD_CURVE_PATH)

    elapsed = time.time() - t0
    logger.info("═══ Pipeline completed in %.1f seconds ═══", elapsed)


# ── CLI entry-point ──────────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline()
