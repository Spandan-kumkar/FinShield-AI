"""
FinShield AI — Prediction module
===================================
Loads the authoritative model bundle and provides single-transaction
and batch-CSV prediction, independently of Streamlit.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

import xgboost as xgb

from src.config import (
    FEATURE_COLUMNS,
    LABEL_MAP,
    MODEL_BUNDLE_PATH,
    MODEL_JSON_PATH,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ── Model bundle cache ───────────────────────────────────────────────

_cached_bundle: Optional[dict] = None


def load_model_bundle(path: Path = MODEL_BUNDLE_PATH) -> dict:
    """Load (and cache) the model bundle from disk.

    Supports both self-contained bundles and universal cross-platform
    bundles where XGBoost model weights are stored in model.json.
    """
    global _cached_bundle
    if _cached_bundle is not None:
        return _cached_bundle
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run 'python -m src.train' first."
        )
    bundle = joblib.load(path)

    # Cross-platform model resolution:
    # If the bundle does not have a live XGBClassifier or model.json exists,
    # load from native JSON format to prevent OS-specific C++ binary corruption.
    current_model = bundle.get("model")
    if current_model is None or not hasattr(current_model, "predict_proba"):
        if MODEL_JSON_PATH.exists():
            model = xgb.XGBClassifier()
            model.load_model(str(MODEL_JSON_PATH))
            bundle["model"] = model
        else:
            raise FileNotFoundError(
                f"Model bundle at {path} contains no model object and native {MODEL_JSON_PATH} was not found."
            )
    elif MODEL_JSON_PATH.exists() and not hasattr(current_model, "predict_proba"):
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_JSON_PATH))
        bundle["model"] = model

    _cached_bundle = bundle
    logger.info("Model bundle loaded from %s", path)
    return _cached_bundle


def clear_bundle_cache() -> None:
    """Clear the cached bundle (useful in tests)."""
    global _cached_bundle
    _cached_bundle = None


def _validate_threshold(threshold: Any) -> float:
    """Validate threshold strictly; never allow missing, non-float, or out-of-range thresholds."""
    if threshold is None:
        raise ValueError("Model bundle is missing required 'threshold' configuration.")
    try:
        t = float(threshold)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid threshold type/value in bundle: {threshold!r}")
    if not (0.0 < t < 1.0):
        raise ValueError(
            f"Classification threshold must be strictly between 0 and 1, got {t}."
        )
    return t


# ── Single prediction ────────────────────────────────────────────────

def predict_single(
    transaction: Dict[str, Any],
    bundle: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Predict a single transaction.

    Parameters
    ----------
    transaction : dict
        Keys must include every name in FEATURE_COLUMNS.
        'Class' must NOT be present.

    Returns
    -------
    dict with predicted_class, predicted_label, fraud_probability,
    threshold, model_version, and any warnings.
    """
    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = _validate_threshold(bundle.get("threshold"))
    feature_cols = bundle["feature_columns"]
    model_version = bundle["model_version"]

    warnings_list: List[str] = []

    # ── Validation ───────────────────────────────────────────────────
    if TARGET_COL in transaction:
        raise ValueError(
            f"'{TARGET_COL}' must not be included in prediction input. "
            "Remove the target column before calling predict."
        )

    missing = [c for c in feature_cols if c not in transaction]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Check for non-numeric / NaN
    row = {c: transaction[c] for c in feature_cols}
    for col, val in row.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"Feature '{col}' has a missing/NaN value.")
        try:
            float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{col}' has non-numeric value: {val!r}"
            )

    extra = [c for c in transaction if c not in feature_cols and c != TARGET_COL]
    if extra:
        warnings_list.append(f"Extra columns ignored: {extra}")

    # ── Predict ──────────────────────────────────────────────────────
    df = pd.DataFrame([row], columns=feature_cols)
    df_pp = preprocessor.transform(df)
    prob = float(model.predict_proba(df_pp)[:, 1][0])
    pred_class = int(prob >= threshold)

    return {
        "predicted_class": pred_class,
        "predicted_label": LABEL_MAP[pred_class],
        "fraud_probability": round(prob, 6),
        "threshold": threshold,
        "model_version": model_version,
        "warnings": warnings_list,
    }


# ── Batch prediction ────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    bundle: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Predict on a DataFrame of transactions.

    - Validates schema.
    - Strips 'Class' if present (never feeds it to the model).
    - Preserves original row order and index.

    Returns a new DataFrame with columns:
        row_index, predicted_class, predicted_label, fraud_probability
    """
    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = _validate_threshold(bundle.get("threshold"))
    feature_cols = bundle["feature_columns"]

    # Separate Class if present
    if TARGET_COL in df.columns:
        logger.warning(
            "'Class' column found in batch input — it will be ignored "
            "for prediction and not passed to the model."
        )
        df = df.drop(columns=[TARGET_COL])

    # Validate required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in CSV: {missing}")

    # Select and order features
    X = df[feature_cols].copy()

    # Check for NaNs
    nan_rows = X.isnull().any(axis=1)
    if nan_rows.any():
        n_bad = int(nan_rows.sum())
        raise ValueError(
            f"{n_bad} row(s) contain missing values in required feature columns."
        )

    # Transform and predict
    X_pp = preprocessor.transform(X)
    probs = model.predict_proba(X_pp)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = pd.DataFrame({
        "row_index": range(len(df)),
        "predicted_class": preds,
        "predicted_label": [LABEL_MAP[p] for p in preds],
        "fraud_probability": np.round(probs, 6),
    })
    return results


def predict_batch_csv(
    csv_path: Union[str, Path],
    bundle: Optional[dict] = None,
) -> pd.DataFrame:
    """Load a CSV file and run batch prediction."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    return predict_batch(df, bundle=bundle)
