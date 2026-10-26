"""
FinShield AI — SHAP Explainability
====================================
Generates local SHAP explanations for individual transaction
predictions using TreeExplainer (fast, exact for tree models).

Important notes
---------------
* SHAP explains model behaviour — it does not establish causation.
* V1–V28 are anonymized PCA features; their SHAP values are displayed
  without inventing real-world interpretations.
* Positive SHAP values push the prediction toward the fraud class;
  negative values push it toward the legitimate class.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.config import FEATURE_COLUMNS, MODEL_BUNDLE_PATH
from src.predict import _validate_threshold, load_model_bundle

logger = logging.getLogger(__name__)


def explain_single(
    transaction: Dict[str, Any],
    bundle: Optional[dict] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Generate a SHAP explanation for one transaction.

    Returns
    -------
    dict with keys:
        base_value       – expected model output (log-odds or probability)
        shap_values      – list of (feature, shap_value) for all features
        top_contributors – top_n features ranked by |shap_value|
        prediction_info  – fraud probability, predicted class, threshold
    """
    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = _validate_threshold(bundle.get("threshold"))
    feature_cols = bundle["feature_columns"]

    # Build a single-row DataFrame
    row = {c: transaction[c] for c in feature_cols}
    df = pd.DataFrame([row], columns=feature_cols)
    df_pp = preprocessor.transform(df)

    # Prediction
    prob = float(model.predict_proba(df_pp)[:, 1][0])
    pred_class = int(prob >= threshold)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(df_pp)

    # shap_vals may be a single array (binary) or list.
    # For XGBoost binary classification, TreeExplainer returns a single
    # array of shape (n_samples, n_features) when the model outputs
    # log-odds.  We want the values for the positive (fraud) class.
    if isinstance(shap_vals, list):
        # Older shap: list of [class0_vals, class1_vals]
        sv = shap_vals[1][0]
        base = float(explainer.expected_value[1])
    elif shap_vals.ndim == 3:
        # Newer shap: shape (n_samples, n_features, n_classes)
        sv = shap_vals[0, :, 1]
        base = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)
    else:
        sv = shap_vals[0]
        base = float(
            explainer.expected_value
            if not hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value[0]
        )

    # Build per-feature SHAP dict
    feature_shap = [
        {"feature": feat, "shap_value": float(sv[i]), "feature_value": float(df_pp.iloc[0, i])}
        for i, feat in enumerate(feature_cols)
    ]
    # Sort by absolute contribution
    ranked = sorted(feature_shap, key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "base_value": base,
        "shap_values": feature_shap,
        "top_contributors": ranked[:top_n],
        "prediction_info": {
            "fraud_probability": prob,
            "predicted_class": pred_class,
            "threshold": threshold,
        },
    }


def get_shap_explanation_dataframe(
    explanation: Dict[str, Any],
) -> pd.DataFrame:
    """Convert top SHAP contributors to a DataFrame for display."""
    rows = explanation["top_contributors"]
    df = pd.DataFrame(rows)
    df["direction"] = df["shap_value"].apply(
        lambda v: "→ Fraud" if v > 0 else "→ Legitimate"
    )
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
