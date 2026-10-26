"""
FinShield AI — Preprocessing
==============================
Fit-once / transform-many preprocessing for the Amount feature.

Design rationale
----------------
* V1–V28 are already PCA-transformed and approximately centred — no
  additional scaling is needed for a tree-based model (XGBoost).
* Time is a raw elapsed-seconds counter; XGBoost can split on raw values,
  so no transformation is applied.  If a linear model were added later,
  Time and Amount would both need scaling.
* Amount has a heavy right skew (many small transactions, a few very
  large ones).  Although XGBoost is invariant to monotone
  transformations, applying a StandardScaler to Amount can improve
  numerical stability and marginally accelerate tree splitting by
  compressing the dynamic range.

The scaler is fit on training data only and is saved inside the model
bundle so that exactly the same transform is applied at inference time.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class Preprocessor:
    """Fit a scaler on training Amount and transform new data identically."""

    def __init__(self):
        self._amount_scaler: Optional[StandardScaler] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── fit (training data only) ─────────────────────────────────────
    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        self._validate_schema(X_train)
        self._amount_scaler = StandardScaler()
        self._amount_scaler.fit(X_train[["Amount"]])
        self._fitted = True
        logger.info(
            "Preprocessor fit on Amount — mean=%.4f  std=%.4f",
            self._amount_scaler.mean_[0],
            self._amount_scaler.scale_[0],
        )
        return self

    # ── transform (any split) ────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        self._validate_schema(X)
        X_out = X.copy()
        X_out["Amount"] = self._amount_scaler.transform(X[["Amount"]])
        return X_out[FEATURE_COLUMNS]       # enforce column order

    # ── fit_transform shortcut ───────────────────────────────────────
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _validate_schema(X: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        # Ensure Class is not in features
        if "Class" in X.columns:
            raise ValueError(
                "'Class' column found in input features — "
                "remove it before preprocessing."
            )
