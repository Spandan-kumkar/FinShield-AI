"""
FinShield AI — Configuration
=============================
Central configuration for paths, random seeds, feature schema,
and model hyperparameters.  All paths are relative to PROJECT_ROOT
so the project is portable.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The original CSV lives at the project root (as provided).
DATASET_PATH = PROJECT_ROOT / "creditcard.csv"

# Model artifacts
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
MODEL_JSON_PATH = ARTIFACTS_DIR / "model.json"
EVALUATION_RESULTS_PATH = REPORTS_DIR / "test_evaluation.json"
VALIDATION_RESULTS_PATH = REPORTS_DIR / "validation_comparison.json"
THRESHOLD_CURVE_PATH = REPORTS_DIR / "threshold_analysis.json"

# ── Reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data splitting ───────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Feature schema ───────────────────────────────────────────────────
TARGET_COL = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]        # V1 … V28
OTHER_FEATURES = ["Time", "Amount"]
FEATURE_COLUMNS = OTHER_FEATURES + PCA_FEATURES         # 30 features
NUM_FEATURES = len(FEATURE_COLUMNS)                      # 30

# ── XGBoost defaults (Experiment A — unweighted) ─────────────────────
XGBOOST_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    eval_metric="aucpr",      # average-precision — good for imbalanced data
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# ── Labels ───────────────────────────────────────────────────────────
LABEL_MAP = {0: "Legitimate", 1: "Fraud"}
MODEL_VERSION = "finshield-v1"
