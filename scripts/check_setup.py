"""
FinShield AI — Setup & Environment Verification Script
========================================================
Run this script to verify that your environment, dependencies,
model artifacts, and file paths are properly configured.

Usage (from repository root):
    python scripts/check_setup.py
"""

import importlib
import json
import sys
from pathlib import Path

# Add project root to sys.path so src imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for clean cross-platform terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_status(component: str, status: str, message: str = ""):
    badge = f"{Colors.GREEN}[PASS]{Colors.RESET}" if status == "PASS" else \
            f"{Colors.YELLOW}[INFO]{Colors.RESET}" if status == "INFO" else \
            f"{Colors.RED}[FAIL]{Colors.RESET}"
    msg = f" - {message}" if message else ""
    print(f"  {badge} {component:<32} {msg}")


def main() -> int:
    print(f"\n{Colors.BOLD}{Colors.BLUE}===================================================={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}       FinShield AI — Setup Verification Tool       {Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}===================================================={Colors.RESET}\n")

    failures = 0
    warnings = 0

    # ── 1. Python Version Check ──────────────────────────────────────
    print(f"{Colors.BOLD}1. Python Environment:{Colors.RESET}")
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver.major == 3 and py_ver.minor >= 10:
        print_status("Python Version", "PASS", f"Detected Python {py_str} (compatible)")
    else:
        print_status("Python Version", "FAIL", f"Detected Python {py_str}. Recommended: Python 3.10 - 3.12.")
        failures += 1

    # ── 2. Required Dependencies Check ───────────────────────────────
    print(f"\n{Colors.BOLD}2. Required Python Packages:{Colors.RESET}")
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("streamlit", "streamlit"),
        ("matplotlib", "matplotlib"),
        ("joblib", "joblib"),
        ("scipy", "scipy"),
        ("pytest", "pytest"),
    ]

    for display_name, import_name in required_packages:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "installed")
            print_status(display_name, "PASS", f"v{ver}")
        except ImportError as e:
            print_status(display_name, "FAIL", f"Not installed ({e})")
            failures += 1

    # ── 3. Repository Structure & Source Code Check ──────────────────
    print(f"\n{Colors.BOLD}3. Project Structure & Core Files:{Colors.RESET}")
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/data.py",
        "src/preprocessing.py",
        "src/predict.py",
        "src/train.py",
        "src/evaluate.py",
        "src/metrics.py",
        "src/explain.py",
        "app/streamlit_app.py",
        "tests/test_model.py",
        "tests/test_data.py",
        "requirements.txt",
        "README.md",
    ]

    for rel_path in required_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            print_status(rel_path, "PASS", "Found")
        else:
            print_status(rel_path, "FAIL", f"Missing file: {rel_path}")
            failures += 1

    # ── 4. Pre-Trained Model Bundle Health Check ─────────────────────
    print(f"\n{Colors.BOLD}4. Model Artifacts & Inference Pipeline:{Colors.RESET}")
    from src.config import FEATURE_COLUMNS, MODEL_BUNDLE_PATH, MODEL_JSON_PATH
    from src.predict import load_model_bundle, predict_single

    if not MODEL_BUNDLE_PATH.exists():
        print_status("Model Bundle File", "FAIL", f"Missing at {MODEL_BUNDLE_PATH.name}")
        failures += 1
    elif not MODEL_JSON_PATH.exists():
        print_status("Model JSON Weights", "FAIL", f"Missing at {MODEL_JSON_PATH.name}")
        failures += 1
    else:
        try:
            bundle = load_model_bundle(MODEL_BUNDLE_PATH)
            required_keys = {"model", "preprocessor", "threshold", "model_version"}
            missing_keys = required_keys - set(bundle.keys())
            has_features = "feature_columns" in bundle or "feature_names" in bundle
            if missing_keys or not has_features:
                if not has_features:
                    missing_keys.add("feature_columns")
                print_status("Model Bundle Keys", "FAIL", f"Missing required keys: {missing_keys}")
                failures += 1
            else:
                threshold = bundle["threshold"]
                model_ver = bundle.get("model_version", "unknown")
                feat_count = len(bundle.get("feature_columns", bundle.get("feature_names", [])))
                print_status("Model Bundle Load", "PASS", f"Version '{model_ver}', Threshold: {threshold:.4f}, Features: {feat_count}")

                # Test synthetic prediction with bundle
                dummy_input = {col: 0.0 for col in FEATURE_COLUMNS}
                dummy_input["Amount"] = 99.50
                res = predict_single(dummy_input, bundle=bundle)
                if "fraud_probability" in res and "predicted_class" in res:
                    print_status("Test Inference", "PASS", f"Sample predict succeeded (P={res['fraud_probability']:.4f})")
                else:
                    print_status("Test Inference", "FAIL", "Invalid prediction output schema")
                    failures += 1
        except Exception as e:
            print_status("Model Bundle Health", "FAIL", f"Exception during validation: {e}")
            failures += 1

    # ── 5. Evaluation & Audit Reports Check ───────────────────────────
    print(f"\n{Colors.BOLD}5. Benchmark Reports:{Colors.RESET}")
    from src.config import EVALUATION_RESULTS_PATH, VALIDATION_RESULTS_PATH, THRESHOLD_CURVE_PATH

    for name, path in [
        ("Test Evaluation Report", EVALUATION_RESULTS_PATH),
        ("Validation Comparison", VALIDATION_RESULTS_PATH),
        ("Threshold Analysis", THRESHOLD_CURVE_PATH),
    ]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                info = f"{len(data)} top-level keys"
                print_status(name, "PASS", info)
            except Exception as e:
                print_status(name, "FAIL", f"Corrupt JSON ({e})")
                failures += 1
        else:
            print_status(name, "INFO", "Not present (can be generated via 'python -m src.train')")
            warnings += 1

    # ── 6. Datasets & Demo Files Check ────────────────────────────────
    print(f"\n{Colors.BOLD}6. Dataset & Test Data Files:{Colors.RESET}")
    sample_batch_path = PROJECT_ROOT / "sample_batch.csv"
    if sample_batch_path.exists():
        print_status("sample_batch.csv", "PASS", f"Found ({sample_batch_path.stat().st_size / 1024:.1f} KB, ready for batch demo)")
    else:
        print_status("sample_batch.csv", "INFO", "Not found")

    demo_samples_path = PROJECT_ROOT / "data" / "sample_transactions.csv"
    if demo_samples_path.exists():
        print_status("data/sample_transactions.csv", "PASS", f"Found ({demo_samples_path.stat().st_size / 1024:.1f} KB, ready for quick examples)")
    else:
        print_status("data/sample_transactions.csv", "INFO", "Not found")

    from src.config import DATASET_PATH
    if DATASET_PATH.exists():
        size_mb = DATASET_PATH.stat().st_size / (1024 * 1024)
        print_status("creditcard.csv (Full Raw)", "PASS", f"Present ({size_mb:.1f} MB) — Full retraining enabled")
    else:
        print_status(
            "creditcard.csv (Full Raw)",
            "INFO",
            "Not present (~144 MB Kaggle dataset is omitted from Git to respect file-size limits). "
            "Pre-trained model bundle is included, so the Streamlit dashboard and batch inference work immediately. "
            "Download creditcard.csv from Kaggle only if you plan to retrain the model."
        )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}===================================================={Colors.RESET}")
    if failures == 0:
        print(f"{Colors.BOLD}{Colors.GREEN}✓ Setup Verification PASSED! FinShield AI is ready to run.{Colors.RESET}")
        print(f"\nQuick Start Commands:")
        print(f"  • Launch Dashboard:  {Colors.BOLD}streamlit run app/streamlit_app.py{Colors.RESET}")
        print(f"  • Run Tests:         {Colors.BOLD}pytest tests/ -v{Colors.RESET}")
        if DATASET_PATH.exists():
            print(f"  • Re-evaluate Model: {Colors.BOLD}python -m src.evaluate{Colors.RESET}")
            print(f"  • Retrain Model:     {Colors.BOLD}python -m src.train{Colors.RESET}")
        print()
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.RED}✗ Setup Verification encountered {failures} error(s).{Colors.RESET}")
        print("Please resolve the missing requirements listed above before proceeding.")
        print("Tip: Install missing dependencies via: pip install -r requirements.txt\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
