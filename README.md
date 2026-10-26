# 🛡️ FinShield AI — Explainable Credit Card Fraud Detection

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Prototype-green.svg)]()
[![Tests](https://img.shields.io/badge/Tests-51%20Passing-brightgreen.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io/)

An academically rigorous, end-to-end machine learning system that detects credit card fraud under extreme class imbalance using **XGBoost**, validation-tuned decision thresholds, and **SHAP** (SHapley Additive exPlanations) for local model explainability.

> **Academic Disclaimer:** This is an educational and research prototype developed for transparent, reproducible evaluation under extreme class imbalance. It is not an operational banking transaction processing system.

---

## Table of Contents

1. [Project Overview & Key Features](#1-project-overview--key-features)
2. [First-Run Quick Start (Pre-Trained Model)](#2-first-run-quick-start-pre-trained-model)
3. [Windows Setup Guide (PowerShell)](#3-windows-setup-guide-powershell)
4. [macOS & Linux Setup Guide (Bash / Zsh)](#4-macos--linux-setup-guide-bash--zsh)
5. [Setup Verification Script](#5-setup-verification-script)
6. [Dataset & Model Distribution Strategy](#6-dataset--model-distribution-strategy)
7. [Running the Application, Tests & Pipelines](#7-running-the-application-tests--pipelines)
8. [Dataset Integrity & Zero-Leakage Deduplication](#8-dataset-integrity--zero-leakage-deduplication)
9. [Machine Learning Methodology & Benchmark Results](#9-machine-learning-methodology--benchmark-results)
10. [Explainability via SHAP](#10-explainability-via-shap)
11. [Repository Structure](#11-repository-structure)
12. [Troubleshooting Guide](#12-troubleshooting-guide)
13. [Limitations & Viva Caveats](#13-limitations--viva-caveats)

---

## 1. Project Overview & Key Features

Credit card fraud presents two formidable engineering challenges:
1. **Severe Class Imbalance:** Fraudulent transactions account for $< 0.2\%$ of records. Traditional accuracy is misleading (a naive classifier predicting "legitimate" on all transactions achieves $99.83\%$ accuracy while detecting 0 fraud cases).
2. **The "Black-Box" Dilemma:** High-capacity gradient boosted trees provide top-tier discrimination, but banking investigators require transparency before taking adverse customer actions.

### Core Capabilities:
- **Zero-Leakage Data Pipeline:** Identifies and resolves 1,081 duplicate transactions in raw benchmark data, guaranteeing **strictly 0 cross-partition record leakage** across Train (70%), Validation (15%), and Test (15%) splits.
- **Validation-Driven Threshold Selection:** Optimizes classification threshold ($0.4119$) on validation minority-class $F_1$, increasing detection rate over the default $0.5$ threshold.
- **Game-Theoretic Explainability:** Computes exact local Shapley values via `TreeExplainer` for every transaction, producing waterfall attribution charts that distinguish risk drivers.
- **Interactive Multi-Page Dashboard:** Built with Streamlit, supporting:
  - 📊 **Overview:** Dataset distribution, audit charts, benchmark metrics, and confusion matrix.
  - 🔍 **Single Transaction Analysis:** Pre-loaded real transactions (no typing 30 numbers manually), editable inputs, probability gauge, and SHAP waterfall plots.
  - 📁 **Batch CSV Analysis:** Bulk transaction screening with one-click demo batch testing and downloadable prediction CSVs.
  - 📈 **Model Performance:** Confusion matrices, ROC curves, and Precision-Recall curves.

---

## 2. First-Run Quick Start (Pre-Trained Model)

**Teammates do NOT need to retrain the model or download the 144 MB dataset just to launch the dashboard.**

The repository includes:
- `artifacts/model_bundle.joblib` (~370 KB): Serialized trained model, fitted scaler, decision threshold, and metadata.
- `sample_batch.csv` (~16 KB): 30 realistic transactions for instant batch screening.
- `data/sample_transactions.csv` (~5 KB): 10 labeled transactions powering the Quick Examples dropdown in the single-transaction view.

### The 3-Step First Run:
1. **Clone** the repository.
2. **Install** dependencies in a virtual environment (`pip install -r requirements.txt`).
3. **Launch** the dashboard (`streamlit run app/streamlit_app.py`).

---

## 3. Windows Setup Guide (PowerShell)

Follow these steps using **Windows PowerShell** (Run as standard user or Administrator):

### Step 1: Prerequisites
Ensure Python (version 3.10, 3.11, or 3.12 recommended; 3.14 compatible) and Git are installed.
```powershell
python --version
git --version
```
*(If `python` is not recognized, ensure "Add Python to PATH" was checked during Python installation).*

### Step 2: Clone the Repository
```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd FinsheildAi
```

### Step 3: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 4: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
> **PowerShell Execution Policy Note:** If Windows displays `Execution_Policies` or `script execution is disabled on this system`, run this command in your current PowerShell session (does not alter global machine security):
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```
> *Alternative (Command Prompt `cmd.exe`):* `venv\Scripts\activate.bat`

### Step 5: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Verify the Environment
```powershell
python scripts\check_setup.py
```
*(Confirms dependencies, bundle health, files, and inference pipeline with clean PASS badges).*

### Step 7: Launch the Streamlit Dashboard
```powershell
streamlit run app\streamlit_app.py
```
Your browser will open automatically at `http://localhost:8501`.

### To Deactivate the Environment When Done:
```powershell
deactivate
```

---

## 4. macOS & Linux Setup Guide (Bash / Zsh)

Follow these steps in Terminal:

### Step 1: Prerequisites
Ensure Python and Git are installed:
```bash
python3 --version
git --version
```

### Step 2: Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd FinsheildAi
```

### Step 3: Create a Virtual Environment
```bash
python3 -m venv venv
```

### Step 4: Activate the Virtual Environment
```bash
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Verify the Environment
```bash
python scripts/check_setup.py
```

### Step 7: Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Your browser will open automatically at `http://localhost:8501`.

### To Deactivate the Environment When Done:
```bash
deactivate
```

---

## 5. Setup Verification Script

FinShield AI includes a cross-platform diagnostic script:
```bash
python scripts/check_setup.py
```

### What It Inspects:
1. **Python Version:** Checks compatibility ($\ge 3.10$).
2. **Dependencies:** Verifies importability and versions of `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `matplotlib`, `joblib`, `scipy`, `pytest`.
3. **Project Files:** Validates repository structure and source modules.
4. **Model Bundle Health:** Loads `artifacts/model_bundle.joblib`, checks dictionary keys, validates threshold, and runs a test single-row prediction.
5. **Benchmark Reports:** Checks parseability of `reports/*.json`.
6. **Dataset Availability:** Confirms test files (`sample_batch.csv`, `data/sample_transactions.csv`) and reports whether `creditcard.csv` is present for retraining.

Prints clean `[PASS]`, `[INFO]`, or `[FAIL]` indicators with actionable remediation steps.

---

## 6. Dataset & Model Distribution Strategy

| Asset | Size | Distributed in Git? | Reason & Instructions |
|---|---|---|---|
| `artifacts/model_bundle.joblib` | ~370 KB | **Yes** | Lightweight serialized bundle allowing immediate dashboard execution and testing. |
| `reports/*.json` | ~5 KB total | **Yes** | Held-out test evaluation and threshold metrics proving model benchmarks. |
| `sample_batch.csv` | ~16 KB | **Yes** | 30-row inference batch for demo testing. |
| `data/sample_transactions.csv` | ~5 KB | **Yes** | 10 real labeled transactions for single-prediction demo quick-loading. |
| `creditcard.csv` | **143.8 MB** | **No** (in `.gitignore`) | Exceeds GitHub's 100 MB hard file limit. Required **only** if you wish to retrain or re-partition the dataset. |

### How to Download the Full Dataset (For Retraining Only)
If you wish to retrain the model or reproduce the full 70/15/15 dataset split:
1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the unzipped `creditcard.csv` in the project root (`FinsheildAi/creditcard.csv`) or in `FinsheildAi/data/`.
3. Run `python -m src.train` to reproduce the entire training pipeline.

---

## 7. Running the Application, Tests & Pipelines

All commands are designed to be run from the **repository root**:

### 1. Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 2. Run Automated Unit Tests (51 Tests)
```bash
pytest tests/ -v
```
- Tests model loading, preprocessing, single prediction, batch prediction, confusion matrix conventions, zero-division safety, report consistency, and SHAP feature attribution.
- *Note:* If `creditcard.csv` is not present, raw dataset validation tests gracefully skip while all unit, model, and metric tests run and pass.

### 3. Re-Evaluate Model on Test Split (Requires `creditcard.csv`)
```bash
python -m src.evaluate
```
Loads the saved model bundle, reproduces the clean held-out test split, and regenerates `reports/test_evaluation.json`.

### 4. Full Model Retraining Pipeline (Requires `creditcard.csv`)
```bash
python -m src.train
```
Executes:
1. Data discovery and validation.
2. Deduplication and zero-leakage verification.
3. Stratified 70/15/15 train/val/test splitting (`seed=42`).
4. Fitting `StandardScaler` on `X_train`.
5. Training Experiment A (unweighted XGBoost) and Experiment B (`scale_pos_weight` weighted XGBoost).
6. Precision-recall threshold sweep on validation predictions to select optimal $F_1$ threshold (`0.4119`).
7. Evaluation on held-out test split.
8. Serialization to `artifacts/model_bundle.joblib` and `reports/*.json`.

---

## 8. Dataset Integrity & Zero-Leakage Deduplication

The benchmark dataset consists of 284,807 European cardholder transactions (September 2013):
- **Features:** 30 input features (`Time`, `Amount`, `V1`–`V28` PCA components).
- **Target:** `Class` ($0$ = Legitimate, $1$ = Fraud).

### Duplicate Row Investigation & Deduplication
During exploratory audit, **1,081 duplicate transactions** were identified. In standard random splitting, duplicate rows cause **cross-partition data leakage** where identical feature vectors appear in both the training set and the test set (evaluating memorization rather than generalization).

FinShield AI applies `src.data.deduplicate_dataset` prior to splitting:

| Metric | Raw Dataset | Clean Modeled Dataset | Impact |
|---|---|---|---|
| Total Transactions | 284,807 | **283,726** | 1,081 duplicates removed |
| Legitimate (`Class = 0`) | 284,315 | **283,253** | 1,062 duplicates removed |
| Fraudulent (`Class = 1`) | 492 | **473** | 19 duplicates removed |
| Fraud Prevalence | 0.1727% | **0.1667%** | Severe imbalance preserved |
| **Cross-Partition Overlap** | **217 rows** (3 fraud) | **0 rows** | **Strict zero-leakage guarantee** |

---

## 9. Machine Learning Methodology & Benchmark Results

### Partitioning & Preprocessing
- **Stratified Split:** 70% Train (198,608), 15% Validation (42,559), 15% Test (42,559).
- **Scaling:** `StandardScaler` fitted strictly on `X_train` for `Amount` ($\mu = 88.32, \sigma = 242.61$) and saved in the bundle. `V1`–`V28` and `Time` are kept in native form.

### Model Selection (Validation Set)
Two models were compared under identical tree architectures (`max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `eval_metric="aucpr"`):
- **Experiment A (Unweighted XGBoost):** Validation $F_1 = 0.8387$, Precision = $0.9811$, 1 false alarm.
- **Experiment B (Weighted `scale_pos_weight=599.02`):** Validation $F_1 = 0.8372$, Precision = $0.9310$, 4 false alarms.

**Selection Decision:** Experiment A was selected for production because it achieved superior balance and reduced false alarms by 75%.

### Threshold Optimization
A threshold sweep on validation precision-recall curve coordinates identified **`0.4119`** as the optimal threshold maximizing minority-class $F_1$ ($0.8571$ vs. $0.8387$ at 0.5).

### Final Held-Out Test Set Performance (42,559 samples, 71 fraud)

| Metric | Score | Practical Meaning |
|---|---|---|
| **Fraud Recall** | **0.7324** | Catches **73.2%** of all fraudulent transactions (52 of 71) |
| **Fraud Precision** | **0.9123** | **91.2%** of flagged transactions are confirmed fraud |
| **Fraud F1-Score** | **0.8125** | Harmonic mean of precision and recall |
| **Average Precision (AP)** | **0.8173** | Weighted mean precision across recall thresholds |
| **PR-AUC (Trapezoidal)** | **0.8167** | Area under precision-recall curve coordinates |
| **ROC-AUC** | **0.9626** | Area under ROC curve |
| **Specificity** | **0.9999** | Correct rejection rate of legitimate transactions |
| **False Positive Rate** | **0.00012** | Only 5 false alarms out of 42,488 legitimate cards (0.012%) |
| **Accuracy** | **0.9994** | 99.94% overall sample accuracy |

### Test Set Confusion Matrix
```text
                      PREDICTED
                 Legitimate    Fraud
ACTUAL  Legit      42,483        5      (Specificity: 99.99%)
        Fraud          19       52      (Recall: 73.24%)
```

---

## 10. Explainability via SHAP

FinShield AI uses **SHAP TreeExplainer** to compute local Shapley values:
- **Base Value ($E[f(x)]$):** Model baseline log-odds output over training distribution ($\approx -7.21$).
- **Color Convention:**
  - 🔴 **Red (+SHAP):** Pushes prediction toward **Fraud**.
  - 🟢 **Green (-SHAP):** Pushes prediction toward **Legitimate**.
- **Scientific Integrity:** Features `V1`–`V28` are anonymized PCA coordinates. In accordance with academic honesty, no speculative real-world labels (e.g. "merchant location") are attributed to them. SHAP explains the *mathematical decision boundary*.

---

## 11. Repository Structure

```text
FinsheildAi/
├── .gitignore                        # Standard Git exclusions (venv, caches, 144MB dataset)
├── README.md                         # Project documentation and setup guide
├── requirements.txt                  # Python dependency specifications
├── sample_batch.csv                  # 30-row inference batch for demo testing
├── app/
│   └── streamlit_app.py              # Interactive multi-page Streamlit dashboard
├── artifacts/
│   └── model_bundle.joblib           # Pre-trained model bundle (XGBoost, Scaler, Threshold)
├── data/
│   ├── README.md                     # Data notes and deduplication audit
│   ├── sample_batch.csv              # Demo batch copy
│   └── sample_transactions.csv       # 10 labeled transactions for quick dashboard loading
├── reports/
│   ├── test_evaluation.json          # Held-out test evaluation report
│   ├── threshold_analysis.json       # Validation threshold sweep metrics
│   └── validation_comparison.json    # Experiment A vs B comparison report
├── scripts/
│   └── check_setup.py                # Cross-platform environment verification script
├── src/
│   ├── __init__.py
│   ├── config.py                     # Portable paths (PROJECT_ROOT), seeds, schemas
│   ├── data.py                       # Loading, validation, deduplication, stratified splitting
│   ├── evaluate.py                   # Standalone evaluation against test split
│   ├── explain.py                    # SHAP TreeExplainer attribution engine
│   ├── metrics.py                    # Classification metrics suite (AP, PR-AUC, CM)
│   ├── predict.py                    # Single-transaction and batch CSV inference engine
│   ├── preprocessing.py              # Training-fitted StandardScaler for Amount
│   └── train.py                      # Reproducible end-to-end training pipeline
└── tests/
    ├── __init__.py
    ├── test_data.py                  # Unit tests for loading, validation, deduplication, splitting
    └── test_model.py                 # Unit tests for bundle, predictions, metrics, threshold, SHAP
```

---

## 12. Troubleshooting Guide

### 1. PowerShell Script Execution Blocked (Windows)
- **Symptom:** `File Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Remedy:** Bypass the policy for the active terminal session without changing system-wide security:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```

### 2. Streamlit Port Conflict (Port 8501 Busy)
- **Symptom:** `Port 8501 is already in use.`
- **Remedy:** Specify an alternative port:
  ```bash
  streamlit run app/streamlit_app.py --server.port 8502
  ```

### 3. Missing Dataset Error When Retraining
- **Symptom:** `FileNotFoundError: Dataset not found at ... creditcard.csv`
- **Remedy:** The pre-trained dashboard does not need this file. If you want to run `python -m src.train`, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 4. `python` Command Not Found (Windows)
- **Symptom:** `'python' is not recognized as an internal or external command.`
- **Remedy:** Use `py` launcher or reinstall Python ensuring **"Add Python to PATH"** is selected.

### 5. Pip Dependency Resolution Conflicts
- **Symptom:** `ERROR: Could not find a version that satisfies the requirement...`
- **Remedy:** Ensure your Python version is between 3.10 and 3.14. Upgrade pip:
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

---

## 13. Limitations & Viva Caveats

During technical viva or academic review, be prepared to discuss:
1. **Accuracy is Deceptive:** 99.94% accuracy is not evidence of overfitting. Because 99.83% of transactions are legitimate, predicting all 0s gives 99.83% accuracy. Fraud recall (73.24%), precision (91.23%), and $F_1$ (0.8125) are the true measures of performance.
2. **Static Global Threshold:** We employ a global threshold ($0.4119$). In production banking, tiered thresholds vary by amount or customer risk tier.
3. **Anonymized Features:** Features `V1`–`V28` cannot be linked to physical customer behaviors, preventing causal policy recommendations.
4. **Historical Benchmark:** Dataset captures European cards across 2 days in 2013; modern fraud vectors (cryptocurrency ramps, MFA intercept) require dynamic online retraining.

---

## Authors & Acknowledgments

- **Developed as an academic demonstration of reproducible, explainable fraud detection.**
- Benchmark dataset provided courtesy of ULB (Université Libre de Bruxelles) Machine Learning Group.
