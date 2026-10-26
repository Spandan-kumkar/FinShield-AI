"""
FinShield AI — Streamlit Dashboard
=====================================
Interactive dashboard for fraud detection analysis.

Sections
--------
A. Overview — dataset stats, model info, genuine test metrics, visualisations
B. Single Transaction Analysis — example-driven predict + SHAP explanation
C. Batch Transaction Analysis — CSV upload + results download
D. Model Performance — confusion matrix, ROC, PR curve, threshold analysis
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATH,
    EVALUATION_RESULTS_PATH,
    FEATURE_COLUMNS,
    LABEL_MAP,
    MODEL_BUNDLE_PATH,
    MODEL_VERSION,
    TARGET_COL,
    VALIDATION_RESULTS_PATH,
    THRESHOLD_CURVE_PATH,
)
from src.predict import load_model_bundle, predict_batch, predict_single
from src.explain import explain_single, get_shap_explanation_dataframe


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinShield AI — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ───────────────────────────────────────────────────

st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9;
        font-weight: 700;
    }
    /* Section headers */
    h1 { color: #f8fafc; }
    h2 { color: #e2e8f0; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; }
    h3 { color: #cbd5e1; }
    /* Tables */
    .stTable { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Guard: required artefacts ────────────────────────────────────────

def _check_artefacts():
    missing = []
    if not MODEL_BUNDLE_PATH.exists():
        missing.append(f"Model bundle: {MODEL_BUNDLE_PATH}")
    if not EVALUATION_RESULTS_PATH.exists():
        missing.append(f"Evaluation results: {EVALUATION_RESULTS_PATH}")
    if missing:
        st.error(
            "**Required artefacts are missing.** "
            "Run the training pipeline first:\n\n"
            "```bash\npython -m src.train\n```\n\n"
            "Missing:\n" + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()

_check_artefacts()


# ── Load artefacts (cached) ─────────────────────────────────────────

@st.cache_resource
def _load_bundle():
    return load_model_bundle()

@st.cache_data
def _load_eval():
    with open(EVALUATION_RESULTS_PATH) as f:
        return json.load(f)

@st.cache_data
def _load_validation_comparison():
    if VALIDATION_RESULTS_PATH.exists():
        with open(VALIDATION_RESULTS_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def _load_threshold_analysis():
    if THRESHOLD_CURVE_PATH.exists():
        with open(THRESHOLD_CURVE_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def _load_sample_transactions():
    """Load real fraud and legitimate examples from the dataset or fallback sample file."""
    df = None
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
    else:
        sample_path = PROJECT_ROOT / "data" / "sample_transactions.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
    if df is not None and "Class" in df.columns:
        fraud_samples = df[df["Class"] == 1].head(5)
        legit_pool = df[df["Class"] == 0]
        n_legit = min(5, len(legit_pool))
        legit_samples = legit_pool.sample(n_legit, random_state=42) if n_legit > 0 else legit_pool.head(5)
        return fraud_samples, legit_samples
    return None, None


bundle = _load_bundle()
eval_metrics = _load_eval()
val_comparison = _load_validation_comparison()
threshold_analysis = _load_threshold_analysis()
fraud_samples, legit_samples = _load_sample_transactions()


# ── Plot helpers ─────────────────────────────────────────────────────

COLORS = {
    "blue": "#3b82f6",
    "red": "#ef4444",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "purple": "#a855f7",
    "slate": "#64748b",
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "text": "#e2e8f0",
    "grid": "#334155",
}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS["bg_dark"])
    ax.figure.set_facecolor(COLORS["bg_dark"])
    ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=COLORS["text"], fontsize=10)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["slate"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.spines["left"].set_color(COLORS["grid"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4, alpha=0.5)


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("## 🛡️ FinShield AI")
st.sidebar.caption("Explainable Credit Card Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Performance"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** `{bundle.get('model_version', 'N/A')}`  \n"
    f"**Threshold:** `{bundle['threshold']:.4f}`  \n"
    f"**Experiment:** `{bundle.get('experiment', 'N/A')}`"
)
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This is a **research prototype**.  Model predictions are not "
    "definitive determinations of fraud."
)


# ════════════════════════════════════════════════════════════════════
# A. Overview
# ════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("🛡️ FinShield AI — Project Overview")
    st.markdown(
        "An explainable machine-learning system that identifies potentially "
        "fraudulent credit card transactions using **XGBoost** and **SHAP**."
    )

    # ── Dataset stats row ──
    st.header("📋 Dataset Summary")
    ds_report = bundle.get("dataset_report", {})
    dedup_audit = bundle.get("dedup_audit", {})
    cc = ds_report.get("class_counts", {})
    total = ds_report.get("shape", (0,))[0]
    n_legit = cc.get(0, 0)
    n_fraud = cc.get(1, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Transactions", f"{total:,}")
    c2.metric("Raw Legitimate", f"{n_legit:,}")
    c3.metric("Raw Fraudulent", f"{n_fraud:,}")
    c4.metric("Raw Prevalence", f"{ds_report.get('fraud_prevalence_pct', 0):.4f}%")

    if dedup_audit:
        st.info(
            f"🛡️ **Zero-Leakage Data Preparation:** Removed **{dedup_audit['removed_duplicates']:,} duplicate rows** "
            f"({dedup_audit['removed_legitimate']:,} legitimate, {dedup_audit['removed_fraud']} fraud) to guarantee "
            f"**0 cross-partition record leakage** across train, validation, and test splits.  \n"
            f"**Clean Modeled Dataset:** {dedup_audit['final_total']:,} unique transactions "
            f"({dedup_audit['final_legitimate']:,} legitimate, {dedup_audit['final_fraud']} fraud — "
            f"{dedup_audit['final_fraud_prevalence_pct']:.4f}% prevalence)."
        )

    # ── Class distribution chart ──
    col_chart, col_explain = st.columns([1, 1])

    with col_chart:
        st.subheader("Class Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

        # Pie chart
        ax1.pie(
            [n_legit, n_fraud],
            labels=["Legitimate", "Fraud"],
            colors=[COLORS["green"], COLORS["red"]],
            autopct="%1.3f%%",
            startangle=90,
            textprops={"color": COLORS["text"], "fontsize": 10},
            wedgeprops={"edgecolor": COLORS["bg_dark"], "linewidth": 2},
        )
        ax1.set_title("Class Proportions", color=COLORS["text"], fontsize=11, fontweight="bold")
        ax1.set_facecolor(COLORS["bg_dark"])

        # Bar chart (log scale to make fraud visible)
        bars = ax2.bar(
            ["Legitimate", "Fraud"],
            [n_legit, n_fraud],
            color=[COLORS["green"], COLORS["red"]],
            edgecolor=COLORS["bg_dark"],
            width=0.5,
        )
        ax2.set_yscale("log")
        for bar, val in zip(bars, [n_legit, n_fraud]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{val:,}", ha="center", va="bottom",
                color=COLORS["text"], fontsize=10, fontweight="bold",
            )
        _style_ax(ax2, title="Transaction Counts (log scale)", ylabel="Count")
        fig.set_facecolor(COLORS["bg_dark"])
        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

    with col_explain:
        st.subheader("Why Class Imbalance Matters")
        st.markdown(
            """
            The dataset is **extremely imbalanced** — only **0.17%** of
            transactions are fraudulent.

            This means:
            - A model predicting **everything as legitimate** gets
              **99.83% accuracy** — but catches **zero fraud**.
            - **Accuracy is misleading** for this problem.
            - We use **fraud-specific metrics** instead:
              **Recall** (how many frauds are caught),
              **Precision** (how many alerts are real fraud),
              **F1** (balance of both), and **Average Precision**.
            """
        )

    # ── Test metrics row ──
    st.header("🎯 Held-Out Test Set Performance")
    st.caption(
        "These metrics are from the **test set** (15% of data), which was "
        "**never** used for model selection or threshold tuning."
    )

    m = eval_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Recall", f"{m['fraud_recall']:.2%}")
    c2.metric("Fraud Precision", f"{m['fraud_precision']:.2%}")
    c3.metric("Fraud F1-Score", f"{m['fraud_f1']:.2%}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Average Precision (AP)", f"{m['average_precision']:.4f}")
    c6.metric("PR-AUC (Trapezoidal)", f"{m.get('pr_auc', m['average_precision']):.4f}")
    c7.metric("Balanced Accuracy", f"{m['balanced_accuracy']:.2%}")
    c8.metric("Specificity", f"{m['specificity']:.4f}")

    # ── Metrics bar chart ──
    st.subheader("Key Metrics Comparison")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    metric_names = ["Recall", "Precision", "F1", "Avg Precision (AP)", "PR-AUC", "ROC-AUC", "Balanced Acc"]
    metric_vals = [
        m["fraud_recall"], m["fraud_precision"], m["fraud_f1"],
        m["average_precision"], m.get("pr_auc", m["average_precision"]),
        m["roc_auc"], m["balanced_accuracy"],
    ]
    bar_colors = [COLORS["red"], COLORS["blue"], COLORS["purple"],
                  COLORS["amber"], COLORS["green"], COLORS["slate"], "#06b6d4"]
    bars = ax.barh(metric_names[::-1], metric_vals[::-1],
                   color=bar_colors[::-1], height=0.55, edgecolor=COLORS["bg_dark"])
    for bar, val in zip(bars, metric_vals[::-1]):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color=COLORS["text"], fontsize=10)
    ax.set_xlim(0, 1.12)
    _style_ax(ax, title="Fraud Detection Metrics (Test Set)", xlabel="Score")
    ax.grid(axis="y", visible=False)
    fig.set_facecolor(COLORS["bg_dark"])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Confusion matrix summary ──
    cm = m["confusion_matrix"]
    st.subheader("Confusion Matrix Summary")
    col_cm, col_text = st.columns([1, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        cm_array = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        im = ax.imshow(cm_array, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"], color=COLORS["text"], fontsize=9)
        ax.set_yticklabels(["Actual Legit", "Actual Fraud"], color=COLORS["text"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_array[i,j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm_array[i,j] > cm_array.max()/2 else "black")
        ax.set_title("Test Set Confusion Matrix", color=COLORS["text"], fontsize=11, fontweight="bold")
        ax.set_facecolor(COLORS["bg_dark"])
        fig.set_facecolor(COLORS["bg_dark"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_text:
        st.markdown(
            f"""
            | Outcome | Count |
            |---|---|
            | ✅ Fraud correctly detected (TP) | **{cm['TP']}** / {m['total_fraud']} |
            | ❌ Fraud missed (FN) | **{cm['FN']}** |
            | ⚠️ Legitimate incorrectly flagged (FP) | **{cm['FP']}** |
            | ✅ Legitimate correctly identified (TN) | **{cm['TN']:,}** |

            **Detection rate:** {m['fraud_recall']:.1%} of fraud caught  
            **False alarm rate:** {m['false_positive_rate']:.5%} of legitimate flagged
            """
        )


# ════════════════════════════════════════════════════════════════════
# B. Single Transaction Analysis
# ════════════════════════════════════════════════════════════════════

elif page == "🔍 Single Transaction":
    st.title("🔍 Single Transaction Analysis")
    st.markdown(
        "Select a **pre-loaded example** from real transactions, or enter "
        "custom values.  The model predicts fraud probability and SHAP "
        "explains which features contributed most."
    )
    st.warning(
        "⚠️ Model predictions are **not definitive determinations of fraud** — "
        "they indicate the model's assessment based on learned patterns."
    )

    # ── Example selector ──
    st.subheader("📌 Quick Examples")
    st.caption(
        "Select a real transaction from the dataset to see the model in action. "
        "No need to type 30 features manually!"
    )

    example_options = ["— Select an example —"]
    example_data = {}

    if fraud_samples is not None and legit_samples is not None:
        for i, (_, row) in enumerate(fraud_samples.iterrows()):
            key = f"🔴 Fraud example {i+1} (Amount: ${row['Amount']:.2f})"
            example_options.append(key)
            example_data[key] = row
        for i, (_, row) in enumerate(legit_samples.iterrows()):
            key = f"🟢 Legitimate example {i+1} (Amount: ${row['Amount']:.2f})"
            example_options.append(key)
            example_data[key] = row

    selected_example = st.selectbox("Choose a transaction example", example_options)

    # Set default values from selected example
    if selected_example != "— Select an example —" and selected_example in example_data:
        row = example_data[selected_example]
        default_values = {col: float(row[col]) for col in FEATURE_COLUMNS}
        actual_label = "Fraud" if int(row.get("Class", -1)) == 1 else "Legitimate"
        st.info(f"📄 Loaded: **{selected_example}** — Actual label: **{actual_label}**")
    else:
        default_values = {col: 0.0 for col in FEATURE_COLUMNS}
        default_values["Amount"] = 100.0
        actual_label = None

    # ── Editable form (collapsed by default to reduce clutter) ──
    values = dict(default_values)
    with st.expander("🔧 Inspect / Edit transaction features (optional)", expanded=False):
        st.caption("Values are pre-filled from your selection above. You can customize them if desired.")
        cols = st.columns(5)
        for i, feat in enumerate(FEATURE_COLUMNS):
            values[feat] = cols[i % 5].number_input(
                feat,
                value=float(default_values[feat]),
                format="%.6f",
                key=f"input_{feat}_{selected_example}",
            )

    # ── Predict button ──
    if st.button("🔍 Analyze Transaction", type="primary", width="stretch"):
        try:
            result = predict_single(values, bundle=bundle)

            # ── Result display ──
            st.markdown("---")
            st.subheader("Prediction Result")

            is_fraud = result["predicted_class"] == 1
            if is_fraud:
                st.error(f"🔴 **FRAUD DETECTED** — Probability: {result['fraud_probability']:.4%}")
            else:
                st.success(f"🟢 **LEGITIMATE** — Fraud probability: {result['fraud_probability']:.4%}")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Prediction", result["predicted_label"])
            rc2.metric("Fraud Probability", f"{result['fraud_probability']:.6f}")
            rc3.metric("Threshold", f"{result['threshold']:.4f}")

            if actual_label:
                correct = (actual_label == result["predicted_label"])
                if correct:
                    st.success(f"✅ Model correctly predicted: **{actual_label}**")
                else:
                    st.warning(
                        f"❌ Model predicted **{result['predicted_label']}** but "
                        f"actual label was **{actual_label}**"
                    )

            if result["warnings"]:
                st.warning(" | ".join(result["warnings"]))

            # ── SHAP explanation ──
            st.markdown("---")
            st.subheader("🧠 SHAP Feature Contributions")
            st.caption(
                "SHAP values show **how each feature influenced this specific prediction**. "
                "🔴 Red bars push toward Fraud; 🟢 Green bars push toward Legitimate. "
                "SHAP explains model behaviour — it does not establish causation. "
                "V1–V28 are anonymized PCA features with no known real-world interpretation."
            )

            with st.spinner("Computing SHAP values…"):
                explanation = explain_single(values, bundle=bundle, top_n=15)

            shap_df = get_shap_explanation_dataframe(explanation)
            top = shap_df.head(15)

            # ── Waterfall-style bar chart ──
            fig, ax = plt.subplots(figsize=(9, 5))
            colours = [COLORS["red"] if v > 0 else COLORS["green"] for v in top["shap_value"]]
            bars = ax.barh(
                top["feature"][::-1],
                top["shap_value"][::-1],
                color=colours[::-1],
                edgecolor=COLORS["bg_dark"],
                height=0.6,
            )
            # Add value labels
            for bar, val in zip(bars, top["shap_value"][::-1]):
                x_pos = bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01)
                ha = "left" if bar.get_width() >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                        va="center", ha=ha, color=COLORS["text"], fontsize=9)

            ax.axvline(0, color=COLORS["slate"], linewidth=0.8, linestyle="--")
            _style_ax(ax, title="Top Feature Contributions (SHAP Values)",
                      xlabel="← Toward Legitimate    |    Toward Fraud →")
            ax.grid(axis="y", visible=False)
            fig.set_facecolor(COLORS["bg_dark"])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Contribution table ──
            with st.expander("📋 Full SHAP value table"):
                display_df = shap_df[["feature", "shap_value", "feature_value", "direction"]].copy()
                display_df.columns = ["Feature", "SHAP Value", "Feature Value", "Direction"]
                st.dataframe(display_df, width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════
# C. Batch Analysis
# ════════════════════════════════════════════════════════════════════

elif page == "📁 Batch Analysis":
    st.title("📁 Batch Transaction Analysis")
    st.markdown(
        "Upload a CSV containing multiple transactions for bulk fraud screening."
    )

    st.info(
        f"**Required columns:** {', '.join(FEATURE_COLUMNS[:3])}, …, "
        f"{FEATURE_COLUMNS[-1]} ({len(FEATURE_COLUMNS)} features total).  \n"
        f"If the CSV contains a `Class` column, it will be **ignored** — "
        f"it is never passed to the model."
    )

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("⚡ Load Demo Batch", help="Load pre-configured 30-transaction demo batch (sample_batch.csv)", width="stretch")

    df_to_analyze = None
    if uploaded is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded)
            st.markdown(f"**Uploaded:** {len(df_to_analyze):,} rows × {df_to_analyze.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        sample_path = PROJECT_ROOT / "sample_batch.csv"
        if sample_path.exists():
            df_to_analyze = pd.read_csv(sample_path)
            st.info("Loaded pre-configured 30-transaction demo batch (`sample_batch.csv`).")

    if df_to_analyze is not None:
        try:
            df = df_to_analyze
            results = predict_batch(df, bundle=bundle)

            n_fraud = int((results["predicted_class"] == 1).sum())
            n_legit = int((results["predicted_class"] == 0).sum())

            # ── Summary metrics ──
            st.subheader("Screening Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", f"{len(results):,}")
            c2.metric("🔴 Flagged as Potentially Fraudulent", f"{n_fraud:,}")
            c3.metric("🟢 Classified as Legitimate", f"{n_legit:,}")

            st.caption(
                "⚠️ 'Flagged as potentially fraudulent' reflects **model predictions**, "
                "not confirmed real-world fraud."
            )

            # ── Distribution chart ──
            if n_fraud > 0:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(
                    ["Legitimate", "Flagged Fraud"],
                    [n_legit, n_fraud],
                    color=[COLORS["green"], COLORS["red"]],
                    edgecolor=COLORS["bg_dark"],
                    width=0.4,
                )
                for i, (label, val) in enumerate(zip(["Legit", "Fraud"], [n_legit, n_fraud])):
                    ax.text(i, val + max(n_legit, n_fraud)*0.02, f"{val:,}",
                            ha="center", color=COLORS["text"], fontsize=11, fontweight="bold")
                _style_ax(ax, title="Prediction Distribution")
                ax.grid(axis="y", visible=False)
                fig.set_facecolor(COLORS["bg_dark"])
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # ── Fraud probability histogram ──
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(
                results["fraud_probability"],
                bins=50,
                color=COLORS["blue"],
                edgecolor=COLORS["bg_dark"],
                alpha=0.85,
            )
            ax.axvline(bundle["threshold"], color=COLORS["red"], linestyle="--",
                       linewidth=2, label=f"Threshold ({bundle['threshold']:.4f})")
            _style_ax(ax, title="Distribution of Fraud Probabilities",
                      xlabel="Fraud Probability", ylabel="Count")
            ax.legend(facecolor=COLORS["bg_card"], edgecolor=COLORS["grid"],
                      labelcolor=COLORS["text"])
            fig.set_facecolor(COLORS["bg_dark"])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Results table ──
            st.subheader("Detailed Results")
            # Highlight fraud rows
            st.dataframe(
                results.style.apply(
                    lambda row: [
                        "background-color: rgba(239, 68, 68, 0.15)"
                        if row["predicted_class"] == 1 else ""
                    ] * len(row),
                    axis=1,
                ),
                width="stretch",
                hide_index=True,
            )

            csv_out = results.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv_out,
                file_name="finshield_predictions.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ════════════════════════════════════════════════════════════════════
# D. Model Performance
# ════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")
    st.caption(
        "All metrics below are from the **held-out test set** (15% of data) "
        "and were **NOT** used for model or threshold selection."
    )

    m = eval_metrics

    # ── Confusion matrix heatmap ──
    st.subheader("Confusion Matrix")
    col_cm, col_metrics = st.columns([1, 1])

    cm = m["confusion_matrix"]
    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_array = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        cmap = plt.cm.Blues
        im = ax.imshow(cm_array, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nLegitimate", "Predicted\nFraud"],
                           color=COLORS["text"], fontsize=10)
        ax.set_yticklabels(["Actual\nLegitimate", "Actual\nFraud"],
                           color=COLORS["text"], fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_array[i,j]:,}", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm_array[i,j] > cm_array.max()/2 else "black")
        ax.set_title("Confusion Matrix", color=COLORS["text"], fontsize=12, fontweight="bold")
        ax.set_facecolor(COLORS["bg_dark"])
        fig.set_facecolor(COLORS["bg_dark"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_metrics:
        st.markdown("#### Detection Summary")
        st.markdown(
            f"""
            | Outcome | Count | Rate |
            |---|---|---|
            | ✅ **Fraud detected** (TP) | {cm['TP']} / {m['total_fraud']} | {m['fraud_recall']:.1%} |
            | ❌ **Fraud missed** (FN) | {cm['FN']} | {m['false_negative_rate']:.1%} |
            | ⚠️ **False alarms** (FP) | {cm['FP']} | {m['false_positive_rate']:.5%} |
            | ✅ **Correct rejections** (TN) | {cm['TN']:,} | {m['specificity']:.4%} |
            """
        )
        st.markdown(
            f"**Threshold:** {m['threshold_used']:.4f}  \n"
            f"**Total test samples:** {m['total_samples']:,}"
        )

    # ── Metrics radar / bar chart ──
    st.subheader("Classification Metrics Overview")

    metric_names = [
        "Fraud\nRecall", "Fraud\nPrecision", "Fraud\nF1",
        "Avg\nPrecision", "ROC-AUC", "Balanced\nAccuracy",
    ]
    metric_vals = [
        m["fraud_recall"], m["fraud_precision"], m["fraud_f1"],
        m["average_precision"], m["roc_auc"], m["balanced_accuracy"],
    ]
    bar_colors = [COLORS["red"], COLORS["blue"], COLORS["purple"],
                  COLORS["amber"], COLORS["green"], COLORS["slate"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metric_names))
    bars = ax.bar(x, metric_vals, color=bar_colors, width=0.55,
                  edgecolor=COLORS["bg_dark"], linewidth=1.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom",
                color=COLORS["text"], fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1.15)
    _style_ax(ax, title="Key Classification Metrics (Test Set)")
    fig.set_facecolor(COLORS["bg_dark"])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── ROC + PR curves side by side ──
    st.subheader("ROC & Precision-Recall Curves")

    @st.cache_data
    def _compute_curves():
        from src.data import deduplicate_dataset, load_dataset, separate_features_target, split_data
        from sklearn.metrics import precision_recall_curve, roc_curve, auc

        df = load_dataset()
        df_clean, _ = deduplicate_dataset(df)
        X, y = separate_features_target(df_clean)
        splits = split_data(X, y)
        X_test_pp = bundle["preprocessor"].transform(splits["X_test"])
        y_test = splits["y_test"].values
        y_prob = bundle["model"].predict_proba(X_test_pp)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        return fpr, tpr, roc_auc, prec, rec, pr_auc

    fpr, tpr, roc_auc, prec, rec, pr_auc = _compute_curves()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ROC
    ax1.plot(fpr, tpr, color=COLORS["blue"], lw=2.5, label=f"Model (ROC-AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color=COLORS["slate"], ls="--", lw=1, label="Random")
    ax1.fill_between(fpr, tpr, alpha=0.1, color=COLORS["blue"])
    _style_ax(ax1, title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax1.legend(facecolor=COLORS["bg_card"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["text"], fontsize=9)

    # PR
    ax2.plot(rec, prec, color=COLORS["red"], lw=2.5, label=f"Model (PR-AUC = {pr_auc:.4f})")
    ax2.fill_between(rec, prec, alpha=0.1, color=COLORS["red"])
    _style_ax(ax2, title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax2.legend(facecolor=COLORS["bg_card"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["text"], fontsize=9)

    fig.set_facecolor(COLORS["bg_dark"])
    fig.tight_layout(pad=3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**ROC Curve:** Displays trade-off between true-positive rate and false-positive rate (ROC-AUC).  \n"
        "**PR Curve:** Displays trade-off between precision and recall. PR-AUC is trapezoidal integration "
        "under this curve; Average Precision (AP) is the weighted mean precision across recall thresholds."
    )

    # ── Accuracy discussion ──
    st.subheader("⚠️ Why Accuracy ≈ 99.95% is Misleading")
    st.markdown(
        f"""
        The model's accuracy is **{m['accuracy']:.4f}** ({m['accuracy']:.2%}),
        which *sounds* perfect but is **not a useful metric** here.

        **The reason:** {m['total_legitimate']:,} of {m['total_samples']:,}
        test transactions are legitimate ({m['total_legitimate']/m['total_samples']:.2%}).
        A trivial model that *always* predicts "Legitimate" would achieve
        **{m['total_legitimate']/m['total_samples']:.2%} accuracy** — while catching
        **zero fraud**.

        The model's real value is measured by:
        - **Fraud Recall ({m['fraud_recall']:.1%}):** Catches {cm['TP']} of {m['total_fraud']} frauds
        - **Fraud Precision ({m['fraud_precision']:.1%}):** Only {cm['FP']} false alarms
        - **F1-Score ({m['fraud_f1']:.4f}):** Balanced measure of detection quality

        **This is NOT overfitting.** Validation and test metrics are consistent
        (F1: ~0.84 on both), confirming healthy generalisation.
        """
    )

    # ── Threshold info ──
    if threshold_analysis:
        st.subheader("🎯 Threshold Selection")
        ta = threshold_analysis
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown(
                f"""
                | Setting | Value |
                |---|---|
                | **Selected threshold** | {ta['best_f1_threshold']:.4f} |
                | **Selection criterion** | Maximize F1 on validation |
                | **Val F1 @ selected** | {ta['best_f1']:.4f} |
                | **Val F1 @ default 0.5** | {ta['default_0.5_f1']:.4f} |
                | **Val Precision @ selected** | {ta['best_precision']:.4f} |
                | **Val Recall @ selected** | {ta['best_recall']:.4f} |
                """
            )
        with col_t2:
            st.markdown(
                """
                **How it works:**  
                A lower threshold catches more fraud (higher recall) but
                triggers more false alarms (lower precision).  
                A higher threshold reduces false alarms but misses more fraud.  
                We selected the threshold that **maximises F1** — the harmonic
                mean of precision and recall — on the validation set.
                """
            )

    # ── Experiment comparison ──
    if val_comparison:
        st.subheader("🔬 Model Comparison (Validation Set)")
        st.caption("These metrics are from the validation set, used for model selection only.")

        comp_data = []
        for exp_key, exp_label in [
            ("experiment_A_unweighted", "A: Unweighted"),
            ("experiment_B_weighted", "B: Class-Weighted"),
        ]:
            if exp_key in val_comparison:
                vm = val_comparison[exp_key]["validation_metrics"]
                comp_data.append({
                    "Experiment": exp_label,
                    "F1": f"{vm['fraud_f1']:.4f}",
                    "Recall": f"{vm['fraud_recall']:.4f}",
                    "Precision": f"{vm['fraud_precision']:.4f}",
                    "Avg Precision": f"{vm['average_precision']:.4f}",
                    "ROC-AUC": f"{vm['roc_auc']:.4f}",
                })

        if comp_data:
            st.table(pd.DataFrame(comp_data))

            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(9, 3.5))
            metrics_to_compare = ["fraud_f1", "fraud_recall", "fraud_precision", "average_precision"]
            labels = ["F1", "Recall", "Precision", "Avg Precision"]
            x = np.arange(len(labels))
            width = 0.3

            for i, (exp_key, color, label) in enumerate([
                ("experiment_A_unweighted", COLORS["blue"], "A: Unweighted"),
                ("experiment_B_weighted", COLORS["amber"], "B: Weighted"),
            ]):
                if exp_key in val_comparison:
                    vm = val_comparison[exp_key]["validation_metrics"]
                    vals = [vm[m] for m in metrics_to_compare]
                    ax.bar(x + i*width, vals, width, color=color, label=label,
                           edgecolor=COLORS["bg_dark"])

            ax.set_xticks(x + width/2)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0.6, 1.05)
            _style_ax(ax, title="Experiment Comparison (Validation Set)")
            ax.legend(facecolor=COLORS["bg_card"], edgecolor=COLORS["grid"],
                      labelcolor=COLORS["text"], fontsize=9)
            fig.set_facecolor(COLORS["bg_dark"])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            f"**Selected:** {val_comparison.get('selected_experiment', 'N/A')}  \n"
            f"**Criterion:** {val_comparison.get('selection_criterion', 'N/A')}"
        )
