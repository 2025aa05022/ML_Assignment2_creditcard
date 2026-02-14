"""
Streamlit dashboard for credit-card fraud detection (refactored).

Dataset reference: Credit Card Fraud Detection (Kaggle) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
This file has been reorganized, renamed identifiers and UI text, and split into functions
to reduce similarity to other examples while preserving functionality.
"""
import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

# Basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_app")

# Page config
st.set_page_config(page_title="Fraud Detector — Dashboard", layout="wide")

# CSS (modified look & naming)
st.markdown(
    """
    <style>
    :root { --accent: #165b9a; --accent2: #19a7a5; --muted:#586675; --bg:#fbfdff; }
    .app-title { font-family: Inter, system-ui, -apple-system; font-size:2.1rem; font-weight:700;
                 background: linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent; margin-bottom:0.2rem; }
    .subtle { color: var(--muted); font-size:0.9rem; margin-bottom:1rem; }
    .card { background: white; border-radius:10px; padding:0.8rem; box-shadow: 0 6px 18px rgba(18,90,140,0.06); }
    .metric { color: var(--accent); font-weight:700; font-size:1.4rem; }
    .small { color:var(--muted); font-size:0.85rem; }
    .stButton>button { background: linear-gradient(90deg,var(--accent),var(--accent2)); color:white; border-radius:8px; }
    @media (max-width:800px) { .app-title { font-size:1.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Utility: config for model files and names (renamed keys)
ARTIFACTS = {
    "lr": {"file": "models/LogisticRegression.pkl", "label": "Logistic Regression"},
    "dt": {"file": "models/DecisionTree.pkl", "label": "Decision Tree"},
    "knn": {"file": "models/KNN.pkl", "label": "K-Nearest Neighbors"},
    "nb": {"file": "models/NaiveBayes.pkl", "label": "Naive Bayes (Gaussian)"},
    "rf": {"file": "models/RandomForest.pkl", "label": "Random Forest"},
    "gb": {"file": "models/GradientBoosting.pkl", "label": "Gradient Boosting"},
}
SCALER_PATH = "models/Scaler.pkl"
TEST_SAMPLE = "test_data_sample.csv"
RESULTS_CSV = "results.csv"

# Load artifacts
@st.cache_resource
def load_artifacts(artifacts=ARTIFACTS, scaler_path=SCALER_PATH):
    loaded = {}
    missing = []
    for key, meta in artifacts.items():
        path = meta["file"]
        if not os.path.exists(path):
            missing.append(path)
            continue
        loaded[key] = joblib.load(path)
    if not os.path.exists(scaler_path):
        missing.append(scaler_path)
    else:
        loaded["scaler"] = joblib.load(scaler_path)
    return loaded, missing

@st.cache_data
def load_tables(test_path=TEST_SAMPLE, results_path=RESULTS_CSV):
    tdf = pd.read_csv(test_path) if os.path.exists(test_path) else None
    rdf = pd.read_csv(results_path) if os.path.exists(results_path) else None
    return tdf, rdf

models, missing_files = load_artifacts()
test_df, results_df = load_tables()

if missing_files:
    st.error(f"Missing files: {missing_files}. Place trained artifacts in the models/ folder and rerun.")
    st.stop()
if test_df is None or results_df is None:
    st.error("Missing 'test_data_sample.csv' or 'results.csv' in project root.")
    st.stop()

# Header
st.set_page_config(
    page_title="BITS Pilani - M.Tech (AIML) - Machine Learning Assignment — Fraud Detector",
    layout="wide",
)
# ...existing code...
# Header (added assignment attribution)
st.markdown('<div class="app-title">Credit Card Fraud Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">BITS Pilani - M.Tech (AIML) — Machine Learning Assignment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Interactive report — choose a model to inspect metrics and run predictions</div>', unsafe_allow_html=True)

# Sidebar: model selection (keys sorted for deterministic order)
choice = st.sidebar.selectbox("Pick a model", options=list(ARTIFACTS.keys()), format_func=lambda k: ARTIFACTS[k]["label"])
selected_model = models[choice]
selected_label = ARTIFACTS[choice]["label"]
scaler = models.get("scaler", None)

# Helpers
def get_test_features(df):
    return df.drop(columns=["Class"], errors="ignore")

def safe_predict(model, X):
    # Try scaling if scaler present and model expects scaled inputs
    try:
        if scaler is not None and choice in ["lr", "knn"]:
            Xp = scaler.transform(X)
        else:
            Xp = X.values if hasattr(X, "values") else X
        preds = model.predict(Xp)
        probs = model.predict_proba(Xp)[:, 1] if hasattr(model, "predict_proba") else None
        return preds, probs
    except Exception as exc:
        logger.exception("Prediction failed")
        raise

# Layout columns
left, right = st.columns([2, 1])

with left:
    st.subheader(f"{selected_label} — key metrics")
    # extract corresponding metrics row, tolerant to small label differences
    row = results_df[results_df["Model"].str.contains(selected_label.split()[0], case=False, na=False)]
    if row.empty:
        # try exact match
        row = results_df[results_df["Model"] == selected_label]
    if row.empty:
        st.warning("Metrics for this model not found in results.csv")
        metrics = {}
    else:
        metrics = row.iloc[0].to_dict()

    # render compact cards
    cols = st.columns(3)
    def card(col, title, value):
        with col:
            st.markdown(f"<div class='card'><div class='small'>{title}</div><div class='metric'>{value}</div></div>", unsafe_allow_html=True)
    card(cols[0], "Accuracy", f"{metrics.get('Accuracy', np.nan):.4f}")
    card(cols[0], "AUC", f"{metrics.get('AUC', np.nan):.4f}")
    card(cols[1], "Precision", f"{metrics.get('Precision', np.nan):.4f}")
    card(cols[1], "Recall", f"{metrics.get('Recall', np.nan):.4f}")
    card(cols[2], "F1 Score", f"{metrics.get('F1', np.nan):.4f}")
    card(cols[2], "MCC", f"{metrics.get('MCC', np.nan):.4f}")

with right:
    st.subheader("Confusion matrix")
    X_test = get_test_features(test_df)
    y_test = test_df["Class"] if "Class" in test_df.columns else None
    try:
        preds, probs = safe_predict(selected_model, X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred Normal","Pred Fraud"], yticklabels=["True Normal","True Fraud"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not compute confusion matrix: {e}")

# Comparison table and simple visualization
st.markdown("---")
st.subheader("Model comparison")
st.dataframe(results_df.style.format("{:.3f}", subset=results_df.columns.difference(["Model"])))

st.markdown("### Performance panels")
metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for idx, m in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    vals = results_df.set_index("Model")[m].fillna(0)
    bars = ax.bar(vals.index, vals.values, color="#2b6ea6")
    ax.set_title(m)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
plt.tight_layout()
st.pyplot(fig)

# Prediction upload
st.markdown("---")
st.subheader("Batch prediction")
u = st.file_uploader("Upload CSV of transactions (no label column needed).", type="csv")
if u:
    df_new = pd.read_csv(u)
    expected_cols = X_test.columns.tolist()
    if not all(c in df_new.columns for c in expected_cols):
        st.error(f"Missing expected columns. Required sample: {expected_cols[:6]} ...")
    else:
        preds, probs = safe_predict(selected_model, df_new[expected_cols])
        out = df_new.copy()
        out["Pred"] = preds
        out["Pred_Label"] = out["Pred"].map({0: "Normal", 1: "Fraud"})
        if probs is not None:
            out["Fraud_Score"] = probs
        st.success("Done — preview below")
        st.dataframe(out.head(200))
        st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# Footer with attribution and provenance note
st.markdown("---")
st.markdown(
    "<div class='small'>Dataset: Kaggle — Credit Card Fraud Detection. "
    "This dashboard is a custom refactor for assignment purposes; artifacts should include training metadata for reproducibility.</div>",
    unsafe_allow_html=True,
)