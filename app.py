
import os
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Monitoring",
    page_icon="💳",
    layout="wide"
)

RANDOM_STATE = 42
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_FILE = "creditcard.csv"
SPLIT_TIME = 24 * 3600
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]
DRIFT_COLS = ["Amount", "V1", "V2", "V3", "V4"]

st.title("💳 Financial Fraud Detection & Model Monitoring")
st.caption("ML_79 | Team G58 | Vanshika Jain & Tanish Pinjani")

st.markdown("""
This dashboard demonstrates credit card fraud detection,
data drift, concept drift and false-positive monitoring.
""")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    data = pd.read_csv(DATA_FILE)
    data = data.drop_duplicates().reset_index(drop=True)
    return data, "Real credit card dataset"


@st.cache_data(show_spinner=False)
def make_synthetic_data(n_rows=100000):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_rows,
        n_features=28,
        n_informative=12,
        n_redundant=4,
        weights=[0.998],
        flip_y=0,
        class_sep=2.0,
        random_state=RANDOM_STATE
    )

    data = pd.DataFrame(
        X, columns=[f"V{i}" for i in range(1, 29)]
    )

    rng = np.random.default_rng(RANDOM_STATE)
    data.insert(
        0, "Time",
        np.sort(rng.uniform(0, 172792, n_rows)).round()
    )
    data["Amount"] = np.round(
        rng.lognormal(3.0, 1.3, n_rows), 2
    )
    data["Class"] = y

    return data


@st.cache_resource(show_spinner="Loading dataset and training Random Forest...")
def prepare_project():
    try:
        df, source = load_data()
    except Exception as error:
        st.warning(
            f"Real dataset could not be downloaded. "
            f"Synthetic fallback is being used. Details: {error}"
        )
        df = make_synthetic_data()
        source = "Synthetic fallback dataset"

    # Feature engineering
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["Period"] = np.where(df["Hour"] < 7, "Night", "Day")

    df["Amount_band"] = pd.cut(
        df["Amount"],
        bins=[-np.inf, 20, 100, np.inf],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    train_rows = df["Time"] < SPLIT_TIME

    if train_rows.sum() < 100 or (~train_rows).sum() < 100:
        raise ValueError("Not enough transactions in both time periods.")

    amount_scaler = StandardScaler().fit(
        df.loc[train_rows, ["Amount"]]
    )
    time_scaler = StandardScaler().fit(
        df.loc[train_rows, ["Time"]]
    )

    df["Amount_scaled"] = amount_scaler.transform(
        df[["Amount"]]
    ).ravel()

    df["Time_scaled"] = time_scaler.transform(
        df[["Time"]]
    ).ravel()

    train_df = df[df["Time"] < SPLIT_TIME].copy()
    live_df = df[df["Time"] >= SPLIT_TIME].copy()

    fit_df, val_df = train_test_split(
        train_df,
        test_size=0.25,
        stratify=train_df["Class"],
        random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=2,
        random_state=RANDOM_STATE
    )

    model.fit(fit_df[FEATURES], fit_df["Class"])

    val_pred = model.predict(val_df[FEATURES])

    baseline = {
        "Precision": precision_score(
            val_df["Class"], val_pred, zero_division=0
        ),
        "Recall": recall_score(
            val_df["Class"], val_pred, zero_division=0
        ),
        "F1": f1_score(
            val_df["Class"], val_pred, zero_division=0
        )
    }

    # Create six incoming monitoring windows
    live = live_df.copy()

    live["Window"] = (
        ((live["Time"] - SPLIT_TIME) // (4 * 3600)).astype(int) + 1
    ).clip(upper=6)

    live["Injected"] = "none"
    rng = np.random.default_rng(RANDOM_STATE)

    # Simulated data drift in window 4
    w4 = live["Window"] == 4
    live.loc[w4, "Amount"] *= 1.5

    for col in ["V1", "V2", "V3", "V4"]:
        live.loc[w4, col] += 0.5 * train_df[col].std()

    live.loc[w4, "Injected"] = "data_drift"

    # Simulated concept drift in windows 5 and 6
    v14_cut = train_df.loc[
        train_df["Class"] == 0, "V14"
    ].quantile(0.997)

    new_fraud = (
        live["Window"].isin([5, 6])
        & (live["Class"] == 0)
        & (live["V14"] > v14_cut)
    )

    live.loc[new_fraud, "Class"] = 1
    live.loc[new_fraud, "Injected"] = "concept_drift"

    # Simulated false-positive spike in window 6
    n_extra = min(150, int(train_df["Class"].sum()))

    if n_extra > 0:
        extra = train_df[
            train_df["Class"] == 1
        ].sample(
            n_extra,
            replace=True,
            random_state=RANDOM_STATE
        ).copy()

        extra[FEATURES[:28]] += rng.normal(
            0, 0.3, size=(n_extra, 28)
        )

        extra["Amount"] = rng.uniform(
            600, 2500, n_extra
        ).round(2)

        extra["Time"] = rng.uniform(
            SPLIT_TIME + 20 * 3600,
            SPLIT_TIME + 24 * 3600 - 1,
            n_extra
        ).round()

        extra["Class"] = 0
        extra["Window"] = 6
        extra["Injected"] = "fp_spike"

        live = pd.concat(
            [live, extra[live.columns]],
            ignore_index=True
        )

    # Recalculate features after simulated changes
    live["Hour"] = ((live["Time"] // 3600) % 24).astype(int)
    live["Period"] = np.where(
        live["Hour"] < 7, "Night", "Day"
    )

    live["Amount_band"] = pd.cut(
        live["Amount"],
        bins=[-np.inf, 20, 100, np.inf],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    live["Amount_scaled"] = amount_scaler.transform(
        live[["Amount"]]
    ).ravel()

    live["Time_scaled"] = time_scaler.transform(
        live[["Time"]]
    ).ravel()

    live["Pred"] = model.predict(live[FEATURES])

    # K-Means clusters for slice monitoring
    kmeans = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=RANDOM_STATE
    )

    kmeans.fit(train_df[FEATURES])
    live["Cluster"] = kmeans.predict(live[FEATURES])

    return df, train_df, live, model, baseline, source


# --------------------------------------------------
# PSI FUNCTION
# --------------------------------------------------

def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    edges = np.unique(
        np.quantile(expected, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_pct = np.histogram(expected, edges)[0] / len(expected)
    actual_pct = np.histogram(actual, edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )


# --------------------------------------------------
# RUN PROJECT
# --------------------------------------------------

try:
    df, train_df, live, model, baseline, source = prepare_project()

    st.success("Model trained and monitoring data prepared successfully.")

except Exception as error:
    st.error(f"Project could not be initialized: {error}")
    st.stop()


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
c3.metric("Baseline F1", f"{baseline['F1']:.3f}")
c4.metric("Model", "Random Forest (200)")

st.caption(f"Dataset source: {source}")

st.divider()

# --------------------------------------------------
# MONITORING SUMMARY
# --------------------------------------------------

st.subheader("🚨 Monitoring Window Analysis")

monitor_rows = []

for w in range(1, 7):
    part = live[live["Window"] == w]

    tn, fp, fn, tp = confusion_matrix(
        part["Class"], part["Pred"], labels=[0, 1]
    ).ravel()

    precision = precision_score(
        part["Class"], part["Pred"], zero_division=0
    )
    recall = recall_score(
        part["Class"], part["Pred"], zero_division=0
    )
    f1 = f1_score(
        part["Class"], part["Pred"], zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0

    monitor_rows.append({
        "Window": w,
        "Transactions": len(part),
        "Fraud cases": int(part["Class"].sum()),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "FPR": fpr
    })

monitor = pd.DataFrame(monitor_rows).set_index("Window")

st.dataframe(
    monitor.style.format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}",
        "FPR": "{:.3%}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# DATA DRIFT
# --------------------------------------------------

st.subheader("📉 Data Drift Detection")

psi_scores = {}
ks_scores = {}

for w in range(1, 7):
    part = live[live["Window"] == w]

    psi_scores[w] = {}
    ks_scores[w] = {}

    for col in DRIFT_COLS:
        psi_scores[w][col] = calculate_psi(
            train_df[col], part[col]
        )

        ks_scores[w][col] = ks_2samp(
            train_df[col], part[col]
        ).pvalue

psi_table = pd.DataFrame(psi_scores).T
psi_table.index.name = "Window"

drift_flags = psi_table.max(axis=1) > 0.25

left, right = st.columns(2)

with left:
    st.markdown("**Maximum PSI by Window**")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(psi_table.index, psi_table.max(axis=1))
    ax.axhline(0.25, color="red", linestyle="--", label="PSI threshold")
    ax.set_xlabel("Monitoring Window")
    ax.set_ylabel("Maximum PSI")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**PSI Feature Heatmap**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        psi_table.T,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# PERFORMANCE MONITORING
# --------------------------------------------------

st.subheader("📈 Model Performance Monitoring")

fig, ax = plt.subplots(figsize=(10, 4))

for metric in ["Precision", "Recall", "F1-score"]:
    ax.plot(
        monitor.index,
        monitor[metric],
        marker="o",
        label=metric
    )

ax.axhline(
    baseline["Recall"],
    linestyle="--",
    label="Baseline Recall"
)

ax.set_xlabel("Monitoring Window")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# FALSE POSITIVE MONITORING
# --------------------------------------------------

st.subheader("🔥 False Positive Rate by Transaction Slice")

slices = []

for w in range(1, 7):
    part = live[live["Window"] == w]

    for band in ["Low", "Medium", "High"]:
        legit = part[
            (part["Amount_band"] == band)
            & (part["Class"] == 0)
        ]

        fpr = (
            legit["Pred"].sum() / len(legit)
            if len(legit) else 0
        )

        slices.append({
            "Slice": f"Amount_{band}",
            "Window": w,
            "FPR (%)": fpr * 100
        })

    for period in ["Day", "Night"]:
        legit = part[
            (part["Period"] == period)
            & (part["Class"] == 0)
        ]

        fpr = (
            legit["Pred"].sum() / len(legit)
            if len(legit) else 0
        )

        slices.append({
            "Slice": f"Period_{period}",
            "Window": w,
            "FPR (%)": fpr * 100
        })

    for cluster in range(4):
        legit = part[
            (part["Cluster"] == cluster)
            & (part["Class"] == 0)
        ]

        fpr = (
            legit["Pred"].sum() / len(legit)
            if len(legit) else 0
        )

        slices.append({
            "Slice": f"Cluster_{cluster}",
            "Window": w,
            "FPR (%)": fpr * 100
        })

fpr_table = pd.DataFrame(slices).pivot(
    index="Slice",
    columns="Window",
    values="FPR (%)"
)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(
    fpr_table,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax
)
ax.set_title("False Positive Rate (%)")
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# ALERTS
# --------------------------------------------------

st.subheader("⚠️ Automated Monitoring Alerts")

for w in range(1, 7):
    messages = []

    if drift_flags.loc[w]:
        messages.append("Data drift detected")

    if (
        baseline["Recall"] - monitor.loc[w, "Recall"]
    ) > 0.15 * baseline["Recall"]:
        messages.append("Recall drop detected")

    if (
        baseline["F1"] - monitor.loc[w, "F1-score"]
    ) > 0.30 * baseline["F1"]:
        messages.append("F1 performance drop detected")

    if messages:
        st.warning(f"Window {w}: " + " | ".join(messages))
    else:
        st.success(f"Window {w}: No major monitoring alert")

# --------------------------------------------------
# TRANSACTION PREDICTION DEMO
# --------------------------------------------------

st.divider()
st.subheader("🔍 Transaction Prediction Demonstration")

st.write(
    "Select a transaction from the simulated monitoring data "
    "to view the model prediction."
)

row_position = st.number_input(
    "Monitoring row number",
    min_value=0,
    max_value=len(live) - 1,
    value=0,
    step=1
)

if st.button("Predict Transaction"):
    transaction = live.iloc[[int(row_position)]]

    prediction = int(
        model.predict(transaction[FEATURES])[0]
    )

    amount = transaction["Amount"].iloc[0]

    st.metric("Transaction Amount", f"{amount:,.2f}")

    if prediction == 1:
        st.error("🚨 FRAUDULENT — Review this transaction")
    else:
        st.success("✅ LEGITIMATE — No fraud alert")

    st.caption(
        "Actual dataset label is used only for offline evaluation, "
        "not shown as a live prediction."
    )

st.divider()

st.caption(
    "ML_79 | Financial Fraud Detection & Model Monitoring System | "
    "Academic demonstration. Windows 4–6 contain simulated changes."
)
