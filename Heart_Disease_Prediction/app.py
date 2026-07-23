"""
Heart Disease Risk Predictor
-----------------------------
Single-file Streamlit application built on top of the heart.csv EDA / ML
pipeline (cleaning, encoding, scaling, multi-model training) developed in
heart.ipynb.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #B91C1C;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #6B7280 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #111827 !important;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 6px solid #DC2626;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border-left: 6px solid #16A34A;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .footer-note {
        color: #9CA3AF;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

MODEL_ZOO = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}


# ----------------------------------------------------------------------------
# DATA LOADING & CLEANING  (mirrors heart.ipynb)
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("heart.csv")

    # Replace physiologically impossible zeros with the column mean
    # (identical logic to the notebook's data-cleaning step)
    mean_chol = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
    df["Cholesterol"] = df["Cholesterol"].replace(0, mean_chol)

    mean_rest = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
    df["RestingBP"] = df["RestingBP"].replace(0, mean_rest)

    return df


@st.cache_resource
def train_all_models(df: pd.DataFrame):
    """Encode, scale, split and train every model in MODEL_ZOO."""
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.astype(int) if df_encoded.select_dtypes(include="bool").shape[1] else df_encoded
    # cast any bool dummy columns to int explicitly
    for c in df_encoded.columns:
        if df_encoded[c].dtype == bool:
            df_encoded[c] = df_encoded[c].astype(int)

    X = df_encoded.drop("HeartDisease", axis=1)
    y = df_encoded["HeartDisease"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained = {}
    results = []
    roc_data = {}

    for name, model in MODEL_ZOO.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_score = model.decision_function(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

        results.append({
            "Model": name, "Accuracy": acc, "F1 Score": f1,
            "Precision": prec, "Recall": rec, "ROC AUC": roc_auc,
        })
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    return {
        "models": trained,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "results_df": results_df,
        "roc_data": roc_data,
        "confusion": {name: confusion_matrix(y_test, trained[name].predict(X_test_scaled)) for name in trained},
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
    }


def build_feature_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    """Turn raw sidebar inputs into a one-hot row matching training columns."""
    base = {col: 0 for col in feature_columns}

    base["Age"] = inputs["Age"]
    base["RestingBP"] = inputs["RestingBP"]
    base["Cholesterol"] = inputs["Cholesterol"]
    base["FastingBS"] = inputs["FastingBS"]
    base["MaxHR"] = inputs["MaxHR"]
    base["Oldpeak"] = inputs["Oldpeak"]

    dummy_flags = {
        "Sex": inputs["Sex"],
        "ChestPainType": inputs["ChestPainType"],
        "RestingECG": inputs["RestingECG"],
        "ExerciseAngina": inputs["ExerciseAngina"],
        "ST_Slope": inputs["ST_Slope"],
    }
    for prefix, value in dummy_flags.items():
        col_name = f"{prefix}_{value}"
        if col_name in base:
            base[col_name] = 1
        # if it's the dropped reference category, all dummy cols stay 0

    row = pd.DataFrame([base])[feature_columns]
    return row


# ----------------------------------------------------------------------------
# LOAD DATA + TRAIN
# ----------------------------------------------------------------------------
df = load_and_clean_data()
bundle = train_all_models(df)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="main-header">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Machine learning powered clinical risk assessment '
    'trained on the UCI Heart Failure Prediction dataset.</p>',
    unsafe_allow_html=True,
)

tab_predict, tab_eda, tab_models, tab_data = st.tabs(
    ["🩺 Predict", "📊 Exploratory Analysis", "🤖 Model Comparison", "🗂️ Dataset"]
)

# ----------------------------------------------------------------------------
# TAB 1 — PREDICTION
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("Patient Information")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 20, 90, 50)
                sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
                cp = st.selectbox(
                    "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
                    format_func=lambda x: {
                        "ATA": "Atypical Angina", "NAP": "Non-Anginal Pain",
                        "ASY": "Asymptomatic", "TA": "Typical Angina",
                    }[x],
                )
                resting_bp = st.slider("Resting BP (mm Hg)", 80, 220, 130)
                cholesterol = st.slider("Cholesterol (mg/dl)", 100, 604, 220)
                fasting_bs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl", [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
            with c2:
                resting_ecg = st.selectbox(
                    "Resting ECG", ["Normal", "ST", "LVH"],
                    format_func=lambda x: {
                        "Normal": "Normal", "ST": "ST-T Wave Abnormality",
                        "LVH": "Left Ventricular Hypertrophy",
                    }[x],
                )
                max_hr = st.slider("Max Heart Rate", 60, 220, 150)
                exercise_angina = st.selectbox(
                    "Exercise-Induced Angina", ["N", "Y"],
                    format_func=lambda x: "Yes" if x == "Y" else "No",
                )
                oldpeak = st.slider("Oldpeak (ST Depression)", -2.6, 6.2, 0.8, step=0.1)
                st_slope = st.selectbox(
                    "ST Slope", ["Up", "Flat", "Down"],
                    format_func=lambda x: {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}[x],
                )

            model_choice = st.selectbox(
                "Prediction Model",
                bundle["results_df"]["Model"].tolist(),
                help="Models are ranked by test-set accuracy; top of the list is the best performer.",
            )

            submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True, type="primary")

    with right:
        st.subheader("Risk Assessment")

        if submitted:
            inputs = {
                "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": resting_bp,
                "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
                "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
            row = build_feature_row(inputs, bundle["feature_columns"])
            row_scaled = bundle["scaler"].transform(row)

            model = bundle["models"][model_choice]
            pred = model.predict(row_scaled)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(row_scaled)[0][1]
            else:
                prob = None

            if pred == 1:
                st.markdown(
                    f"""<div class="risk-card-high">
                    <h3 style="margin-top:0;color:#991B1B;">⚠️ Elevated Risk Detected</h3>
                    <p style="color:#7F1D1D;margin-bottom:0;">The model predicts a higher likelihood of heart disease
                    based on the entered clinical parameters. This is not a diagnosis — please consult a
                    cardiologist for proper evaluation.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="risk-card-low">
                    <h3 style="margin-top:0;color:#166534;">✅ Low Risk Indicated</h3>
                    <p style="color:#14532D;margin-bottom:0;">The model does not detect strong indicators of heart
                    disease based on the entered clinical parameters. Routine checkups are still recommended.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", "Disease" if pred == 1 else "No Disease")
            m2.metric("Confidence" if prob is not None else "Model", f"{prob*100:.1f}%" if prob is not None else model_choice)
            m3.metric("Model Used", model_choice)

            if prob is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Predicted Risk Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#B91C1C" if prob > 0.5 else "#16A34A"},
                        "steps": [
                            {"range": [0, 40], "color": "#DCFCE7"},
                            {"range": [40, 70], "color": "#FEF9C3"},
                            {"range": [70, 100], "color": "#FEE2E2"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                    },
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill in the patient details on the left and click **Assess Risk** to generate a prediction.")
            st.markdown("##### How it works")
            st.write(
                "This tool reproduces the preprocessing pipeline from the original notebook "
                "(zero-value imputation, one-hot encoding, standard scaling) and lets you compare "
                "predictions across five classic ML models trained on the same data split."
            )

# ----------------------------------------------------------------------------
# TAB 2 — EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients", len(df))
    k2.metric("Heart Disease Cases", int(df["HeartDisease"].sum()))
    k3.metric("Healthy Cases", int((df["HeartDisease"] == 0).sum()))
    k4.metric("Disease Rate", f"{df['HeartDisease'].mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="Age", color="HeartDisease", nbins=25, marginal="box",
            barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Age Distribution by Heart Disease Status",
            labels={"HeartDisease": "Heart Disease"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig3 = px.box(
            df, x="HeartDisease", y="Cholesterol", color="HeartDisease",
            color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Cholesterol by Heart Disease Status",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        cp_counts = df.groupby(["ChestPainType", "HeartDisease"]).size().reset_index(name="Count")
        fig2 = px.bar(
            cp_counts, x="ChestPainType", y="Count", color="HeartDisease",
            barmode="group", color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Chest Pain Type vs Heart Disease",
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig4 = px.violin(
            df, x="HeartDisease", y="MaxHR", color="HeartDisease", box=True,
            color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Max Heart Rate by Heart Disease Status",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Feature Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig5 = px.imshow(
        corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
        title="Correlation Matrix (Numeric Features)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — MODEL COMPARISON
# ----------------------------------------------------------------------------
with tab_models:
    st.subheader("Model Performance Comparison")
    results_df = bundle["results_df"].copy()
    display_df = results_df.copy()
    for c in ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]:
        display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best_model = results_df.iloc[0]["Model"]
    st.success(f"🏆 Best performing model: **{best_model}** ({results_df.iloc[0]['Accuracy']*100:.1f}% accuracy)")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            results_df, x="Model", y=["Accuracy", "F1 Score", "Precision", "Recall"],
            barmode="group", title="Metric Comparison Across Models",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig_roc = go.Figure()
        for name, (fpr, tpr, roc_auc) in bundle["roc_data"].items():
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.2f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select a model", results_df["Model"].tolist(), key="cm_select")
    cm = bundle["confusion"][selected_model]
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Reds",
        labels=dict(x="Predicted", y="Actual"),
        x=["No Disease", "Disease"], y=["No Disease", "Disease"],
        title=f"Confusion Matrix — {selected_model}",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — RAW DATA
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Cleaned Dataset")
    st.caption("Zero-value Cholesterol and RestingBP entries have been imputed with column means, matching the source notebook.")
    st.dataframe(df, use_container_width=True, height=450)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned CSV", csv, "heart_cleaned.csv", "text/csv")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    '<p class="footer-note">Built with Streamlit · Models: Logistic Regression, KNN, SVM, Naive Bayes, '
    'Decision Tree · For educational purposes only, not a medical device.</p>',
    unsafe_allow_html=True,
)