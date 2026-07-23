"""
app.py
------
Customer Churn Predictor — UI ONLY.
All training logic lives in train_model.py, which saves model_bundle.pkl.
This file just loads that bundle and renders the Streamlit interface.

First run:
    python train_model.py      # creates model_bundle.pkl
    streamlit run app.py       # launches the UI
"""

import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

MODEL_PATH = "model_bundle.pkl"

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
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
        color: #1D4ED8;
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


# ----------------------------------------------------------------------------
# LOAD PRE-TRAINED BUNDLE
# ----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


bundle = load_bundle()

if bundle is None:
    st.error(
        f"⚠️ Could not find **{MODEL_PATH}**. Run `python train_model.py` first "
        "to train the models and generate this file, then restart the app."
    )
    st.stop()

df = bundle["cleaned_df"]
CATEGORICAL_COLS = bundle["categorical_cols"]


def build_feature_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    """Turn raw sidebar inputs into a one-hot row matching training columns."""
    base = {col: 0 for col in feature_columns}

    base["SeniorCitizen"] = inputs["SeniorCitizen"]
    base["tenure"] = inputs["tenure"]
    base["MonthlyCharges"] = inputs["MonthlyCharges"]
    base["TotalCharges"] = inputs["TotalCharges"]

    dummy_flags = {k: v for k, v in inputs.items() if k in CATEGORICAL_COLS}
    for prefix, value in dummy_flags.items():
        col_name = f"{prefix}_{value}"
        if col_name in base:
            base[col_name] = 1
        # if it's the dropped reference category, all dummy cols stay 0

    row = pd.DataFrame([base])[feature_columns]
    return row


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="main-header">📉 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Machine learning powered churn-risk assessment '
    'trained on the IBM Telco Customer Churn dataset (7,043 customers).</p>',
    unsafe_allow_html=True,
)

tab_predict, tab_eda, tab_models, tab_data = st.tabs(
    ["🔮 Predict", "📊 Exploratory Analysis", "🤖 Model Comparison", "🗂️ Dataset"]
)

# ----------------------------------------------------------------------------
# TAB 1 — PREDICTION
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("Customer Profile")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Has Partner", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            with c2:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=9000.0,
                value=float(tenure) * monthly_charges, step=10.0,
                help="Defaults to an estimate of tenure × monthly charges; adjust if you know the exact figure.",
            )

            model_choice = st.selectbox(
                "Prediction Model",
                bundle["results_df"]["Model"].tolist(),
                help="Models are ranked by test-set ROC AUC; top of the list is the best performer.",
            )

            submitted = st.form_submit_button("🔮 Assess Churn Risk", use_container_width=True, type="primary")

    with right:
        st.subheader("Churn Risk Assessment")

        if submitted:
            inputs = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
                "InternetService": internet_service, "OnlineSecurity": online_security,
                "OnlineBackup": online_backup, "DeviceProtection": device_protection,
                "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
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
                    <h3 style="margin-top:0;color:#991B1B;">⚠️ High Churn Risk</h3>
                    <p style="color:#7F1D1D;margin-bottom:0;">The model predicts this customer is likely to churn.
                    Consider proactive retention actions — loyalty discounts, contract upgrade offers, or a
                    support outreach call.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="risk-card-low">
                    <h3 style="margin-top:0;color:#166534;">✅ Low Churn Risk</h3>
                    <p style="color:#14532D;margin-bottom:0;">The model does not detect strong churn indicators
                    for this customer profile. Standard engagement is likely sufficient.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
            m2.metric("Confidence" if prob is not None else "Model", f"{prob*100:.1f}%" if prob is not None else model_choice)
            m3.metric("Model Used", model_choice)

            if prob is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Predicted Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#DC2626" if prob > 0.5 else "#16A34A"},
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
            st.info("Fill in the customer profile on the left and click **Assess Churn Risk** to generate a prediction.")
            st.markdown("##### How it works")
            st.write(
                "This tool loads a pre-trained model bundle (see `train_model.py`) and lets you compare "
                "predictions across five classic ML models trained on the IBM Telco Customer Churn dataset."
            )

# ----------------------------------------------------------------------------
# TAB 2 — EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", len(df))
    k2.metric("Churned", int(df["Churn"].sum()))
    k3.metric("Retained", int((df["Churn"] == 0).sum()))
    k4.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="tenure", color="Churn", nbins=30, marginal="box",
            barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Tenure Distribution by Churn Status",
            labels={"Churn": "Churn"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig3 = px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Monthly Charges by Churn Status",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        contract_counts = df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig2 = px.bar(
            contract_counts, x="Contract", y="Count", color="Churn",
            barmode="group", color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Contract Type vs Churn",
        )
        st.plotly_chart(fig2, use_container_width=True)

        internet_counts = df.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
        fig4 = px.bar(
            internet_counts, x="InternetService", y="Count", color="Churn",
            barmode="group", color_discrete_map={0: "#16A34A", 1: "#DC2626"},
            title="Internet Service Type vs Churn",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Feature Correlation Heatmap")
    corr = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].corr()
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
    st.success(f"🏆 Best performing model: **{best_model}** ({results_df.iloc[0]['ROC AUC']*100:.1f}% ROC AUC)")

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
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        selected_model = st.selectbox("Select a model", results_df["Model"].tolist(), key="cm_select")
        cm = bundle["confusion"][selected_model]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            title=f"Confusion Matrix — {selected_model}",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.subheader("Top Churn Drivers")
        if bundle["feature_importance"] is not None:
            fi = bundle["feature_importance"].sort_values(ascending=True)
            fig_fi = px.bar(
                fi, orientation="h",
                title="Random Forest Feature Importance (Top 12)",
                labels={"value": "Importance", "index": "Feature"},
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — RAW DATA
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Cleaned Dataset")
    st.caption("customerID dropped; TotalCharges coerced to numeric with blanks filled as 0 (new customers, 0 tenure).")
    st.dataframe(df, use_container_width=True, height=450)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned CSV", csv, "churn_cleaned.csv", "text/csv")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    '<p class="footer-note">Built with Streamlit · Models: Logistic Regression, Random Forest, Gradient '
    'Boosting, KNN, Decision Tree · Dataset: IBM Telco Customer Churn · For educational purposes only.</p>',
    unsafe_allow_html=True,
)