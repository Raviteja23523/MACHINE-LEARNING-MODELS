"""
train_model.py
--------------
Loads and cleans telco_churn.csv, trains 5 classifiers, and saves every
artifact the Streamlit UI needs (models, scaler, feature columns, evaluation
results, feature importance) into a single model_bundle.pkl file.

Run this once (or whenever telco_churn.csv changes):
    python train_model.py
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = "telco_churn.csv"
MODEL_OUT_PATH = "model_bundle.pkl"

BINARY_YESNO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
TRISTATE_COLS = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL_COLS = ["gender", "InternetService", "Contract", "PaymentMethod"] + BINARY_YESNO_COLS + TRISTATE_COLS

MODEL_ZOO = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text; 11 rows are blank (customers with 0 tenure)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target as 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def train_all_models(df: pd.DataFrame) -> dict:
    """Encode, scale, split and train every model in MODEL_ZOO."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    for c in df_encoded.columns:
        if df_encoded[c].dtype == bool:
            df_encoded[c] = df_encoded[c].astype(int)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
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

    results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    # feature importance from Random Forest (works well for a churn business narrative)
    rf_importances = None
    if "Random Forest" in trained:
        rf_importances = pd.Series(
            trained["Random Forest"].feature_importances_, index=feature_columns
        ).sort_values(ascending=False).head(12)

    confusion = {name: confusion_matrix(y_test, trained[name].predict(X_test_scaled)) for name in trained}

    return {
        "models": trained,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "categorical_cols": CATEGORICAL_COLS,
        "results_df": results_df,
        "roc_data": roc_data,
        "confusion": confusion,
        "feature_importance": rf_importances,
    }


def main():
    print(f"Loading and cleaning data from {DATA_PATH} ...")
    df = load_and_clean_data(DATA_PATH)
    print(f"  -> {len(df)} rows, {df['Churn'].mean()*100:.1f}% churn rate")

    print("Training models ...")
    bundle = train_all_models(df)

    # Save the cleaned dataframe too, so the UI doesn't need to re-clean raw data
    bundle["cleaned_df"] = df

    print("Results:")
    print(bundle["results_df"].to_string(index=False))

    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved trained models + artifacts to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()