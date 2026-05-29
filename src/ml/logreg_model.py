import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

# LOAD CUSTOMER HISTORY

BASE_DIR = Path(__file__).resolve().parents[2]

customer_history_path = BASE_DIR / "data" / "transactions.csv"

df=pd.read_csv(customer_history_path)

# Feature Selection
features = [
    "amount_idr",
    "merchant_category",
    "city",
    "is_foreign",
    "is_new_device",
    "hour",
    "day_of_week",
    "customer_risk_profile",
    "customer_account_age_months",
    "customer_typical_p95",
    "customer_previous_fraud_reports",
    "customer_device_count",
    "amount_vs_p95_ratio",
    "category_in_typical",
    "velocity_flag"
]

target = "is_fraud"

X = df[features]
y = df[target]

# Feature Types

categorical_features = [
    "merchant_category",
    "city",
    "customer_risk_profile"
]

numeric_features = [
    "amount_idr",
    "is_foreign",
    "is_new_device",
    "hour",
    "day_of_week",
    "customer_account_age_months",
    "customer_typical_p95",
    "customer_previous_fraud_reports",
    "customer_device_count",
    "amount_vs_p95_ratio",
    "category_in_typical",
    "velocity_flag"
]

# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, y_pred))

print("=" * 50)
print("ROC AUC")
print("=" * 50)

print(roc_auc_score(y_test, y_prob))

print("=" * 50)
print("PR AUC")
print("=" * 50)

print(average_precision_score(y_test, y_prob))

# Save Model
joblib.dump(model, "fraud_logistic_regression.pkl")
print("Model saved!")