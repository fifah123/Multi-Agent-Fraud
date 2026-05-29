import json
import joblib
from pathlib import Path
from typing import TypedDict, Optional

import pandas as pd

from langgraph.graph import StateGraph, END

# BASE DIRECTORY
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# LOAD MODEL
MODEL_PATH = BASE_DIR / "models" / "fraud_logistic_regression.pkl"

fraud_model = joblib.load(MODEL_PATH)

# LOAD CUSTOMER HISTORY
CUSTOMER_HISTORY_PATH = BASE_DIR / "data" / "customer_history.json"

with open(CUSTOMER_HISTORY_PATH, "r", encoding="utf-8") as f:
    customer_history_data = json.load(f)

CUSTOMERS = customer_history_data["customers"]

# GRAPH STATE
class FraudState(TypedDict):

    transaction: dict
    customer_context: Optional[dict]
    fraud_scoring_result: Optional[dict]


# CUSTOMER CONTEXT AGENT 
def customer_context_agent(state: FraudState):

    transaction = state["transaction"]
    customer_id = transaction["customer_id"]
    customer = next(
        (
            c
            for c in CUSTOMERS
            if c["customer_id"] == customer_id
        ),
        None
    )

    return {
        "customer_context": customer
    }

# FRAUD SCORING AGENT

def fraud_scoring_agent(state: FraudState):

    transaction = state["transaction"]
    customer = state["customer_context"]

    # FEATURE ENGINEERING
    
    amount_idr = transaction["amount_idr"]
    merchant_category = transaction["merchant_category"]
    city = transaction["city"]
    is_foreign = int(transaction.get("is_foreign", False))
    device_id = transaction.get("device_id","")
    hour = transaction["hour"]
    day_of_week = transaction["day_of_week"]
    customer_risk_profile = customer["risk_profile"]
    customer_account_age_months = customer["account_age_months"]
    customer_typical_p95 = customer["typical_transaction_amount_idr"]["p95"]
    customer_previous_fraud_reports = customer["previous_fraud_reports"]
    customer_device_count = customer["active_devices"]

    # DERIVED FEATURES
    amount_vs_p95_ratio = (amount_idr / customer_typical_p95)
    category_in_typical = int(merchant_category in customer["typical_merchant_categories"])
    is_new_device = int(device_id not in customer["device_fingerprints"])

    # Simple velocity logic
    recent_tx = customer["recent_transactions"]
    recent_cities = [tx["city"] for tx in recent_tx[:5]]

    velocity_flag = int(city not in recent_cities)

    # BUILD FEATURE DF
    feature_df = pd.DataFrame([{

        "amount_idr": amount_idr,
        "merchant_category": merchant_category,
        "city": city,
        "is_foreign": is_foreign,
        "is_new_device": is_new_device,
        "hour": hour,
        "day_of_week": day_of_week,
        "customer_risk_profile":customer_risk_profile,
        "customer_account_age_months":customer_account_age_months,
        "customer_typical_p95":customer_typical_p95,
        "customer_previous_fraud_reports":customer_previous_fraud_reports,
        "customer_device_count":customer_device_count,
        "amount_vs_p95_ratio":amount_vs_p95_ratio,
        "category_in_typical":category_in_typical,
        "velocity_flag":velocity_flag
    }])

    # =====================================================
    # MODEL INFERENCE
    # =====================================================

    fraud_probability = float(fraud_model.predict_proba(feature_df)[0][1])
    fraud_prediction = int(fraud_probability >= 0.5)

    # RETURN RESULT

    return {

        "fraud_scoring_result": {
            "fraud_probability":round(fraud_probability, 4),
            "fraud_prediction":fraud_prediction,
            "model_threshold": 0.5,
            "engineered_features": {
                "amount_vs_p95_ratio":round(amount_vs_p95_ratio,2),
                "category_in_typical":category_in_typical,
                "is_new_device": is_new_device,
                "velocity_flag":velocity_flag
            }
        }
    }

# BUILD WORKFLOW

workflow = StateGraph(FraudState)
workflow.add_node(
    "customer_context_agent",
    customer_context_agent
)

workflow.add_node(
    "fraud_scoring_agent",
    fraud_scoring_agent
)

workflow.set_entry_point(
    "customer_context_agent"
)

workflow.add_edge(
    "customer_context_agent",
    "fraud_scoring_agent"
)

workflow.add_edge(
    "fraud_scoring_agent",
    END
)

app = workflow.compile()

# =========================================================
# TEST TRANSACTION
# =========================================================

if __name__ == "__main__":

    transaction_input = {

        "transaction_id": "TXN-90001",
        "customer_id": "CUST-00001",
        "amount_idr": 7000000,
        "merchant_category": "CRYPTO",
        "city": "Bangkok",
        "is_foreign": True,
        "device_id": "DEV-NEW-999",
        "hour": 3,
        "day_of_week": 6
    }

    response = app.invoke({

        "transaction": transaction_input
    })

    print(
        json.dumps(
            response["fraud_scoring_result"],
            indent=2
        )
    )