import json
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

# LOAD CUSTOMER HISTORY

BASE_DIR = Path(__file__).resolve().parents[2]

customer_history_path = BASE_DIR / "data" / "customer_history.json"

with open(customer_history_path, "r", encoding="utf-8") as f:
    customer_history_data = json.load(f)

CUSTOMERS = customer_history_data["customers"]

# GRAPH STATE
class FraudState(TypedDict):

    transaction: dict

    customer_context: Optional[dict]


# CUSTOMER CONTEXT AGENT
def customer_context_agent(state: FraudState):

    transaction = state["transaction"]

    customer_id = transaction["customer_id"]


    # SEARCH CUSTOMER

    customer = next(
        (
            c
            for c in CUSTOMERS
            if c["customer_id"] == customer_id
        ),
        None
    )

    # CUSTOMER NOT FOUND
    if customer is None:

        return {
            "customer_context": {
                "customer_found": False,
                "customer_id": customer_id,
                "message": "Customer not found"
            }
        }

    # BUILD STRUCTURED CONTEXT

    customer_context = {

        "customer_found": True,

        "customer_id": customer["customer_id"],

        "name_masked": customer.get(
            "name_masked"
        ),

        "risk_profile": customer.get(
            "risk_profile"
        ),

        "kyc_status": customer.get(
            "kyc_status"
        ),

        "account_age_months": customer.get(
            "account_age_months"
        ),

        "home_city": customer.get(
            "home_city"
        ),

        "typical_transaction_amount_idr": customer.get(
            "typical_transaction_amount_idr"
        ),

        "typical_transaction_locations": customer.get(
            "typical_transaction_locations"
        ),

        "typical_merchant_categories": customer.get(
            "typical_merchant_categories"
        ),

        "active_devices": customer.get(
            "active_devices"
        ),

        "previous_fraud_reports": customer.get(
            "previous_fraud_reports"
        ),

        "device_fingerprints": customer.get(
            "device_fingerprints"
        ),

        "recent_transactions": customer.get(
            "recent_transactions"
        )
    }

    return {
        "customer_context": customer_context
    }



# BUILD LANGGRAPH
workflow = StateGraph(FraudState)

workflow.add_node(
    "customer_context_agent",
    customer_context_agent
)

workflow.set_entry_point(
    "customer_context_agent"
)

workflow.add_edge(
    "customer_context_agent",
    END
)

app = workflow.compile()
# TEST TRANSACTION
transaction_input = {

    "transaction_id": "TXN-90001",
    "customer_id": "CUST-00001",
    "amount_idr": 7000000,
    "merchant_category": "CRYPTO",
    "city": "Bangkok",
    "device_id": "DEV-NEW-999",
    "is_foreign": True
}

response = app.invoke({

    "transaction": transaction_input
})

print(
    json.dumps(
        response["customer_context"],
        indent=2
    )
)

