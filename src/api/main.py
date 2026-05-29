from fastapi import FastAPI

from pydantic import BaseModel

from src.agents.orchestrator import (
    app as fraud_graph
)

# =========================================================
# FASTAPI
# =========================================================

api = FastAPI(
    title="Fraud Investigation API"
)

# =========================================================
# REQUEST SCHEMA
# =========================================================

class TransactionRequest(BaseModel):

    transaction_id: str

    customer_id: str

    amount_idr: int

    merchant_category: str

    city: str

    is_foreign: bool

    device_id: str

    hour: int

    day_of_week: int

# =========================================================
# INVESTIGATE ENDPOINT
# =========================================================

@api.post("/investigate")

def investigate_transaction(
    request: TransactionRequest
):

    transaction = request.dict()

    # Create investigation question
    question = f"""
    Investigate suspicious transaction:

    Amount:
    {transaction['amount_idr']}

    Merchant:
    {transaction['merchant_category']}

    City:
    {transaction['city']}

    Foreign:
    {transaction['is_foreign']}

    Device:
    {transaction['device_id']}
    """

    result = fraud_graph.invoke({

        "transaction": transaction,

        "question": question
    })

    return {

        "transaction":
            transaction,

        "customer_context":
            result["customer_context"],

        "fraud_scoring_result":
            result["fraud_scoring_result"],

        "retrieved_context":
            result["retrieved_context"],

        "investigation_result":
            result["investigation_result"]
    }