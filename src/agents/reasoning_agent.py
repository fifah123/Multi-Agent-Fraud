from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================================================
# STATE
# =========================================================

class GraphState(TypedDict):

    transaction: dict

    customer_context: Optional[dict]

    fraud_scoring_result: Optional[dict]

    retrieved_context: Optional[str]

    investigation_result: Optional[dict]

# =========================================================
# PROMPT
# =========================================================

reasoning_prompt = ChatPromptTemplate.from_template(
"""
You are a Senior Fraud Investigation Agent for Allo Bank.

Your task:
Analyze transaction risk using:

1. Transaction data
2. Customer behavioral profile
3. Fraud scoring model result
4. Fraud SOP / policy retrieval knowledge

You MUST:
- identify suspicious behavior
- explain anomalies
- identify likely fraud pattern
- determine risk severity
- recommend action
- recommend escalation tier

Use ONLY provided context.

Return output in JSON format.

--------------------------------------------------

TRANSACTION:
{transaction}

--------------------------------------------------

CUSTOMER CONTEXT:
{customer_context}

--------------------------------------------------

FRAUD SCORING RESULT:
{fraud_scoring_result}

--------------------------------------------------

RETRIEVED KNOWLEDGE:
{retrieved_context}

--------------------------------------------------

Required JSON Schema:

{{
    "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",

    "fraud_probability": float,

    "likely_fraud_pattern": string,

    "recommended_action": string,

    "recommended_escalation_tier": string,

    "supporting_signals": [
        string
    ],

    "reasoning": string
}}
"""
)

# =========================================================
# AGENT
# =========================================================

def reasoning_agent(state: GraphState):

    chain = reasoning_prompt | llm

    result = chain.invoke({

        "transaction":
            state["transaction"],

        "customer_context":
            state["customer_context"],

        "fraud_scoring_result":
            state["fraud_scoring_result"],

        "retrieved_context":
            state["retrieved_context"]
    })

    return {

        "investigation_result":
            result.content
    }
