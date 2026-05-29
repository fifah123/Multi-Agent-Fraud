from typing import TypedDict, Optional

from langgraph.graph import (
    StateGraph,
    END
)

from src.agents.customer_context_agent import (
    customer_context_agent
)

from src.agents.fraud_scoring_agent import (
    fraud_scoring_agent
)

from src.agents.rag_agent import (
    refine_query,
    retrieve_docs
)

from src.agents.reasoning_agent import (
    reasoning_agent
)

# =========================================================
# GLOBAL STATE
# =========================================================

class FraudState(TypedDict):

    transaction: dict

    question: Optional[str]

    customer_context: Optional[dict]

    fraud_scoring_result: Optional[dict]

    refined_query: Optional[str]

    retrieved_context: Optional[str]

    investigation_result: Optional[dict]

# =========================================================
# BUILD GRAPH
# =========================================================

workflow = StateGraph(FraudState)

# Nodes
workflow.add_node(
    "customer_context_agent",
    customer_context_agent
)

workflow.add_node(
    "fraud_scoring_agent",
    fraud_scoring_agent
)

workflow.add_node(
    "refine_query",
    refine_query
)

workflow.add_node(
    "retrieve_docs",
    retrieve_docs
)

workflow.add_node(
    "reasoning_agent",
    reasoning_agent
)

# Entry
workflow.set_entry_point(
    "customer_context_agent"
)

# Flow
workflow.add_edge(
    "customer_context_agent",
    "fraud_scoring_agent"
)

workflow.add_edge(
    "fraud_scoring_agent",
    "refine_query"
)

workflow.add_edge(
    "refine_query",
    "retrieve_docs"
)

workflow.add_edge(
    "retrieve_docs",
    "reasoning_agent"
)

workflow.add_edge(
    "reasoning_agent",
    END
)

# Compile
app = workflow.compile()