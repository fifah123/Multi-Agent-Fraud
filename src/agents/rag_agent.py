from typing import TypedDict, Optional
from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import (StateGraph,END)
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings)
from langchain_chroma import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate)


# LOAD ENV

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

# INITIALIZE EMBEDDINGS
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
# LOAD VECTOR STORE
vector_store = Chroma(
    collection_name="fraud-rag",
    embedding_function=embeddings,
    persist_directory="../vector_db"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# =========================================================
# INITIALIZE LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================================================
# GRAPH STATE
# =========================================================

class GraphState(TypedDict):

    question: str

    refined_query: Optional[str]

    retrieved_context: Optional[str]


# =========================================================
# QUERY REFINER PROMPT
# =========================================================

refine_prompt = ChatPromptTemplate.from_template(
    """
You are a fraud knowledge retrieval query refiner.

Your ONLY task:
convert user fraud investigation questions into optimized semantic retrieval queries.

User Question:
{question}

Rules:
- output plain text only
- no explanations
- preserve fraud terminology
- include fraud pattern keywords
- optimize for semantic retrieval
"""
)

# =========================================================
# QUERY REFINER NODE
# =========================================================

def refine_query(state: GraphState):

    chain = refine_prompt | llm

    result = chain.invoke({

        "question": state["question"]
    })

    return {

        "refined_query": result.content.strip()
    }

# =========================================================
# RETRIEVAL NODE
# =========================================================

def retrieve_docs(state: GraphState):

    refined_query = state["refined_query"]

    docs = retriever.invoke(
        refined_query
    )

    retrieved_context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    return {

        "retrieved_context":
            retrieved_context
    }

# =========================================================
# BUILD GRAPH
# =========================================================

workflow = StateGraph(GraphState)

workflow.add_node(
    "refine_query",
    refine_query
)

workflow.add_node(
    "retrieve_docs",
    retrieve_docs
)

workflow.set_entry_point(
    "refine_query"
)

workflow.add_edge(
    "refine_query",
    "retrieve_docs"
)

workflow.add_edge(
    "retrieve_docs",
    END
)

app = workflow.compile()


def run_rag(question: str):

    result = app.invoke({
        "question": question
    })

    return {
        "refined_query": result["refined_query"],
        "retrieved_context": result["retrieved_context"]
    }