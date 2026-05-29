import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =========================================================
# BASE DIR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# =========================================================
# PATHS
# =========================================================

DOCS_PATH = BASE_DIR / "data"

VECTOR_DB_PATH = BASE_DIR / "vector_db"

print("DOCS_PATH:", DOCS_PATH)
print("VECTOR_DB_PATH:", VECTOR_DB_PATH)

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# =========================================================
# TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["===[TEXT]==="]
)

all_docs = []

# =========================================================
# LOAD TXT FILES
# =========================================================

for file in DOCS_PATH.glob("*.txt"):

    print("Processing:", file.name)

    with open(file, encoding="utf-8") as f:
        content = f.read()

    texts = text_splitter.create_documents([content])

    for d in texts:

        cleaned = d.page_content.replace(
            "===[TEXT]===",
            ""
        ).strip()

        if cleaned:

            all_docs.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": file.name
                    }
                )
            )

print("Total chunks:", len(all_docs))

# =========================================================
# DELETE OLD DB (OPTIONAL)
# =========================================================

if VECTOR_DB_PATH.exists():

    import shutil

    shutil.rmtree(VECTOR_DB_PATH)

# =========================================================
# CREATE VECTOR DB
# =========================================================

vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="fraud-rag",
    persist_directory=str(VECTOR_DB_PATH)
)

print("Vector DB saved!")
print("Collection:", vector_store._collection.name)