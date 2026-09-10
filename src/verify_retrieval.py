"""
Phase 1, Step 2: verify that ingested issues are actually retrievable.

This does a plain similarity search directly against the Astra DB
collection — no LLM, no agent yet. The point is to isolate retrieval
quality from everything else: if this works, we know embeddings + storage
are solid before adding an agent on top that could mask a retrieval bug.

Run: python src/verify_retrieval.py
"""

import os

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

COLLECTION_NAME = "github_issues"

# Test queries chosen to match the 3 sample issues by meaning, not by
# repeating their exact title wording — a good retrieval system should
# find them even when the phrasing differs.
TEST_QUERIES = [
    "Why does filtering tasks by priority not work with capital letters?",
    "What happens if I delete a task that doesn't exist?",
    "Is there a way to modify a task after creating it?",
]


def get_vector_store() -> AstraDBVectorStore:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


if __name__ == "__main__":
    store = get_vector_store()

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        results = store.similarity_search(query, k=2)
        if not results:
            print("  No results returned — retrieval is broken, stop here.")
            continue
        for doc in results:
            title_line = doc.page_content.splitlines()[0]
            print(f"  Match: {title_line} (issue #{doc.metadata.get('issue_number')}, {doc.metadata.get('url')})")
