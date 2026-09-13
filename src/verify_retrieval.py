"""
Phase 1, Step 2 / Phase 3: verify that ingested issues AND source code
chunks are actually retrievable.

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

# Test queries chosen to match known issues/behavior by meaning, not by
# repeating exact title or function-name wording — a good retrieval
# system should find them even when the phrasing differs.
ISSUE_QUERIES = [
    "Why does filtering tasks by priority not work with capital letters?",
    "What happens if I delete a task that doesn't exist?",
    "Is there a way to modify a task after creating it?",
]

# Phrased around the same known behaviors as the issue queries above, but
# aimed at the actual implementation rather than the bug report about it —
# a good test of whether code chunking retrieves the relevant function
# rather than an unrelated part of the file.
CODE_QUERIES = [
    "How is a new task added in the code?",
    "What does the code do when you try to delete a task that isn't there?",
    "How is task priority filtering implemented?",
]

# Phrased around whatever the README actually documents (installation,
# usage, known limitations) — a good test of whether heading-based
# chunking retrieves the right section rather than an unrelated one.
DOC_QUERIES = [
    "How do I install this project?",
    "How do I run this from the command line?",
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


def run_queries(store: AstraDBVectorStore, queries: list[str]) -> None:
    for query in queries:
        print(f"\nQuery: {query}")
        results = store.similarity_search(query, k=2)
        if not results:
            print("  No results returned — retrieval is broken, stop here.")
            continue
        for doc in results:
            title_line = doc.page_content.splitlines()[0]
            source = doc.metadata.get("source", "unknown")
            citation = doc.metadata.get("citation", "no citation")
            print(f"  Match [{source}]: {title_line} ({citation})")


if __name__ == "__main__":
    store = get_vector_store()

    print("=== Issue retrieval ===")
    run_queries(store, ISSUE_QUERIES)

    print("\n=== Code retrieval ===")
    run_queries(store, CODE_QUERIES)

    print("\n=== Doc retrieval ===")
    run_queries(store, DOC_QUERIES)
