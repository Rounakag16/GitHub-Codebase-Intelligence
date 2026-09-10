"""
Phase 1, Step 1: fetch GitHub issues for TARGET_REPO and load them into
Astra DB as embedded documents.

This is the tutorial's original data source (issues), kept as-is for now.
Phase 3 will expand ingestion to source files, README, and docs.

Run: python src/ingest_issues.py
"""

import os

from dotenv import load_dotenv
from github import Auth, Github
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

COLLECTION_NAME = "github_issues"


def fetch_issues(repo_name: str) -> list[Document]:
    """Pull all issues (open + closed) for repo_name as LangChain Documents."""
    gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
    repo = gh.get_repo(repo_name)

    documents = []
    for issue in repo.get_issues(state="all"):
        # Skip pull requests — the GitHub API returns PRs as issues too.
        if issue.pull_request is not None:
            continue

        content = f"Title: {issue.title}\n\n{issue.body or ''}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "github_issue",
                    "issue_number": issue.number,
                    "url": issue.html_url,
                    "state": issue.state,
                },
            )
        )
    return documents


def load_into_astra(documents: list[Document]) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vector_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )

    ids = [f"issue-{doc.metadata['issue_number']}" for doc in documents]
    added_ids = vector_store.add_documents(documents, ids=ids)
    print(f"Loaded {len(added_ids)} issues into Astra DB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    target_repo = os.environ["TARGET_REPO"]
    print(f"Fetching issues from {target_repo}...")
    docs = fetch_issues(target_repo)
    print(f"Fetched {len(docs)} issues (excluding pull requests).")

    load_into_astra(docs)
