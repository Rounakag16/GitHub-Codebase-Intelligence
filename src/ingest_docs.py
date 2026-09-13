"""
Phase 3 (continued): fetch TARGET_REPO's markdown docs (README, anything
under a docs/ folder, etc.) and load them into the same Astra DB
collection used for issues and code.

Chunking strategy: split each file on level-2 headings ("## ..."), so a
long README with several distinct sections doesn't get embedded as one
blob where an unrelated section pollutes the vector for the section
someone's actually asking about — same principle as chunking code by
function instead of by file. If a file has no "## " headings at all
(e.g. a short README), it's embedded as a single whole-file chunk instead
of being forced into an empty split.

No changes were needed in agent.py or ingest_issues.py/ingest_code.py for
this: every ingestion script already writes a precomputed "citation"
metadata field in its own format, and the agent's document_prompt just
prints whatever citation is attached — a new source type (source="doc")
slots in for free.

Run: python src/ingest_docs.py
"""

import os
import re

from dotenv import load_dotenv
from github import Auth, Github
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Same collection as ingest_issues.py / ingest_code.py — see the comment
# there for why the name stays "github_issues" despite being
# general-purpose now.
COLLECTION_NAME = "github_issues"

SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".mypy_cache"}

# Matches a level-2 markdown heading line, e.g. "## Installation".
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _chunk_markdown(text: str) -> list[dict]:
    """Split markdown into (heading, content) chunks on '## ' headings.

    Returns a list of dicts with keys: section, text. If there are no
    level-2 headings, returns a single chunk covering the whole file.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [{"section": "full_file", "text": text}]

    chunks = []
    # Anything before the first "## " heading (e.g. the title and intro
    # under a top-level "# " heading) becomes its own leading chunk, if
    # there's meaningful content there.
    intro = text[: matches[0].start()].strip()
    if intro:
        chunks.append({"section": "intro", "text": intro})

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append({"section": section_title, "text": text[start:end].strip()})

    return chunks


def fetch_doc_chunks(repo_name: str) -> list[Document]:
    """Walk TARGET_REPO's markdown files via the GitHub API and return one
    Document per section chunk."""
    gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
    repo = gh.get_repo(repo_name)
    default_branch = repo.default_branch

    documents = []
    contents = repo.get_contents("")
    while contents:
        item = contents.pop(0)
        if item.type == "dir":
            if item.name in SKIP_DIRS:
                continue
            contents.extend(repo.get_contents(item.path))
            continue

        if not item.path.lower().endswith(".md"):
            continue

        text = item.decoded_content.decode("utf-8", errors="ignore")
        blob_url = f"{repo.html_url}/blob/{default_branch}/{item.path}"

        for chunk in _chunk_markdown(text):
            citation = f"{item.path} § {chunk['section']}, {blob_url}"
            documents.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": "doc",
                        "file_path": item.path,
                        "section": chunk["section"],
                        "url": blob_url,
                        "citation": citation,
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

    def make_id(doc: Document) -> str:
        safe_path = doc.metadata["file_path"].replace("/", "_").replace(".", "_")
        safe_section = re.sub(r"[^a-zA-Z0-9_]", "_", doc.metadata["section"])
        return f"doc-{safe_path}-{safe_section}"

    ids = [make_id(doc) for doc in documents]
    added_ids = vector_store.add_documents(documents, ids=ids)
    print(f"Loaded {len(added_ids)} doc sections into Astra DB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    target_repo = os.environ["TARGET_REPO"]
    print(f"Fetching markdown docs from {target_repo}...")
    docs = fetch_doc_chunks(target_repo)
    print(f"Parsed {len(docs)} doc sections.")

    load_into_astra(docs)
