"""
Phase 3: fetch TARGET_REPO's Python source files, split each one into
function/class-level chunks (not naive text splitting), and load them into
the same Astra DB collection used for issues.

Why chunk by function/class instead of by file: embedding a whole file as
one vector blends together everything in it — an unrelated helper function,
the main class, and a CLI entrypoint all get squashed into a single point
in vector space. That makes retrieval worse (a question about one function
pulls back the whole file, most of it irrelevant) and is exactly the naive
approach this project is meant to avoid. Splitting by function/class means
a question like "how does delete_task work" retrieves that one function,
not the whole module.

Chunking strategy (Python's `ast` module):
  - Each top-level function -> its own chunk.
  - Each class -> one small chunk for the class itself (signature +
    docstring, not the full body — the body is covered by its methods),
    plus one chunk per method.
  - Anything left at module level that isn't inside a function/class
    (imports, constants, an `if __name__ == "__main__":` block) -> one
    fallback chunk per file, so nothing is silently dropped.

Run: python src/ingest_code.py
"""

import ast
import os

from dotenv import load_dotenv
from github import Auth, Github
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Same collection as ingest_issues.py — issues and code coexist here,
# distinguished by metadata["source"]. Kept as "github_issues" (rather
# than renamed) so existing ingested issues don't need a collection
# migration; the name is legacy, the collection is now general-purpose.
COLLECTION_NAME = "github_issues"

# Directories to skip when walking the repo tree — not source code.
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".mypy_cache"}


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    return segment if segment is not None else ""


def _chunk_file(file_path: str, source: str) -> list[dict]:
    """Parse one file's source into function/class/module-level chunks.

    Returns a list of dicts with keys: name, kind, start_line, end_line,
    text — not yet wrapped as Documents (that happens once we also have
    the repo's URL info, in fetch_code_chunks).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Skip files that don't parse as valid Python rather than crashing
        # the whole ingestion run over one bad file.
        return []

    chunks = []
    module_leftover = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                {
                    "name": node.name,
                    "kind": "function",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": _source_segment(source, node),
                }
            )
        elif isinstance(node, ast.ClassDef):
            # One small chunk for the class itself: signature + docstring
            # only, not the full body (methods are chunked separately
            # below, so including the full body here would duplicate
            # nearly everything).
            docstring = ast.get_docstring(node)
            class_header = f"class {node.name}:"
            class_text = class_header + (f"\n    \"\"\"{docstring}\"\"\"" if docstring else "")
            chunks.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "start_line": node.lineno,
                    "end_line": node.lineno,
                    "text": class_text,
                }
            )

            # Scoped to THIS class only — must never be merged with
            # module_leftover, or a class's own leftover statements (e.g.
            # class-level attributes) get mislabeled as module-level code
            # from an unrelated part of the file.
            class_leftover = []
            for i, sub_node in enumerate(node.body):
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        {
                            "name": f"{node.name}.{sub_node.name}",
                            "kind": "method",
                            "start_line": sub_node.lineno,
                            "end_line": sub_node.end_lineno,
                            "text": _source_segment(source, sub_node),
                        }
                    )
                elif i == 0 and docstring and isinstance(sub_node, ast.Expr):
                    # The docstring itself — already captured in class_text
                    # above, skip so it isn't duplicated as "leftover".
                    continue
                else:
                    class_leftover.append(sub_node)

            if class_leftover:
                leftover_text = "\n".join(
                    _source_segment(source, n) for n in class_leftover if _source_segment(source, n)
                )
                if leftover_text.strip():
                    chunks.append(
                        {
                            "name": f"{node.name} (class-level attributes)",
                            "kind": "class_attributes",
                            "start_line": min(n.lineno for n in class_leftover),
                            "end_line": max(getattr(n, "end_lineno", n.lineno) for n in class_leftover),
                            "text": leftover_text,
                        }
                    )
        else:
            module_leftover.append(node)

    if module_leftover:
        leftover_text = "\n".join(_source_segment(source, n) for n in module_leftover if _source_segment(source, n))
        if leftover_text.strip():
            chunks.append(
                {
                    "name": "module_level",
                    "kind": "module_level",
                    "start_line": min(n.lineno for n in module_leftover),
                    "end_line": max(getattr(n, "end_lineno", n.lineno) for n in module_leftover),
                    "text": leftover_text,
                }
            )

    return chunks


def fetch_code_chunks(repo_name: str) -> list[Document]:
    """Walk TARGET_REPO's Python files via the GitHub API and return one
    Document per function/class/method/module-level chunk."""
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

        if not item.path.endswith(".py"):
            continue

        source = item.decoded_content.decode("utf-8", errors="ignore")
        blob_url = f"{repo.html_url}/blob/{default_branch}/{item.path}"

        for chunk in _chunk_file(item.path, source):
            line_ref = f"#L{chunk['start_line']}-L{chunk['end_line']}"
            url = blob_url + line_ref
            citation = f"{item.path}:{chunk['start_line']}-{chunk['end_line']} ({chunk['name']}), {url}"
            documents.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": "code",
                        "file_path": item.path,
                        "name": chunk["name"],
                        "kind": chunk["kind"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "url": url,
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

    # Deterministic IDs (same pattern as ingest_issues.py) so re-running
    # this script upserts instead of duplicating chunks.
    def make_id(doc: Document) -> str:
        safe_path = doc.metadata["file_path"].replace("/", "_").replace(".", "_")
        safe_name = doc.metadata["name"].replace(".", "_")
        return f"code-{safe_path}-{safe_name}"

    ids = [make_id(doc) for doc in documents]
    added_ids = vector_store.add_documents(documents, ids=ids)
    print(f"Loaded {len(added_ids)} code chunks into Astra DB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    target_repo = os.environ["TARGET_REPO"]
    print(f"Fetching Python source files from {target_repo}...")
    docs = fetch_code_chunks(target_repo)
    print(f"Parsed {len(docs)} function/class/method/module-level chunks.")

    load_into_astra(docs)
