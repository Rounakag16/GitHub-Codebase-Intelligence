"""
Phase 1, Step 3: the agent itself.

Takes a natural-language question, decides (via tool-calling) whether to
search the issue knowledge base, and answers using Gemini.

Built on LangChain 1.0's `create_agent` (the current standard since
LangChain's October 2025 v1.0 release) — NOT the older `AgentExecutor` /
`create_tool_calling_agent` pattern, which moved to the separate
`langchain-classic` package and is no longer part of the main `langchain`
import surface. `create_agent` runs on LangGraph under the hood and
returns a compiled graph rather than a Runnable + separate executor.

Run: python src/agent.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

COLLECTION_NAME = "github_issues"
NOTES_FILE = "notes.txt"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a GitHub "
    "repository. Right now you only have access to this repository's "
    "GitHub issues — you do NOT have access to its source code. "
    "Always use the search_github_issues tool before answering a question "
    "that could relate to a bug or feature request — do not guess. Your "
    "answer may combine several separate facts (e.g. what currently "
    "works, and what's missing). Treat each fact in your answer "
    "independently: for every individual fact that comes from a "
    "retrieved issue, cite that issue's number and URL right next to it — "
    "even if the question's main topic is something else, and even if "
    "another part of the same answer already has a citation. Never let "
    "one cited fact make a nearby uncited fact look sourced by "
    "association. If nothing relevant is found in the issues, say so "
    "plainly. "
    "Critically: if a question asks about implementation details, function "
    "signatures, arguments, return values, or how code actually works, and "
    "that information is not explicitly stated in a retrieved issue, do "
    "NOT invent or guess an answer — clearly say you don't have access to "
    "the source code and cannot confirm that detail. A confident wrong "
    "answer is much worse than admitting you don't know."
)


def get_retriever_tool():
    """Wrap the Astra DB vector store as a tool the agent can call by choice."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    document_prompt = PromptTemplate.from_template(
        "{page_content}\n(Issue #{issue_number}, {url})"
    )
    return create_retriever_tool(
        retriever,
        name="search_github_issues",
        description=(
            "Search this repository's GitHub issues. Use this whenever the "
            "question might relate to a known bug, limitation, or feature "
            "request — i.e. almost always, before answering from general "
            "knowledge."
        ),
        document_prompt=document_prompt,
    )


@tool
def save_note(note: str) -> str:
    """Save a short note to a local notes file, for remembering something
    important across the conversation (e.g. a conclusion reached, or a
    detail the user asked to keep track of)."""
    with open(NOTES_FILE, "a") as f:
        f.write(note.strip() + "\n")
    return "Note saved."


def build_llm():
    provider = os.environ.get("CHAT_PROVIDER", "gemini").lower()
    if provider == "ollama":
        return ChatOllama(model=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"), temperature=0)
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)


def build_agent():
    llm = build_llm()
    tools = [get_retriever_tool(), save_note]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    provider = os.environ.get("CHAT_PROVIDER", "gemini").lower()
    print(f"Using chat provider: {provider}")
    agent = build_agent()
    messages = []
    print("Ask a question about the repo's issues (type 'exit' to quit).")
    while True:
        query = input("\n> ")
        if query.strip().lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": query})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print("\n" + messages[-1].text)
