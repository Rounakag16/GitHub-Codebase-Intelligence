"""
Phase 4: a thin FastAPI wrapper around the agent.

One real endpoint: POST /ask. Multi-turn conversation is supported by
having the CLIENT hold and resend the full message history on each
request — the exact same pattern src/agent.py's own CLI loop already
uses internally (append the new message, call agent.invoke, keep the
returned message list for next time). The server itself stays
stateless: no session storage, no per-user state to manage, nothing that
breaks on a server restart. "Multi-turn" is a property of how the API is
used, not something the server tracks on its own.

Only user/assistant text turns are sent over the wire (tool calls and
their results are filtered out of what's returned and what's expected
back in). This keeps the request/response shape simple for any future
frontend, at the cost of the agent re-running retrieval on each turn
rather than reusing a prior tool result — an acceptable trade for a v1
thin API, not a hidden correctness problem: each new question is still
answered from a fresh, grounded search either way.

Run: uvicorn src.api:app --reload
Then: POST http://127.0.0.1:8000/ask with a JSON body like
    {"messages": [{"role": "user", "content": "Can I add a task?"}]}
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import build_agent

app = FastAPI(title="Codebase Intelligence Agent API")

# Built once at startup and reused across requests — rebuilding the
# vector store connection and LLM client on every call would be wasteful,
# and would also burn through Gemini's daily rate limit faster than
# necessary.
agent = build_agent()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    messages: list[Message]


class AskResponse(BaseModel):
    answer: str
    messages: list[Message]


def _to_langchain_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _from_langchain_messages(raw_messages: list) -> list[Message]:
    """Keep only human/ai turns with actual text, in the API's role
    naming (user/assistant).

    Tool calls and tool results are internal reasoning steps, not
    conversation turns a frontend needs to display or resend — dropping
    them keeps the wire format simple. An AI message whose only content
    is a tool-call request (no text yet) has empty `.text` and is
    dropped too, so it doesn't show up as a blank assistant turn.
    """
    role_map = {"human": "user", "ai": "assistant"}
    return [
        Message(role=role_map[m.type], content=m.text)
        for m in raw_messages
        if m.type in role_map and m.text.strip()
    ]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question, given the full conversation so far.

    The caller sends the entire message history (including their newest
    question) each time — this endpoint doesn't remember anything
    between requests on its own. Returns both the final answer text and
    the updated message list; the caller just persists what comes back
    and sends it straight back in on the next turn.
    """
    result = agent.invoke({"messages": _to_langchain_messages(request.messages)})
    response_messages = _from_langchain_messages(result["messages"])
    answer_text = response_messages[-1].content if response_messages else ""

    return AskResponse(answer=answer_text, messages=response_messages)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
