"""FastAPI app: serves the /chat endpoint used by the frontend in docs/."""

from __future__ import annotations

import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from studybot.cache import answer_cache, rate_limiter
from studybot.config import settings
from studybot.llm import answer_question
from studybot.rag import RetrievedChunk, merge_unique, retriever

app = FastAPI(title="StudyBot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Prior turns of this conversation, oldest first. Optional — omitting it
    # (or sending an empty list) gets you the old stateless behavior.
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _simplify_location(location: str) -> str:
    """Collapses internal chunk-tracking detail (row/column numbers) into a
    label that actually means something to a student, e.g. "table 3, row 12"
    and "table 3, row 28" both just become "Table 3" — the specific row
    isn't meaningful to them, but which table is."""
    if location == "document text":
        return "General syllabus text"
    if location.startswith("document text — "):
        return location.removeprefix("document text — ")
    match = re.match(r"table (\d+)", location)
    if match:
        return f"Table {match.group(1)}"
    return location


def _build_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Groups sources by file, listing each file once with its distinct
    sections/tables comma-separated, instead of repeating the full filename
    for every single retrieved chunk."""
    grouped: dict[str, set[str]] = {}
    for c in chunks:
        grouped.setdefault(c.source, set()).add(_simplify_location(c.location))
    return [f"{source}: {', '.join(sorted(locs))}" for source, locs in sorted(grouped.items())]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Cap history server-side regardless of what the client sends, so a
    # long-running conversation can't grow the prompt (and Groq token usage)
    # without bound.
    history = req.history[-settings.max_history_messages :]

    # 1. Serve from cache — but only for the first question in a conversation.
    # Once there's history, the same question text can have a different
    # correct answer depending on what preceded it (e.g. "lay it out in
    # steps" means something different after a question about neurons vs.
    # one about extensions), so caching by question text alone would be
    # actively wrong here.
    if not history:
        cached_answer = answer_cache.get(question)
        if cached_answer is not None:
            return ChatResponse(answer=cached_answer, sources=[], cached=True)

    # 2. Protect the free Groq quota with a soft rate limit.
    if not rate_limiter.allow():
        wait = int(rate_limiter.seconds_until_available()) + 1
        raise HTTPException(
            status_code=429,
            detail=f"We're getting a lot of questions right now — please try again "
                   f"in about {wait} seconds.",
        )

    # 3. Retrieve relevant course material. Search twice when there's history:
    # once with just the new question (catches a clean topic switch, like
    # asking about exams right after asking about neurons), and once with
    # recent turns folded in (catches genuine follow-ups like "lay it out in
    # steps", where the question alone has no topic of its own). Merging the
    # two avoids having to guess in advance which situation we're in.
    try:
        chunks = retriever.retrieve(question)
        if history:
            recent_user_turns = [m.content for m in history if m.role == "user"][-2:]
            retrieval_query = " ".join([*recent_user_turns, question])
            contextual_chunks = retriever.retrieve(retrieval_query)
            chunks = merge_unique(chunks, contextual_chunks, limit=settings.top_k)
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - misconfiguration, missing deps, etc.
        raise HTTPException(
            status_code=503,
            detail="The course assistant isn't set up correctly yet. "
                   "Please let the instructor know.",
        ) from e

    # 4. Ask the LLM, grounded in the retrieved context and aware of the
    # conversation so far.
    try:
        answer = answer_question(
            question,
            chunks,
            history=[{"role": m.role, "content": m.content} for m in history],
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"The AI service had a problem answering that: {e}",
        ) from e

    if not history:
        answer_cache.set(question, answer)

    sources = _build_sources(chunks)
    return ChatResponse(answer=answer, sources=sources, cached=False)
