"""FastAPI app: serves the /chat endpoint used by the frontend in docs/."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from studybot.cache import answer_cache, rate_limiter
from studybot.config import settings
from studybot.llm import answer_question
from studybot.rag import retriever

app = FastAPI(title="StudyBot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Serve from cache if we've seen this (or a near-identical) question.
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

    # 3. Retrieve relevant course material.
    try:
        chunks = retriever.retrieve(question)
    except (RuntimeError, FileNotFoundError) as e:
         raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - misconfiguration, missing deps, etc.
        raise HTTPException(
            status_code=503,
            detail="The course assistant isn't set up correctly yet. "
                   "Please let the instructor know.",
        ) from e

    # 4. Ask the LLM, grounded in the retrieved context.
    try:
        answer = answer_question(question, chunks)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"The AI service had a problem answering that: {e}",
        ) from e

    answer_cache.set(question, answer)

    sources = sorted({f"{c.source} ({c.location})" for c in chunks})
    return ChatResponse(answer=answer, sources=sources, cached=False)
