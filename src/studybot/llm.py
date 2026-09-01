"""Calls Groq's chat completion API with retrieved course context."""

from __future__ import annotations

from groq import Groq

from studybot.config import settings
from studybot.rag import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful teaching assistant for this course. \
Answer the student's question using ONLY the course material context provided below. \
If the answer isn't in the context, say you don't have that information in the \
course materials and suggest they check with the instructor — do not make anything up. \
Keep answers concise and clear. When useful, mention which slide or section the \
information came from.
"""


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
    return Groq(api_key=settings.groq_api_key)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant course material was found for this question.)"
    parts = []
    for c in chunks:
        parts.append(f"[{c.source}, {c.location}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _build_context(chunks)
    user_prompt = f"Course material context:\n\n{context}\n\nStudent question: {question}"

    client = _client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
