"""Calls Groq's chat completion API with retrieved course context."""

from __future__ import annotations

from groq import Groq

from studybot.config import settings
from studybot.rag import RetrievedChunk, simplify_location

SYSTEM_PROMPT_TEMPLATE = """You are Study Bot, a helpful teaching assistant for \
__COURSE_NAME__. \
Answer the student's question using ONLY the course material context provided below. \
If the answer isn't in the context, say you don't have that information in the \
course materials and suggest they check with the instructor — do not make anything up.

Formatting:
- Your answer is rendered as Markdown in a chat bubble, so use it purposefully, \
not decoratively.
- For a simple one-fact answer (a date, a percentage, a single rule), just write \
one or two plain sentences. Don't add a heading or table for a single fact.
- Use a Markdown table only when the student is asking about genuinely tabular \
information (a full schedule, a multi-item grading breakdown, comparing several items).
- Use a bullet list when there are several distinct items to enumerate, not for \
a single point.
- Use **bold** sparingly, only for the specific value that answers the question \
(a date, a number, a name) — not for restating the question as a heading.
- Keep the whole answer short. Mention which part of the course material it came \
from in one short trailing note, not as a heading.
- Never mention internal identifiers like row numbers, column numbers, or chunk \
labels (e.g. "row 28", "table 4 column 2") in your answer — a student has no idea \
what those refer to. If you want to reference where something came from, describe \
it in plain terms instead (e.g. "the course schedule", "the grading breakdown").
- For mathematical notation (equations, formulas, statistical notation), use LaTeX \
with $ for inline math and $$ for standalone/display equations — do not use square \
brackets for this. Write correct, complete LaTeX syntax (e.g. proper \\frac{}{} for \
fractions, _{} for subscripts, ^{} for exponents) rather than approximating it with \
punctuation.
- Exception: if the student explicitly asks for the exact wording, to quote it, \
or to reproduce something "word for word", quote the retrieved text precisely and \
completely rather than paraphrasing or summarizing it — this is the instructor's own \
material, so reproducing it accurately for a student is expected, not something to \
avoid or shorten.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("__COURSE_NAME__", settings.course_name)


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
        parts.append(f"[{c.source} — {simplify_location(c.location)}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> str:
    context = _build_context(chunks)
    user_prompt = f"Course material context:\n\n{context}\n\nStudent question: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Prior turns give the model conversational continuity (so "lay it
        # out in steps" is understood in relation to what was just asked),
        # without re-injecting a full context block for every past turn.
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    client = _client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()
