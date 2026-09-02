"""Calls Groq's chat completion API with retrieved course context."""

from __future__ import annotations

import re

from groq import Groq

from studybot.config import settings
from studybot.rag import RetrievedChunk, simplify_location

SYSTEM_PROMPT_TEMPLATE = """You are Study Bot, a helpful teaching assistant for \
__COURSE_NAME__. \
Answer the student's question using ONLY the course material context provided below. \
If the answer isn't in the context, say you don't have that information in the \
course materials and suggest they check with the instructor — do not make anything up.
- This applies per-item, not just per-question: if a student asks about several \
related things (e.g. "show the equation for X and Y") and the materials only cover \
some of them, answer the ones you have and explicitly say the rest aren't in the \
materials. Do NOT fill a gap with a generic, "standard", or textbook version of \
something just because it's common knowledge or you're confident it's correct — \
if it isn't in the retrieved context for this specific course, it doesn't go in the \
answer. A confident-sounding fabricated formula is worse than admitting a gap.
- When presenting an equation or formula that appears in the retrieved context, \
reproduce it EXACTLY as given — same notation, same level of detail. Do not \
"complete" or improve it with conventional notation that isn't there (e.g. don't \
add explicit summation bounds like i=1 to N if the source just has a bare Σ; \
don't add a subscript that isn't present). Exact reproduction matters more than \
looking like a textbook here — if the source's notation is unusual or minimal, \
that's what the student should see.

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
with $ for inline math and $$ for standalone/display equations — never bare square \
brackets, and never \\[ \\] or \\( \\). Write correct, complete LaTeX syntax (proper \
\\frac{}{} for fractions, _{} for subscripts, ^{} for exponents), not an \
approximation using semicolons or commas.
  Example of correctly formatted math: "The sample mean is $\\bar{x} = \\frac{\\sum x_i}{n}$. \
As a standalone equation: $$SD = \\sqrt{\\frac{\\sum (x_i - \\bar{x})^2}{n-1}}$$"
- Exception: if the student explicitly asks for the exact wording, to quote it, \
or to reproduce something "word for word", quote the retrieved text precisely and \
completely rather than paraphrasing or summarizing it — this is the instructor's own \
material, so reproducing it accurately for a student is expected, not something to \
avoid or shorten.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("__COURSE_NAME__", settings.course_name)

# A line containing only "[" (with optional whitespace) is the strict
# signal for "this bracket starts a standalone equation block" — but a
# markdown list bullet before it ("- [") breaks that strict match, since the
# line then contains "- [" rather than just "[". Tolerating a short leading
# prefix (whitespace, and optionally a list marker) catches that case too,
# while still requiring the bracket be immediately followed by a line break
# so an ordinary mid-sentence "[" is never touched.
_BARE_BRACKET_MATH = re.compile(
    r"^[ \t]*(?:[-*•]\s*)?\[[ \t]*$\n(.*?)\n^[ \t]*\][ \t]*$", re.MULTILINE | re.DOTALL
)

# The model has repeatedly inserted stray commas immediately inside LaTeX
# braces (e.g. "\frac{SS}{,n-1,}" instead of "\frac{SS}{n-1}") — a comma
# right after "{" or right before "}" is never intentional LaTeX, so this
# strips it. Deliberately narrow (only touches commas touching a brace) so
# it can't affect a legitimate comma elsewhere, like inside prose or a
# number such as "1,000".
_STRAY_BRACE_COMMA = re.compile(r"\{\s*,|,\s*\}")

# Once a math block's delimiters are correct, its CONTENT can still break
# rendering: marked.js (breaks:true) converts a literal newline into a <br>
# tag before KaTeX ever runs, which splits a multi-line equation (e.g. a
# chain of equalities across several lines) across separate DOM text nodes
# — KaTeX's delimiter matching can't find a closing "$$" that isn't in the
# same text node, so the whole block silently fails to render at all, even
# though the delimiters themselves are exactly correct. Collapsing internal
# newlines to spaces before the text ever reaches the browser sidesteps
# this entirely. Separately, the model has a persistent habit of writing
# "X ;=; Y" instead of "X = Y" — harmless to KaTeX's parser, but it renders
# as a literal, confusing semicolon in the output.
_MATH_BLOCK = re.compile(r"(\${1,2})(.*?)\1", re.DOTALL)

# The model sometimes wraps an equation in backticks (Markdown inline code)
# instead of $ / $$, which renders it as a literal, unstyled code span
# rather than math — and it can do this for ONE equation in a response while
# correctly using $$ for another, so this isn't a global formatting choice
# to reason about, just another delimiter variant to normalize. Only convert
# backtick content that actually looks like LaTeX (a backslash command, or a
# sub/superscript brace) so a genuine code snippet is never touched.
_BACKTICK_MATH = re.compile(r"`([^`\n]*(?:\\[a-zA-Z]+|[_^]\{)[^`\n]*)`")

# The model also wraps small inline equations in plain parentheses instead
# of $ — "(SD = 2.9700)", "(\bar{x}=5.0000)". This is riskier to normalize
# than backticks or brackets: parentheses are extremely common in ordinary
# prose, and something like "(s)" is a normal English plural marker
# ("student(s)") far more often than a stray variable reference. So this
# only converts parenthetical content that's unambiguously math: either it
# contains a LaTeX command (same signal as backticks), or its ENTIRE
# content is just "short token = number" with nothing else — a shape that
# essentially never occurs in ordinary prose. A bare symbol reference with
# no "=" and no backslash, like "(n-1)", is deliberately left alone rather
# than risk a false positive.
_PAREN_MATH = re.compile(
    r"\((\\[a-zA-Z]+[^()\n]*|[A-Za-z][A-Za-z0-9_^{}\\]{0,20}\s*=\s*-?[\d.]+)\)"
)


def _clean_math_content(latex: str) -> str:
    latex = re.sub(r"\s*\n\s*", " ", latex).strip()
    # The model scatters stray ";" around "=" inconsistently -- ";=;", ";=",
    # AND "=;" all show up across different responses, not just one fixed
    # pattern. Matching semicolons on EITHER side in one pass covers all of
    # them (and is a harmless no-op on a plain, already-clean "=").
    latex = re.sub(r"\s*;*\s*=\s*;*\s*", " = ", latex)
    # Same habit shows up as a dangling "," or ";" at the very end of an
    # expression (e.g. "SP/N," or "...N}} ;."). Strip it, but keep a real
    # trailing period if the model correctly ended the sentence with one.
    latex = re.sub(r"[;,]+(\s*\.?)\s*$", r"\1", latex)
    return re.sub(r"\s+", " ", latex).strip()


def _normalize_math_delimiters(text: str) -> str:
    text = _BACKTICK_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    text = _PAREN_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)
    text = _BARE_BRACKET_MATH.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    text = _STRAY_BRACE_COMMA.sub(lambda m: "{" if m.group(0).startswith("{") else "}", text)
    text = _MATH_BLOCK.sub(lambda m: f"{m.group(1)}{_clean_math_content(m.group(2))}{m.group(1)}", text)
    return text


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
    return _normalize_math_delimiters(response.choices[0].message.content.strip())
