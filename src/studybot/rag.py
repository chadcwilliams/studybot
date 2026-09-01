"""Retrieval: embed a question, fetch the most relevant chunks from the
local vector store (see store.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from studybot.config import settings
from studybot.store import VectorStore

if TYPE_CHECKING:
    from fastembed import TextEmbedding


@dataclass
class RetrievedChunk:
    text: str
    source: str
    location: str
    score: float  # cosine similarity, higher = more relevant


_term_aliases_cache: dict[str, list[str]] | None = None


def _load_term_aliases() -> dict[str, list[str]]:
    """Loads data/term_aliases.json if present — an optional, per-course
    file mapping a term to synonyms actually used in that course's
    materials (e.g. {"exam": ["midterm"]} for a course that calls its exams
    "midterms"). Embedding search only finds text that's semantically close
    to the query; if a course uses different vocabulary than students do,
    no amount of prompt tuning fixes that after the fact — the right chunk
    just never gets retrieved. This lets each course correct for its own
    known vocabulary gaps without touching any code.
    """
    global _term_aliases_cache
    if _term_aliases_cache is not None:
        return _term_aliases_cache
    path = settings.term_aliases_path
    if not path.exists():
        _term_aliases_cache = {}
        return _term_aliases_cache
    try:
        with open(path, encoding="utf-8") as f:
            _term_aliases_cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _term_aliases_cache = {}
    return _term_aliases_cache


def _expand_query(text: str) -> str:
    aliases = _load_term_aliases()
    if not aliases:
        return text
    lower = text.lower()
    extra_terms = [syn for term, syns in aliases.items() if term.lower() in lower for syn in syns]
    return f"{text} {' '.join(extra_terms)}" if extra_terms else text


class Retriever:
    """Lazily loads the embedding model and vector store on first use, so
    the FastAPI app starts fast and only pays the (one-time) model load
    cost when the first question actually comes in.
    """

    def __init__(self) -> None:
        self._model: "TextEmbedding | None" = None
        self._store: VectorStore | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(settings.embedding_model)
        if self._store is None:
            store = VectorStore(settings.index_dir)
            store.load()  # raises FileNotFoundError with a clear message if missing
            self._store = store

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self._ensure_loaded()
        top_k = top_k or settings.top_k

        # query_embed() (vs. embed()) applies the model's recommended query-side
        # instruction/prefix internally, which matters for asymmetric retrieval
        # models like BGE — queries and passages are embedded slightly differently.
        query_text = _expand_query(question)
        query_embedding = next(iter(self._model.query_embed(query_text))).tolist()
        results = self._store.query(query_embedding, top_k=top_k)

        return [
            RetrievedChunk(text=chunk.text, source=chunk.source, location=chunk.location, score=score)
            for chunk, score in results
        ]


retriever = Retriever()


def merge_unique(*chunk_lists: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    """Combines multiple retrieval passes (e.g. a standalone-question search
    and a history-augmented search), dropping duplicates and keeping each
    chunk's best score, then returns the top `limit` overall.
    """
    best: dict[tuple[str, str], RetrievedChunk] = {}
    for chunks in chunk_lists:
        for c in chunks:
            key = (c.source, c.location)
            if key not in best or c.score > best[key].score:
                best[key] = c
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]
