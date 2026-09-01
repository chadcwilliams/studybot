"""Retrieval: embed a question, fetch the most relevant chunks from the
local vector store (see store.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from studybot.config import settings
from studybot.store import VectorStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    source: str
    location: str
    score: float  # cosine similarity, higher = more relevant


class Retriever:
    """Lazily loads the embedding model and vector store on first use, so
    the FastAPI app starts fast and only pays the (one-time) model load
    cost when the first question actually comes in.
    """

    def __init__(self) -> None:
        self._model: "SentenceTransformer | None" = None
        self._store: VectorStore | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        if self._store is None:
            store = VectorStore(settings.index_dir)
            store.load()  # raises FileNotFoundError with a clear message if missing
            self._store = store

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self._ensure_loaded()
        top_k = top_k or settings.top_k

        query_embedding = self._model.encode([question])[0].tolist()
        results = self._store.query(query_embedding, top_k=top_k)

        return [
            RetrievedChunk(text=chunk.text, source=chunk.source, location=chunk.location, score=score)
            for chunk, score in results
        ]


retriever = Retriever()
