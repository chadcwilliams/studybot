"""
A minimal local vector store: no external vector database required.

We avoid chromadb here on purpose. Its `chroma-hnswlib` dependency is a
compiled C++ extension that doesn't ship a prebuilt wheel for every Python
version/OS combination (notably: no Windows wheel for Python 3.13 at time of
writing), which makes it a fragile install on some machines.

For a single course's worth of material — a few hundred to a few thousand
text chunks — a plain numpy array searched by cosine similarity is more than
fast enough, and numpy has reliable wheels everywhere. This trades a small
amount of "roll your own" code for a much simpler, more portable dependency
footprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class StoredChunk:
    text: str
    source: str
    location: str


class VectorStore:
    """Persists embeddings + their source chunks to disk, and answers
    nearest-neighbor queries by cosine similarity.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vectors: np.ndarray | None = None
        self._chunks: list[StoredChunk] = []

    @property
    def _vectors_path(self) -> Path:
        return self.path / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.path / "chunks.json"

    def exists(self) -> bool:
        return self._vectors_path.exists() and self._meta_path.exists()

    def save(self, vectors: list[list[float]], chunks: list[StoredChunk]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        arr = np.array(vectors, dtype=np.float32)
        arr = _normalize_rows(arr)
        np.save(self._vectors_path, arr)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f)

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(
                f"No course material index found at {self.path}. "
                "Add files to data/materials/ and run `python -m studybot.ingest` first."
            )
        self._vectors = np.load(self._vectors_path)
        with open(self._meta_path, encoding="utf-8") as f:
            self._chunks = [StoredChunk(**d) for d in json.load(f)]

    def query(self, query_vector: list[float], top_k: int) -> list[tuple[StoredChunk, float]]:
        if self._vectors is None:
            self.load()
        q = _normalize_rows(np.array([query_vector], dtype=np.float32))[0]
        # Both sides are unit vectors, so the dot product IS cosine similarity.
        scores = self._vectors @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
