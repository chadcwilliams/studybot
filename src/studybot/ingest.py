"""
Parse course materials (.pptx, .pdf, .docx) in data/materials/, split them
into overlapping text chunks, embed them locally, and persist them to a
local vector store on disk (see store.py).

Run this once after adding/updating files:

    python -m studybot.ingest

Re-run it any time you add new slides or update the syllabus — it rebuilds
the index from scratch each time, so it's always safe to re-run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument

from studybot.config import settings
from studybot.store import StoredChunk, VectorStore

# fastembed is imported lazily inside run() below, so the text-extraction
# functions above can be unit-tested or reused without downloading models.


@dataclass
class Chunk:
    text: str
    source: str  # filename
    location: str  # e.g. "slide 4" or "page 2"


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------

def extract_pptx(path: Path) -> list[tuple[str, str]]:
    """Returns list of (text, location) per slide, including speaker notes."""
    prs = Presentation(path)
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells)
                    if row_text.strip(" |"):
                        parts.append(row_text)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("Speaker notes: " + slide.notes_slide.notes_text_frame.text.strip())
        if parts:
            out.append(("\n".join(parts), f"slide {i}"))
    return out


def extract_pdf(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append((text, f"page {i}"))
    return out


def extract_docx(path: Path) -> list[tuple[str, str]]:
    doc = DocxDocument(path)
    out = []

    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if full_text.strip():
        out.append((full_text, "document text"))

    # Tables (grading breakdowns, exam schedules, etc.) live in doc.tables,
    # completely separate from doc.paragraphs — easy to silently miss.
    for i, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                rows.append(row_text)
        if rows:
            out.append(("\n".join(rows), f"table {i}"))

    return out


EXTRACTORS = {
    ".pptx": extract_pptx,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-based sliding-window chunker."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(materials_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = [f for f in materials_dir.rglob("*") if f.suffix.lower() in EXTRACTORS]

    if not files:
        print(f"No supported files found in {materials_dir}. "
              f"Add .pptx / .pdf / .docx files there and re-run.")
        return chunks

    for f in files:
        extractor = EXTRACTORS[f.suffix.lower()]
        try:
            sections = extractor(f)
        except Exception as e:  # noqa: BLE001 - keep ingesting other files
            print(f"  ! Failed to parse {f.name}: {e}", file=sys.stderr)
            continue

        for text, location in sections:
            for piece in chunk_text(text, chunk_size, chunk_overlap):
                if piece.strip():
                    chunks.append(Chunk(text=piece, source=f.name, location=location))

        print(f"  parsed {f.name}: {len(sections)} section(s)")

    return chunks


# --------------------------------------------------------------------------
# Embedding + storage
# --------------------------------------------------------------------------

def run() -> None:
    from fastembed import TextEmbedding

    print(f"Reading materials from: {settings.materials_dir}")
    chunks = build_chunks(settings.materials_dir, settings.chunk_size, settings.chunk_overlap)

    if not chunks:
        print("Nothing to index. Exiting.")
        return

    print(f"Built {len(chunks)} chunks. Loading embedding model "
          f"'{settings.embedding_model}' (first run downloads it)...")
    model = TextEmbedding(settings.embedding_model)

    print("Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = [e.tolist() for e in model.embed(texts)]

    store = VectorStore(settings.index_dir)
    stored_chunks = [StoredChunk(text=c.text, source=c.source, location=c.location) for c in chunks]
    store.save(embeddings, stored_chunks)

    print(f"Done. Indexed {len(chunks)} chunks at {settings.index_dir}")


if __name__ == "__main__":
    run()
