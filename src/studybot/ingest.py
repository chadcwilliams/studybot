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
    # separate from doc.paragraphs. We chunk them row-by-row — each row
    # paired with its header row — rather than joining the whole table into
    # one blob. A long table joined into one string can get sliced by the
    # generic character-based chunker at an arbitrary point, severing a fact
    # (e.g. "Exam 1" from its date "Oct 15") across two separate chunks.
    # Row-level chunks are small enough to never get split, and keeping the
    # header attached preserves what each column means.
    for ti, table in enumerate(doc.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            continue

        if len(rows) == 1:
            # A single-row table is usually several side-by-side categories
            # (e.g. "Exams: 45%" | "Article Summary: 45%" | "Participation:
            # 2%") stuffed into one row of cells. Joining them into one
            # chunk lets facts from one category bleed into another when the
            # model reads it — keep each column as its own chunk instead.
            for ci, cell_text in enumerate(rows[0], start=1):
                if cell_text.strip():
                    out.append((cell_text, f"table {ti}, column {ci}"))
            continue

        header = rows[0]
        header_text = " | ".join(header)
        for ri, row in enumerate(rows[1:], start=2):
            row_text = " | ".join(row)
            if not row_text.strip(" |"):
                continue
            out.append((f"{header_text}\n{row_text}", f"table {ti}, row {ri}"))

        # A long schedule table with many near-identical rows (same columns,
        # slightly different dates/topics) is hard for embedding search to
        # pick out of — a row like "Summary 1" differs from its 27 neighbors
        # by one word, which is too weak a signal to reliably rank near the
        # top for a question like "when are the assignments due". If a
        # column's header suggests it marks deadlines, build one consolidated
        # chunk listing every non-empty entry in that column together — a
        # single dense, unambiguous chunk beats hoping semantic search finds
        # all the scattered needle rows on its own.
        due_col = next(
            (i for i, h in enumerate(header) if any(kw in h.lower() for kw in ("due", "deadline"))),
            None,
        )
        if due_col is not None:
            entries = []
            for row in rows[1:]:
                value = row[due_col].strip()
                if not value:
                    continue
                context = ", ".join(
                    f"{header[i]}: {row[i]}"
                    for i in range(len(row))
                    if i != due_col and row[i].strip()
                )
                entries.append(f"{value} — {context}")
            if entries:
                due_header = header[due_col]
                summary = f"{due_header}:\n" + "\n".join(entries)
                out.append((summary, f"table {ti} — {due_header.strip()} (summary)"))

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
