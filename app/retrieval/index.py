"""Ingestion: chunk the corpus, embed it, and store it two ways.

Chroma gets the vectors (for semantic search). `data/chunks.json` gets the
plain chunk records (for BM25, which cannot persist itself and is rebuilt
from this file at process start — Phase 2 spec). Both are rebuilt from
scratch on every run: re-ingesting after `docs/` or the chunking parameters
change must never leave stale chunks behind under old IDs.

Run:
    python -m app.retrieval.index
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

COLLECTION_NAME = "northbay_docs"

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    # {source_file_stem}::{chunk_index} — decided in Phase 2's spec so the
    # format is stable before any node (Phase 3) validates a citation against
    # it. Readable in a trace, carries its own provenance.
    id: str
    text: str
    source: str
    chunk_index: int


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Returns (source_stem, raw_text) pairs, sorted for a deterministic order."""
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")
    return [(p.stem, p.read_text(encoding="utf-8")) for p in paths]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy, paragraph-respecting chunking with a word-count overlap.

    A paragraph is never split mid-paragraph, even when that occasionally
    makes a chunk larger than `chunk_size` — the alternative would cut a
    sentence in half to hit a target that is already just a rough guideline,
    which is a worse trade.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current and len(current) + len(para_words) > chunk_size:
            chunks.append(" ".join(current))
            current = (current[-overlap:] if overlap else []) + para_words
        else:
            current.extend(para_words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunks(documents: list[tuple[str, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in documents:
        for i, piece in enumerate(chunk_text(text, settings.chunk_size_words, settings.chunk_overlap_words)):
            chunks.append(Chunk(id=f"{source}::{i}", text=piece, source=source, chunk_index=i))
    return chunks


def build_index(chunks: list[Chunk]) -> None:
    settings.ensure_dirs()

    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode([c.text for c in chunks], show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    # No embedding_function registered: embeddings are always supplied
    # explicitly, both here and at query time in hybrid.py, so there is one
    # embedding model used consistently rather than two potentially different
    # ones (Chroma's default vs. ours).
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )

    settings.chunks_file.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    documents = load_documents(settings.docs_dir)
    chunks = build_chunks(documents)
    build_index(chunks)

    print(f"{len(documents)} docs -> {len(chunks)} chunks indexed")
    print(f"Chroma store: {settings.chroma_dir}")
    print(f"Chunk records: {settings.chunks_file}")


if __name__ == "__main__":
    main()
