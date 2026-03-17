from __future__ import annotations

"""Chunking pipeline: load corpus documents and emit hierarchical chunks."""

from pathlib import Path
from typing import Iterable, Dict, Any, List
import json

from crawler.corpus import get_corpus_dir, load_corpus_docs
from .strategies import Chunk, build_chunker_registry, FallbackChunker


class ChunkingPipeline:
    """Load corpus.jsonl, apply source-aware chunking, and yield chunks."""

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self.corpus_dir = corpus_dir or get_corpus_dir()
        self._registry = build_chunker_registry()
        self._fallback = FallbackChunker()

    def _iter_docs(self) -> Iterable[Dict[str, Any]]:
        docs = load_corpus_docs(self.corpus_dir)
        for doc in docs:
            if not doc.get("is_valid", True):
                continue
            yield doc

    def iter_chunks(self) -> Iterable[Chunk]:
        for doc in self._iter_docs():
            source_type = doc.get("source_type") or "unknown"
            chunker = self._registry.get(source_type, self._fallback)
            for chunk in chunker.chunk(doc):
                yield chunk


def write_chunks_jsonl(chunks: Iterable[Chunk], output_path: Path) -> int:
    """Write chunks to a JSONL file. Returns count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            obj = {
                "chunk_id": ch.chunk_id,
                "parent_section_id": ch.parent_section_id,
                "section_title": ch.section_title,
                "chunk_text": ch.chunk_text,
                "chunk_type": ch.chunk_type,
                "token_count": ch.token_count,
                **(ch.metadata or {}),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count

