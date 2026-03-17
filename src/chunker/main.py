from __future__ import annotations

"""CLI entry point for hierarchical, source-aware chunking.

Usage:
    python -m chunker.main
"""

from pathlib import Path
import sys

from .pipeline import ChunkingPipeline, write_chunks_jsonl


def main() -> None:
    """Run chunking over corpus.jsonl and write chunks.jsonl."""
    pipeline = ChunkingPipeline()
    corpus_dir = pipeline.corpus_dir
    output_path = corpus_dir / "chunks.jsonl"

    chunks = pipeline.iter_chunks()
    count = write_chunks_jsonl(chunks, output_path)

    print(f"Wrote {count} chunks to {output_path}")


if __name__ == "__main__":
    main()
