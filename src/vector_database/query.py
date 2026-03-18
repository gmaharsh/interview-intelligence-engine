from __future__ import annotations

"""Production-oriented query interface for the interview vector store.

This module encapsulates:
- Vector store construction (Qdrant + embeddings).
- A high-level `search_chunks` function with common filters.
- A simple CLI for manual queries during development.
"""

import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from operational.constants import VECTOR_DATABASE_COLLECTION_NAME, EMBEDDING_MODEL


load_dotenv()


def _get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY") or None
    if not url:
        raise RuntimeError("QDRANT_URL is not set in environment.")
    return QdrantClient(url=url, api_key=api_key)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> QdrantVectorStore:
    """Return a QdrantVectorStore attached to the production collection."""
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    return QdrantVectorStore(
        client=client,
        collection_name=VECTOR_DATABASE_COLLECTION_NAME,
        embedding=embeddings,
    )


def _build_filter(
    company: Optional[str],
    source_type: Optional[str],
    only_official: bool,
) -> Optional[Filter]:
    """Build a Qdrant Filter from optional metadata constraints."""
    must = []
    if company:
        must.append(FieldCondition(key="metadata.company", match=MatchValue(value=company)))
    if source_type:
        must.append(FieldCondition(key="metadata.source_type", match=MatchValue(value=source_type)))
    if only_official:
        must.append(FieldCondition(key="metadata.is_official", match=MatchValue(value=True)))
    return Filter(must=must) if must else None


def search_chunks(
    query: str,
    *,
    top_k: int = 8,
    company: Optional[str] = None,
    source_type: Optional[str] = None,
    only_official: bool = False,
) -> List[Document]:
    """Semantic search; returns Documents without scores."""
    vector_store = get_vector_store()
    qdrant_filter = _build_filter(company, source_type, only_official)
    return vector_store.similarity_search(query, k=top_k, filter=qdrant_filter)


def search_chunks_with_score(
    query: str,
    *,
    top_k: int = 8,
    company: Optional[str] = None,
    source_type: Optional[str] = None,
    only_official: bool = False,
) -> List[Tuple[Document, float]]:
    """Semantic search; returns (Document, cosine_score) pairs, highest score first."""
    from vector_database.query_rewrite import _is_navigation_noise

    vector_store = get_vector_store()
    qdrant_filter = _build_filter(company, source_type, only_official)
    # Fetch extra candidates so filtering doesn't leave us short.
    fetch_k = max(top_k * 2, 16)
    raw = vector_store.similarity_search_with_score(query, k=fetch_k, filter=qdrant_filter)
    filtered = [(doc, score) for doc, score in raw if not _is_navigation_noise(doc)]
    return filtered[:top_k]


def main() -> None:
    """CLI: quick manual query against the vector store."""
    import argparse

    parser = argparse.ArgumentParser(description="Query the interview vector store.")
    parser.add_argument("query", type=str, help="Natural language query.")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks to return.")
    parser.add_argument("--company", type=str, default=None, help="Filter by company.")
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        help="Filter by source_type (e.g. official_company_page, social).",
    )
    parser.add_argument(
        "--only-official",
        action="store_true",
        help="Restrict to official company sources.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "Enable query rewriting: Groq-based paraphrase expansion + HyDE + "
            "RRF deduplication. Requires GROQ_API_KEY in environment."
        ),
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="(With --rewrite) Disable LLM paraphrase expansion.",
    )
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="(With --rewrite) Disable HyDE hypothetical answer generation.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Generate a synthesised answer from retrieved chunks using Groq. "
            "Automatically enables --rewrite. Requires GROQ_API_KEY."
        ),
    )
    args = parser.parse_args()

    # --generate implies --rewrite (need the best possible retrieval for generation)
    if args.generate:
        args.rewrite = True

    if args.rewrite:
        from vector_database.query_rewrite import rewrite_and_search

        print("[INFO] Query rewriting enabled (expansion + HyDE + RRF).")
        results = rewrite_and_search(
            args.query,
            top_k=args.k,
            use_expansion=not args.no_expansion,
            use_hyde=not args.no_hyde,
            override_company=args.company,
            override_source_type=args.source_type,
            only_official=bool(args.only_official),
        )
        score_label = "rrf"
    else:
        results = search_chunks_with_score(
            args.query,
            top_k=args.k,
            company=args.company,
            source_type=args.source_type,
            only_official=bool(args.only_official),
        )
        score_label = "cosine"

    if args.generate:
        from llm.generate import generate_answer

        print("\n[INFO] Generating answer from retrieved chunks...\n")
        result = generate_answer(args.query, results)
        print(result)
        return

    for idx, (doc, score) in enumerate(results, start=1):
        print(f"\n--- Result {idx} ({score_label}: {score:.4f}) ---")
        print("Source:", doc.metadata.get("source_type"), "| Company:", doc.metadata.get("company"))
        print("Title:", doc.metadata.get("title"))
        print("Section:", doc.metadata.get("section_title"))
        print("URL:", doc.metadata.get("url"))
        print("Content:\n", doc.page_content[:600])


if __name__ == "__main__":
    main()

