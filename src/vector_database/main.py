from __future__ import annotations

"""Vector database setup and chunk ingestion."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from operational.constants import VECTOR_DATABASE_COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client from environment variables."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise RuntimeError("QDRANT_URL is not set in environment.")

    return QdrantClient(url=url, api_key=api_key)


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Return the embedding model used for chunks."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """
    Create the collection if it does not exist.
    If it exists, validate schema compatibility.
    """
    collections = client.get_collections()
    names = {c.name for c in collections.collections}

    if VECTOR_DATABASE_COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=VECTOR_DATABASE_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    info = client.get_collection(VECTOR_DATABASE_COLLECTION_NAME)
    config = info.config.params.vectors

    # Handle unnamed dense vector config
    existing_size = config.size
    existing_distance = config.distance

    if existing_size != vector_size:
        raise RuntimeError(
            f"Collection '{VECTOR_DATABASE_COLLECTION_NAME}' exists with vector size "
            f"{existing_size}, expected {vector_size}."
        )

    if existing_distance != Distance.COSINE:
        raise RuntimeError(
            f"Collection '{VECTOR_DATABASE_COLLECTION_NAME}' exists with distance "
            f"{existing_distance}, expected {Distance.COSINE}."
        )


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create keyword/bool payload indexes needed for filtered search.

    langchain_qdrant nests all metadata under a 'metadata' key in the payload,
    so filter fields must be prefixed with 'metadata.'.
    """
    keyword_fields = [
        "metadata.company",
        "metadata.source_type",
        "metadata.section_title",
        "metadata.chunk_type",
    ]
    bool_fields = ["metadata.is_official"]

    for field in keyword_fields:
        client.create_payload_index(
            collection_name=VECTOR_DATABASE_COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    for field in bool_fields:
        client.create_payload_index(
            collection_name=VECTOR_DATABASE_COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.BOOL,
        )

    print(f"Payload indexes ensured for collection '{VECTOR_DATABASE_COLLECTION_NAME}'.")


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load all chunks from a JSONL file and report malformed lines."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: List[Dict[str, Any]] = []
    malformed_count = 0

    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    chunks.append(obj)
                else:
                    malformed_count += 1
                    print(f"[WARN] Line {line_num}: JSON is not an object, skipping.")
            except json.JSONDecodeError as exc:
                malformed_count += 1
                print(f"[WARN] Line {line_num}: malformed JSON, skipping. Error: {exc}")

    if malformed_count:
        print(f"[WARN] Skipped {malformed_count} malformed lines from {path}")

    return chunks


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep metadata Qdrant-friendly.
    Converts unsupported values to strings where needed.
    """
    sanitized: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [
                item if isinstance(item, (str, int, float, bool)) else str(item)
                for item in value
            ]
        else:
            sanitized[key] = str(value)

    return sanitized


def build_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Return a stable chunk ID.
    Prefer provided chunk_id, otherwise derive one deterministically.
    """
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    raw = {
        "url": chunk.get("url"),
        "section_title": chunk.get("section_title"),
        "parent_section_id": chunk.get("parent_section_id"),
        "chunk_text": chunk.get("chunk_text", ""),
    }
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest


def ingest_chunks(chunks_path: Path | None = None, batch_size: int = 8) -> None:
    """
    Ingest chunks into Qdrant.

    - Reads chunks JSONL
    - Embeds chunk_text
    - Upserts into Qdrant with metadata payload
    """
    from crawler.corpus import get_corpus_dir

    corpus_dir = get_corpus_dir()
    chunks_path = chunks_path or (corpus_dir / "chunks.jsonl")

    chunks = load_chunks(chunks_path)
    if not chunks:
        print(f"No chunks found at {chunks_path}")
        return

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    embeddings = get_embeddings_model()
    sample_vec = embeddings.embed_query("dimension probe")
    vector_size = len(sample_vec)

    client = get_qdrant_client()
    ensure_collection(client, vector_size=vector_size)
    ensure_payload_indexes(client)
    print(f"Collection '{VECTOR_DATABASE_COLLECTION_NAME}' is ready.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=VECTOR_DATABASE_COLLECTION_NAME,
        embedding=embeddings,
    )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    total_ingested = 0
    skipped_empty = 0

    def flush_batch(max_retries: int = 4, backoff_base: float = 2.0) -> None:
        nonlocal texts, metadatas, ids, total_ingested

        if not texts:
            return

        batch_texts = texts[:]
        batch_metadatas = metadatas[:]
        batch_ids = ids[:]
        texts.clear()
        metadatas.clear()
        ids.clear()

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
                total_ingested += len(batch_texts)
                print(f"[INFO] Ingested batch of {len(batch_texts)} chunks. Total: {total_ingested}")
                return
            except Exception as exc:
                last_exc = exc
                wait = backoff_base ** attempt
                print(
                    f"[WARN] Attempt {attempt}/{max_retries} failed for batch "
                    f"(first ID: {batch_ids[0] if batch_ids else 'N/A'}). "
                    f"Retrying in {wait:.0f}s... Error: {exc}"
                )
                time.sleep(wait)

        print(f"[ERROR] Batch permanently failed after {max_retries} retries. Skipping.")
        raise RuntimeError(f"Batch ingestion failed after {max_retries} retries.") from last_exc

    for ch in chunks:
        text = (ch.get("chunk_text") or "").strip()
        if not text:
            skipped_empty += 1
            continue

        metadata = sanitize_metadata({k: v for k, v in ch.items() if k != "chunk_text"})
        chunk_id = build_chunk_id(ch)

        texts.append(text)
        metadatas.append(metadata)
        ids.append(chunk_id)

        if len(texts) >= batch_size:
            try:
                flush_batch()
            except RuntimeError:
                pass  # already logged; continue with next batch

    try:
        flush_batch()
    except RuntimeError:
        pass

    print(
        f"Finished ingestion into '{VECTOR_DATABASE_COLLECTION_NAME}'. "
        f"Ingested={total_ingested}, Skipped empty={skipped_empty}"
    )


def main() -> None:
    """CLI: ingest all chunks into the vector store."""
    ingest_chunks()


if __name__ == "__main__":
    main()