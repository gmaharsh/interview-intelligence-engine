"""Save extracted page content to the corpus (JSONL). Valid docs only in corpus; rejected logged separately."""

import json
from datetime import UTC, datetime
from pathlib import Path

from .search import SearchResult


def get_corpus_dir() -> Path:
    """Corpus directory: project root / data / corpus."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "corpus"


def load_corpus_docs(corpus_dir: Path | None = None) -> list[dict]:
    """Load all documents from corpus.jsonl. Returns list of doc dicts."""
    directory = corpus_dir or get_corpus_dir()
    path = directory / "corpus.jsonl"
    docs: list[dict] = []
    if not path.exists():
        return docs
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def write_corpus_docs(docs: list[dict], corpus_dir: Path | None = None) -> Path:
    """Overwrite corpus.jsonl with the given list of doc dicts. Use after cleaning."""
    directory = corpus_dir or get_corpus_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path


def get_corpus_urls(corpus_dir: Path | None = None) -> set[str]:
    """Return set of URLs already in corpus.jsonl. Use to skip duplicates on re-runs."""
    directory = corpus_dir or get_corpus_dir()
    path = directory / "corpus.jsonl"
    urls: set[str] = set()
    if not path.exists():
        return urls
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if isinstance(doc.get("url"), str):
                    urls.add(doc["url"])
            except json.JSONDecodeError:
                continue
    return urls


def save_document(
    result: SearchResult,
    content: str,
    corpus_dir: Path | None = None,
    *,
    source_type: str = "unknown",
    content_quality: float = 0.0,
    is_valid: bool = True,
    rejection_reason: str | None = None,
    existing_urls: set[str] | None = None,
) -> Path | None:
    """Append one document to corpus. If existing_urls is set and result.url is in it, skip and return None."""
    directory = corpus_dir or get_corpus_dir()
    path = directory / "corpus.jsonl"
    if existing_urls is not None and result.url in existing_urls:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    doc = {
        "url": result.url,
        "title": result.title,
        "company": result.company,
        "query": result.query,
        "score": result.score,
        "content": content,
        "source_type": source_type,
        "content_quality": round(content_quality, 4),
        "is_valid": is_valid,
        "rejection_reason": rejection_reason,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    if existing_urls is not None:
        existing_urls.add(result.url)
    return path


def log_rejected(
    result: SearchResult,
    content: str,
    rejection_reason: str,
    corpus_dir: Path | None = None,
    *,
    source_type: str = "unknown",
    content_quality: float = 0.0,
) -> Path:
    """Append a rejected document to rejected.jsonl for audit. Content may be truncated."""
    directory = corpus_dir or get_corpus_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rejected.jsonl"
    doc = {
        "url": result.url,
        "title": result.title,
        "company": result.company,
        "query": result.query,
        "score": result.score,
        "content": (content[:2000] + "...") if len(content) > 2000 else content,
        "source_type": source_type,
        "content_quality": round(content_quality, 4),
        "is_valid": False,
        "rejection_reason": rejection_reason,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path
