"""CorpusManager: clean existing corpus and fetch new documents."""

from datetime import UTC, datetime
from pathlib import Path

from .corpus import (
    get_corpus_dir,
    get_corpus_urls,
    load_corpus_docs,
    log_rejected,
    save_document,
    write_corpus_docs,
)
from .quality import (
    UNSUPPORTED_SOURCE_PATTERNS,
    clean_content,
    infer_source_type,
    is_valid_document,
    score_quality,
)
from .search import SearchResult, get_tavily_client, search_all_companies, crawl_url, should_skip_url


class CorpusManager:
    """
    Clean the existing dataset and/or fetch new documents.
    - clean_existing(): re-clean content, drop invalid docs, rewrite corpus.
    - fetch_new(): run search → crawl → validate → save for URLs not in corpus.
    """

    def __init__(self, corpus_dir: Path | None = None):
        self.corpus_dir = corpus_dir or get_corpus_dir()

    def clean_existing(self, verbose: bool = True) -> dict:
        """
        Re-run cleaning and validation on every document in the corpus.
        Keeps only valid docs; overwrites corpus.jsonl with cleaned content and updated metadata.
        Returns summary: kept, removed, by_reason.
        """
        docs = load_corpus_docs(self.corpus_dir)
        if not docs:
            if verbose:
                print("Corpus is empty; nothing to clean.")
            return {"kept": 0, "removed": 0, "by_reason": {}}

        kept: list[dict] = []
        by_reason: dict[str, int] = {}
        for doc in docs:
            url = doc.get("url", "")
            title = doc.get("title", "")
            if any(p in url.lower() for p in UNSUPPORTED_SOURCE_PATTERNS):
                by_reason["unsupported_source_type"] = by_reason.get("unsupported_source_type", 0) + 1
                continue
            raw_content = (doc.get("content") or "").strip()
            content = clean_content(raw_content)
            valid, rejection_reason = is_valid_document(content, url, title)
            if not valid:
                reason = rejection_reason or "unknown"
                by_reason[reason] = by_reason.get(reason, 0) + 1
                continue
            quality = score_quality(content, url, title)
            source_type = infer_source_type(url, title, content)
            doc["content"] = content
            doc["source_type"] = source_type
            doc["content_quality"] = round(quality, 4)
            doc["is_valid"] = True
            doc["rejection_reason"] = None
            doc["fetched_at"] = datetime.now(UTC).isoformat()
            kept.append(doc)

        removed = len(docs) - len(kept)
        write_corpus_docs(kept, self.corpus_dir)

        if verbose:
            print(f"Cleaned corpus: kept {len(kept)}, removed {removed}")
            if by_reason:
                for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
                    print(f"  removed ({reason}): {count}")

        return {"kept": len(kept), "removed": removed, "by_reason": by_reason}

    def fetch_new(self, verbose: bool = True) -> dict:
        """
        Run Tavily search, crawl new URLs, validate, and append to corpus.
        Skips URLs already in corpus and excluded domains.
        Returns summary: saved, rejected, skipped_already, skipped_domain, no_content.
        """
        client = get_tavily_client()
        results = search_all_companies()
        corpus_urls = get_corpus_urls(self.corpus_dir)
        saved = 0
        rejected = 0
        skipped_already = 0
        skipped_domain = 0
        no_content = 0

        if verbose:
            print(f"Found {len(results)} URLs from search ({len(corpus_urls)} already in corpus)\n")

        for r in results:
            if r.url in corpus_urls:
                skipped_already += 1
                continue
            if should_skip_url(r.url):
                skipped_domain += 1
                continue
            raw = crawl_url(client, r.url)
            if not raw:
                no_content += 1
                if verbose:
                    print(f"  [no content] {r.url[:60]}...")
                continue
            content = clean_content(raw)
            valid, rejection_reason = is_valid_document(content, r.url, r.title)
            if not valid:
                rejected += 1
                quality = score_quality(content, r.url, r.title)
                source_type = infer_source_type(r.url, r.title, content)
                log_rejected(
                    r,
                    content,
                    rejection_reason or "unknown",
                    corpus_dir=self.corpus_dir,
                    source_type=source_type,
                    content_quality=quality,
                )
                if verbose:
                    print(f"  [rejected {rejection_reason}] {r.title[:50]}...")
                continue
            quality = score_quality(content, r.url, r.title)
            source_type = infer_source_type(r.url, r.title, content)
            path = save_document(
                r,
                content,
                corpus_dir=self.corpus_dir,
                source_type=source_type,
                content_quality=quality,
                is_valid=True,
                rejection_reason=None,
                existing_urls=corpus_urls,
            )
            if path:
                saved += 1
                if verbose:
                    print(f"  [saved] {r.title[:50]}... ({len(content)} chars, quality={quality:.2f})")

        if verbose:
            print(f"\nFetch summary: saved={saved}, rejected={rejected}, skipped_already={skipped_already}, skipped_domain={skipped_domain}, no_content={no_content}")

        return {
            "saved": saved,
            "rejected": rejected,
            "skipped_already": skipped_already,
            "skipped_domain": skipped_domain,
            "no_content": no_content,
        }

    def run(
        self,
        clean_first: bool = True,
        fetch_after: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Optionally clean the existing corpus, then optionally fetch new documents.
        Returns combined summary dict.
        """
        summary: dict = {}
        if clean_first:
            summary["clean"] = self.clean_existing(verbose=verbose)
        if fetch_after:
            summary["fetch"] = self.fetch_new(verbose=verbose)
        return summary


def main() -> None:
    """CLI: python -m crawler.dataset [clean|fetch|run]. Default: run (clean then fetch)."""
    import sys
    manager = CorpusManager()
    mode = (sys.argv[1] if len(sys.argv) > 1 else "run").lower()
    if mode == "clean":
        manager.clean_existing(verbose=True)
    elif mode == "fetch":
        manager.fetch_new(verbose=True)
    elif mode == "run":
        manager.run(clean_first=True, fetch_after=True, verbose=True)
    else:
        print("Usage: python -m crawler.dataset [clean|fetch|run]")
        print("  clean  - re-clean corpus, drop invalid docs, rewrite corpus.jsonl")
        print("  fetch  - fetch new URLs from Tavily and append valid docs")
        print("  run    - clean first, then fetch (default)")
        sys.exit(1)


if __name__ == "__main__":
    main()
