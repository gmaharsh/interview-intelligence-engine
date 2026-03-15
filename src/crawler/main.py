"""Crawler entrypoint: run Tavily search, extract content, clean, validate, save to corpus."""

from .corpus import get_corpus_dir, get_corpus_urls, log_rejected, save_document
from .quality import clean_content, infer_source_type, is_valid_document, score_quality
from .search import get_tavily_client, search_all_companies, crawl_url, should_skip_url


def main() -> None:
    client = get_tavily_client()
    results = search_all_companies()
    corpus_dir = get_corpus_dir()
    corpus_urls = get_corpus_urls(corpus_dir)
    print(f"Found {len(results)} URLs from Tavily search")
    print(f"Corpus: {corpus_dir / 'corpus.jsonl'} ({len(corpus_urls)} URLs already in corpus)\n")
    for r in results:
        print(f"  [{r.company}] {r.title}")
        print(f"    {r.url}")
        if r.url in corpus_urls:
            print("    -> skipped (already in corpus)")
            print()
            continue
        if should_skip_url(r.url):
            print("    -> skipped (excluded domain)")
            print()
            continue
        raw = crawl_url(client, r.url)
        if not raw:
            print("    -> (no content extracted)")
            print()
            continue
        content = clean_content(raw)
        valid, rejection_reason = is_valid_document(content, r.url, r.title)
        if not valid:
            quality = score_quality(content, r.url, r.title)
            source_type = infer_source_type(r.url, r.title, content)
            log_rejected(
                r,
                content,
                rejection_reason or "unknown",
                corpus_dir=corpus_dir,
                source_type=source_type,
                content_quality=quality,
            )
            print(f"    -> rejected ({rejection_reason})")
            print()
            continue
        quality = score_quality(content, r.url, r.title)
        source_type = infer_source_type(r.url, r.title, content)
        saved_path = save_document(
            r,
            content,
            corpus_dir=corpus_dir,
            source_type=source_type,
            content_quality=quality,
            is_valid=True,
            rejection_reason=None,
            existing_urls=corpus_urls,
        )
        if not saved_path:
            print("    -> skipped (already in corpus)")
            print()
            continue
        print(f"    -> saved ({len(content)} chars, quality={quality:.2f}, source={source_type})")
        print(content[:300], "..." if len(content) > 300 else "")
        print()

if __name__ == "__main__":
    main()
