"""QA: run quality checks on the existing corpus and report results."""

import json
import sys
from pathlib import Path

from .corpus import get_corpus_dir
from .quality import infer_source_type, is_valid_document, score_quality


def load_corpus_docs(corpus_path: Path) -> list[dict]:
    """Load corpus.jsonl into a list of doc dicts."""
    docs: list[dict] = []
    if not corpus_path.exists():
        return docs
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


# Pass rate threshold: exit 0 if pass_rate >= this (e.g. 0.9 for dev), else exit 1.
PASS_RATE_THRESHOLD = 0.9


def run_qa(corpus_dir: Path | None = None, report_path: Path | None = None) -> dict:
    """
    Run quality checks on every document in corpus.jsonl.
    Returns summary dict. Optionally writes report_path (JSONL: one line per doc with qa fields).
    """
    directory = corpus_dir or get_corpus_dir()
    path = directory / "corpus.jsonl"
    docs = load_corpus_docs(path)
    if not docs:
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_urls": [],
            "by_reason": {},
            "by_source_type": {},
            "valid_by_source_type": {},
            "avg_quality": 0.0,
            "lowest_quality_valid": [],
        }

    results: list[dict] = []
    by_reason: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    valid_by_source_type: dict[str, int] = {}
    valid_count = 0
    qualities: list[float] = []

    for doc in docs:
        url = doc.get("url", "")
        title = doc.get("title", "")
        content = (doc.get("content") or "").strip()
        valid, rejection_reason = is_valid_document(content, url, title)
        quality = score_quality(content, url, title)
        source_type = infer_source_type(url, title, content)
        qualities.append(quality)
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
        if valid:
            valid_count += 1
            valid_by_source_type[source_type] = valid_by_source_type.get(source_type, 0) + 1
        else:
            by_reason[rejection_reason or "unknown"] = by_reason.get(rejection_reason or "unknown", 0) + 1
        row = {
            "url": url,
            "title": title,
            "company": doc.get("company"),
            "is_valid": valid,
            "rejection_reason": rejection_reason,
            "content_quality": round(quality, 4),
            "source_type": source_type,
            "content_length": len(content),
        }
        results.append(row)

    valid_results = [r for r in results if r["is_valid"]]
    lowest_quality_valid = sorted(valid_results, key=lambda x: x["content_quality"])[:10]
    avg_quality = round(sum(qualities) / len(qualities), 4) if qualities else 0.0

    summary = {
        "total": len(docs),
        "valid": valid_count,
        "invalid": len(docs) - valid_count,
        "invalid_urls": [r["url"] for r in results if not r["is_valid"]],
        "by_reason": by_reason,
        "by_source_type": by_source_type,
        "valid_by_source_type": valid_by_source_type,
        "avg_quality": avg_quality,
        "lowest_quality_valid": [
            {"url": r["url"], "title": r["title"], "content_quality": r["content_quality"]}
            for r in lowest_quality_valid
        ],
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary


def main() -> None:
    corpus_dir = get_corpus_dir()
    report_path = corpus_dir / "qa_report.jsonl"
    summary = run_qa(corpus_dir=corpus_dir, report_path=report_path)
    total = summary["total"]
    valid = summary["valid"]
    invalid = summary["invalid"]
    print("Corpus QA report")
    print("=" * 50)
    print(f"Corpus: {corpus_dir / 'corpus.jsonl'}")
    print(f"Total documents: {total}")
    print(f"Valid:   {valid}")
    print(f"Invalid: {invalid}")
    pass_rate = valid / total if total else 0.0
    if total:
        print(f"Pass rate: {pass_rate * 100:.1f}%")
    if summary.get("by_source_type"):
        print("\nBy source type:")
        for st, count in sorted(summary["by_source_type"].items(), key=lambda x: -x[1]):
            print(f"  {st}: {count}")
    if summary.get("by_reason"):
        print("\nInvalid by reason:")
        for reason, count in sorted(summary["by_reason"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    avg_q = summary.get("avg_quality")
    if avg_q is not None and total:
        print(f"\nAverage content quality: {avg_q:.2f}")
    lowest = summary.get("lowest_quality_valid", [])
    if lowest:
        print("\nLowest-quality valid docs (first 5):")
        for r in lowest[:5]:
            title_short = r["title"][:60] + "..." if len(r["title"]) > 60 else r["title"]
            print(f"  [{r['content_quality']:.2f}] {title_short}")
            print(f"    {r['url']}")
    if summary.get("invalid_urls"):
        print("\nInvalid URLs (first 20):")
        for url in summary["invalid_urls"][:20]:
            print(f"  {url}")
        if len(summary["invalid_urls"]) > 20:
            print(f"  ... and {len(summary['invalid_urls']) - 20} more")
    print(f"\nReport written to: {report_path}")
    sys.exit(0 if pass_rate >= PASS_RATE_THRESHOLD else 1)


if __name__ == "__main__":
    main()
