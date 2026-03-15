"""Tavily search: company-based queries and URL collection."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from tavily import TavilyClient
import trafilatura

from .load_config import (
    get_companies,
    get_exclude_domains,
    get_query_template,
    get_search_options,
)


@dataclass
class SearchResult:
    """Single search result (one URL to crawl later)."""
    url: str
    title: str
    company: str
    query: str
    score: float = 0.0


def get_tavily_client() -> TavilyClient:
    """Build TavilyClient from TAVILY_API_KEY env."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Add it to .env or export it."
        )
    return TavilyClient(api_key=api_key)


def search_for_company(
    client: TavilyClient,
    company: str,
    query_template: str,
    max_results: int = 10,
    search_depth: str = "basic",
) -> list[SearchResult]:
    """Run Tavily search for one company; return list of SearchResult."""
    query = query_template.format(company=company)
    opts = get_search_options()
    max_results = opts.get("max_results", max_results)
    search_depth = opts.get("search_depth", search_depth)

    response = client.search(
        query,
        max_results=max_results,
        search_depth=search_depth,
    )
    if isinstance(response, dict):
        raw = response.get("results", [])
    else:
        raw = getattr(response, "results", []) or []
    results = []
    for r in raw:
        url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
        title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
        score = r.get("score", 0) if isinstance(r, dict) else getattr(r, "score", 0)
        if url:
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    company=company,
                    query=query,
                    score=score,
                )
            )
    return results


def search_all_companies() -> list[SearchResult]:
    """Load config, run Tavily search per company, return deduplicated URL list."""
    client = get_tavily_client()
    companies = get_companies()
    template = get_query_template()
    seen_urls: set[str] = set()
    out: list[SearchResult] = []
    for company in companies:
        for r in search_for_company(client, company, template):
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                out.append(r)
    return out


def should_skip_url(url: str) -> bool:
    """True if URL should be skipped (excluded domain). Use in main to avoid calling Tavily."""
    return _domain_excluded(url)


def _domain_excluded(url: str) -> bool:
    """True if URL's domain is in config exclude list."""
    try:
        netloc = urlparse(url).netloc or ""
        netloc = netloc.lower().lstrip("www.")
        for d in get_exclude_domains():
            d = d.lstrip("www.")
            if netloc == d or netloc.endswith("." + d):
                return True
    except Exception:
        pass
    return False


def _fetch_with_trafilatura(url: str, timeout: int = 15) -> str:
    """Fetch URL with requests and extract main text with trafilatura. Returns "" on failure."""
    headers = {"User-Agent": "InterviewRAGBot/1.0 (interview prep corpus; +https://github.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        text = trafilatura.extract(resp.content, output_format="text", include_comments=False)
        return (text or "").strip()
    except Exception:
        return ""


def crawl_url(client: TavilyClient, url: str, timeout: float | None = 15) -> str:
    """Extract main content: try Tavily Extract, then requests + trafilatura if empty. Returns "" on failure."""
    if _domain_excluded(url):
        return ""
    try:
        resp = client.extract(urls=url, format="text", timeout=min(60, max(1, int(timeout or 15))))
        if isinstance(resp, dict):
            results = resp.get("results", [])
        else:
            results = getattr(resp, "results", []) or []
        if results:
            r = results[0]
            content = r.get("raw_content", "") if isinstance(r, dict) else getattr(r, "raw_content", "")
            if (content or "").strip():
                return content.strip()
    except Exception:
        pass
    return _fetch_with_trafilatura(url, timeout=int(timeout or 15))