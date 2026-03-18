from __future__ import annotations

"""Query rewriting layer for the interview RAG pipeline.

Components:
1. extract_intent        - regex-based company/topic/source detection (no LLM).
2. expand_queries        - Groq LLM paraphrase expansion.
3. generate_hypothetical_answer - HyDE via Groq LLM.
4. reciprocal_rank_fusion       - RRF deduplication and reranking.
5. rewrite_and_search    - top-level orchestrator.
"""

import difflib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document

from operational.constants import GROQ_MODEL, KNOWN_COMPANIES

load_dotenv()


# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Topic keyword map
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "behavioral": [
        "behavioral",
        "behaviour",
        "star method",
        "star format",
        "tell me about a time",
        "leadership principles",
        "lp",
        "situational",
        "soft skills",
        "conflict",
        "teamwork",
        "failure",
        "success story",
    ],
    "coding": [
        "coding",
        "leetcode",
        "algorithm",
        "data structure",
        "dsa",
        "array",
        "linked list",
        "tree",
        "graph",
        "dynamic programming",
        "dp",
        "time complexity",
        "space complexity",
        "big o",
    ],
    "system_design": [
        "system design",
        "lld",
        "hld",
        "low level design",
        "high level design",
        "scalability",
        "distributed",
        "architecture",
        "database design",
        "api design",
        "microservices",
        "load balancer",
        "caching",
    ],
    "resume": [
        "resume",
        "cv",
        "cover letter",
        "application",
        "portfolio",
        "linkedin",
    ],
    "offer": [
        "offer",
        "salary",
        "negotiation",
        "compensation",
        "comp",
        "tc",
        "stock",
        "equity",
        "rsu",
    ],
}

_OFFICIAL_SIGNALS = [
    r"\bofficial\b",
    r"\bcompany says\b",
    r"\baccording to\b",
    r"\bfrom amazon\b",
    r"\bfrom google\b",
    r"\bfrom meta\b",
    r"\bfrom microsoft\b",
    r"\bjob description\b",
    r"\bcareers page\b",
]

_SOCIAL_SIGNALS = [
    r"\breddit\b",
    r"\bcandidates say\b",
    r"\bpeople say\b",
    r"\bfrom experience\b",
    r"\breal candidates\b",
    r"\bactual interview\b",
    r"\bblind\b",
    r"\blevels\.fyi\b",
    r"\binterviewing\.io\b",
]


# ---------------------------------------------------------------------------
# 1. Intent extraction
# ---------------------------------------------------------------------------


def extract_intent(query: str) -> Dict[str, Any]:
    """
    Extract structured intent from the raw query without an LLM.

    Returns a dict with keys:
      - company: str | None
      - topics: list[str]
      - prefer_official: bool
      - prefer_social: bool
    """
    lower = query.lower()
    tokens = re.findall(r"[a-z0-9]+", lower)

    # Company detection: exact substring first, then per-token fuzzy fallback.
    detected_company: Optional[str] = None
    for company in KNOWN_COMPANIES:
        if company.lower() in lower:
            detected_company = company
            break

    if detected_company is None:
        # Fuzzy match: check each query token against each known company name.
        # Use a high cutoff (0.82) so short tokens don't over-match.
        for token in tokens:
            if len(token) < 4:
                continue
            matches = difflib.get_close_matches(
                token,
                [c.lower() for c in KNOWN_COMPANIES],
                n=1,
                cutoff=0.82,
            )
            if matches:
                matched_lower = matches[0]
                detected_company = next(
                    c for c in KNOWN_COMPANIES if c.lower() == matched_lower
                )
                break

    # Topic detection: all matching topics.
    detected_topics: List[str] = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            detected_topics.append(topic)

    prefer_official = any(re.search(p, lower) for p in _OFFICIAL_SIGNALS)
    prefer_social = any(re.search(p, lower) for p in _SOCIAL_SIGNALS)

    # If both signals exist, don't restrict.
    if prefer_official and prefer_social:
        prefer_official = False
        prefer_social = False

    return {
        "company": detected_company,
        "topics": detected_topics,
        "prefer_official": prefer_official,
        "prefer_social": prefer_social,
    }


# ---------------------------------------------------------------------------
# 2. Query paraphrase expansion via Groq
# ---------------------------------------------------------------------------


def expand_queries(query: str, n: int = 4) -> List[str]:
    """
    Generate n alternative phrasings of query using Groq LLM.
    Returns the original query prepended to the paraphrases.
    """
    client = _get_groq_client()
    system_prompt = (
        "You are a query optimizer for an interview preparation RAG system. "
        f"Given a user question, output exactly {n} alternative phrasings that cover "
        "different angles of the same intent. "
        "Output only the alternative queries, one per line, no numbering, no extra text."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        paraphrases = [line.strip() for line in raw.splitlines() if line.strip()][:n]
    except Exception as exc:
        print(f"[WARN] Query expansion failed: {exc}. Falling back to original query.")
        paraphrases = []

    return [query] + paraphrases


# ---------------------------------------------------------------------------
# 3. HyDE — Hypothetical Document Embedding
# ---------------------------------------------------------------------------


def generate_hypothetical_answer(query: str) -> str:
    """
    Generate a short hypothetical expert answer to query using Groq.
    The returned text is embedded rather than the raw query — closer
    to dense corpus passages than a short question string.

    Applied selectively: only called when the query is under-specified
    (fewer than 8 words).
    """
    client = _get_groq_client()
    system_prompt = (
        "You are an expert interview coach. "
        "Write a concise 2–3 sentence answer to the following question as if it came "
        "from a professional interview preparation guide. "
        "Be specific and actionable. Output only the answer, no preamble."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[WARN] HyDE generation failed: {exc}. Falling back to original query.")
        return query


# ---------------------------------------------------------------------------
# 4. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    *,
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Merge and rerank multiple result lists using Reciprocal Rank Fusion.

    RRF score for a document d across n lists:
        score(d) = sum_i( 1 / (rank_i(d) + k) )

    Deduplicates by chunk_id from doc.metadata. Returns list sorted
    by RRF score descending, carrying the original cosine score of the
    best-ranked occurrence as secondary info.
    """
    rrf_scores: Dict[str, float] = {}
    best_doc: Dict[str, Document] = {}
    best_cosine: Dict[str, float] = {}

    for result_list in result_lists:
        for rank, (doc, cosine_score) in enumerate(result_list, start=1):
            chunk_id = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or doc.page_content[:80]
            )
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
            if chunk_id not in best_doc or cosine_score > best_cosine.get(chunk_id, 0.0):
                best_doc[chunk_id] = doc
                best_cosine[chunk_id] = cosine_score

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(best_doc[cid], score) for cid, score in ranked]


# ---------------------------------------------------------------------------
# Navigation noise detector
# ---------------------------------------------------------------------------

_NOISE_SECTION_TITLES = re.compile(
    r"popular articles|related articles|you might also like|see also|"
    r"trending|more from|site navigation|footer|header|breadcrumb",
    re.IGNORECASE,
)

_CAMEL_RUN = re.compile(r"[a-z][A-Z]")  # "InterviewTop" pattern


def _is_navigation_noise(doc: Document) -> bool:
    """Return True for chunks that look like sidebar/navigation boilerplate."""
    section = doc.metadata.get("section_title", "")
    if section and _NOISE_SECTION_TITLES.search(section):
        return True

    text = (doc.page_content or "").strip()
    if not text:
        return True

    # If the text has no sentence-ending punctuation AND contains multiple
    # CamelCase runs glued together ("InterviewTopQuestions"), it's almost
    # certainly a concatenated link list from a sidebar.
    camel_runs = len(_CAMEL_RUN.findall(text))
    has_sentence_end = bool(re.search(r"[.!?]", text))
    if camel_runs >= 3 and not has_sentence_end:
        return True

    # Discard chunks with almost no real content (< 10 tokens).
    token_count = doc.metadata.get("token_count", 0)
    if isinstance(token_count, (int, float)) and token_count < 10:
        return True

    return False


# ---------------------------------------------------------------------------
# 5. Top-level orchestrator
# ---------------------------------------------------------------------------


def rewrite_and_search(
    query: str,
    *,
    top_k: int = 8,
    use_expansion: bool = True,
    use_hyde: bool = True,
    override_company: Optional[str] = None,
    override_source_type: Optional[str] = None,
    only_official: bool = False,
) -> List[Tuple[Document, float]]:
    """
    Full query rewriting pipeline:
      1. Extract intent (company, topics, source preference).
      2. Expand the query into multiple paraphrases via Groq.
      3. Optionally add a HyDE hypothetical answer for short queries.
      4. Run similarity_search_with_score for each query variant.
      5. Fuse results with RRF and return top_k.

    Manual overrides (override_company, override_source_type, only_official)
    take priority over extracted intent.
    """
    from vector_database.query import _build_filter, get_vector_store  # local import avoids circular

    intent = extract_intent(query)

    company = override_company or intent["company"]
    source_type = override_source_type
    if not source_type:
        if intent["prefer_official"]:
            source_type = "official_company_page"
        elif intent["prefer_social"]:
            source_type = "social"

    if only_official:
        qdrant_filter = _build_filter(company, None, only_official=True)
    else:
        qdrant_filter = _build_filter(company, source_type, only_official=False)

    # Build the list of query strings to search.
    query_variants: List[str] = (
        expand_queries(query) if use_expansion else [query]
    )

    # Add HyDE hypothetical answer for under-specified queries.
    if use_hyde and len(query.split()) < 8:
        hypothetical = generate_hypothetical_answer(query)
        if hypothetical and hypothetical != query:
            query_variants.append(hypothetical)

    vector_store = get_vector_store()
    # Fetch more candidates per variant to give RRF enough signal.
    fetch_k = max(top_k * 2, 16)

    result_lists: List[List[Tuple[Document, float]]] = []
    for q in query_variants:
        try:
            results = vector_store.similarity_search_with_score(
                q, k=fetch_k, filter=qdrant_filter
            )
            results = [(doc, score) for doc, score in results if not _is_navigation_noise(doc)]
            if results:
                result_lists.append(results)
        except Exception as exc:
            print(f"[WARN] Search failed for query variant '{q[:60]}': {exc}")

    if not result_lists:
        return []

    fused = reciprocal_rank_fusion(result_lists)
    return fused[:top_k]
