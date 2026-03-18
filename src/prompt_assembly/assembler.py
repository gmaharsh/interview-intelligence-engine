from __future__ import annotations

"""Prompt assembly for the RAG generation layer.

Responsibilities
----------------
1. Accept a user query + a ranked list of retrieved (Document, score) pairs.
2. Deduplicate and truncate chunks so the total context fits within the model's
   context window.
3. Format each chunk as a numbered, attributed source block.
4. Return an ``AssembledPrompt`` dataclass containing:
   - the final system prompt
   - the final user message (query + context block)
   - the ordered list of source citations used
"""

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Token budget — llama-3.3-70b-versatile context is 128 k tokens.
# We reserve space for the system prompt, the query, and the answer.
# ---------------------------------------------------------------------------
_MAX_CONTEXT_TOKENS: int = 6_000   # conservative budget for the context block
_APPROX_CHARS_PER_TOKEN: float = 3.8  # rough heuristic (English prose)
_MAX_CONTEXT_CHARS: int = int(_MAX_CONTEXT_TOKENS * _APPROX_CHARS_PER_TOKEN)

# How many characters to show per chunk (hard cap to keep context diverse)
_MAX_CHUNK_CHARS: int = 1_200

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert interview preparation coach with deep knowledge of
    hiring processes at top technology companies.

    You will be given a question and a set of numbered source passages
    retrieved from a knowledge base of official company pages, blog posts,
    and community discussions about interviews.

    Rules:
    1. Answer the question thoroughly using ONLY the information in the
       provided sources.
    2. Cite every factual claim with its source number in square brackets,
       e.g. [1], [2].
    3. If the sources do not contain enough information to answer, say so
       clearly — do NOT fabricate details.
    4. Structure your answer with clear sections when the question has
       multiple parts.
    5. Be concise but complete. Prefer bullet points for lists of tips or
       steps.
""")


@dataclass
class SourceCitation:
    """A single source used in the assembled prompt."""
    index: int
    title: str
    url: str
    company: Optional[str]
    source_type: Optional[str]
    section_title: Optional[str]
    score: float
    snippet: str          # the exact text included in the context block


@dataclass
class AssembledPrompt:
    """Everything needed for a single LLM generation call."""
    system_prompt: str
    user_message: str               # query + formatted context block
    sources: List[SourceCitation]   # ordered list of sources cited in context
    query: str
    total_context_chars: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.total_context_chars = len(self.user_message)


class PromptAssembler:
    """
    Converts (query, ranked_chunks) → AssembledPrompt.

    Parameters
    ----------
    max_context_chars:
        Hard character budget for the combined context block.
    max_chunk_chars:
        Maximum characters taken from a single chunk.
    max_sources:
        Maximum number of source passages to include.
    """

    def __init__(
        self,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
        max_chunk_chars: int = _MAX_CHUNK_CHARS,
        max_sources: int = 8,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars
        self.max_sources = max_sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        query: str,
        ranked_chunks: List[Tuple[Document, float]],
    ) -> AssembledPrompt:
        """
        Build an AssembledPrompt from a query and a ranked list of chunks.

        Deduplicates by URL+section so the same passage doesn't appear
        twice (can happen when RRF merges overlapping retrieval results).
        Truncates each chunk to ``max_chunk_chars`` and stops adding sources
        once the character budget is exhausted.
        """
        selected = self._select_chunks(ranked_chunks)
        sources = self._build_citations(selected)
        context_block = self._format_context_block(sources)
        user_message = self._format_user_message(query, context_block)

        return AssembledPrompt(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
            sources=sources,
            query=query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_chunks(
        self,
        ranked_chunks: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Deduplicate and budget-cap the chunk list.

        Deduplication key: (url, section_title) — prevents the same
        section from contributing two near-identical passages.
        """
        seen_keys: set[tuple[str, str]] = set()
        selected: List[Tuple[Document, float]] = []
        budget = self.max_context_chars

        for doc, score in ranked_chunks:
            if len(selected) >= self.max_sources:
                break

            meta = doc.metadata or {}
            key = (
                meta.get("url", ""),
                meta.get("section_title", ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            snippet = (doc.page_content or "").strip()[: self.max_chunk_chars]
            if not snippet:
                continue

            if budget - len(snippet) < 0:
                # Include a truncated version if there's at least 200 chars left.
                if budget >= 200:
                    snippet = snippet[:budget]
                else:
                    break

            budget -= len(snippet)
            selected.append((doc, score, snippet))  # type: ignore[arg-type]

        return selected  # type: ignore[return-value]

    def _build_citations(
        self,
        selected: List[Tuple[Document, float]],
    ) -> List[SourceCitation]:
        citations: List[SourceCitation] = []
        for idx, (doc, score, snippet) in enumerate(selected, start=1):  # type: ignore[misc]
            meta = doc.metadata or {}
            citations.append(
                SourceCitation(
                    index=idx,
                    title=meta.get("title") or "Untitled",
                    url=meta.get("url") or "",
                    company=meta.get("company"),
                    source_type=meta.get("source_type"),
                    section_title=meta.get("section_title"),
                    score=float(score),
                    snippet=snippet,
                )
            )
        return citations

    @staticmethod
    def _format_context_block(sources: List[SourceCitation]) -> str:
        """Render sources as a numbered block the LLM can cite from."""
        if not sources:
            return "(No relevant sources were retrieved.)"

        lines: List[str] = ["### Retrieved Sources\n"]
        for src in sources:
            header_parts = [f"[{src.index}]"]
            if src.company:
                header_parts.append(src.company)
            if src.title:
                header_parts.append(src.title)
            if src.section_title and src.section_title != src.title:
                header_parts.append(f"§ {src.section_title}")
            header_parts.append(f"({src.source_type or 'unknown'})")
            header_parts.append(f"<{src.url}>")

            lines.append(" | ".join(header_parts))
            lines.append(src.snippet)
            lines.append("")   # blank line separator

        return "\n".join(lines).rstrip()

    @staticmethod
    def _format_user_message(query: str, context_block: str) -> str:
        return (
            f"Question: {query}\n\n"
            f"{context_block}\n\n"
            "Please answer the question using the sources above."
        )
