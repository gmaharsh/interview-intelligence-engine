from __future__ import annotations

"""Source-aware hierarchical chunking strategies.

This module defines:
- Section: logical document sections (Level A).
- Chunk: final chunks ready for embedding.
- BaseChunker and concrete per-source implementations.
- Shared helpers for structure detection and sub-chunking.
"""

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Tuple
import re
import uuid


TOKEN_APPROX_PER_WORD = 0.75


@dataclass
class Section:
    section_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    parent_section_id: str | None
    section_title: str | None
    chunk_text: str
    chunk_type: str
    token_count: int
    metadata: Dict[str, Any]


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\w+", text)
    return int(len(words) * TOKEN_APPROX_PER_WORD)


def _split_into_paragraphs(text: str) -> List[str]:
    """Split on blank lines into paragraph blocks."""
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.rstrip())
        else:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _split_into_sentences(text: str) -> List[str]:
    """Very simple sentence splitter; good enough for sub-chunking."""
    if not text:
        return []
    # Split on ., ?, ! followed by space or end.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _clean_official_content(text: str) -> str:
    """Drop bylines, read-time labels, and obvious image captions from official pages."""
    if not text:
        return ""
    cleaned: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        # Byline / author + read time.
        if lower.startswith("written by "):
            continue
        if " min read" in lower or "minute read" in lower:
            continue
        # Obvious photo / image captions: short, no period, and contains visual words.
        if (
            len(line) < 80
            and "." not in line
            and any(
                w in lower
                for w in (
                    "professional",
                    "smiling",
                    "colleagues",
                    "photo",
                    "image",
                    "portrait",
                    "headshot",
                )
            )
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def subchunk_section(
    section_text: str,
    *,
    target_tokens: int = 400,
    max_tokens: int = 700,
    overlap_tokens: int = 50,
    parent_section_id: str | None = None,
    section_title: str | None = None,
    base_metadata: Dict[str, Any] | None = None,
    chunk_type: str = "section_paragraph_group",
) -> List[Chunk]:
    """Split a section into size-controlled chunks using paragraphs→sentences."""
    base_metadata = dict(base_metadata or {})
    chunks: List[Chunk] = []

    # Fast path if small.
    total_tokens = _approx_token_count(section_text)
    if total_tokens <= max_tokens:
        chunk_id = str(uuid.uuid4())
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                parent_section_id=parent_section_id,
                section_title=section_title,
                chunk_text=section_text.strip(),
                chunk_type=chunk_type,
                token_count=total_tokens,
                metadata=base_metadata,
            )
        )
        return chunks

    paragraphs = _split_into_paragraphs(section_text)
    current: List[str] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current, current_tokens, chunks
        if not current:
            return
        text = "\n\n".join(current).strip()
        tokens = _approx_token_count(text)
        chunk_id = str(uuid.uuid4())
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                parent_section_id=parent_section_id,
                section_title=section_title,
                chunk_text=text,
                chunk_type=chunk_type,
                token_count=tokens,
                metadata=base_metadata,
            )
        )
        current = []
        current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_token_count(para)
        if para_tokens > max_tokens:
            # Break this huge paragraph by sentences.
            flush_current()
            sentences = _split_into_sentences(para)
            sent_buf: List[str] = []
            sent_tokens = 0
            for sent in sentences:
                t = _approx_token_count(sent)
                if sent_tokens + t > max_tokens and sent_buf:
                    text = " ".join(sent_buf).strip()
                    tokens = _approx_token_count(text)
                    chunk_id = str(uuid.uuid4())
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            parent_section_id=parent_section_id,
                            section_title=section_title,
                            chunk_text=text,
                            chunk_type=chunk_type,
                            token_count=tokens,
                            metadata=base_metadata,
                        )
                    )
                    sent_buf = []
                    sent_tokens = 0
                sent_buf.append(sent)
                sent_tokens += t
            if sent_buf:
                text = " ".join(sent_buf).strip()
                tokens = _approx_token_count(text)
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        parent_section_id=parent_section_id,
                        section_title=section_title,
                        chunk_text=text,
                        chunk_type=chunk_type,
                        token_count=tokens,
                        metadata=base_metadata,
                    )
                )
            continue

        if current_tokens + para_tokens > max_tokens and current:
            flush_current()

        current.append(para)
        current_tokens += para_tokens

    flush_current()

    # Add small overlap between adjacent chunks if requested.
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped: List[Chunk] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            prev = overlapped[-1]
            # Take tail from previous chunk.
            prev_sents = _split_into_sentences(prev.chunk_text)
            tail: List[str] = []
            acc = 0
            for sent in reversed(prev_sents):
                t = _approx_token_count(sent)
                if acc + t > overlap_tokens:
                    break
                tail.insert(0, sent)
                acc += t
            if tail:
                new_text = " ".join(tail + ["", ch.chunk_text]).strip()
                ch = Chunk(
                    chunk_id=ch.chunk_id,
                    parent_section_id=ch.parent_section_id,
                    section_title=ch.section_title,
                    chunk_text=new_text,
                    chunk_type=ch.chunk_type,
                    token_count=_approx_token_count(new_text),
                    metadata=ch.metadata,
                )
            overlapped.append(ch)
        chunks = overlapped

    return chunks


def detect_sections(text: str) -> List[Section]:
    """Detect coarse sections using headings and step markers."""
    lines = text.splitlines()
    sections: List[Section] = []
    current_title = ""
    current_lines: List[str] = []

    heading_pattern = re.compile(
        r"^(#+\s+.+|[A-Z][A-Za-z0-9 \-]{0,80}:?|Step\s+\d+[:.\-].+)$"
    )

    def flush_section() -> None:
        nonlocal current_title, current_lines
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            current_lines = []
            return
        sec_id = str(uuid.uuid4())
        sections.append(
            Section(
                section_id=sec_id,
                title=current_title.strip() or None or "",
                text=section_text,
            )
        )
        current_lines = []
        current_title = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            current_lines.append("")
            continue
        if heading_pattern.match(line):
            flush_section()
            current_title = re.sub(r"^#+\s*", "", line).strip()
            continue
        current_lines.append(line)

    flush_section()

    if not sections and text.strip():
        sec_id = str(uuid.uuid4())
        sections.append(Section(section_id=sec_id, title="", text=text.strip()))

    return sections


class BaseChunker:
    """Base interface for per-source chunkers."""

    source_types: Tuple[str, ...] = ()

    def chunk(self, doc: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

    @staticmethod
    def _base_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
        meta_keys = [
            "url",
            "title",
            "company",
            "query",
            "source_type",
            "content_quality",
            "fetched_at",
        ]
        meta = {k: doc.get(k) for k in meta_keys if k in doc}
        source_type = doc.get("source_type") or ""
        meta["is_official"] = source_type == "official_company_page"
        return meta


class OfficialCompanyPageChunker(BaseChunker):
    source_types = ("official_company_page",)

    def chunk(self, doc: Dict[str, Any]) -> List[Chunk]:
        content = _clean_official_content((doc.get("content") or "").strip())
        if not content:
            return []
        base_meta = self._base_metadata(doc)
        sections = detect_sections(content)
        chunks: List[Chunk] = []
        for sec in sections:
            if not sec.text.strip():
                continue
            sec_chunks = subchunk_section(
                sec.text,
                parent_section_id=sec.section_id,
                section_title=sec.title or None,
                base_metadata=base_meta,
                target_tokens=400,
                max_tokens=700,
                overlap_tokens=50,
                chunk_type="section_paragraph_group",
            )
            chunks.extend(sec_chunks)
        return chunks


class SocialPageChunker(BaseChunker):
    """Chunker for social / reddit-like content."""

    source_types = ("social",)

    def chunk(self, doc: Dict[str, Any]) -> List[Chunk]:
        content = (doc.get("content") or "").strip()
        if not content:
            return []
        base_meta = self._base_metadata(doc)

        # Heuristic: split original post vs comments using simple markers.
        lines = content.splitlines()
        op_lines: List[str] = []
        comment_blocks: List[str] = []
        current_comment: List[str] = []
        in_comments = False

        for raw in lines:
            line = raw.strip()
            if not line:
                if in_comments and current_comment:
                    comment_blocks.append("\n".join(current_comment).strip())
                    current_comment = []
                elif not in_comments and op_lines:
                    op_lines.append("")
                continue
            if re.match(r"^(Comment|Reply)\b", line, re.IGNORECASE):
                in_comments = True
                if current_comment:
                    comment_blocks.append("\n".join(current_comment).strip())
                    current_comment = []
                continue
            if not in_comments:
                op_lines.append(line)
            else:
                current_comment.append(line)

        if current_comment:
            comment_blocks.append("\n".join(current_comment).strip())

        chunks: List[Chunk] = []

        if op_lines:
            op_text = "\n".join(op_lines).strip()
            sec_id = str(uuid.uuid4())
            op_chunks = subchunk_section(
                op_text,
                parent_section_id=sec_id,
                section_title=doc.get("title") or "Original post",
                base_metadata=base_meta,
                target_tokens=350,
                max_tokens=650,
                overlap_tokens=40,
                chunk_type="post_or_comment",
            )
            chunks.extend(op_chunks)

        # Take top-N longest comments as separate sections; rest grouped.
        if comment_blocks:
            sorted_comments = sorted(
                comment_blocks, key=lambda t: len(t), reverse=True
            )
            top = sorted_comments[:5]
            tail = sorted_comments[5:]

            for idx, txt in enumerate(top, start=1):
                sec_id = str(uuid.uuid4())
                sec_title = f"Top comment {idx}"
                comment_chunks = subchunk_section(
                    txt,
                    parent_section_id=sec_id,
                    section_title=sec_title,
                    base_metadata=base_meta,
                    target_tokens=300,
                    max_tokens=600,
                    overlap_tokens=30,
                    chunk_type="post_or_comment",
                )
                chunks.extend(comment_chunks)

            if tail:
                sec_id = str(uuid.uuid4())
                misc_text = "\n\n".join(tail)
                misc_chunks = subchunk_section(
                    misc_text,
                    parent_section_id=sec_id,
                    section_title="Other comments",
                    base_metadata=base_meta,
                    target_tokens=350,
                    max_tokens=650,
                    overlap_tokens=30,
                    chunk_type="post_or_comment",
                )
                chunks.extend(misc_chunks)

        return chunks


class FallbackChunker(BaseChunker):
    """Conservative paragraph/sentence-based chunker for unknown/article/job_board."""

    source_types = ("article", "job_board", "unknown", "video_page")

    def chunk(self, doc: Dict[str, Any]) -> List[Chunk]:
        content = (doc.get("content") or "").strip()
        if not content:
            return []
        base_meta = self._base_metadata(doc)
        sections = detect_sections(content)
        chunks: List[Chunk] = []
        for sec in sections:
            if not sec.text.strip():
                continue
            sec_chunks = subchunk_section(
                sec.text,
                parent_section_id=sec.section_id,
                section_title=sec.title or None,
                base_metadata=base_meta,
                target_tokens=400,
                max_tokens=700,
                overlap_tokens=40,
                chunk_type="section_paragraph_group",
            )
            chunks.extend(sec_chunks)
        return chunks


def build_chunker_registry() -> Dict[str, BaseChunker]:
    """Return mapping from source_type to chunker instance."""
    registry: Dict[str, BaseChunker] = {}
    all_chunkers: List[BaseChunker] = [
        OfficialCompanyPageChunker(),
        SocialPageChunker(),
        FallbackChunker(),
    ]
    for ch in all_chunkers:
        for st in ch.source_types:
            registry[st] = ch
    return registry

