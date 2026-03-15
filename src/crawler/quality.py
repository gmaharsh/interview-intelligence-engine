"""Document quality scoring, validation, and cleaning. Reject junk before it hits the corpus."""

import re
from urllib.parse import urlparse

# Reject if content is mostly or contains these (video/social boilerplate).
BAD_PHRASES = [
    "AboutPressCopyrightContact us",
    "AboutPressCopyrightContact usCreators",
    "TermsPrivacyPolicy",
    "How YouTube works",
    "Test new features",
    "CopyrightContact usCreatorsAdvertiseDevelopers",
]

# Min length to consider content substantive (chars).
MIN_CONTENT_LENGTH = 300

# Lines/phrases to strip from content during cleaning (nav, footer, promo).
NOISE_LINE_PATTERNS = [
    r"^Trending\s+news\s+and\s+stories?\s*$",
    r"^Related\s+[Tt]ags?\s*$",
    r"^Amazon\s+named\s+among\s+.*most\s+admired",
    r"^©\s*\d{4}\s+",
    r"^All\s+rights\s+reserved",
    r"^Privacy\s+[Pp]olicy",
    r"^Terms\s+of\s+[Ss]ervice",
    r"^Cookie\s+[Pp]olicy",
    r"^Careers\s*$",
    r"^Press\s*$",
    r"^Contact\s+us\s*$",
    r"^Advertise\s*$",
    r"^Developers\s*$",
    r"^Subscribe\s*$",
    r"^Sign\s+in\s*$",
    r"^Sign\s+up\s*$",
    r"^Follow\s+us\s*$",
    r"^Share\s+this\s*$",
    r"^Loading\.\.\.\s*$",
]
NOISE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in NOISE_LINE_PATTERNS]

# Domains that suggest official company / careers content.
OFFICIAL_DOMAINS = (
    "aboutamazon.com",
    "amazon.jobs",
    "careers.google",
    "jobs.lever.co",
    "company.linkedin.com",
    "blog.",
    "careers.",
    ".careers",
    "jobs.",
    ".jobs",
)


def clean_content(content: str) -> str:
    """Remove common nav/footer/promo lines. Returns cleaned text."""
    if not content or not content.strip():
        return ""
    lines: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop lines that are purely noise.
        if any(p.search(line) for p in NOISE_PATTERNS_COMPILED):
            continue
        # Drop very short lines that look like nav (single words).
        if len(line) < 25 and line.count(" ") < 2 and any(
            line.lower().startswith(w) for w in ("home", "about", "contact", "more", "legal", "follow")
        ):
            continue
        lines.append(line)
    return "\n\n".join(lines).strip()


def infer_source_type(url: str, title: str, content: str) -> str:
    """Infer source_type for metadata. Prefer URL/title over content."""
    url_lower = url.lower()
    domain = (urlparse(url).netloc or "").lower().replace("www.", "")
    for d in OFFICIAL_DOMAINS:
        if d in domain or d in url_lower:
            return "official_company_page"
    if any(x in url_lower for x in ("youtube.com", "youtu.be", "vimeo.com", "spotify.com")):
        return "video_page"
    if any(x in url_lower for x in ("medium.com", "substack.com", "blog.", "article")):
        return "article"
    if any(x in url_lower for x in ("reddit.com", "twitter.com", "x.com", "linkedin.com")):
        return "social"
    if "glassdoor" in url_lower or "indeed" in url_lower:
        return "job_board"
    return "unknown"


# URL patterns that we do not support (video/social with no transcript extraction).
UNSUPPORTED_SOURCE_PATTERNS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "spotify.com",
    "open.spotify.com",
    "instagram.com",
    "facebook.com",
    "fb.com",
    "tiktok.com",
)


def is_valid_document(content: str, url: str, title: str) -> tuple[bool, str | None]:
    """
    Returns (is_valid, rejection_reason).
    rejection_reason is None when valid.
    Uses specific reasons: unsupported_source_type, empty_content, too_short, boilerplate_only.
    """
    if not content or not content.strip():
        return False, "empty_content"
    url_lower = url.lower()
    if any(p in url_lower for p in UNSUPPORTED_SOURCE_PATTERNS):
        return False, "unsupported_source_type"
    text = content.strip()
    if len(text) < MIN_CONTENT_LENGTH:
        return False, "too_short"
    for phrase in BAD_PHRASES:
        if phrase in text:
            return False, "boilerplate_only"
    return True, None


def score_quality(content: str, url: str, title: str) -> float:
    """
    Rough quality score 0.0--1.0 for retrieval usefulness.
    Dense, relevant, clean content scores higher.
    """
    if not content or not content.strip():
        return 0.0
    text = content.strip()
    score = 0.5  # base
    # Length: longer substantive content is better (cap benefit).
    length = len(text)
    if length >= 1000:
        score += 0.15
    elif length >= 500:
        score += 0.1
    elif length >= MIN_CONTENT_LENGTH:
        score += 0.05
    # Penalize known junk.
    for phrase in BAD_PHRASES:
        if phrase in text:
            score -= 0.4
            break
    # Bonus for official / article sources (by URL).
    source = infer_source_type(url, title, text)
    if source == "official_company_page":
        score += 0.2
    elif source == "article":
        score += 0.1
    # Penalize very short or all-caps lines (noise).
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        short_lines = sum(1 for l in lines if len(l.strip()) < 20)
        if short_lines / len(lines) > 0.6:
            score -= 0.15
    return max(0.0, min(1.0, score))
