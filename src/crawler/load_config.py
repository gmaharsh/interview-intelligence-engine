"""Load crawler config from YAML and env."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root (one level up from src/crawler).
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


def get_config_path() -> Path:
    """Path to config.yaml (next to this package)."""
    return Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    """Load config.yaml and return merged dict. Raises if file missing."""
    path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def get_companies() -> list[str]:
    """List of company names to search (flattens categories if present)."""
    cfg = load_config()
    raw = cfg.get("companies") or []
    if isinstance(raw, list):
        return raw
    # Categories: { "FAANG": [...], "Big Tech": [...], ... }
    out: list[str] = []
    for names in raw.values():
        if isinstance(names, list):
            out.extend(names)
        else:
            out.append(str(names))
    return out


def get_query_template() -> str:
    """Query template with {company} placeholder."""
    cfg = load_config()
    return cfg.get("query_template") or "{company} interviewing"


def get_search_options() -> dict:
    """Tavily search options from config (max_results, search_depth, etc.)."""
    cfg = load_config()
    return cfg.get("search") or {}


def get_exclude_domains() -> set[str]:
    """Domains to skip when extracting (video, social, paywalled)."""
    cfg = load_config()
    extract = cfg.get("extract") or {}
    raw = extract.get("exclude_domains") or []
    return set(str(d).strip().lower() for d in raw)
