from __future__ import annotations

"""Generation layer — the 'G' in RAG.

Uses Mistral via LangChain (ChatMistralAI) for answer synthesis.
Groq remains in query_rewrite.py for query expansion and HyDE.

Public surface
--------------
``generate_answer(query, ranked_chunks)``  →  ``GenerationResult``

``GenerationResult`` carries:
  - answer:   the synthesised text response
  - sources:  the SourceCitation objects that fed the prompt (for display / API)
  - query:    the original question (echo for convenience)
  - model:    the Mistral model that produced the answer
  - usage:    token usage dict (prompt / completion / total)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document

from operational.constants import MISTRAL_MODEL
from prompt_assembly.assembler import AssembledPrompt, PromptAssembler, SourceCitation

load_dotenv()

# ---------------------------------------------------------------------------
# ChatMistralAI client — lazy singleton
# ---------------------------------------------------------------------------

_mistral_llm: Optional[ChatMistralAI] = None


def _get_mistral_llm(temperature: float = 0.2) -> ChatMistralAI:
    """
    Return a ChatMistralAI instance.

    A new instance is created if the temperature differs from the cached one,
    since temperature is baked into the LangChain object at construction time.
    """
    global _mistral_llm
    if _mistral_llm is None or _mistral_llm.temperature != temperature:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in environment.")
        _mistral_llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            api_key=api_key,
            temperature=temperature,
        )
    return _mistral_llm


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Encapsulates a complete RAG generation response."""

    query: str
    answer: str
    sources: List[SourceCitation]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)

    def formatted_sources(self) -> str:
        """Human-readable source list for CLI / API responses."""
        if not self.sources:
            return "No sources."
        lines = []
        for src in self.sources:
            parts = [f"[{src.index}]"]
            if src.company:
                parts.append(src.company)
            parts.append(src.title)
            if src.url:
                parts.append(f"<{src.url}>")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def __str__(self) -> str:
        header = f"Answer (via {self.model})"
        sep = "-" * len(header)
        usage_str = (
            f"tokens — prompt: {self.usage.get('prompt_tokens', '?')}, "
            f"completion: {self.usage.get('completion_tokens', '?')}"
        )
        return (
            f"{header}\n{sep}\n"
            f"{self.answer}\n\n"
            f"Sources:\n{self.formatted_sources()}\n\n"
            f"[{usage_str}]"
        )


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

_DEFAULT_ASSEMBLER = PromptAssembler()


def generate_answer(
    query: str,
    ranked_chunks: List[Tuple[Document, float]],
    *,
    assembler: Optional[PromptAssembler] = None,
    temperature: float = 0.2,
    max_tokens: int = 1_024,
) -> GenerationResult:
    """
    Full RAG generation step using Mistral via LangChain.

    Parameters
    ----------
    query:
        The raw user question.
    ranked_chunks:
        Ordered list of (Document, score) pairs from the retrieval layer.
        The assembler deduplicates, truncates, and formats these into a
        numbered context block for the LLM.
    assembler:
        Optional custom PromptAssembler; defaults to the module-level
        singleton with standard settings.
    temperature:
        Controls response creativity. 0.2 suits factual interview answers.
    max_tokens:
        Maximum tokens in the generated answer.

    Returns
    -------
    GenerationResult
        Contains the answer text, source citations, model name, and
        token usage statistics.
    """
    asm = assembler or _DEFAULT_ASSEMBLER
    prompt: AssembledPrompt = asm.assemble(query, ranked_chunks)

    if not prompt.sources:
        return GenerationResult(
            query=query,
            answer=(
                "I could not find relevant information in the knowledge base "
                "to answer your question. Try rephrasing or broadening your query."
            ),
            sources=[],
            model=MISTRAL_MODEL,
        )

    llm = _get_mistral_llm(temperature=temperature)

    messages = [
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.user_message),
    ]

    response = llm.invoke(messages, max_tokens=max_tokens)

    answer_text = (response.content or "").strip()

    # LangChain surfaces token usage in response_metadata
    usage: Dict[str, int] = {}
    meta = getattr(response, "response_metadata", {}) or {}
    usage_meta = meta.get("token_usage") or meta.get("usage") or {}
    if usage_meta:
        usage = {
            "prompt_tokens": usage_meta.get("prompt_tokens", 0),
            "completion_tokens": usage_meta.get("completion_tokens", 0),
            "total_tokens": usage_meta.get("total_tokens", 0),
        }

    model_name = meta.get("model", MISTRAL_MODEL)

    return GenerationResult(
        query=query,
        answer=answer_text,
        sources=prompt.sources,
        model=model_name,
        usage=usage,
    )
