"""Optional semantic embedding providers for research analyses (§45).

Embeddings stay an *optional* analysis family — never required for DFM,
frequencies, classification baselines, or topic models. Research code must
not treat RAG retrieval chunks as canonical document text.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding provider contract."""

    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class UnavailableEmbeddingProvider:
    """Default provider: honest failure until a real backend is configured."""

    name = "unavailable"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise ValueError(
            "Semantic embeddings are not configured for text research. "
            "Configure an EmbeddingProvider (optional analysis family) — "
            "embeddings are not part of the standard quantitative path."
        )


def get_default_embedding_provider() -> EmbeddingProvider:
    """Resolve the active provider (currently: unavailable stub)."""
    return UnavailableEmbeddingProvider()


def describe_embedding_capabilities() -> dict[str, Any]:
    return {
        "available": False,
        "provider": get_default_embedding_provider().name,
        "notes": [
            "Embeddings are optional and not wired into the default DFM/classifier path.",
            "Do not use RAG chunk embeddings as research canonical text.",
        ],
    }
