from __future__ import annotations

import logging

from backend.modules.ai.providers import AiProviderRegistry
from backend.modules.rag.infrastructure.rag_config import RagConfig

logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter:
    """Embedding adapter — uses existing AI provider registry, not hardcoded OpenAI."""

    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()
        self._providers = AiProviderRegistry()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        provider = self._providers.get(self.config.embedding_provider)
        return await provider.embed_texts(texts, model=self.config.embedding_model)
