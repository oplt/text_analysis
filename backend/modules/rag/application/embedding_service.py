from __future__ import annotations

from time import perf_counter

from backend.lib.embedding_cache import embed_texts_with_cache
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.langchain_embeddings import LangChainEmbeddingAdapter
from backend.modules.rag.infrastructure.rag_config import RagConfig


class EmbeddingService:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()
        self._adapter = LangChainEmbeddingAdapter(self.config)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        started = perf_counter()
        vectors = await embed_texts_with_cache(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
            texts=texts,
            embed_fn=self._adapter.embed_texts,
            dimensions=getattr(self.config, "embedding_dimensions", None),
            model_version=getattr(self.config, "embedding_model_version", None),
            preprocessing_version=getattr(self.config, "embedding_preprocessing_version", None),
        )
        expected = getattr(self.config, "embedding_dimensions", None)
        invalid = (
            [index for index, vector in enumerate(vectors) if len(vector) != expected]
            if isinstance(expected, int)
            else []
        )
        if invalid and expected is not None:
            raise ValueError(
                "embedding provider returned vectors with unexpected dimension; "
                f"expected {expected}"
            )
        metrics.rag_embedding_latency_ms.observe((perf_counter() - started) * 1000)
        return vectors
