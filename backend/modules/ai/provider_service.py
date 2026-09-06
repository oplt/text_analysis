from __future__ import annotations

from backend.core.config import settings
from backend.modules.ai.base_service import AiBaseService
from backend.modules.ai.schemas import AiProviderDescriptor


class AiProviderService(AiBaseService):
    @staticmethod
    def list_provider_descriptors() -> list[AiProviderDescriptor]:
        return [
            AiProviderDescriptor(
                key="local",
                label="Local heuristic",
                supports_generation=True,
                supports_embeddings=True,
            ),
            AiProviderDescriptor(
                key="openai",
                label="OpenAI",
                supports_generation=True,
                supports_embeddings=True,
            ),
            AiProviderDescriptor(
                key="anthropic",
                label="Anthropic",
                supports_generation=True,
                supports_embeddings=settings.AI_EMBEDDING_PROVIDER == "anthropic",
            ),
        ]
