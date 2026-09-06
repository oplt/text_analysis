from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from backend.core.config import settings
from backend.core.pagination import MAX_PAGE_LIMIT
from backend.modules.ai.repository import AiRepository
from backend.modules.identity_access.models import User

DEFAULT_RAG_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using only the provided context when relevant. "
    "Cite sources using the [Source N] labels in the document context (include filename when helpful). "
    "If the context does not contain enough information to answer confidently, say so "
    "explicitly instead of guessing."
)


@dataclass(frozen=True, slots=True)
class RagAnswerPromptSpec:
    template_id: str | None
    version_id: str | None
    provider_key: str
    model_name: str
    system_prompt: str
    response_format: str
    temperature: float
    input_cost_per_million: int
    output_cost_per_million: int

    @property
    def execution_version(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_name=self.model_name,
            response_format=self.response_format,
            temperature=self.temperature,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )


def _default_model_name(provider_key: str) -> str:
    if provider_key == "openai":
        return settings.OPENAI_DEFAULT_MODEL
    if provider_key == "anthropic":
        return settings.ANTHROPIC_DEFAULT_MODEL
    return settings.AI_LOCAL_MODEL_NAME


def _default_prompt_spec() -> RagAnswerPromptSpec:
    provider_key = settings.AI_DEFAULT_PROVIDER
    return RagAnswerPromptSpec(
        template_id=None,
        version_id=None,
        provider_key=provider_key,
        model_name=_default_model_name(provider_key),
        system_prompt=DEFAULT_RAG_ANSWER_SYSTEM_PROMPT,
        response_format="text",
        temperature=0.2,
        input_cost_per_million=0,
        output_cost_per_million=0,
    )


async def resolve_rag_answer_prompt(repo: AiRepository, user: User) -> RagAnswerPromptSpec:
    template_key = settings.RAG_ASK_PROMPT_TEMPLATE_KEY.strip()
    if not template_key:
        return _default_prompt_spec()

    template = await repo.get_prompt_template_by_key_for_user(user.id, template_key)
    if not template:
        return _default_prompt_spec()

    versions, _ = await repo.list_prompt_versions(
        template.id, limit=MAX_PAGE_LIMIT, offset=0
    )
    version = None
    if template.active_version_id:
        version = next((item for item in versions if item.id == template.active_version_id), None)
    if version is None:
        version = next((item for item in versions if item.is_published), None)
    if version is None and versions:
        version = versions[0]
    if version is None:
        return _default_prompt_spec()

    return RagAnswerPromptSpec(
        template_id=template.id,
        version_id=version.id,
        provider_key=version.provider_key,
        model_name=version.model_name,
        system_prompt=version.system_prompt or DEFAULT_RAG_ANSWER_SYSTEM_PROMPT,
        response_format=version.response_format,
        temperature=version.temperature,
        input_cost_per_million=version.input_cost_per_million,
        output_cost_per_million=version.output_cost_per_million,
    )
