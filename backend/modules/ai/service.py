from __future__ import annotations

import asyncio

from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.ai.evaluation_service import AiEvaluationService, _EvaluationCaseResult
from backend.modules.ai.provider_service import AiProviderService
from backend.modules.ai.review_service import AiReviewService
from backend.modules.identity_access.models import User


class AiService(AiEvaluationService, AiReviewService, AiProviderService):
    """Compatibility facade and cross-capability overview coordinator."""

    async def get_overview(self, user: User):
        (
            prompt_templates_result,
            recent_runs_result,
            documents_result,
            datasets_result,
        ) = await asyncio.gather(
            self.repo.list_prompt_templates_for_user(user.id, limit=DEFAULT_PAGE_LIMIT, offset=0),
            self.repo.list_runs_for_user(user.id, limit=10, offset=0),
            self.list_documents(user, limit=DEFAULT_PAGE_LIMIT, offset=0),
            self.repo.list_datasets_for_user(user.id, limit=DEFAULT_PAGE_LIMIT, offset=0),
        )
        prompt_templates, _ = prompt_templates_result
        recent_runs, _ = recent_runs_result
        documents, _ = documents_result
        datasets, _ = datasets_result
        return {
            "providers": self.list_provider_descriptors(),
            "prompt_templates": prompt_templates,
            "recent_runs": recent_runs,
            "documents": documents,
            "datasets": datasets,
        }


__all__ = ["AiService", "_EvaluationCaseResult"]
