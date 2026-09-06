from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.ai.provider_service import AiProviderService
from backend.modules.ai.schemas import AiModuleOverviewResponse, AiProviderDescriptor
from backend.modules.ai.serializers import (
    _dataset_to_response,
    _document_to_response,
    _prompt_template_to_response,
    run_to_response,
)
from backend.modules.ai.service import AiService
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/overview", response_model=AiModuleOverviewResponse)
async def get_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiService(db)
    overview = await service.get_overview(current_user)
    return AiModuleOverviewResponse(
        providers=overview["providers"],
        prompt_templates=[
            _prompt_template_to_response(item) for item in overview["prompt_templates"]
        ],
        recent_runs=[run_to_response(item) for item in overview["recent_runs"]],
        documents=[_document_to_response(item) for item in overview["documents"]],
        datasets=[_dataset_to_response(item) for item in overview["datasets"]],
    )


@router.get("/providers", response_model=list[AiProviderDescriptor])
async def list_providers():
    return AiProviderService.list_provider_descriptors()
