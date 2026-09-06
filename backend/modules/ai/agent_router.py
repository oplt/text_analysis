import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.modules.ai.application.agent_service import AgentService
from backend.modules.ai.dependencies import enforce_ai_generation_rate_limit
from backend.modules.ai.schemas import AiRunRequest, AiRunResponse
from backend.modules.ai.serializers import run_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


class AgentRunRequest(AiRunRequest):
    agent_id: str = "default"
    run_id: str | None = None
    project_id: str | None = None
    user_message: str | None = Field(default=None, max_length=8000)


class AgentRunResponse(AiRunResponse):
    memory_run_id: str


@router.post("/runs", response_model=AgentRunResponse, status_code=201)
async def create_agent_run(
    payload: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_ai_generation_rate_limit),
):
    service = AgentService(db)
    try:
        run, memory_run_id, _working = await asyncio.wait_for(
            service.run_agent_prompt(
                current_user,
                prompt_template_key=payload.prompt_template_key,
                prompt_version_id=payload.prompt_version_id,
                variables=payload.variables,
                retrieval_query=payload.retrieval_query,
                document_ids=payload.document_ids,
                top_k=payload.top_k,
                review_required=payload.review_required,
                agent_id=payload.agent_id,
                run_id=payload.run_id,
                project_id=payload.project_id,
                user_message=payload.user_message,
            ),
            timeout=settings.AI_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Agent run timed out") from exc
    base = run_to_response(run)
    return AgentRunResponse(**base.model_dump(), memory_run_id=memory_run_id)
