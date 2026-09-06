from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.identity_access.models import User
from backend.modules.memory.api.schemas import (
    MemoryAuditLogResponse,
    MemoryForgetRequest,
    MemoryItemResponse,
    MemoryMetadataResponse,
    MemorySearchRequestBody,
    MemoryWriteRequest,
    MemoryWriteResponse,
)
from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.models import MemoryItem
from backend.modules.memory.infrastructure.memory_config import MemoryConfig
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _memory_to_response(item: MemoryItem) -> MemoryItemResponse:
    meta = item.metadata
    return MemoryItemResponse(
        id=item.id,
        content=item.content,
        metadata=MemoryMetadataResponse(
            memory_level=meta.memory_level.value,
            memory_type=meta.memory_type.value,
            user_id=meta.user_id,
            agent_id=meta.agent_id,
            run_id=meta.run_id,
            project_id=meta.project_id,
            source=meta.source.value,
            source_ref=meta.source_ref,
            confidence=meta.confidence,
            privacy=meta.privacy.value,
            version=meta.version,
            created_at=meta.created_at,
            last_seen_at=meta.last_seen_at,
            last_confirmed_at=meta.last_confirmed_at,
        ),
        score=item.score,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _require_memory_enabled() -> None:
    if not MemoryConfig.from_settings().enabled:
        raise HTTPException(status_code=503, detail="Memory service is disabled")


@router.get("", response_model=PaginatedResponse[MemoryItemResponse])
async def list_memories(
    memory_level: str | None = None,
    project_id: str | None = None,
    agent_id: str = "default",
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    items, total = await service.list_user_memories(
        user_id=current_user.id,
        memory_level=memory_level,
        project_id=project_id,
        agent_id=agent_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_memory_to_response(item) for item in items],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/search", response_model=list[MemoryItemResponse])
async def search_memories(
    payload: MemorySearchRequestBody,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    items = await service.recall(
        user_id=current_user.id,
        agent_id=payload.agent_id,
        query=payload.query,
        run_id=payload.run_id,
        project_id=payload.project_id,
        memory_levels=payload.memory_levels,
        limit=payload.limit,
    )
    return [_memory_to_response(item) for item in items]


@router.get("/audit", response_model=PaginatedResponse[MemoryAuditLogResponse])
async def list_audit_logs(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    logs, total = await service.audit_repo.list_for_user(
        current_user.id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [MemoryAuditLogResponse.model_validate(log) for log in logs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("", response_model=MemoryWriteResponse, status_code=201)
async def create_memory(
    payload: MemoryWriteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    if not MemoryConfig.from_settings().write_enabled:
        raise HTTPException(status_code=503, detail="Memory writes are disabled")
    service = MemoryService(db)
    result = await service.remember(
        user_id=current_user.id,
        agent_id=payload.agent_id,
        content=payload.content,
        memory_level=payload.memory_level,
        memory_type=payload.memory_type,
        run_id=payload.run_id,
        project_id=payload.project_id,
        metadata=payload.metadata,
        source=payload.source,
        source_ref=payload.source_ref,
        confidence=payload.confidence,
        privacy=payload.privacy,
    )
    if not result.accepted and result.rejection_reason == "mem0_unavailable":
        raise HTTPException(status_code=503, detail="Memory engine unavailable")
    return MemoryWriteResponse(
        memory_id=result.memory_id,
        memory_level=result.memory_level.value,
        memory_type=result.memory_type.value,
        accepted=result.accepted,
        rejection_reason=result.rejection_reason,
    )


@router.post("/forget/{memory_id}", status_code=204)
async def forget_memory(
    memory_id: str,
    payload: MemoryForgetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    await service.forget(
        user_id=current_user.id,
        memory_id=memory_id,
        reason=payload.reason,
        is_admin=current_user.is_admin,
    )


@router.get("/{memory_id}", response_model=MemoryItemResponse)
async def get_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    item = await service.get_memory(user_id=current_user.id, memory_id=memory_id)
    return _memory_to_response(item)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_memory_enabled()
    service = MemoryService(db)
    await service.forget(
        user_id=current_user.id,
        memory_id=memory_id,
        reason="user_deleted",
        is_admin=current_user.is_admin,
    )
