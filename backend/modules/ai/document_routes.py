from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.ai.document_service import AiDocumentService
from backend.modules.ai.schemas import (
    AiChunkMatchResponse,
    AiDocumentCreate,
    AiDocumentResponse,
    AiRetrieveRequest,
)
from backend.modules.ai.serializers import _document_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/documents", response_model=PaginatedResponse[AiDocumentResponse])
async def list_documents(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiDocumentService(db)
    documents, total = await service.list_documents(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [_document_to_response(item) for item in documents],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/documents/{document_id}", response_model=AiDocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiDocumentService(db)
    document = await service.get_document(current_user, document_id)
    return _document_to_response(document)


@router.post("/documents", response_model=AiDocumentResponse, status_code=202)
async def create_document(
    payload: AiDocumentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiDocumentService(db)
    document = await service.create_document_from_text(
        current_user,
        title=payload.title,
        description=payload.description,
        content=payload.content,
        content_type=payload.content_type,
        metadata=payload.metadata,
    )
    return _document_to_response(document)


@router.post("/documents/upload", response_model=AiDocumentResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiDocumentService(db)
    document = await service.create_document_from_upload(current_user, file, description)
    return _document_to_response(document)


@router.post("/retrieve", response_model=list[AiChunkMatchResponse])
async def retrieve_chunks(
    payload: AiRetrieveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiDocumentService(db)
    matches = await service.retrieve_chunks(
        current_user,
        query=payload.query,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
    )
    return [AiChunkMatchResponse(**item) for item in matches]
