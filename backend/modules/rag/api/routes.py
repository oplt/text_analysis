import asyncio
import json

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.core.text_snippet import text_snippet
from backend.modules.ai.dependencies import enforce_ai_generation_rate_limit
from backend.modules.identity_access.models import User
from backend.modules.rag.api.schemas import (
    RagAskRequest,
    RagAskResponse,
    RagChunkResponse,
    RagCitationResponse,
    RagCoverageResponse,
    RagDocumentResponse,
    RagDocumentUploadResponse,
    RagIngestionJobResponse,
    RagQueryResponse,
    RagRetrievedChunkResponse,
    RagRetrieveRequest,
    RagRetrieveResponse,
)
from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.rag.workers import queue_document_indexing
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _require_rag_enabled() -> None:
    if not RagConfig.from_settings().enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")


def _document_to_response(document) -> RagDocumentResponse:
    return RagDocumentResponse(
        id=document.id,
        user_id=document.user_id,
        project_id=document.project_id,
        organization_id=document.organization_id,
        filename=document.filename,
        original_filename=document.original_filename,
        content_type=document.content_type,
        storage_path=document.storage_path,
        status=document.status,
        source_type=document.source_type,
        metadata=json.loads(document.metadata_json or "{}"),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post("/documents/upload", response_model=RagDocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    project_id: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    service = DocumentIngestionService(db)
    document, job, _ = await service.upload_document(
        user_id=current_user.id,
        filename=file.filename or "upload.bin",
        content_type=file.content_type or "application/octet-stream",
        project_id=project_id,
        upload=file,
    )
    queue_document_indexing(
        document_id=document.id,
        user_id=current_user.id,
        job_id=job.id,
    )
    return RagDocumentUploadResponse(
        document=_document_to_response(document),
        ingestion_job=RagIngestionJobResponse.model_validate(job),
    )


@router.get("/documents", response_model=PaginatedResponse[RagDocumentResponse])
async def list_documents(
    project_id: str | None = None,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    repo = RagRepository(db)
    docs, total = await repo.list_documents_for_user(
        current_user.id,
        project_id=project_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_document_to_response(doc) for doc in docs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/documents/{document_id}", response_model=RagDocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    repo = RagRepository(db)
    document = await repo.get_document(document_id)
    if not document or document.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return _document_to_response(document)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    service = DocumentIngestionService(db)
    await service.delete_document(
        document_id=document_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin,
    )


@router.post(
    "/documents/{document_id}/index",
    response_model=RagIngestionJobResponse,
    status_code=202,
)
async def index_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    service = DocumentIngestionService(db)
    job = await service.enqueue_document_indexing(
        document_id=document_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin,
    )
    return RagIngestionJobResponse.model_validate(job)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=PaginatedResponse[RagChunkResponse],
)
async def list_document_chunks(
    document_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    content_mode: str = Query(
        default="snippet",
        pattern="^(snippet|full)$",
        description="Return truncated chunk previews (snippet) or full chunk text (full).",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    repo = RagRepository(db)
    document = await repo.get_document(document_id)
    if not document or document.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks, total = await repo.list_chunks_for_document(
        document_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )

    def chunk_content(raw: str) -> str:
        if content_mode == "full":
            return raw
        return text_snippet(raw)

    return paginated_response(
        [
            RagChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk_content(chunk.content),
                token_count=chunk.token_count,
                metadata=json.loads(chunk.metadata_json or "{}"),
                revision_id=chunk.revision_id,
            )
            for chunk in chunks
        ],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/retrieve", response_model=RagRetrieveResponse)
async def retrieve_chunks(
    payload: RagRetrieveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    service = RetrievalService(db)
    filters: dict | None = {"owner_scoped": True}
    if payload.document_ids is not None:
        filters["document_ids"] = payload.document_ids
    outcome = await service.retrieve(
        payload.query,
        user_id=current_user.id,
        project_id=payload.project_id,
        top_k=payload.top_k,
        filters=filters,
        intent=payload.intent,
        persist_trace=False,
    )
    coverage = None
    if outcome.coverage is not None:
        coverage = RagCoverageResponse(
            documents_in_scope=outcome.coverage.documents_in_scope,
            documents_with_retrieved_evidence=outcome.coverage.documents_with_retrieved_evidence,
            retrieved_passage_count=outcome.coverage.retrieved_passage_count,
            coverage_ratio=outcome.coverage.coverage_ratio,
        )
    return RagRetrieveResponse(
        chunks=[
            RagRetrievedChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=chunk.score,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                rank=chunk.rank,
                used_in_answer=chunk.used_in_answer,
                index_revision_id=chunk.index_revision_id,
                source_span_ids=(chunk.metadata or {}).get("source_span_ids"),
                char_start=(chunk.metadata or {}).get("char_start"),
                char_end=(chunk.metadata or {}).get("char_end"),
                offset_coordinate_system=(chunk.metadata or {}).get(
                    "offset_coordinate_system"
                ),
                offset_scope=(chunk.metadata or {}).get("offset_scope"),
                offset_scope_id=(chunk.metadata or {}).get("offset_scope_id"),
                source_spans=(chunk.metadata or {}).get("source_spans"),
            )
            for chunk in outcome.chunks
        ],
        degraded=outcome.degraded,
        degradation_reason=outcome.degradation_reason,
        no_matches=outcome.no_matches,
        injection_chunks_filtered=outcome.injection_chunks_filtered,
        intent=outcome.intent.value if outcome.intent else None,
        fusion_method=outcome.fusion_method,
        coverage=coverage,
        retrieval_trace_id=outcome.retrieval_trace_id,
        index_revision_ids=outcome.index_revision_ids,
    )


@router.post("/ask", response_model=RagAskResponse)
async def ask_rag(
    payload: RagAskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_ai_generation_rate_limit),
):
    _require_rag_enabled()
    service = RagAnswerService(db)
    try:
        result = await asyncio.wait_for(
            service.answer(
                payload.query,
                user=current_user,
                project_id=payload.project_id,
                run_id=payload.run_id,
                agent_id=payload.agent_id,
                document_ids=payload.document_ids,
                intent=payload.intent,
            ),
            timeout=settings.RAG_ASK_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="RAG answer timed out") from exc
    coverage = None
    if result.coverage is not None:
        coverage = RagCoverageResponse(
            documents_in_scope=result.coverage.documents_in_scope,
            documents_with_retrieved_evidence=result.coverage.documents_with_retrieved_evidence,
            retrieved_passage_count=result.coverage.retrieved_passage_count,
            coverage_ratio=result.coverage.coverage_ratio,
        )
    return RagAskResponse(
        query=result.query,
        answer=result.answer,
        citations=[
            RagCitationResponse(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                filename=c.filename,
                score=c.score,
                snippet=c.snippet,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                citation_number=c.citation_number,
                used_in_answer=c.used_in_answer,
                section_heading=c.section_heading,
                char_start=c.char_start,
                char_end=c.char_end,
                source_span_ids=c.source_span_ids,
                parent_context_id=c.parent_context_id,
                offset_coordinate_system=c.offset_coordinate_system,
                offset_scope=c.offset_scope,
                offset_scope_id=c.offset_scope_id,
                source_spans=c.source_spans,
            )
            for c in result.citations
        ],
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        model_name=result.model_name,
        latency_ms=result.latency_ms,
        no_context_found=result.no_context_found,
        ai_run_id=result.ai_run_id,
        retrieval_degraded=result.retrieval_degraded,
        memory_degraded=result.memory_degraded,
        degradation_reason=result.degradation_reason,
        injection_chunks_filtered=result.injection_chunks_filtered,
        retrieval_trace_id=result.retrieval_trace_id,
        evidence_revision_hash=result.evidence_revision_hash,
        index_revision_ids=result.index_revision_ids,
        citation_validation_failed=result.citation_validation_failed,
        coverage=coverage,
    )


@router.get("/queries", response_model=PaginatedResponse[RagQueryResponse])
async def list_queries(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    repo = RagRepository(db)
    rows, total = await repo.list_queries_for_user(
        current_user.id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [RagQueryResponse.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/jobs/{job_id}", response_model=RagIngestionJobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_rag_enabled()
    repo = RagRepository(db)
    job = await repo.get_ingestion_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return RagIngestionJobResponse.model_validate(job)
