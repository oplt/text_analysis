"""Ask Corpus / research assistant routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.schemas import (
    AssistantConversationDetailResponse,
    AssistantMessageRequest,
    AssistantMessageResponse,
    AssistantRetrieveRequest,
    AssistantScopeResponse,
    AssistantSynthesizeRequest,
    AssistantThreadCreateRequest,
    AssistantThreadResponse,
)
from backend.modules.text_research.application.corpus_assistant_service import (
    CorpusAssistantService,
)
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService

router = APIRouter(tags=["research-assistant"])


def _thread_response(thread) -> AssistantThreadResponse:
    return AssistantThreadResponse(
        id=thread.id,
        corpus_id=thread.corpus_id,
        project_id=thread.project_id,
        rag_conversation_id=thread.rag_conversation_id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


@router.post(
    "/corpora/{corpus_id}/assistant/conversations",
    response_model=AssistantThreadResponse,
)
async def create_assistant_conversation(
    corpus_id: str,
    body: AssistantThreadCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    thread, _scope = await CorpusAssistantService(db).create_thread(
        corpus_id=corpus_id,
        user=current_user,
        title=body.title,
        document_subset=body.document_ids,
    )
    return _thread_response(thread)


@router.get(
    "/corpora/{corpus_id}/assistant/conversations",
    response_model=list[AssistantThreadResponse],
)
async def list_assistant_conversations(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    threads = await CorpusAssistantService(db).list_threads(
        corpus_id=corpus_id, user_id=current_user.id
    )
    return [_thread_response(t) for t in threads]


@router.get(
    "/assistant/conversations/{thread_id}",
    response_model=AssistantConversationDetailResponse,
)
async def get_assistant_conversation(
    thread_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = CorpusAssistantService(db)
    thread, messages = await service.get_thread(thread_id, user_id=current_user.id)
    scope = await CorpusScopeService(db).resolve(
        corpus_id=thread.corpus_id, user_id=current_user.id
    )
    return AssistantConversationDetailResponse(
        thread=_thread_response(thread),
        messages=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "model_name": m.model_name,
                "retrieval_trace_id": m.retrieval_trace_id,
                "citations": json.loads(m.citations_json or "[]"),
                "claims": json.loads(m.claims_json or "[]"),
                "metadata": json.loads(m.metadata_json or "{}"),
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
        scope=AssistantScopeResponse(**scope.to_dict()),
    )


@router.post(
    "/corpora/{corpus_id}/assistant/messages",
    response_model=AssistantMessageResponse,
)
async def post_assistant_message(
    corpus_id: str,
    body: AssistantMessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await CorpusAssistantService(db).ask(
        corpus_id=corpus_id,
        user=current_user,
        query=body.query,
        thread_id=body.thread_id,
        intent=body.intent,
        document_subset=body.document_ids,
    )
    return AssistantMessageResponse(
        thread_id=result["thread_id"],
        conversation_id=result["conversation_id"],
        message_id=result["message_id"],
        retrieval_trace_id=result["retrieval_trace_id"],
        query=result["query"],
        original_query=result.get("original_query"),
        resolved_retrieval_query=result.get("resolved_retrieval_query"),
        answer=result["answer"],
        citations=result["citations"],
        claims=result["claims"],
        retrieved_chunk_ids=result["retrieved_chunk_ids"],
        model_name=result["model_name"],
        latency_ms=result["latency_ms"],
        no_context_found=result["no_context_found"],
        retrieval_degraded=result["retrieval_degraded"],
        degradation_reason=result["degradation_reason"],
        citation_validation_failed=result["citation_validation_failed"],
        citation_validation_status=result.get("citation_validation_status", "valid"),
        injection_chunks_filtered=result["injection_chunks_filtered"],
        scope=AssistantScopeResponse(**result["scope"]),
        coverage=result["coverage"],
        context_message_ids=result.get("context_message_ids") or [],
        ai_run_id=result.get("ai_run_id"),
        fusion_method=result.get("fusion_method"),
    )


@router.post("/corpora/{corpus_id}/assistant/retrieve")
async def assistant_retrieve(
    corpus_id: str,
    body: AssistantRetrieveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scope, outcome = await CorpusAssistantService(db).retrieve(
        corpus_id=corpus_id,
        user=current_user,
        query=body.query,
        intent=body.intent,
        document_subset=body.document_ids,
        top_k=body.top_k,
        retrieval_mode=body.retrieval_mode,
    )
    return {
        "scope": scope.to_dict(),
        "retrieval_trace_id": outcome.retrieval_trace_id,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "content": c.content,
                "score": c.score,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "rank": c.rank,
                "retrieval_sources": list(c.retrieval_sources),
            }
            for c in outcome.chunks
        ],
        "degraded": outcome.degraded,
        "degradation_reason": outcome.degradation_reason,
        "no_matches": outcome.no_matches,
        "coverage": {
            "documents_in_scope": outcome.coverage.documents_in_scope if outcome.coverage else 0,
            "documents_with_retrieved_evidence": (
                outcome.coverage.documents_with_retrieved_evidence if outcome.coverage else 0
            ),
            "retrieved_passage_count": (
                outcome.coverage.retrieved_passage_count if outcome.coverage else 0
            ),
            "coverage_ratio": outcome.coverage.coverage_ratio if outcome.coverage else 0.0,
        },
        "intent": outcome.intent.value if outcome.intent else None,
        "fusion_method": outcome.fusion_method,
    }


@router.post("/corpora/{corpus_id}/assistant/synthesize")
async def assistant_synthesize(
    corpus_id: str,
    body: AssistantSynthesizeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from backend.modules.text_research.application.corpus_synthesis_service import (
        CorpusSynthesisService,
    )

    return await CorpusSynthesisService(db).synthesize(
        corpus_id=corpus_id,
        user=current_user,
        query=body.query,
        document_subset=body.document_ids,
        async_mode=body.async_mode,
    )


@router.get("/corpora/{corpus_id}/assistant/scope", response_model=AssistantScopeResponse)
async def assistant_scope(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scope = await CorpusScopeService(db).resolve(corpus_id=corpus_id, user_id=current_user.id)
    return AssistantScopeResponse(**scope.to_dict())
