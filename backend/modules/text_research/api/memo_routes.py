"""Research memo CRUD and source-backed save endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.schemas import (
    ResearchMemoCreate,
    ResearchMemoResponse,
    ResearchMemoSourceRequest,
    ResearchMemoUpdate,
)
from backend.modules.text_research.application.research_memo_service import ResearchMemoService
from backend.modules.text_research.domain.enums import ResearchMemoSourceType

router = APIRouter(tags=["research-memos"])


def _response(memo) -> ResearchMemoResponse:
    return ResearchMemoResponse(**ResearchMemoService.to_dict(memo))


@router.post("/projects/{project_id}/memos", response_model=ResearchMemoResponse)
async def create_research_memo(
    project_id: str,
    body: ResearchMemoCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    memo = await ResearchMemoService(db).create(
        user_id=current_user.id,
        project_id=project_id,
        corpus_id=body.corpus_id,
        title=body.title,
        body=body.body,
        source_type=body.source_type,
        citations=body.citations,
        claims=body.claims,
        provenance=body.provenance,
        evidence_revision_hash=body.evidence_revision_hash,
        originating_assistant_message_id=body.originating_assistant_message_id,
        originating_synthesis_run_id=body.originating_synthesis_run_id,
    )
    return _response(memo)


@router.get("/projects/{project_id}/memos", response_model=list[ResearchMemoResponse])
async def list_research_memos(
    project_id: str,
    corpus_id: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    memos = await ResearchMemoService(db).list(
        user_id=current_user.id,
        project_id=project_id,
        corpus_id=corpus_id,
        include_archived=include_archived,
    )
    return [_response(memo) for memo in memos]


@router.get("/memos/{memo_id}", response_model=ResearchMemoResponse)
async def get_research_memo(
    memo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _response(await ResearchMemoService(db).get(memo_id=memo_id, user_id=current_user.id))


@router.patch("/memos/{memo_id}", response_model=ResearchMemoResponse)
async def update_research_memo(
    memo_id: str,
    body: ResearchMemoUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _response(
        await ResearchMemoService(db).update(
            memo_id=memo_id, user_id=current_user.id, title=body.title, body=body.body
        )
    )


@router.post("/memos/{memo_id}/archive", response_model=ResearchMemoResponse)
async def archive_research_memo(
    memo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _response(
        await ResearchMemoService(db).archive(memo_id=memo_id, user_id=current_user.id)
    )


@router.post("/corpora/{corpus_id}/memos/from-assistant", response_model=ResearchMemoResponse)
async def save_assistant_as_research_memo(
    corpus_id: str,
    body: ResearchMemoSourceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ResearchMemoService(db)
    corpus = await service.get_corpus_or_404(corpus_id, user_id=current_user.id)
    if not body.message_id:
        raise HTTPException(status_code=422, detail="message_id is required")
    memo = await service.create(
        user_id=current_user.id,
        project_id=corpus.project_id,
        corpus_id=corpus_id,
        title=body.title,
        body=body.body,
        source_type=ResearchMemoSourceType.ASSISTANT_ANSWER.value,
        originating_assistant_message_id=body.message_id,
    )
    return _response(memo)


@router.post("/corpora/{corpus_id}/memos/from-synthesis", response_model=ResearchMemoResponse)
async def save_synthesis_as_research_memo(
    corpus_id: str,
    body: ResearchMemoSourceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ResearchMemoService(db)
    corpus = await service.get_corpus_or_404(corpus_id, user_id=current_user.id)
    if not body.run_id:
        raise HTTPException(status_code=422, detail="run_id is required")
    memo = await service.create(
        user_id=current_user.id,
        project_id=corpus.project_id,
        corpus_id=corpus_id,
        title=body.title,
        body=body.body,
        source_type=ResearchMemoSourceType.ANALYSIS_RESULT.value,
        originating_synthesis_run_id=body.run_id,
    )
    return _response(memo)
