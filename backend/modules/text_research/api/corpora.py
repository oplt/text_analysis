"""Corpus-scoped text-research API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.application.corpus_service import CorpusService

router = APIRouter()


@router.get("/corpora/{corpus_id}/facets")
async def corpus_metadata_facets(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Distinct metadata values and SQL-aggregated document counts."""
    return await CorpusService(db).metadata_facets(corpus_id, user_id=current_user.id)
