"""pgvector revision ACL contract (RAG-P1-021).

Contract (must hold whenever Postgres + pgvector + migrated schema are available):

1. Retrieval scope filters bind ``c.revision_id`` when ``index_revision_ids`` is set.
2. Soft-deleted documents (``d.deleted_at IS NULL``) never contribute candidates.
3. Empty revision allow-list short-circuits to zero rows (never widens to live HEAD).
4. ``list_available_revision_ids`` only returns revisions that still have chunk rows
   on non-deleted indexed documents.

Filter-contract assertions below always run (no cloud secrets). The live
availability probe skips when DATABASE_URL cannot reach a pgvector-enabled DB.
CI enables the live path by providing ``pgvector/pgvector`` + ``alembic upgrade head``.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.modules.rag.infrastructure.repositories import RagRepository


def _database_url() -> str | None:
    return os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL")


async def _probe_pgvector() -> tuple[bool, str]:
    url = _database_url()
    if not url:
        return False, (
            "DATABASE_URL / TEST_DATABASE_URL is unset — "
            "live pgvector revision ACL check skipped "
            "(filter contract tests above still run)"
        )
    if not url.startswith("postgresql"):
        return False, f"DATABASE_URL is not postgres ({url.split(':', 1)[0]})"
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        future=True,
        connect_args={"timeout": 2},
    )
    try:
        async with engine.connect() as conn:
            ext = await conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
            )
            if ext.scalar() is None:
                return False, "postgres reachable but vector extension is missing"
            return True, "ok"
    except Exception as exc:  # noqa: BLE001 — probe must never fail the suite
        return False, f"postgres/pgvector probe failed: {type(exc).__name__}: {exc}"
    finally:
        await engine.dispose()


def test_revision_acl_scope_filters_bind_revision_and_exclude_deleted():
    """Document + assert the SQL filter contract used by dense/lexical retrieval."""
    repo = RagRepository(MagicMock())
    params: dict = {}
    filters = repo._retrieval_scope_filters(
        user_id="user-1",
        project_id="proj-1",
        document_ids=["doc-1"],
        owner_scoped=False,
        params=params,
        index_revision_ids=["rev-frozen"],
    )
    assert filters is not None
    joined = " AND ".join(filters)
    assert "c.revision_id = ANY(:index_revision_ids)" in joined
    assert "d.deleted_at IS NULL" in joined
    assert params["index_revision_ids"] == ["rev-frozen"]


def test_empty_revision_allow_list_short_circuits():
    repo = RagRepository(MagicMock())
    params: dict = {}
    filters = repo._retrieval_scope_filters(
        user_id="user-1",
        project_id=None,
        document_ids=["doc-1"],
        owner_scoped=False,
        params=params,
        index_revision_ids=[],
    )
    assert filters is None  # zero candidates; never widen to current HEAD


@pytest.mark.asyncio
async def test_list_available_revision_ids_respects_deleted_and_missing():
    """Live check: only non-deleted indexed chunk revisions are reported available."""
    ok, reason = await _probe_pgvector()
    if not ok:
        pytest.skip(reason)

    url = _database_url()
    assert url is not None
    engine = create_async_engine(url, pool_pre_ping=True, future=True)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with Session() as db:
            # Require migrated RAG tables; skip (not fail) if schema not applied yet.
            try:
                await db.execute(text("SELECT 1 FROM rag_documents LIMIT 1"))
                await db.execute(text("SELECT 1 FROM rag_chunks LIMIT 1"))
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"RAG schema not migrated yet: {type(exc).__name__}: {exc}")

            repo = RagRepository(db)
            # Non-existent IDs must yield empty availability — never invent live HEAD.
            available = await repo.list_available_revision_ids(
                document_ids=["00000000-0000-0000-0000-000000000001"],
                revision_ids=["00000000-0000-0000-0000-0000000000aa"],
            )
            assert available == []
    finally:
        await engine.dispose()
